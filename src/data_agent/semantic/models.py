"""语义层的数据模型。

语义层不是 DDL 的翻版——它要回答的是"这个字段在业务上意味着什么、
什么时候不能用"。所以模型里刻意保留了 notes / caveats / rationale
这类"给模型看的解释性信息"。

指标分两类（对应行业里的 base measure / derived metric）：

  - **基础指标**：`definition` 给出基础聚合片段，落在某一张表上
  - **派生指标**：`expression` 由其他指标组合而成，SQL 由编译器生成

派生指标的 SQL 不手写。比率、差值这类指标只要把业务关系写清楚，
join、归期过滤、粒度对齐都由 `semantic/compiler.py` 推导。
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Relation(BaseModel):
    """表间关系（等价于外键声明）。

    这份声明是 join 的唯一来源：指标定义里不再手写 JOIN 语句，
    编译器沿关系图找路径并生成。真实数仓里这层信息来自外键或 ER
    元数据，本项目的数据是用 DataFrame 直接写入的，没有外键约束，
    所以显式补在这里。
    """

    to: str
    on: str
    kind: Literal["many_to_one", "one_to_many"] = "many_to_one"


class Column(BaseModel):
    name: str
    type: str
    description: str
    notes: str | None = None
    values: dict[str, str] | None = None


class Table(BaseModel):
    name: str
    description: str
    grain: str
    columns: list[Column] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def relation_to(self, table: str) -> Relation | None:
        return next((r for r in self.relations if r.to == table), None)


class Metric(BaseModel):
    name: str
    display_name: str
    description: str
    base_table: str
    time_column: str
    # 归期字段所在的表。多数指标与 base_table 相同；像商品销售额这种
    # 需要按订单支付时间归期的指标，归期字段在关联表上。
    time_table: str | None = None
    filters: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    owner: str | None = None

    # 二选一：基础指标的聚合片段，或派生指标的指标名表达式
    definition: str | None = None
    expression: str | None = None

    @model_validator(mode="after")
    def _definition_xor_expression(self) -> Metric:
        has_def = bool(self.definition and self.definition.strip())
        has_expr = bool(self.expression and self.expression.strip())
        if has_def == has_expr:
            raise ValueError(
                f"指标 {self.name} 必须且只能声明 definition 或 expression 之一"
            )
        return self

    @property
    def is_derived(self) -> bool:
        return bool(self.expression and self.expression.strip())

    def time_table_name(self) -> str:
        return self.time_table or self.base_table

    def referenced_metrics(self) -> list[str]:
        """表达式里用 {name} 显式引用了哪些指标。

        显式语法（对齐 dbt 的 ref()、Cube 的 {metric}）换来两件事：
        拼写错误能在加载时报出来，而不是变成一句非法 SQL；
        指标名和 SQL 标识符不会互相误伤。
        """
        if not self.expression:
            return []
        found: list[str] = []
        for name in re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", self.expression):
            if name != self.name and name not in found:
                found.append(name)
        return found


class Rule(BaseModel):
    id: str
    scope: str
    severity: Literal["error", "warning", "info"]
    description: str
    rationale: str


class SemanticLayer(BaseModel):
    tables: list[Table]
    metrics: list[Metric]
    rules: list[Rule]

    # ---- 查询辅助 ----
    def table(self, name: str) -> Table | None:
        return next((t for t in self.tables if t.name == name), None)

    def metric(self, name: str) -> Metric | None:
        return next((m for m in self.metrics if m.name == name), None)

    def metric_names(self) -> set[str]:
        return {m.name for m in self.metrics}

    def dependencies(self, metric: Metric) -> list[str]:
        """派生指标直接依赖了哪些指标。"""
        return metric.referenced_metrics()

    def global_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.scope == "global"]

    def rules_for(self, *scopes: str) -> list[Rule]:
        wanted = {"global", *scopes}
        return [r for r in self.rules if r.scope in wanted]
