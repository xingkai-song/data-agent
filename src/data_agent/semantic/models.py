"""语义层的数据模型。

语义层不是 DDL 的翻版——它要回答的是"这个字段在业务上意味着什么、
什么时候不能用"。所以模型里刻意保留了 notes / caveats / rationale
这类"给模型看的解释性信息"。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class Metric(BaseModel):
    name: str
    display_name: str
    description: str
    definition: str
    base_table: str
    time_column: str
    # 归期字段所在的表。多数指标与 base_table 相同；像商品销售额这种
    # 需要按订单支付时间归期的指标，归期字段在关联表上。
    time_table: str | None = None
    filters: list[str] = Field(default_factory=list)
    joins: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    owner: str | None = None

    def time_table_name(self) -> str:
        return self.time_table or self.base_table


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

    def global_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.scope == "global"]

    def rules_for(self, *scopes: str) -> list[Rule]:
        wanted = {"global", *scopes}
        return [r for r in self.rules if r.scope in wanted]
