"""指标编译器：把声明式指标定义编译成可执行 SQL。

对齐行业里的语义层做法（dbt metrics / Cube / LookML）：人只声明
"业务关系"和"归期主体"，join 路径、归期过滤、粒度对齐由编译器推导。

为什么值得这么做：

  1. **join 是数据模型元数据，不该在指标里手抄。** 关系声明一次
     （`tables.yaml` 的 relations），所有指标共用。
  2. **同一个口径不该被写很多遍。** 比率、差值类指标如果手写 SQL，
     "净成交额 = 成交额 - 退款"这条业务关系会在多个地方重复出现，
     改一处漏一处。
  3. **编译产物可以被自动校验。** 编译出的 SQL 跑出的结果必须和评测集
     里的参考 SQL 一致——这一步把"口径"从文档变成了可回归的断言。

编译策略：

  - 所有叶子指标锚定在同一张表 → 展开成单表查询（比率类指标走这条）
  - 否则 → 每个直接依赖编译成一个 CTE，按维度对齐后在外层计算
    （净成交额这类跨表差值走这条）

编译器会主动报告 fan-out 风险：如果为了拿到某个维度必须从父表走到
子表，行数会膨胀，聚合结果会偏高。这类问题在真实数仓里很常见，
所以只警告、不静默处理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from data_agent.semantic.models import Metric, Relation, SemanticLayer


class CompilerError(ValueError):
    """指标定义无法编译成 SQL。"""


@dataclass(frozen=True)
class Anchor:
    """归期主体：统计区间用哪张表、哪个时间字段界定。

    派生指标的锚点会覆盖子指标的默认归期。例：退款金额默认按
    refund_time 归期，但算净成交额时要改用订单的 pay_time——
    这条业务决定写在父指标的 base_table / time_column 上。
    """

    table: str
    time_column: str
    filters: tuple[str, ...] = ()

    def time_ref(self) -> str:
        return f"{self.table}.{self.time_column}"


@dataclass
class CompiledMetric:
    name: str
    sql: str
    tables: list[str] = field(default_factory=list)
    anchor_table: str = ""
    anchor_column: str = ""
    merged: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Step:
    source: str
    target: str
    relation: Relation
    forward: bool  # True = 从声明表走向父表（多对一，不会放大行数）


class MetricCompiler:
    """把语义层里的指标定义编译成可执行 SQL。"""

    def __init__(self, layer: SemanticLayer) -> None:
        self.layer = layer

    # ------------------------------------------------------------------
    # 关系图
    # ------------------------------------------------------------------
    def _neighbours(self, table: str) -> list[_Step]:
        steps: list[_Step] = []
        own = self.layer.table(table)
        if own:
            for r in own.relations:
                steps.append(_Step(table, r.to, r, forward=True))
        for other in self.layer.tables:
            if other.name == table:
                continue
            for r in other.relations:
                if r.to == table:
                    # 反向走（父表 → 子表）：起点永远是当前表
                    steps.append(_Step(table, other.name, r, forward=False))
        return steps

    def find_path(self, src: str, dst: str) -> list[_Step] | None:
        """在关系图上找最短 join 路径（BFS）。"""
        if src == dst:
            return []
        seen = {src}
        queue: list[tuple[str, list[_Step]]] = [(src, [])]
        while queue:
            current, path = queue.pop(0)
            for step in self._neighbours(current):
                if step.target in seen:
                    continue
                new_path = [*path, step]
                if step.target == dst:
                    return new_path
                seen.add(step.target)
                queue.append((step.target, new_path))
        return None

    def _from_and_joins(
        self, base: str, needed: Iterable[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """从 base 出发把 needed 里的表 join 进来，返回 (子句, 警告, 涉及的表)。"""
        clauses = [f"FROM {base}"]
        notes: list[str] = []
        used = {base}
        tables = [base]

        for target in sorted(set(needed)):
            if target in used:
                continue
            path = self.find_path(base, target)
            if path is None:
                raise CompilerError(
                    f"表 {base} 与 {target} 之间没有关系路径，"
                    f"无法自动生成 JOIN（检查 tables.yaml 的 relations）"
                )
            for step in path:
                if step.target in used:
                    continue
                keyword = "JOIN" if step.forward else "LEFT JOIN"
                clauses.append(f"{keyword} {step.target} ON {step.relation.on}")
                if not step.forward:
                    notes.append(
                        f"经过 {step.target} 是从父表走向子表（一对多），"
                        f"会让 {base} 的行数膨胀、聚合结果偏高"
                    )
                used.add(step.target)
                tables.append(step.target)
        return clauses, notes, tables

    # ------------------------------------------------------------------
    # 表达式处理
    # ------------------------------------------------------------------
    def _replace_metric_refs(self, expr: str, mapping: dict[str, str]) -> str:
        """把表达式里的 {指标名} 替换成目标文本。"""

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in mapping:
                raise CompilerError(f"表达式引用了未知指标 {name}：{expr}")
            return mapping[name]

        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, expr)

    def _tables_in(self, expr: str) -> set[str]:
        """表达式里显式引用了哪些表（形如 orders.pay_time）。"""
        return {
            t.name
            for t in self.layer.tables
            if re.search(rf"(?<![\w.]){re.escape(t.name)}\.", expr)
        }

    def leaves(self, metric: Metric) -> list[Metric]:
        """递归展开，返回这个指标最终依赖的基础指标。"""
        out: list[Metric] = []
        for name in self.layer.dependencies(metric):
            dep = self.layer.metric(name)
            if dep is None:
                raise CompilerError(f"指标 {metric.name} 引用了不存在的指标 {name}")
            if dep.is_derived:
                out.extend(self.leaves(dep))
            else:
                out.append(dep)
        return out

    def _expand_to_sql_expr(self, metric: Metric) -> str:
        """把派生指标递归展开成可以直接放进 SELECT 的 SQL 表达式。"""
        assert metric.expression is not None
        mapping: dict[str, str] = {}
        for name in self.layer.dependencies(metric):
            dep = self.layer.metric(name)
            if dep is None:
                raise CompilerError(f"指标 {metric.name} 引用了不存在的指标 {name}")
            inner = (
                self._expand_to_sql_expr(dep) if dep.is_derived else str(dep.definition)
            )
            mapping[name] = f"({inner})"
        return self._replace_metric_refs(metric.expression, mapping)

    # ------------------------------------------------------------------
    # SQL 渲染
    # ------------------------------------------------------------------
    def _time_predicate(
        self, anchor: Anchor, start: str | None, end: str | None
    ) -> str:
        ref = anchor.time_ref()
        if start is None and end is None:
            # 模板模式：留给使用者填具体区间
            return f"{ref} >= TIMESTAMP :start AND {ref} < TIMESTAMP :end"
        parts: list[str] = []
        if start is not None:
            parts.append(f"{ref} >= TIMESTAMP '{start}'")
        if end is not None:
            parts.append(f"{ref} < TIMESTAMP '{end}'")
        return " AND ".join(parts)

    # ------------------------------------------------------------------
    # 维度渲染
    # ------------------------------------------------------------------
    def _dim_expr(self, qualified: str) -> str:
        """维度的分组表达式：字段声明了 `null_label` 就补 COALESCE。

        规则 R005 要求显式说明 NULL 怎么归组。声明一次（tables.yaml），
        编译器统一生效——否则编译出的 SQL 与参考 SQL 的分组集合对不上
        （一个出现 NULL 组，一个是 '未知' 组）。
        """
        table, _, col = qualified.partition(".")
        t = self.layer.table(table)
        if t is not None:
            column = next((c for c in t.columns if c.name == col), None)
            if column is not None and column.null_label:
                return f"COALESCE({qualified}, '{column.null_label}')"
        return qualified

    def _dim_projection(self, dims: list[str], *, friendly: bool) -> str:
        """维度列的输出。

        `friendly=True` 时用字段原名做别名（给用户看的），否则用 d0..dn
        ——后者是编译器内部 CTE 之间对齐用的，不能动。
        """
        parts: list[str] = []
        for i, d in enumerate(dims):
            alias = d.rpartition(".")[2] if friendly else f"d{i}"
            parts.append(f"{self._dim_expr(d)} AS {alias}")
        return ", ".join(parts)

    @staticmethod
    def _uniq(items: Iterable[str]) -> list[str]:
        out: list[str] = []
        for item in items:
            if item not in out:
                out.append(item)
        return out

    def _where(self, anchor: Anchor, metric: Metric, start, end) -> str:
        conditions = self._uniq(
            [self._time_predicate(anchor, start, end), *anchor.filters, *metric.filters]
        )
        return " AND ".join(conditions)

    def _select_for(
        self,
        name: str,
        anchor: Anchor,
        dims: list[str],
        start: str | None,
        end: str | None,
        *,
        value_alias: str = "v",
        friendly: bool = False,
    ) -> str:
        """编译成 SELECT，输出列固定为 d0..dn（若有维度）+ value_alias。"""
        metric = self.layer.metric(name)
        if metric is None:
            raise CompilerError(f"指标 {name} 不存在")

        dim_cols = self._dim_projection(dims, friendly=friendly)
        group_by = f" GROUP BY {', '.join(str(i + 1) for i in range(len(dims)))}" if dims else ""
        dim_prefix = f"{dim_cols}, " if dims else ""

        if not metric.is_derived:
            needed = {anchor.table, metric.base_table}
            # 定义和过滤条件里显式用到的表也要 join 进来：VIP 成交额这类
            # 指标靠 definition 引用 users.is_vip，编译器要自己发现
            needed |= self._tables_in(metric.definition or "")
            for condition in (*anchor.filters, *metric.filters):
                needed |= self._tables_in(condition)
            for d in dims:
                needed |= self._tables_in(d)
            clauses, _, _ = self._from_and_joins(metric.base_table, needed)
            where = self._where(anchor, metric, start, end)
            return (
                f"SELECT {dim_prefix}{metric.definition} AS {value_alias}\n"
                + "\n".join(clauses)
                + f"\nWHERE {where}"
                + group_by
            )

        deps = self.layer.dependencies(metric)
        if not deps:
            raise CompilerError(f"派生指标 {name} 没有引用任何指标")

        ctes: list[str] = []
        cte_names: list[str] = []
        mapping: dict[str, str] = {}
        joins: list[str] = []
        for i, dep in enumerate(deps):
            cte_name = f"{name}__d{i}"
            cte_names.append(cte_name)
            inner = self._select_for(dep, anchor, dims, start, end, value_alias="v")
            ctes.append(f"{cte_name} AS (\n  {inner.replace(chr(10), chr(10) + '  ')}\n)")
            mapping[dep] = f"{cte_name}.v"
            if i == 0:
                joins.append(f"FROM {cte_name}")
            else:
                on = " AND ".join(
                    f"{cte_names[0]}.d{j} = {cte_name}.d{j}" for j in range(len(dims))
                ) or "1 = 1"
                joins.append(f"FULL OUTER JOIN {cte_name} ON {on}")

        assert metric.expression is not None
        expr = self._replace_metric_refs(metric.expression, mapping)

        if dims:
            select_dims = []
            for j in range(len(dims)):
                refs = ", ".join(f"{c}.d{j}" for c in cte_names)
                alias = dims[j].rpartition(".")[2] if friendly else f"d{j}"
                select_dims.append(f"COALESCE({refs}) AS {alias}")
            prefix = ", ".join(select_dims) + ", "
        else:
            prefix = ""

        return (
            "WITH " + ",\n".join(ctes) + "\n"
            f"SELECT {prefix}{expr} AS {value_alias}\n"
            + "\n".join(joins)
        )

    def _can_merge(self, metric: Metric, anchor: Anchor) -> bool:
        """所有叶子都落在锚点表上时，可以合成一条单表查询。"""
        if not metric.is_derived:
            # 基础指标只有在自己的表就是归期主体时才能直接单表查询，
            # 否则 FROM 会挂错表（商品销售额的明细在 order_items、
            # 归期在 orders，必须走 join）
            return metric.base_table == anchor.table
        leaves = self.leaves(metric)
        if not leaves:
            return False
        return all(
            leaf.base_table == anchor.table and leaf.time_table_name() == anchor.table
            for leaf in leaves
        )

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def anchor_for(self, metric: Metric) -> Anchor:
        return Anchor(
            table=metric.time_table_name(),
            time_column=metric.time_column,
            filters=tuple(metric.filters),
        )

    def tables_for(self, metric: Metric) -> list[str]:
        """这个指标最终会用到哪些表（供上下文检索确定必选表）。"""
        if not metric.is_derived:
            needed = {metric.base_table, metric.time_table_name()}
            needed |= self._tables_in(metric.definition or "")
            for condition in metric.filters:
                needed |= self._tables_in(condition)
            return sorted(needed)
        out: list[str] = [self.anchor_for(metric).table]
        for name in self.layer.dependencies(metric):
            dep = self.layer.metric(name)
            if dep is not None:
                out.extend(self.tables_for(dep))
        return self._uniq(out)

    def compile(
        self,
        name: str,
        *,
        dimensions: Iterable[str] = (),
        start: str | None = None,
        end: str | None = None,
        friendly: bool = False,
    ) -> CompiledMetric:
        """编译一个指标。

        `start` / `end` 为 None 时输出带 :start / :end 占位符的模板，
        用于放进提示词；给定具体时间则输出可直接执行的 SQL。

        `friendly=True` 时维度列用字段原名做别名（`city` 而不是 `d0`），
        并给声明了 `null_label` 的字段补 COALESCE——这是给用户/模型直接
        执行的形态。默认 False 保留 d0..dn，供内部 CTE 对齐使用。
        """
        metric = self.layer.metric(name)
        if metric is None:
            raise CompilerError(f"指标 {name} 不存在")

        dims = list(dimensions)
        anchor = self.anchor_for(metric)
        notes: list[str] = []

        if self._can_merge(metric, anchor):
            if metric.is_derived:
                expr = self._expand_to_sql_expr(metric)
            else:
                if metric.definition is None:
                    raise CompilerError(f"基础指标 {name} 缺少 definition")
                expr = metric.definition
            needed = {anchor.table}
            needed |= self._tables_in(expr)
            for condition in (*anchor.filters, *metric.filters):
                needed |= self._tables_in(condition)
            for d in dims:
                needed |= self._tables_in(d)
            clauses, notes, tables = self._from_and_joins(anchor.table, needed)
            dim_cols = self._dim_projection(dims, friendly=friendly)
            prefix = f"{dim_cols}, " if dims else ""
            group_by = (
                f" GROUP BY {', '.join(str(i + 1) for i in range(len(dims)))}"
                if dims
                else ""
            )
            sql = (
                f"SELECT {prefix}{expr} AS {name}\n"
                + "\n".join(clauses)
                + f"\nWHERE {self._where(anchor, metric, start, end)}"
                + group_by
            )
            return CompiledMetric(
                name=name,
                sql=sql,
                tables=tables,
                anchor_table=anchor.table,
                anchor_column=anchor.time_column,
                merged=True,
                notes=notes,
            )

        inner = self._select_for(
            name, anchor, dims, start, end, value_alias=name, friendly=friendly
        )
        tables: list[str] = []
        for m in [metric, *self.leaves(metric)]:
            tables = self._uniq([*tables, m.base_table, m.time_table_name()])
        return CompiledMetric(
            name=name,
            sql=inner,
            tables=tables,
            anchor_table=anchor.table,
            anchor_column=anchor.time_column,
            merged=False,
            notes=notes,
        )

    def compile_all(self, **kwargs) -> dict[str, CompiledMetric]:
        return {m.name: self.compile(m.name, **kwargs) for m in self.layer.metrics}


def build_compiler(layer: SemanticLayer) -> MetricCompiler:
    return MetricCompiler(layer)
