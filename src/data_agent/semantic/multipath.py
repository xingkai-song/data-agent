"""多路径对算：用语义层编译器去校验模型手写的 SQL。

静默错误里最难抓的一类是"**数值自洽但口径被替换**"——例如算客单价时把分子
从 GMV 悄悄换成净成交额。`agent/sanity.py` 的四项不变量抓不住它（四个等式
全都不破），只有拿第二条独立路径对账才看得见。

这里就是那条独立路径：**同一个指标，用编译器再算一遍，对不上就报警。**

只在能可靠判定时触发，宁可少报也不误报：

  * 结果是**单行标量**——编译对照只做标量，分组结果不做
  * 结果里有列名能对上语义层里的**任一定义过的指标**。派生指标（比率、差值）
    是口径替换的高发区，但基础指标同样会中招——实测里就有把 gmv 算成
    "扣掉退款后的金额"、列名却仍写 `gmv` 的情况
  * SQL 里的时间区间能解析出来
  * SQL 的过滤条件**不超出**该指标声明的过滤条件——否则本来就是另一个口径
    （例如"只算 VIP 的客单价"），对照没有意义

**已知边界**：分组结果、加了额外过滤的查询、语义层里没有的临时指标，都跳过。
所以它是"能抓一类"，不是"全都能抓"。
"""

from __future__ import annotations

import re
from typing import Any, Callable

import sqlglot
from sqlglot import exp

from data_agent.semantic.compiler import MetricCompiler
from data_agent.semantic.models import Metric, SemanticLayer

_DATE = re.compile(r"'(\d{4}-\d{2}-\d{2})'")
# 允许的相对误差：口径被换掉通常带来百分之几的差异（净额 vs 总额约 4%），
# 而四舍五入带来的差异远小于此。2% 能既容忍舍入、又抓住换口径。
_REL_TOL = 0.02
_ABS_TOL = 0.01


def _resolve_metric(layer: SemanticLayer, text: str) -> Metric | None:
    """列名 → 指标。名字/展示名精确匹配，最后退到最长别名包含。"""
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    for m in layer.metrics:
        if m.name.lower() == low:
            return m
    for m in layer.metrics:
        if m.display_name == t:
            return m
    best, best_len = None, 0
    for m in layer.metrics:
        for alias in m.aliases:
            a = alias.lower()
            if len(a) >= 2 and (a in low or low in a) and len(alias) > best_len:
                best, best_len = m, len(alias)
    return best


def _where_columns(sql: str) -> set[str] | None:
    """SQL 的 WHERE 里引用了哪些列。解析失败返回 None（表示不判定）。"""
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:  # noqa: BLE001
        return None
    cols: set[str] = set()
    for where in tree.find_all(exp.Where):
        for col in where.find_all(exp.Column):
            cols.add(col.name.lower())
    return cols


def _declared_columns(metric: Metric) -> set[str]:
    """该指标口径里本来就有的列（归期字段 + filters 里引用的列）。"""
    allowed = {metric.time_column.lower()}
    for condition in metric.filters:
        try:
            node = sqlglot.parse_one(f"SELECT 1 WHERE {condition}", read="duckdb")
        except Exception:  # noqa: BLE001
            continue
        for col in node.find_all(exp.Column):
            allowed.add(col.name.lower())
    return allowed


def cross_check(
    layer: SemanticLayer,
    sql: str,
    columns: list[str],
    rows: list[list[Any]],
    *,
    execute: Callable[[str], list[tuple]],
    rel_tol: float = _REL_TOL,
    abs_tol: float = _ABS_TOL,
) -> list[str]:
    """用语义层口径复核标量结果，返回不一致的地方（最多 2 条）。"""
    if not columns or len(rows) != 1:
        return []

    dates = sorted(set(_DATE.findall(sql)))
    if len(dates) < 2:
        return []
    start, end = dates[0], dates[-1]

    where_cols = _where_columns(sql)
    if where_cols is None:
        return []

    row = rows[0]
    warnings: list[str] = []
    compiler = MetricCompiler(layer)

    for i, name in enumerate(columns):
        metric = _resolve_metric(layer, name)
        if metric is None:
            continue
        value = row[i] if i < len(row) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        # 该 SQL 额外加了过滤条件 → 本来就是另一个口径，不对照
        if where_cols - _declared_columns(metric):
            continue
        try:
            compiled = compiler.compile(metric.name, start=start, end=end)
            got = execute(compiled.sql)
        except Exception:  # noqa: BLE001 - 对照失败不影响主流程，静默跳过
            continue
        if not got or not got[0]:
            continue
        expected = got[0][0]
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            continue
        expected = float(expected)
        if abs(float(value) - expected) <= max(abs_tol, abs(expected) * rel_tol):
            continue
        warnings.append(
            f"列 `{name}`={float(value):,.6g} 与语义层口径算出的 {expected:,.6g} 对不上"
            f"（{metric.display_name} 的标准算式是 "
            f"{metric.expression or metric.definition}）。"
            "**两条路径不一致说明口径可能被换掉了**，请核对后再下结论"
        )
    return warnings[:2]
