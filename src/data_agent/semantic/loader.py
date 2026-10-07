"""从 YAML 加载语义层，并做一致性校验。

校验的意义：语义层本身写错了，后面所有生成的 SQL 都是错的，而且很难发现。
所以宁可加载时就报错。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from data_agent.config import DEFINITIONS_DIR
from data_agent.semantic.models import Metric, Rule, SemanticLayer, Table


class SemanticLayerError(ValueError):
    """语义层定义不一致。"""


def _read_yaml(path: Path) -> list[dict]:
    if not path.exists():
        raise SemanticLayerError(f"语义层文件不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SemanticLayerError(f"{path.name} 顶层必须是列表")
    return data


def load_semantic_layer(definitions_dir: Path | None = None) -> SemanticLayer:
    d = definitions_dir or DEFINITIONS_DIR
    tables = _parse(Table, _read_yaml(d / "tables.yaml"), "tables.yaml")
    metrics = _parse(Metric, _read_yaml(d / "metrics.yaml"), "metrics.yaml")
    rules = _parse(Rule, _read_yaml(d / "rules.yaml"), "rules.yaml")
    layer = SemanticLayer(tables=tables, metrics=metrics, rules=rules)
    _validate(layer)
    return layer


def _parse(model, rows: list[dict], filename: str):
    """解析单条定义，把 pydantic 的校验错误转成统一的可读错误。"""
    out = []
    for row in rows:
        try:
            out.append(model(**row))
        except ValidationError as exc:
            name = row.get("name") or row.get("id") or "?"
            first = exc.errors()[0]
            raise SemanticLayerError(
                f"{filename} 中 `{name}` 的定义有误：{first['msg']}"
            ) from exc
    return out


def _validate(layer: SemanticLayer) -> None:
    table_names = {t.name for t in layer.tables}
    metric_names = {m.name for m in layer.metrics}
    problems: list[str] = []

    if len(table_names) != len(layer.tables):
        problems.append("存在重名的表")

    rule_ids = [r.id for r in layer.rules]
    if len(set(rule_ids)) != len(rule_ids):
        problems.append("存在重复的规则 id")

    if len(metric_names) != len(layer.metrics):
        problems.append("存在重名的指标")

    # 表关系（join 的唯一来源）
    for t in layer.tables:
        for r in t.relations:
            if r.to == t.name:
                problems.append(f"表 {t.name} 的关系指向自己")
            elif r.to not in table_names:
                problems.append(f"表 {t.name} 的关系指向不存在的表 `{r.to}`")

    for m in layer.metrics:
        if m.base_table not in table_names:
            problems.append(f"指标 {m.name} 的 base_table `{m.base_table}` 不存在")
            continue
        time_table_name = m.time_table_name()
        if time_table_name not in table_names:
            problems.append(
                f"指标 {m.name} 的 time_table `{time_table_name}` 不存在"
            )
            continue
        table = layer.table(time_table_name)
        assert table is not None
        if m.time_column not in table.column_names():
            problems.append(
                f"指标 {m.name} 的 time_column `{m.time_column}` "
                f"不在表 {time_table_name} 中"
            )

        # 派生指标：引用的指标必须存在
        for dep in m.referenced_metrics():
            if dep not in metric_names:
                problems.append(f"派生指标 {m.name} 引用了不存在的指标 `{dep}`")

    cycle = _find_cycle(layer)
    if cycle:
        problems.append("指标存在循环依赖：" + " → ".join(cycle))

    for r in layer.rules:
        if r.scope not in ("global", *metric_names, *table_names):
            problems.append(f"规则 {r.id} 的 scope `{r.scope}` 未匹配任何指标或表")

    if problems:
        raise SemanticLayerError("语义层定义有问题：\n  - " + "\n  - ".join(problems))


def _find_cycle(layer: SemanticLayer) -> list[str] | None:
    """深度优先找派生指标的循环依赖，返回环上的指标名。"""
    state: dict[str, str] = {}
    stack: list[str] = []

    def visit(name: str) -> list[str] | None:
        if state.get(name) == "done":
            return None
        if state.get(name) == "visiting":
            return [*stack[stack.index(name) :], name]
        state[name] = "visiting"
        stack.append(name)
        metric = layer.metric(name)
        if metric is not None and metric.is_derived:
            for dep in metric.referenced_metrics():
                found = visit(dep)
                if found:
                    return found
        stack.pop()
        state[name] = "done"
        return None

    for m in layer.metrics:
        found = visit(m.name)
        if found:
            return found
    return None
