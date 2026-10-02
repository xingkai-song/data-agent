"""从 YAML 加载语义层，并做一致性校验。

校验的意义：语义层本身写错了，后面所有生成的 SQL 都是错的，而且很难发现。
所以宁可加载时就报错。
"""

from __future__ import annotations

from pathlib import Path

import yaml

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
    tables = [Table(**row) for row in _read_yaml(d / "tables.yaml")]
    metrics = [Metric(**row) for row in _read_yaml(d / "metrics.yaml")]
    rules = [Rule(**row) for row in _read_yaml(d / "rules.yaml")]
    layer = SemanticLayer(tables=tables, metrics=metrics, rules=rules)
    _validate(layer)
    return layer


def _validate(layer: SemanticLayer) -> None:
    table_names = {t.name for t in layer.tables}
    problems: list[str] = []

    if len(table_names) != len(layer.tables):
        problems.append("存在重名的表")

    rule_ids = [r.id for r in layer.rules]
    if len(set(rule_ids)) != len(rule_ids):
        problems.append("存在重复的规则 id")

    metric_names = {m.name for m in layer.metrics}
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

    for r in layer.rules:
        if r.scope not in ("global", *metric_names, *table_names):
            problems.append(f"规则 {r.id} 的 scope `{r.scope}` 未匹配任何指标或表")

    if problems:
        raise SemanticLayerError("语义层定义有问题：\n  - " + "\n  - ".join(problems))
