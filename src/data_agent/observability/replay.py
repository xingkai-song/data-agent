"""轨迹回放与失败回流。

两件事：

1. **回放**：把 JSONL 轨迹还原成树并按时间线打印。出了问题不用翻散落的
   日志，直接看这一步是谁调用的、花了多久、是模型选错了工具还是工具报错。

2. **失败回流**：把失败的轮次抽出来作为候选评测任务。真实失败是最有价值
   的测试用例来源——比凭空造样本准得多。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SpanNode:
    record: dict[str, Any]
    children: list["SpanNode"] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.record.get("name", "?")

    @property
    def failed(self) -> bool:
        return self.record.get("status") == "ERROR"


def load_spans(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"轨迹文件不存在：{p}")
    records: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def group_by_trace(spans: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in spans:
        groups.setdefault(s.get("trace_id") or "unknown", []).append(s)
    return groups


def build_tree(spans: list[dict[str, Any]]) -> list[SpanNode]:
    nodes = {s["span_id"]: SpanNode(record=s) for s in spans if s.get("span_id")}
    roots: list[SpanNode] = []
    for s in spans:
        node = nodes.get(s.get("span_id"))
        if node is None:
            continue
        parent_id = s.get("parent_span_id")
        parent = nodes.get(parent_id) if parent_id else None
        if parent is not None:
            parent.children.append(node)
        else:
            roots.append(node)
    for node in nodes.values():
        node.children.sort(key=lambda n: n.record.get("start_time") or "")
    roots.sort(key=lambda n: n.record.get("start_time") or "")
    return roots


def render_tree(roots: list[SpanNode], *, show_attributes: bool = True) -> str:
    lines: list[str] = []
    for root in roots:
        _render(root, lines, prefix="", show_attributes=show_attributes)
    return "\n".join(lines)


def _render(node: SpanNode, lines: list[str], *, prefix: str, show_attributes: bool) -> None:
    rec = node.record
    mark = "x" if node.failed else "."
    duration = rec.get("duration_ms")
    dur = f"{duration:.0f}ms" if isinstance(duration, (int, float)) else ""
    lines.append(f"{prefix}{mark} {rec.get('name')}  {dur}".rstrip())

    if show_attributes:
        for key, value in _interesting_attrs(rec).items():
            lines.append(f"{prefix}    {key} = {value}")
    if rec.get("status_message"):
        lines.append(f"{prefix}    ! {rec['status_message']}")

    child_prefix = prefix + "  "
    for child in node.children:
        _render(child, lines, prefix=child_prefix, show_attributes=show_attributes)


def _interesting_attrs(rec: dict[str, Any]) -> dict[str, Any]:
    """只显示有用的属性，避免把 trace 打成一大坨。"""
    attrs = rec.get("attributes") or {}
    keep = (
        "gen_ai.tool.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "error.type",
        "agent.question",
        "agent.argument_keys",
        "agent.tool_call_count",
    )
    return {k: attrs[k] for k in keep if k in attrs}


# ----------------------------------------------------------------------
# 失败回流
# ----------------------------------------------------------------------


def find_failed_turns(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """抽出失败的轮次，作为评测集的候选样本。

    判据（满足其一）：
      * 这一轮没有正常作答（status 不是 answered）——包括熔断、澄清、模型故障
      * 这一轮的任意子 span 状态是 ERROR

    覆盖"没答出来"而不仅仅是"抛异常"：一个把模型绕进死循环、
    最后被熔断停下的任务，同样是最该进评测集的样本。
    """
    failures: list[dict[str, Any]] = []
    for trace_id, group in group_by_trace(spans).items():
        turn = next((s for s in group if s.get("name") == "turn"), None)
        if not turn:
            continue
        attrs = turn.get("attributes") or {}
        status = attrs.get("agent.status")
        error_spans = [s for s in group if s.get("status") == "ERROR"]
        if status == "answered" and not error_spans:
            continue
        failures.append(
            {
                "trace_id": trace_id,
                "question": attrs.get("agent.question"),
                "as_of": attrs.get("agent.as_of"),
                "status": status,
                "steps": attrs.get("agent.steps"),
                "failed_tool_calls": attrs.get("agent.failed_tool_calls"),
                "error_count": len(error_spans),
                "errors": [
                    {
                        "span": s.get("name"),
                        "tool": (s.get("attributes") or {}).get("gen_ai.tool.name"),
                        "type": (s.get("attributes") or {}).get("error.type"),
                        "message": s.get("status_message"),
                    }
                    for s in error_spans
                ],
            }
        )
    return failures


def append_candidates(path: str | Path, failures: list[dict[str, Any]]) -> int:
    """把失败样本追加到候选文件，供人工整理成正式评测任务。

    刻意不直接写进 tasks.jsonl：评测任务需要人工补标准 SQL 和口径依据。
    自动生成的任务如果没经过人工校验，只会污染评测集。
    """
    if not failures:
        return 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(failures)
