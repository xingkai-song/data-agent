"""Span 导出器。

核心是 **LocalJsonSpanExporter**：把每个 span 写成一行 JSON。
它让可观测性在完全离线的情况下也能用——不需要 Docker、不需要网络、
不需要 Langfuse，而且产出的文件可以回放、可以进测试、可以作为
"失败 case 自动回流到评测集"的数据源。

需要接入 Langfuse / Jaeger / Datadog 时，再加一个 OTLP 导出器即可，
埋点代码一行不用改。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

# 这些属性包含完整 prompt，体积大且可能含敏感数据，默认不落盘
_HEAVY_KEYS = {
    "gen_ai.prompt",
    "gen_ai.completion",
    "input.messages",
    "output.messages",
}


def span_to_record(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.context
    parent = span.parent
    attrs = {
        k: _jsonable(v)
        for k, v in (span.attributes or {}).items()
        if k not in _HEAVY_KEYS
    }
    start_ns = span.start_time or 0
    end_ns = span.end_time or start_ns
    return {
        "trace_id": f"{ctx.trace_id:032x}" if ctx else None,
        "span_id": f"{ctx.span_id:016x}" if ctx else None,
        "parent_span_id": f"{parent.span_id:016x}" if parent else None,
        "name": span.name,
        "kind": span.kind.name if span.kind else None,
        "start_time": _iso(start_ns),
        "end_time": _iso(end_ns),
        "duration_ms": round((end_ns - start_ns) / 1e6, 3),
        "status": span.status.status_code.name if span.status else "UNSET",
        "status_message": span.status.description if span.status else None,
        "attributes": attrs,
        "events": [
            {
                "name": e.name,
                "time": _iso(e.timestamp or 0),
                "attributes": {k: _jsonable(v) for k, v in (e.attributes or {}).items()},
            }
            for e in (span.events or [])
        ],
    }


class LocalJsonSpanExporter(SpanExporter):
    """把 span 追加写入 JSONL 文件。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._written = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                for span in spans:
                    rec = span_to_record(span)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self._records.append(rec)
            self._written += len(spans)
            return SpanExportResult.SUCCESS
        except Exception:  # pragma: no cover - 落盘失败不该影响主流程
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    @property
    def written(self) -> int:
        return self._written


class CollectingSpanExporter(SpanExporter):
    """把 span 留在内存里，供测试断言。"""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.records.extend(span_to_record(s) for s in spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def clear(self) -> None:
        self.records.clear()

    def by_name(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["name"] == name]


def build_otlp_exporter(endpoint: str):
    """可选的 OTLP 导出器。

    指向 Langfuse 的 OTLP 端点、OpenTelemetry Collector 或任何兼容后端。
    没装对应依赖时明确报错，不静默降级。
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 OTLP 导出依赖，请执行："
            "pip install opentelemetry-exporter-otlp-proto-http"
        ) from exc
    return OTLPSpanExporter(endpoint=endpoint)


def _iso(ns: int) -> str:
    if not ns:
        return ""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)
