"""埋点门面。

Trace 结构（按 OpenTelemetry GenAI 语义规范）：

    turn          一次提问（gen_ai.conversation.id）
     ├─ understand  意图理解
     ├─ step 1      一个推理步骤
     │   ├─ llm_call    模型调用（模型名、token 用量）
     │   └─ execute_tool 工具调用（工具名、调用 id、参数、结果摘要）
     └─ synthesize  结论组装

设计上刻意把"没启用时"也做成零成本：`enabled=False` 时返回空上下文，
连 span 对象都不创建，免得在生产里白白产生开销。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Status, StatusCode

from data_agent.observability.attributes import (
    ERROR_TYPE,
    GEN_AI,
    OP_CHAT,
    OP_EXECUTE_TOOL,
    OP_INVOKE_AGENT,
)
from data_agent.observability.exporters import LocalJsonSpanExporter


@dataclass
class SpanHandle:
    """包一层，避免调用方到处写 set_attribute。"""

    span: Any
    enabled: bool = True

    def set(self, **attrs: Any) -> "SpanHandle":
        if self.enabled and self.span is not None:
            for k, v in attrs.items():
                if v is not None:
                    self.span.set_attribute(k, _attr_value(v))
        return self

    def set_response(self, response: Any) -> "SpanHandle":
        """记录模型返回：模型名、token 用量、工具调用数。"""
        usage = getattr(response, "usage", None)
        self.set(
            **{
                GEN_AI["response_model"]: getattr(response, "model", None),
                GEN_AI["input_tokens"]: getattr(usage, "input_tokens", None),
                GEN_AI["output_tokens"]: getattr(usage, "output_tokens", None),
                "agent.tool_call_count": len(getattr(response, "tool_calls", []) or []),
                "agent.has_final_answer": not bool(
                    getattr(response, "tool_calls", None)
                ),
            }
        )
        return self

    def fail(self, exc: BaseException | str, *, kind: str | None = None) -> "SpanHandle":
        name = kind or (type(exc).__name__ if isinstance(exc, BaseException) else "error")
        if self.enabled and self.span is not None:
            self.span.set_attribute(ERROR_TYPE, name)
            self.span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
        return self

    def event(self, name: str, **attrs: Any) -> "SpanHandle":
        if self.enabled and self.span is not None:
            self.span.add_event(name, {k: _attr_value(v) for k, v in attrs.items()})
        return self


class AgentTracer:
    def __init__(self, tracer: Any, *, enabled: bool = True, agent_name: str = "data-agent"):
        self._tracer = tracer
        self.enabled = enabled
        self.agent_name = agent_name

    # ---- 会话 / 轮次 ----
    @contextmanager
    def turn(self, question: str, *, as_of: str | None = None) -> Iterator[SpanHandle]:
        with self._span(
            "turn",
            {
                GEN_AI["operation_name"]: OP_INVOKE_AGENT,
                GEN_AI["agent_name"]: self.agent_name,
                "agent.question": question[:500],
                "agent.as_of": as_of,
            },
        ) as h:
            h.event("question", length=len(question))
            yield h

    @contextmanager
    def node(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        with self._span(f"node.{name}", attrs) as h:
            yield h

    # ---- 模型调用 ----
    @contextmanager
    def llm_call(
        self, *, model: str, message_count: int, tool_count: int = 0, temperature: float = 0.0
    ) -> Iterator[SpanHandle]:
        with self._span(
            "llm_call",
            {
                GEN_AI["operation_name"]: OP_CHAT,
                GEN_AI["request_model"]: model,
                "agent.message_count": message_count,
                "agent.tool_count": tool_count,
                "agent.temperature": temperature,
            },
        ) as h:
            yield h

    # ---- 工具调用 ----
    @contextmanager
    def tool_call(
        self, *, name: str, call_id: str | None = None, arguments: dict | None = None
    ) -> Iterator[SpanHandle]:
        with self._span(
            "execute_tool",
            {
                GEN_AI["operation_name"]: OP_EXECUTE_TOOL,
                GEN_AI["tool_name"]: name,
                GEN_AI["tool_call_id"]: call_id,
                GEN_AI["tool_type"]: "function",
                # 参数只记键名，不记值：值里可能有敏感数据，而且体积不可控
                "agent.argument_keys": sorted((arguments or {}).keys()),
            },
        ) as h:
            yield h

    # ---- 内部 ----
    @contextmanager
    def _span(self, name: str, attrs: dict[str, Any]) -> Iterator[SpanHandle]:
        if not self.enabled:
            yield SpanHandle(span=None, enabled=False)
            return
        with self._tracer.start_as_current_span(name) as span:
            handle = SpanHandle(span=span)
            handle.set(**{k: v for k, v in attrs.items() if v is not None})
            yield handle


def _attr_value(v: Any) -> Any:
    """OTel 属性只接受标量或标量序列。"""
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return str(v)


def setup_tracing(
    *,
    json_path: str | Path | None = None,
    otlp_endpoint: str | None = None,
    console: bool = False,
    extra_exporters: list[SpanExporter] | None = None,
    service_name: str = "data-agent",
    enabled: bool = True,
) -> AgentTracer:
    """装配 tracing。

    默认只挂本地 JSON 导出器——离线可用、可回放、可测试。
    需要接 Langfuse / Collector 时传 otlp_endpoint 即可，埋点代码不动。
    """
    if not enabled:
        return AgentTracer(trace.get_tracer(service_name), enabled=False)

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name,
                "service.version": "0.1.0",
            }
        )
    )

    exporters: list[SpanExporter] = []
    if extra_exporters:
        exporters.extend(extra_exporters)
    if json_path is not None:
        exporters.append(LocalJsonSpanExporter(json_path))
    if otlp_endpoint:
        from data_agent.observability.exporters import build_otlp_exporter

        exporters.append(build_otlp_exporter(otlp_endpoint))
    if console:
        exporters.append(ConsoleSpanExporter())

    for exp in exporters:
        provider.add_span_processor(SimpleSpanProcessor(exp))

    return AgentTracer(provider.get_tracer(service_name), enabled=bool(exporters))
