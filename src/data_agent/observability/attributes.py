"""OpenTelemetry GenAI 语义规范的属性常量。

为什么要单独抽一层：GenAI 语义规范目前还在 **incubating**（实验）阶段，
属性名可能随版本调整。如果代码里到处硬编码 `"gen_ai.tool.name"`，
规范一变就要全局搜索替换。

这里统一从 semconv 包取；取不到时退回字符串字面量，
这样即使某个版本把模块挪走了，埋点也不会直接崩。
"""

from __future__ import annotations


def _load() -> dict[str, str]:
    try:
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as g,
        )
    except Exception:  # pragma: no cover - 取决于 semconv 版本
        g = None

    def attr(name: str, default: str) -> str:
        return getattr(g, name, default) if g is not None else default

    return {
        "agent_name": attr("GEN_AI_AGENT_NAME", "gen_ai.agent.name"),
        "conversation_id": attr("GEN_AI_CONVERSATION_ID", "gen_ai.conversation.id"),
        "operation_name": attr("GEN_AI_OPERATION_NAME", "gen_ai.operation.name"),
        "tool_name": attr("GEN_AI_TOOL_NAME", "gen_ai.tool.name"),
        "tool_call_id": attr("GEN_AI_TOOL_CALL_ID", "gen_ai.tool.call.id"),
        "tool_description": attr("GEN_AI_TOOL_DESCRIPTION", "gen_ai.tool.description"),
        "tool_type": attr("GEN_AI_TOOL_TYPE", "gen_ai.tool.type"),
        "request_model": attr("GEN_AI_REQUEST_MODEL", "gen_ai.request.model"),
        "response_model": attr("GEN_AI_RESPONSE_MODEL", "gen_ai.response.model"),
        "input_tokens": attr("GEN_AI_USAGE_INPUT_TOKENS", "gen_ai.usage.input_tokens"),
        "output_tokens": attr("GEN_AI_USAGE_OUTPUT_TOKENS", "gen_ai.usage.output_tokens"),
        "provider_name": attr("GEN_AI_PROVIDER_NAME", "gen_ai.provider.name"),
    }


GEN_AI = _load()

# 语义规范里预定义的 operation name 取值
OP_INVOKE_AGENT = "invoke_agent"
OP_EXECUTE_TOOL = "execute_tool"
OP_CHAT = "chat"

# 错误类型属性（语义规范通用约定）
ERROR_TYPE = "error.type"
