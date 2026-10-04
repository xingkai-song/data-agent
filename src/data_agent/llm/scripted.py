"""脚本化的假模型，用于测试 Agent 循环。

它的价值不是"假装能跑"，而是让循环逻辑（重试、熔断、预算、状态转移）
变成可确定性验证的东西——这些逻辑用真模型反而测不准。

典型用法：

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("1", "run_sql", {"sql": "SELECT 1"})]),
        LLMResponse(content="成交额是 100"),
    ])
"""

from __future__ import annotations

import json
from typing import Any, Callable

from data_agent.llm.base import LLMError, LLMResponse, TokenUsage


class ScriptedLLM:
    def __init__(
        self,
        responses: list[LLMResponse | Callable[[list[dict], list[dict] | None], LLMResponse]],
        *,
        input_tokens_per_call: int = 100,
        output_tokens_per_call: int = 50,
    ):
        self._responses = list(responses)
        self._calls: list[dict[str, Any]] = []
        self._i = 0
        self._in = input_tokens_per_call
        self._out = output_tokens_per_call

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._calls

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def push(self, response: LLMResponse) -> None:
        self._responses.append(response)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._calls.append(
            {"messages": messages, "tools": tools, "response_format": response_format}
        )
        if self._i >= len(self._responses):
            raise LLMError("脚本化模型没有更多预设响应了", transient=False)
        item = self._responses[self._i]
        self._i += 1
        if callable(item):
            resp = item(messages, tools)
        else:
            resp = item
        if resp.usage.total == 0:
            resp.usage = TokenUsage(self._in, self._out)
        resp.model = resp.model or "scripted"
        return resp


def json_response(payload: dict[str, Any]) -> LLMResponse:
    """构造一个返回 JSON 文本的响应（用于 understand / plan 这类结构化节点）。"""
    return LLMResponse(content=json.dumps(payload, ensure_ascii=False))
