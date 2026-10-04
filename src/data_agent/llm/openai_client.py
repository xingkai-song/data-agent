"""OpenAI 兼容协议的客户端。

任何兼容 OpenAI Chat Completions 协议的服务都能用：
    OpenAI     https://api.openai.com/v1
    DeepSeek   https://api.deepseek.com/v1
    通义千问    https://dashscope.aliyuncs.com/compatible-mode/v1
    智谱       https://open.bigmodel.cn/api/paas/v4
只要设置 OPENAI_BASE_URL 和 OPENAI_API_KEY 即可。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from data_agent.llm.base import LLMError, LLMResponse, TokenUsage, ToolCall

# 粗略的价格表（美元 / 1K tokens），用于估算成本。
# 估算就是估算——拿不到官方账单时不要把它当成精确值。
_PRICES = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "deepseek-chat": (0.00027, 0.0011),
    "qwen-plus": (0.0004, 0.0012),
}


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        from openai import OpenAI

        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
        )
        self.last_usage = TokenUsage()
        self._total_usage = TokenUsage()

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        if response_format:
            payload["response_format"] = response_format

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**payload)
                return self._to_response(resp)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_retryable(exc) or attempt == self.max_retries - 1:
                    break
                # 指数退避 + 抖动
                time.sleep(min(2**attempt * 0.5, 4.0))
        raise LLMError(
            f"模型调用失败：{type(last_exc).__name__}: {last_exc}",
            transient=_is_retryable(last_exc),
        )

    def _to_response(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__parse_error__": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = TokenUsage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        self.last_usage = usage
        self._total_usage = self._total_usage + usage
        return LLMResponse(
            content=msg.content,
            tool_calls=calls,
            usage=usage,
            model=getattr(resp, "model", self.model),
        )

    def estimate_cost_usd(self, usage: TokenUsage) -> float:
        inp, out = _PRICES.get(self.model, (0.0, 0.0))
        return usage.input_tokens / 1000 * inp + usage.output_tokens / 1000 * out


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in ("APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    return False
