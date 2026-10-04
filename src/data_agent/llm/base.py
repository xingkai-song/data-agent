"""LLM 客户端的抽象层。

为什么要有这一层：Agent 循环的逻辑（状态转移、重试、熔断、预算）
和"用哪个模型"是两件独立的事。把模型调用抽象出来之后：

  * 循环逻辑可以用 ScriptedLLM 做确定性测试，不需要联网、不花 token
  * 换模型只改一个环境变量，不改循环代码
  * 成本统计有统一入口，不会漏记
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    raw: dict[str, Any] | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    """模型调用失败。由上层按 transient/permanent 分类处理。"""

    def __init__(self, message: str, *, transient: bool = True):
        super().__init__(message)
        self.transient = transient
