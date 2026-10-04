"""资源预算。

没有预算上限的 Agent 在生产里是危险的：一个循环能烧掉几百块。
这里限制 token、工具调用次数和总时长，任一超限就停下来并说明原因。
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Budget:
    max_tokens: int = 60_000
    max_tool_calls: int = 12
    max_seconds: float = 120.0

    def exceeded(self, usage: dict[str, float], started_at: float) -> str | None:
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if tokens > self.max_tokens:
            return f"token 用量 {int(tokens)} 超过上限 {self.max_tokens}"
        if usage.get("tool_calls", 0) > self.max_tool_calls:
            return f"工具调用 {int(usage['tool_calls'])} 次超过上限 {self.max_tool_calls}"
        elapsed = time.time() - started_at
        if elapsed > self.max_seconds:
            return f"耗时 {elapsed:.1f}s 超过上限 {self.max_seconds}s"
        return None

    def remaining_tokens(self, usage: dict[str, float]) -> int:
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return max(0, int(self.max_tokens - tokens))
