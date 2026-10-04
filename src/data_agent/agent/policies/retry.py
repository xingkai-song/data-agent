"""重试策略。

核心原则：**只重试瞬态错误**。

把"整轮对话"重试是常见但危险的做法——它会重放已经成功的工具调用，
如果中间有写操作（发通知、导出报告），重试一次就产生一次重复副作用。
所以重试必须下沉到**单次工具调用**，并且只对瞬态错误生效；
参数错误之类的应该把错误回灌给模型让它自己改，而不是原样重试。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from data_agent.tools.base import ErrorKind, ToolResult


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0
    sleep: Callable[[float], None] = time.sleep
    attempts: list[int] = field(default_factory=list)

    def delay_for(self, attempt: int) -> float:
        """指数退避。第 attempt 次失败后的等待时长（attempt 从 0 开始）。"""
        return min(self.base_delay_s * (2**attempt), self.max_delay_s)

    def call(self, fn: Callable[[], ToolResult]) -> tuple[ToolResult, int]:
        """执行 fn，必要时重试。

        返回 (最后一次结果, 实际尝试次数)。
        """
        attempts = 0
        result: ToolResult | None = None
        while attempts < self.max_attempts:
            attempts += 1
            result = fn()
            if result.ok:
                break
            if result.error_kind is not ErrorKind.TRANSIENT:
                # 不可重试的错误：原样返回，交给上层回灌给模型
                break
            if attempts < self.max_attempts:
                self.sleep(self.delay_for(attempts - 1))
        self.attempts.append(attempts)
        assert result is not None
        return result, attempts
