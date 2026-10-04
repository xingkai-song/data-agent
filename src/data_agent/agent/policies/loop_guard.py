"""死循环防护。

单一手段都不够，所以做三重检测：

  1. **最大步数**：兜底，任何情况下都会停
  2. **重复调用**：同一个 (工具, 参数) 反复出现，说明模型在原地打转
  3. **无进展**：连续若干步的观测结果没有新增信息（比如反复查同一张表结构）

只有最大步数的话，模型可能在第 3 步就开始无效循环，白烧 5 步的 token；
只有重复检测的话，模型换个参数就能绕过去。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


class LoopDetected(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class LoopGuard:
    max_steps: int = 8
    max_repeats: int = 2
    no_progress_window: int = 3

    @staticmethod
    def signature(tool: str, args: dict[str, Any]) -> str:
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        return f"{tool}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"

    def check(
        self,
        *,
        step: int,
        tool_calls: Iterable[dict[str, Any]],
        observations: Iterable[dict[str, Any]],
    ) -> str | None:
        """返回停止原因；None 表示可以继续。"""
        if step >= self.max_steps:
            return f"达到最大步数 {self.max_steps}"

        sigs = [c.get("signature") for c in tool_calls if c.get("signature")]
        if sigs:
            last = sigs[-1]
            if sigs.count(last) > self.max_repeats:
                return f"同一个调用重复了 {sigs.count(last)} 次：{last}"

        obs = list(observations)
        if len(obs) >= self.no_progress_window:
            recent = [o.get("digest") for o in obs[-self.no_progress_window :]]
            if recent and len(set(recent)) == 1:
                return f"连续 {self.no_progress_window} 步没有新信息"
        return None

    @staticmethod
    def digest(result: Any) -> str:
        """把一次工具结果压缩成摘要，用于判断"有没有新信息"。

        必须先剔除易变字段。踩过的坑：结果里带一个每次新建的句柄
        （`rs_xxxx`），两次完全相同的查询摘要必然不同，
        "连续 N 步没有新信息"这个检测就永远不会触发。
        """
        payload = json.dumps(
            _strip_volatile(result), ensure_ascii=False, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# 这些字段每次调用都会变，不代表"有没有新信息"
_VOLATILE_KEYS = {
    "handle",
    "path",
    "log_path",
    "latency_ms",
    "elapsed_ms",
    "elapsed_s",
    "created_at",
    "timestamp",
    "bytes_written",
}


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_volatile(v) for k, v in value.items() if k not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_volatile(v) for v in value]
    return value
