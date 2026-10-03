"""幂等键存储。

解决的问题：工具调用失败重试时，有副作用的操作会被执行两次
（典型症状是用户收到两封通知、生成两份报告）。

做法：调用方提供一个幂等键，同一个键只真正执行一次。

为什么用 SQLite 而不是内存字典：重试常常发生在**进程重启之后**
（比如 Agent 崩溃后恢复）。内存字典一重启就没了，幂等保护随之失效。
SQLite 落盘才能覆盖这个场景。生产环境可以换成 Redis，
接口保持一致即可。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class IdempotencyStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False)
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency (
                key         TEXT PRIMARY KEY,
                result      TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
            """
        )
        self._con.commit()

    # ---- 基础读写 ----
    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._con.execute(
                "SELECT result FROM idempotency WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def has(self, key: str) -> bool:
        with self._lock:
            return (
                self._con.execute(
                    "SELECT 1 FROM idempotency WHERE key = ?", (key,)
                ).fetchone()
                is not None
            )

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO idempotency (key, result, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False, default=str), _now()),
            )
            self._con.commit()

    def clear(self) -> None:
        with self._lock:
            self._con.execute("DELETE FROM idempotency")
            self._con.commit()

    def size(self) -> int:
        with self._lock:
            return self._con.execute("SELECT COUNT(*) FROM idempotency").fetchone()[0]

    # ---- 核心：只执行一次 ----
    def run_once(self, key: str, fn: Callable[[], Any]) -> tuple[Any, bool]:
        """执行 fn 并把结果记在 key 下。

        返回 (结果, 是否命中已有记录)。命中时**不会**再次执行 fn。
        """
        existing = self.get(key)
        if existing is not None:
            return existing, True
        value = fn()
        self.put(key, value)
        return value, False

    def close(self) -> None:
        self._con.close()


def make_key(namespace: str, *parts: Any) -> str:
    """根据工具名和参数生成稳定的幂等键。

    不传幂等键时用它兜底：只要参数相同，重复调用就不会产生重复副作用。
    """
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{namespace}:{digest}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
