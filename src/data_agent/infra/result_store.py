"""中间结果存储。

为什么需要它：查询结果可能有几千行，直接塞进模型上下文既浪费 token
又不可能准确复述。所以 run_sql 把结果存起来返回一个 handle，
后续的分析、画图工具通过 handle 取数据。
"""

from __future__ import annotations

import threading
import uuid
from typing import Any


class ResultStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def put(self, columns: list[str], rows: list[tuple], *, source: str = "") -> str:
        handle = f"rs_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._data[handle] = {
                "columns": list(columns),
                "rows": [tuple(r) for r in rows],
                "source": source,
            }
        return handle

    def get(self, handle: str) -> dict[str, Any]:
        with self._lock:
            if handle not in self._data:
                raise KeyError(
                    f"结果句柄不存在或已过期：{handle}。"
                    f"可用句柄：{list(self._data)[:5]}"
                )
            return self._data[handle]

    def preview(self, handle: str, n: int = 5) -> dict[str, Any]:
        entry = self.get(handle)
        return {
            "columns": entry["columns"],
            "row_count": len(entry["rows"]),
            "preview": [list(r) for r in entry["rows"][:n]],
        }

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
