"""中间结果存储。

为什么需要它：查询结果可能有几千行，直接塞进模型上下文既浪费 token
又不可能准确复述。所以 run_sql 把结果存起来返回一个 handle，
后续的分析、画图工具通过 handle 取数据。

提供两个实现：
  * `ResultStore`：进程内内存字典。快，但进程重启就没了，
    而且多 worker 之间不共享——本地开发和单进程够用。
  * `SqliteResultStore`：落盘 + TTL。服务化时必须用这个，
    否则"提交任务的那个 worker"和"查询结果的那个 worker"
    拿不到同一份数据。

两者接口一致，通过 `build_result_store()` 按配置选择。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _new_handle() -> str:
    return f"rs_{uuid.uuid4().hex[:12]}"


class ResultStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def put(self, columns: list[str], rows: list[tuple], *, source: str = "") -> str:
        handle = _new_handle()
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


class SqliteResultStore:
    """落盘的中间结果存储，带 TTL。

    几个刻意的设计：

      * **过期清理是惰性的**（读取时顺手删），不另起后台线程。
        后台线程在多进程下会互相打架，而惰性清理足够用。
      * **每次操作独立开关连接**，避免跨进程共享连接。
        SQLite 连接不是线程安全的，也不该跨进程复用。
      * 行数据用 JSON 存储。中间结果本来就是给模型看的，
        精度够用，且避免了 pickle 的安全问题。
    """

    def __init__(self, path: str | Path, *, ttl_seconds: float = 3600.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=10.0)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    handle      TEXT PRIMARY KEY,
                    columns     TEXT NOT NULL,
                    rows        TEXT NOT NULL,
                    source      TEXT,
                    created_at  REAL NOT NULL
                )
                """
            )

    def put(self, columns: list[str], rows: list[tuple], *, source: str = "") -> str:
        handle = _new_handle()
        payload = json.dumps(
            [[_jsonable(v) for v in row] for row in rows], ensure_ascii=False
        )
        with self._connect() as con:
            con.execute(
                "INSERT INTO results (handle, columns, rows, source, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (handle, json.dumps(columns, ensure_ascii=False), payload, source, time.time()),
            )
        return handle

    def get(self, handle: str) -> dict[str, Any]:
        self._evict_expired()
        with self._connect() as con:
            row = con.execute(
                "SELECT columns, rows, source FROM results WHERE handle = ?",
                (handle,),
            ).fetchone()
        if row is None:
            raise KeyError(
                f"结果句柄不存在或已过期：{handle}。"
                f"句柄有效期 {int(self.ttl)} 秒。"
            )
        return {
            "columns": json.loads(row[0]),
            "rows": [tuple(r) for r in json.loads(row[1])],
            "source": row[2],
        }

    def preview(self, handle: str, n: int = 5) -> dict[str, Any]:
        entry = self.get(handle)
        return {
            "columns": entry["columns"],
            "row_count": len(entry["rows"]),
            "preview": [list(r) for r in entry["rows"][:n]],
        }

    def size(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    def clear(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM results")

    def _evict_expired(self) -> int:
        cutoff = time.time() - self.ttl
        with self._connect() as con:
            cur = con.execute("DELETE FROM results WHERE created_at < ?", (cutoff,))
            return cur.rowcount


def _jsonable(v: Any) -> Any:
    """把中间结果转成 JSON 可存的类型。

    日期、Decimal 这类值直接用 str 存——中间结果不参与精确比对，
    真正需要精度的比对发生在 SQL 层。
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def build_result_store(backend: str = "memory", **kwargs: Any):
    """按配置选择实现。"""
    if backend == "sqlite":
        return SqliteResultStore(**kwargs)
    if backend == "memory":
        return ResultStore()
    raise ValueError(f"未知的结果存储类型：{backend}")
