"""本地数仓访问层。

读连接默认只读，避免评测脚本意外改到数据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from data_agent.config import warehouse_path


class Warehouse:
    """薄封装，便于后续替换为真实数仓连接。"""

    def __init__(self, path: Path | str | None = None, *, read_only: bool = True):
        self.path = Path(path) if path else warehouse_path()
        if not self.path.exists():
            raise FileNotFoundError(
                f"数仓不存在：{self.path}\n请先运行 python scripts/build_dataset.py"
            )
        self._con = duckdb.connect(str(self.path), read_only=read_only)

    def query(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        cur = self._con.execute(sql, params or [])
        return cur.fetchall()

    def query_df(self, sql: str, params: list[Any] | None = None):
        return self._con.execute(sql, params or []).df()

    def columns_of(self, sql: str) -> list[str]:
        cur = self._con.execute(sql)
        return [d[0] for d in cur.description] if cur.description else []

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "Warehouse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
