"""异步任务的状态存储。

为什么任务状态必须落盘：一次分析要 5–15 秒。用同步 HTTP 接口会超时，
所以改成"提交 → 返回 task_id → 轮询"。一旦是轮询，
状态就不能只放在提交请求所在进程的内存里——否则多 worker 部署时，
轮询可能落到另一个进程，查不到任务。

状态机很简单，但必须显式：

    queued ──> running ──> succeeded
                     └───> failed
                     └───> stopped   （熔断：超出步数/预算）
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
STOPPED = "stopped"

TERMINAL = {SUCCEEDED, FAILED, STOPPED}


@dataclass
class TaskRecord:
    task_id: str
    question: str
    as_of: str | None
    status: str = QUEUED
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    answer: str | None = None
    clarification: str | None = None
    error: str | None = None
    trace_id: str | None = None
    steps: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def elapsed_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return round(end - self.started_at, 3)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["elapsed_s"] = self.elapsed_s
        data["is_terminal"] = self.is_terminal
        return data


class TaskStore:
    """SQLite 版本的任务存储。"""

    COLUMNS = (
        "task_id",
        "question",
        "as_of",
        "status",
        "created_at",
        "started_at",
        "finished_at",
        "answer",
        "clarification",
        "error",
        "trace_id",
        "steps",
        "usage",
        "selection",
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS analysis_tasks (
                    task_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    as_of TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    answer TEXT,
                    clarification TEXT,
                    error TEXT,
                    trace_id TEXT,
                    steps INTEGER DEFAULT 0,
                    usage TEXT,
                    selection TEXT
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON analysis_tasks(status)"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=10.0)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def create(self, record: TaskRecord) -> TaskRecord:
        record.created_at = record.created_at or time.time()
        with self._connect() as con:
            con.execute(
                f"INSERT INTO analysis_tasks ({', '.join(self.COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(self.COLUMNS))})",
                self._row(record),
            )
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        with self._connect() as con:
            row = con.execute(
                f"SELECT {', '.join(self.COLUMNS)} FROM analysis_tasks "
                "WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def update(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        for key in ("usage", "selection"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], ensure_ascii=False, default=str)
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as con:
            con.execute(
                f"UPDATE analysis_tasks SET {assignments} WHERE task_id = ?",
                (*fields.values(), task_id),
            )

    def list_recent(self, limit: int = 20) -> list[TaskRecord]:
        with self._connect() as con:
            rows = con.execute(
                f"SELECT {', '.join(self.COLUMNS)} FROM analysis_tasks "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._from_row(r) for r in rows]

    def size(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM analysis_tasks").fetchone()[0]

    @staticmethod
    def _row(r: TaskRecord) -> tuple:
        return (
            r.task_id,
            r.question,
            r.as_of,
            r.status,
            r.created_at,
            r.started_at,
            r.finished_at,
            r.answer,
            r.clarification,
            r.error,
            r.trace_id,
            r.steps,
            json.dumps(r.usage, ensure_ascii=False, default=str),
            json.dumps(r.selection, ensure_ascii=False, default=str),
        )

    @staticmethod
    def _from_row(row: tuple) -> TaskRecord:
        (
            task_id,
            question,
            as_of,
            status,
            created_at,
            started_at,
            finished_at,
            answer,
            clarification,
            error,
            trace_id,
            steps,
            usage,
            selection,
        ) = row
        return TaskRecord(
            task_id=task_id,
            question=question,
            as_of=as_of,
            status=status,
            created_at=created_at or 0.0,
            started_at=started_at,
            finished_at=finished_at,
            answer=answer,
            clarification=clarification,
            error=error,
            trace_id=trace_id,
            steps=steps or 0,
            usage=json.loads(usage) if usage else {},
            selection=json.loads(selection) if selection else {},
        )
