"""分析服务：把 Agent 包装成可异步调度的服务。

设计要点：

  * **每个任务新建一个 DataAgent。** Agent 实例上有 `_started_at` 这种
    运行期状态，多个线程共用一个实例会互相污染预算判断。
    新建实例很便宜，共享的是底层的工具上下文。

  * **依赖可注入。** 测试时传入脚本化模型和一个假 warehouse，
    就能在毫秒级跑完整条服务链路，不需要真实模型。

  * **提交即返回。** 一次分析 5–15 秒，同步接口会超时；
    提交返回 task_id，客户端轮询状态。
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from data_agent.service.tasks import (
    FAILED,
    QUEUED,
    RUNNING,
    STOPPED,
    SUCCEEDED,
    TaskRecord,
    TaskStore,
)

logger = logging.getLogger(__name__)

# Agent 结束状态到任务状态的映射。
# Agent 的 "clarified" 是正常结束（把问题交回给用户），算成功；
# "stopped" 单独记一类，因为「熔断」和「失败」是两件事，排查时要能区分。
_STATUS_MAP = {
    "answered": SUCCEEDED,
    # 拒绝越权/写操作请求是**正常结束**——把问题交回用户，和澄清同理
    "refused": SUCCEEDED,
    "clarified": SUCCEEDED,
    "stopped": STOPPED,
    "failed": FAILED,
}


class AnalysisService:
    def __init__(
        self,
        *,
        agent_factory,
        store: TaskStore,
        max_workers: int = 4,
        trace_path: str | Path | None = None,
    ):
        self._agent_factory = agent_factory
        self.store = store
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="analysis"
        )
        self.trace_path = Path(trace_path) if trace_path else None

    def submit(self, question: str, *, as_of: str | None = None) -> TaskRecord:
        record = TaskRecord(
            task_id=uuid.uuid4().hex[:16],
            question=question,
            as_of=as_of,
        )
        self.store.create(record)
        self._executor.submit(self._run, record.task_id, question, as_of)
        return record

    def _run(self, task_id: str, question: str, as_of: str | None) -> None:
        self.store.update(task_id, status=RUNNING, started_at=_now())
        try:
            agent = self._agent_factory()
            result = agent.run(question, as_of=as_of)
            self.store.update(
                task_id,
                status=_STATUS_MAP.get(result.status, FAILED),
                finished_at=_now(),
                answer=result.answer,
                clarification=result.clarification,
                error=result.stop_reason if result.status == "failed" else None,
                trace_id=result.trace_id,
                steps=result.steps,
                usage=result.usage,
                selection=result.selection,
            )
        except Exception as exc:  # noqa: BLE001 - 后台任务不能把异常抛出去
            logger.exception("分析任务执行失败：%s", task_id)
            self.store.update(
                task_id,
                status=FAILED,
                finished_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )

    def get(self, task_id: str) -> TaskRecord | None:
        return self.store.get(task_id)

    def list_recent(self, limit: int = 20) -> list[TaskRecord]:
        return self.store.list_recent(limit)

    def get_trace(self, task_id: str) -> list[dict[str, Any]] | None:
        """按 task 取完整轨迹。"""
        record = self.store.get(task_id)
        if not record or not record.trace_id or not self.trace_path:
            return None
        from data_agent.observability.replay import (
            build_tree,
            group_by_trace,
            load_spans,
            render_tree,
        )

        if not self.trace_path.exists():
            return None
        spans = [
            s
            for s in load_spans(self.trace_path)
            if s.get("trace_id") == record.trace_id
        ]
        if not spans:
            return None
        return {
            "trace_id": record.trace_id,
            "span_count": len(spans),
            "spans": spans,
            "tree": render_tree(build_tree(spans)),
            "groups": len(group_by_trace(spans)),
        }

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


def _now() -> float:
    import time

    return time.time()
