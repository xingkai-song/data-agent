"""HTTP 接口层。

只有薄薄一层：参数校验 → 交给 AnalysisService → 序列化响应。
真正的逻辑都在 service 和 agent 里，这样接口层不需要单独写一堆测试。

接口设计成异步的（提交返回 task_id）而不是同步阻塞，原因是
一次分析 5–15 秒，同步请求会超时，而且没法在网关层做重试。
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from data_agent.api.schemas import (
    AnalysisAccepted,
    AnalysisRequest,
    AnalysisResult,
    HealthResponse,
    TraceResponse,
)
from data_agent.service.analysis import AnalysisService
from data_agent.service.tasks import QUEUED, RUNNING


def create_app(service: AnalysisService) -> FastAPI:
    """按依赖创建 app。

    不在这里 new 依赖——服务实例由调用方传入，
    测试时可以注入假模型和临时存储。
    """
    app = FastAPI(
        title="数据分析 Agent",
        description="把业务问题转成数据仓库查询并给出可追溯的结论",
        version="0.1.0",
    )

    @app.post("/analyses", response_model=AnalysisAccepted, status_code=202)
    def submit_analysis(req: AnalysisRequest) -> AnalysisAccepted:
        record = service.submit(req.question, as_of=req.as_of)
        return AnalysisAccepted(task_id=record.task_id, status=record.status)

    @app.get("/analyses/{task_id}", response_model=AnalysisResult)
    def get_analysis(task_id: str) -> AnalysisResult:
        record = service.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
        return AnalysisResult(**record.to_dict())

    @app.get("/analyses", response_model=list[AnalysisResult])
    def list_analyses(limit: int = 20) -> list[AnalysisResult]:
        return [AnalysisResult(**r.to_dict()) for r in service.list_recent(limit)]

    @app.get("/analyses/{task_id}/trace", response_model=TraceResponse)
    def get_trace(task_id: str) -> TraceResponse:
        record = service.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
        trace = service.get_trace(task_id)
        if trace is None:
            raise HTTPException(
                status_code=404,
                detail="轨迹不可用：任务还没结束、没有开启埋点，或者轨迹文件已被清理",
            )
        return TraceResponse(**trace)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        recent = service.list_recent(limit=1000)
        pending = sum(1 for r in recent if r.status in (QUEUED, RUNNING))
        return HealthResponse(tasks_total=len(recent), pending=pending)

    return app
