"""接口的请求 / 响应模型。

单独放一个文件是为了让接口契约显式化——FastAPI 会据此生成 OpenAPI 文档，
前端或者别的调用方不用读代码就知道字段是什么。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="业务问题")
    as_of: str | None = Field(
        default=None,
        description="基准日期（YYYY-MM-DD），用于解析「上个月」这类相对时间",
    )


class AnalysisAccepted(BaseModel):
    task_id: str
    status: str
    message: str = "任务已提交，请轮询 /analyses/{task_id} 获取结果"


class AnalysisResult(BaseModel):
    task_id: str
    status: str
    is_terminal: bool
    question: str
    answer: str | None = None
    clarification: str | None = None
    error: str | None = None
    trace_id: str | None = None
    steps: int = 0
    elapsed_s: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    selection: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(BaseModel):
    trace_id: str
    span_count: int
    tree: str
    spans: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str = "ok"
    tasks_total: int
    pending: int
