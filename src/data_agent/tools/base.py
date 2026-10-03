"""工具层的公共契约。

设计要点：

1. **错误必须分类。** 直接抛异常给上层，重试策略就只能"要么全重试、
   要么全不重试"。这里把错误分成五类，让重试逻辑能按类型决策。

2. **入参出参都用 Pydantic 模型。** JSON Schema 由模型自动生成，
   既不手写也不容易写错；同时模型传错参数时能给出结构化错误。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel


class ErrorKind(str, Enum):
    """错误分类，直接决定重试策略。"""

    VALIDATION = "validation"    # 参数不合法：不重试，把错误回灌给模型让它改
    DENIED = "denied"            # 安全校验拦截：不重试，需要人工确认
    TRANSIENT = "transient"      # 网络超时 / 限流 / 5xx：可以退避重试
    BUSINESS = "business"        # 业务规则不满足：交给模型决策
    PERMANENT = "permanent"      # 权限、资源不存在等：不重试
    INTERNAL = "internal"        # 我们自己的 bug：不重试，需要修

    @property
    def retryable(self) -> bool:
        return self is ErrorKind.TRANSIENT


class ToolResult(BaseModel):
    """所有工具的统一返回结构。"""

    ok: bool
    data: Any = None
    error: str | None = None
    error_kind: ErrorKind | None = None
    # 给模型看的可读提示（会被回灌进上下文，让它自己修正）
    hint: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data: Any = None, **meta: Any) -> "ToolResult":
        return cls(ok=True, data=data, meta=meta)

    @classmethod
    def failure(
        cls,
        kind: ErrorKind,
        error: str,
        *,
        hint: str | None = None,
        **meta: Any,
    ) -> "ToolResult":
        return cls(
            ok=False, error=error, error_kind=kind, hint=hint, meta=meta
        )


class ToolError(Exception):
    """工具内部抛出，由 registry 转成 ToolResult。"""

    def __init__(self, kind: ErrorKind, message: str, hint: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.hint = hint


@dataclass
class ToolContext:
    """工具执行时需要的外部依赖，统一注入便于测试和替换。"""

    warehouse: Any
    semantic: Any
    idempotency: Any | None = None
    results: Any | None = None
    audit: Any | None = None
    # 工具产出物目录（报告、图表）
    artifact_dir: Any | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    category: str            # query / analysis / viz / side_effect
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    func: Callable[[BaseModel, ToolContext], Any]
    side_effect: bool = False
    permission: str = "read"  # read / write / dangerous
    timeout_s: int = 30

    def json_schema(self) -> dict[str, Any]:
        """生成给模型看的工具描述（兼容 OpenAI function calling 格式）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }
