"""工具层入口。

导入这个包会注册全部工具。提供一个工厂函数把依赖装配好，
便于测试时替换成假实现。
"""

from __future__ import annotations

from pathlib import Path

from data_agent.config import PROJECT_ROOT
from data_agent.infra.db import Warehouse
from data_agent.infra.idempotency import IdempotencyStore
from data_agent.infra.result_store import ResultStore
from data_agent.semantic import load_semantic_layer
from data_agent.tools.base import ErrorKind, ToolContext, ToolError, ToolResult
from data_agent.tools.registry import DEFAULT_REGISTRY, ToolRegistry

# 导入即注册（顺序不重要，但需要显式导入）
from data_agent.tools import analysis_tools, query_tools, side_effect_tools, viz_tools  # noqa: F401,E402

__all__ = [
    "DEFAULT_REGISTRY",
    "ErrorKind",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_context",
    "default_registry",
]


def default_registry() -> ToolRegistry:
    return DEFAULT_REGISTRY


def build_context(
    *,
    warehouse: Warehouse | None = None,
    data_dir: Path | None = None,
    semantic=None,
) -> ToolContext:
    """装配工具运行所需的所有依赖。"""
    base = data_dir or (PROJECT_ROOT / "data")
    base.mkdir(parents=True, exist_ok=True)
    return ToolContext(
        warehouse=warehouse or Warehouse(),
        semantic=semantic or load_semantic_layer(),
        idempotency=IdempotencyStore(base / "idempotency.sqlite"),
        results=ResultStore(),
        audit=None,
        artifact_dir=PROJECT_ROOT / "artifacts",
    )
