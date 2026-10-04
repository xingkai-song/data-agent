"""Agent 的状态定义。

状态是这个项目里最该显式化的东西：每一步发生了什么、为什么停，
都必须能从状态里读出来。否则出了问题只能靠翻日志猜。
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ---- 输入 ----
    question: str
    as_of: str | None          # 基准日期，用于解析"9 月"这类相对时间

    # ---- 意图与计划 ----
    intent: dict[str, Any]     # 指标、时间范围、维度、是否需要澄清
    plan: list[str]            # 子任务列表
    needs_clarification: bool
    clarification: str | None

    # ---- 上下文 ----
    context: str               # 由语义层检索组装出来的提示
    selection: dict[str, Any]  # 命中的表/指标/规则（便于评测定位）
    messages: list[dict[str, Any]]

    # ---- 执行轨迹 ----
    step: int
    tool_calls: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    reflections: list[str]

    # LangGraph 用这两个字段做节点间通信。必须显式声明——
    # 未声明的键在状态合并时会被丢掉（踩过：_diagnosis 丢失导致
    # 空结果诊断永远传不到 reflect 节点）。
    _next: str
    diagnosis: str | None

    # ---- 资源 ----
    usage: dict[str, float]

    # ---- 结果 ----
    status: str                # running / answered / clarified / failed / stopped
    stop_reason: str | None
    final_answer: str | None


def new_state(question: str, *, as_of: str | None = None) -> AgentState:
    return AgentState(
        question=question,
        as_of=as_of,
        intent={},
        plan=[],
        needs_clarification=False,
        clarification=None,
        context="",
        selection={},
        messages=[],
        step=0,
        tool_calls=[],
        observations=[],
        errors=[],
        reflections=[],
        _next="act",
        diagnosis=None,
        usage={"input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "llm_calls": 0},
        status="running",
        stop_reason=None,
        final_answer=None,
    )
