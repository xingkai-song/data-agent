"""Agent 执行循环的集成测试。

用 ScriptedLLM 驱动，把"模型会不会犯错"变成可控变量，
这样循环逻辑（状态转移、错误回灌、熔断、预算）可以被确定性验证。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.agent import DataAgent
from data_agent.agent.policies import Budget, LoopGuard, RetryPolicy
from data_agent.infra.idempotency import IdempotencyStore
from data_agent.infra.result_store import ResultStore
from data_agent.llm import ScriptedLLM
from data_agent.llm.base import LLMError, LLMResponse, ToolCall
from data_agent.llm.scripted import json_response
from data_agent.semantic import load_semantic_layer
from data_agent.tools import ToolContext, build_context, default_registry
from data_agent.tools.base import ErrorKind, ToolResult, ToolSpec
from data_agent.tools.registry import ToolRegistry, tool

REF_SQL = (
    "SELECT ROUND(SUM(amount), 2) AS gmv FROM orders "
    "WHERE status IN ('paid','refunded') "
    "AND pay_time >= TIMESTAMP '2026-09-01' AND pay_time < TIMESTAMP '2026-10-01'"
)


@pytest.fixture
def ctx(tmp_path):
    base = build_context()
    return ToolContext(
        warehouse=base.warehouse,
        semantic=load_semantic_layer(),
        idempotency=IdempotencyStore(tmp_path / "idem.sqlite"),
        results=ResultStore(),
        artifact_dir=tmp_path / "artifacts",
    )


def agent_with(responses, ctx, **kwargs) -> DataAgent:
    llm = ScriptedLLM(responses)
    return DataAgent(llm=llm, registry=default_registry(), tool_ctx=ctx, **kwargs)


def plan_response(needs_clarification: bool = False, question: str | None = None):
    return json_response(
        {
            "metrics": ["成交额"],
            "time_range": "2026-09-01 至 2026-09-30",
            "dimensions": [],
            "needs_clarification": needs_clarification,
            "clarification_question": question,
            "plan": ["确认口径", "查询数据"],
        }
    )


# ---------- 正常路径 ----------


def test_happy_path_answers(ctx):
    agent = agent_with(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="9 月成交总额为 4,693,111.19 元。"),
        ],
        ctx,
    )
    r = agent.run("9 月的成交总额是多少？", as_of="2026-10-01")
    assert r.status == "answered"
    assert r.succeeded
    assert "4,693,111.19" in (r.answer or "")
    assert [c["name"] for c in r.tool_calls] == ["run_sql"]
    assert r.tool_calls[0]["ok"]


def test_answer_carries_data_sources(ctx):
    """结论必须能追溯到具体查询——这是"可验证"的基本要求。"""
    agent = agent_with(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="成交额是 4,693,111.19 元。"),
        ],
        ctx,
    )
    r = agent.run("9 月的成交总额是多少？", as_of="2026-10-01")
    assert "数据来源" in (r.answer or "")
    assert "SUM(amount)" in (r.answer or "")


def test_intent_and_plan_are_recorded(ctx):
    agent = agent_with(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        ctx,
    )
    r = agent.run("9 月的成交总额是多少？", as_of="2026-10-01")
    assert r.intent["metrics"] == ["成交额"]
    assert r.plan == ["确认口径", "查询数据"]
    assert r.selection["metrics"] == ["gmv"]


def test_usage_is_accumulated(ctx):
    agent = agent_with(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        ctx,
    )
    r = agent.run("9 月的成交总额是多少？")
    assert r.usage["llm_calls"] == 3
    assert r.usage["tool_calls"] == 1
    assert r.usage["input_tokens"] > 0


# ---------- 澄清分支 ----------


def test_clarification_branch_skips_tool_execution(ctx):
    agent = agent_with(
        [plan_response(True, "你是指按支付时间还是下单时间统计成交额？")],
        ctx,
    )
    r = agent.run("上个月的销售额是多少？")
    assert r.status == "clarified"
    assert r.clarification and "支付时间" in r.clarification
    assert r.tool_calls == []
    assert r.stop_reason == "需要澄清口径"


def test_understand_failure_degrades_gracefully(ctx):
    """意图识别那一步失败，不应该让整个任务挂掉，降级为直接执行。"""

    def boom(messages, tools):
        raise LLMError("模型超时", transient=True)

    agent = agent_with(
        [
            boom,
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        ctx,
    )
    r = agent.run("9 月的成交总额是多少？")
    assert r.status == "answered"
    assert "error" in r.intent


# ---------- 错误回灌 ----------


def test_tool_error_is_reflowed_to_model(ctx):
    """工具报错不能只记日志，必须作为 tool 消息回灌，模型才能自我修正。"""
    llm = ScriptedLLM(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": "SELECT * FROM orderz"})]),
            LLMResponse(tool_calls=[ToolCall("c2", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="已修正表名，成交额 4,693,111.19 元。"),
        ]
    )
    agent = DataAgent(llm=llm, registry=default_registry(), tool_ctx=ctx)
    r = agent.run("9 月的成交总额是多少？", as_of="2026-10-01")

    assert r.status == "answered"
    assert len(r.tool_calls) == 2
    assert not r.tool_calls[0]["ok"]
    assert r.tool_calls[0]["error_kind"] == "validation"
    assert r.tool_calls[1]["ok"]

    # 第三次模型调用时，上下文里必须已经有那次失败的 tool 消息
    third_call_messages = llm.calls[2]["messages"]
    tool_msgs = [m for m in third_call_messages if m.get("role") == "tool"]
    assert tool_msgs, "错误没有回灌给模型"
    assert "错误类型" in tool_msgs[0]["content"]


def test_denied_error_also_reflowed(ctx):
    llm = ScriptedLLM(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": "DELETE FROM orders"})]),
            LLMResponse(content="这个操作被安全策略拦截了，我改成只读查询。"),
        ]
    )
    agent = DataAgent(llm=llm, registry=default_registry(), tool_ctx=ctx)
    r = agent.run("把订单表清空")
    assert r.tool_calls[0]["error_kind"] == "denied"
    tool_msgs = [m for m in llm.calls[2]["messages"] if m.get("role") == "tool"]
    assert "只读" in tool_msgs[0]["content"]


# ---------- observe / reflect ----------


def test_empty_result_triggers_reflection(ctx):
    """空结果通常是筛选条件写错了，系统要主动提示模型复查。"""
    empty_sql = (
        "SELECT ROUND(SUM(amount), 2) AS gmv FROM orders "
        "WHERE status IN ('paid','refunded') "
        "AND pay_time >= TIMESTAMP '2019-01-01' AND pay_time < TIMESTAMP '2019-02-01'"
    )
    llm = ScriptedLLM(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": empty_sql})]),
            LLMResponse(content="时间范围超出数据覆盖区间，改查 2026 年 9 月。"),
        ]
    )
    agent = DataAgent(llm=llm, registry=default_registry(), tool_ctx=ctx)
    r = agent.run("9 月的成交总额是多少？")
    assert r.tool_calls[0]["ok"]

    last_messages = llm.calls[-1]["messages"]
    assert any("系统检查" in (m.get("content") or "") for m in last_messages), (
        "空结果没有触发系统检查提示"
    )


def test_normal_aggregate_does_not_trigger_false_truncation(ctx):
    """聚合查询只返回 1 行，不能被误判为"被 LIMIT 截断"。"""
    llm = ScriptedLLM(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ]
    )
    agent = DataAgent(llm=llm, registry=default_registry(), tool_ctx=ctx)
    agent.run("9 月的成交总额是多少？")
    last_messages = llm.calls[-1]["messages"]
    assert not any(
        "被强制加了 LIMIT" in (m.get("content") or "") for m in last_messages
    )


# ---------- 循环防护与预算 ----------


def test_max_steps_stops_and_reports_reason(ctx):
    responses = [plan_response()] + [
        LLMResponse(tool_calls=[ToolCall(f"c{i}", "run_sql", {"sql": f"SELECT {i} AS x FROM orders"})])
        for i in range(6)
    ]
    agent = agent_with(responses, ctx, loop_guard=LoopGuard(max_steps=3))
    r = agent.run("随便问问")
    assert r.status == "stopped"
    assert "最大步数" in (r.stop_reason or "")
    assert len(r.tool_calls) == 3


def test_repeated_call_stops(ctx):
    same = ToolCall("c", "run_sql", {"sql": REF_SQL})
    responses = [plan_response()] + [LLMResponse(tool_calls=[same]) for _ in range(5)]
    agent = agent_with(responses, ctx, loop_guard=LoopGuard(max_repeats=2))
    r = agent.run("9 月的成交总额是多少？")
    assert r.status == "stopped"
    assert "重复" in (r.stop_reason or "")


def test_no_progress_stops(ctx):
    """参数变但结果不变的循环也要能识别。"""
    responses = [plan_response()] + [
        LLMResponse(tool_calls=[ToolCall(f"c{i}", "run_sql", {"sql": "SELECT COUNT(*) AS n FROM orders", "purpose": f"第{i}次"})])
        for i in range(5)
    ]
    agent = agent_with(
        responses, ctx,
        loop_guard=LoopGuard(max_steps=10, max_repeats=10, no_progress_window=3),
    )
    r = agent.run("订单有多少")
    assert r.status == "stopped"
    assert "没有新信息" in (r.stop_reason or "")


def test_token_budget_stops(ctx):
    responses = [plan_response()] + [
        LLMResponse(tool_calls=[ToolCall(f"c{i}", "run_sql", {"sql": f"SELECT {i} AS x FROM orders"})])
        for i in range(6)
    ]
    agent = agent_with(
        responses, ctx, budget=Budget(max_tokens=250, max_tool_calls=99, max_seconds=999)
    )
    r = agent.run("随便问问")
    assert r.status == "stopped"
    assert "token" in (r.stop_reason or "")


def test_stopped_run_gives_honest_partial_answer(ctx):
    """被熔断时不能编一个完整答案，要说明中断原因。"""
    same = ToolCall("c", "run_sql", {"sql": REF_SQL})
    responses = [plan_response()] + [LLMResponse(tool_calls=[same]) for _ in range(5)]
    agent = agent_with(responses, ctx, loop_guard=LoopGuard(max_repeats=1))
    r = agent.run("9 月的成交总额是多少？")
    assert r.status == "stopped"
    assert "未能完成" in (r.answer or "")
    assert "重复" in (r.answer or "")


# ---------- 模型故障 ----------


def test_model_failure_after_planning(ctx):
    def boom(messages, tools):
        raise LLMError("服务不可用", transient=True)

    agent = agent_with([plan_response(), boom], ctx)
    r = agent.run("9 月的成交总额是多少？")
    assert r.status == "failed"
    assert "模型调用失败" in (r.stop_reason or "")


# ---------- 重试与副作用 ----------


class _Args(BaseModel):
    value: int = 0


class _Out(BaseModel):
    value: int = 0


def _make_flaky_registry(fail_times: int, kind: ErrorKind):
    reg = ToolRegistry()
    state = {"n": 0}

    def impl(inp: _Args, ctx: ToolContext):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise _FakeToolError(kind)
        return _Out(value=inp.value)

    reg.register(
        ToolSpec(
            name="flaky",
            description="测试用：前几次失败",
            category="query",
            input_model=_Args,
            output_model=_Out,
            func=impl,
        )
    )
    return reg, state


class _FakeToolError(Exception):
    pass


def test_transient_tool_error_is_retried_inside_the_loop(ctx):
    """瞬态错误在单次工具调用内部重试，而不是重放整轮对话。"""
    reg, state = _make_flaky_registry(fail_times=2, kind=ErrorKind.TRANSIENT)
    spec = reg.get("flaky")
    original = spec.func  # 必须先取出原函数，否则包装后会递归调用自己

    def wrapped(inp, c):
        try:
            return original(inp, c)
        except _FakeToolError:
            from data_agent.tools.base import ToolError

            raise ToolError(ErrorKind.TRANSIENT, "模拟超时") from None

    spec.func = wrapped

    llm = ScriptedLLM(
        [
            plan_response(),
            LLMResponse(tool_calls=[ToolCall("c1", "flaky", {"value": 7})]),
        ]
    )
    agent = DataAgent(
        llm=llm, registry=reg, tool_ctx=ctx,
        retry=RetryPolicy(max_attempts=3, sleep=lambda _: None),
    )
    r = agent.run("测试重试")
    assert r.tool_calls[0]["ok"], r.tool_calls[0]
    assert r.tool_calls[0]["attempts"] == 3
    assert state["n"] == 3

    # 关键：重试发生在单次工具调用内部，没有重放对话。
    # 所以上下文里只会有一条"调用 flaky"的 assistant 消息，
    # 而不是三条。
    last_messages = llm.calls[-1]["messages"]
    flaky_calls = [
        m for m in last_messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert len(flaky_calls) == 1, "重试似乎重放了对话，而不是在工具内部重试"


def test_side_effect_not_duplicated_across_retry(ctx):
    """最关键的一条：重试不能让有副作用的操作执行两次。"""
    llm = ScriptedLLM(
        [
            plan_response(),
            LLMResponse(
                tool_calls=[
                    ToolCall("c1", "run_sql", {"sql": REF_SQL}),
                    ToolCall(
                        "c2", "send_notification",
                        {"channel": "email", "message": "日报已生成",
                         "idempotency_key": "job-1"},
                    ),
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "c3", "send_notification",
                        {"channel": "email", "message": "日报已生成（重试）",
                         "idempotency_key": "job-1"},
                    ),
                ]
            ),
            LLMResponse(content="完成"),
        ]
    )
    agent = DataAgent(llm=llm, registry=default_registry(), tool_ctx=ctx)
    r = agent.run("生成日报并通知我")

    notif = [c for c in r.tool_calls if c["name"] == "send_notification"]
    assert len(notif) == 2
    assert notif[0]["ok"] and notif[1]["ok"]

    import pathlib

    log = pathlib.Path(ctx.artifact_dir) / "notifications.log"
    lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, f"副作用重复执行了：{lines}"
