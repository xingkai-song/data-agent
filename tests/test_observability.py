"""可观测性测试。

重点验证两件事：
  1. 埋点结构正确（turn → step → llm_call / execute_tool），
     否则回放时看不出"失败在第几步"
  2. 失败判定不误报也不漏报——尤其不能让"会被自动修正的参数错误"
     污染失败率统计
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.agent import DataAgent  # noqa: E402
from data_agent.agent.policies import LoopGuard  # noqa: E402
from data_agent.infra.idempotency import IdempotencyStore  # noqa: E402
from data_agent.infra.result_store import ResultStore  # noqa: E402
from data_agent.llm import ScriptedLLM  # noqa: E402
from data_agent.llm.base import LLMResponse, ToolCall  # noqa: E402
from data_agent.llm.scripted import json_response  # noqa: E402
from data_agent.observability.attributes import GEN_AI, ERROR_TYPE  # noqa: E402
from data_agent.observability.exporters import (  # noqa: E402
    CollectingSpanExporter,
    span_to_record,
)
from data_agent.observability.replay import (  # noqa: E402
    append_candidates,
    build_tree,
    find_failed_turns,
    group_by_trace,
    load_spans,
    render_tree,
)
from data_agent.observability.tracing import setup_tracing  # noqa: E402
from data_agent.semantic import load_semantic_layer  # noqa: E402
from data_agent.tools import ToolContext, build_context, default_registry  # noqa: E402

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


def plan():
    return json_response(
        {"metrics": ["成交额"], "needs_clarification": False, "plan": ["查数"]}
    )


def run_agent(ctx, responses, *, tracer, **kwargs):
    agent = DataAgent(
        llm=ScriptedLLM(responses),
        registry=default_registry(),
        tool_ctx=ctx,
        tracer=tracer,
        **kwargs,
    )
    return agent.run("9 月的成交总额是多少？", as_of="2026-10-01")


# ---------- 语义规范属性 ----------


def test_genai_attribute_names_follow_the_spec():
    """属性名必须符合 OTel GenAI 语义规范，否则接不到任何后端。"""
    assert GEN_AI["tool_name"] == "gen_ai.tool.name"
    assert GEN_AI["tool_call_id"] == "gen_ai.tool.call.id"
    assert GEN_AI["input_tokens"] == "gen_ai.usage.input_tokens"
    assert GEN_AI["output_tokens"] == "gen_ai.usage.output_tokens"
    assert GEN_AI["operation_name"] == "gen_ai.operation.name"
    assert ERROR_TYPE == "error.type"


# ---------- 埋点结构 ----------


def test_trace_has_expected_structure(ctx):
    collector = CollectingSpanExporter()
    tracer = setup_tracing(extra_exporters=[collector])

    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )

    names = [r["name"] for r in collector.records]
    assert "turn" in names
    assert "node.understand" in names
    assert "node.step" in names
    assert "execute_tool" in names
    assert "llm_call" in names
    assert "node.synthesize" in names


def test_turn_span_carries_run_summary(ctx):
    collector = CollectingSpanExporter()
    tracer = setup_tracing(extra_exporters=[collector])
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    turn = collector.by_name("turn")[0]
    attrs = turn["attributes"]
    assert attrs["agent.status"] == "answered"
    assert attrs["agent.question"] == "9 月的成交总额是多少？"
    assert attrs["agent.tool_calls"] == 1
    assert attrs["agent.llm_calls"] == 3
    assert attrs["agent.failed_tool_calls"] == 0


def test_tool_span_uses_semconv_attributes(ctx):
    collector = CollectingSpanExporter()
    tracer = setup_tracing(extra_exporters=[collector])
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("call-42", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    tool_span = collector.by_name("execute_tool")[0]
    attrs = tool_span["attributes"]
    assert attrs["gen_ai.tool.name"] == "run_sql"
    assert attrs["gen_ai.tool.call.id"] == "call-42"
    assert attrs["agent.ok"] is True
    assert attrs["agent.row_count"] == 1


def test_llm_span_records_tokens(ctx):
    collector = CollectingSpanExporter()
    tracer = setup_tracing(extra_exporters=[collector])
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    for span in collector.by_name("llm_call"):
        attrs = span["attributes"]
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 50


def test_tool_span_records_argument_keys_not_values(ctx):
    """参数值可能有敏感数据，只记键名。"""
    collector = CollectingSpanExporter()
    tracer = setup_tracing(extra_exporters=[collector])
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    tool_span = collector.by_name("execute_tool")[0]
    assert tool_span["attributes"]["agent.argument_keys"] == ["sql"]
    assert "SELECT" not in json.dumps(tool_span["attributes"], ensure_ascii=False)


def test_step_spans_group_llm_and_tools(ctx):
    collector = CollectingSpanExporter()
    tracer = setup_tracing(extra_exporters=[collector])
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    steps = collector.by_name("node.step")
    # 两步：第一步取数，第二步基于结果给出结论
    assert len(steps) == 2

    first = steps[0]
    first_children = {c["name"] for c in collector.records if c["parent_span_id"] == first["span_id"]}
    assert first_children == {"llm_call", "execute_tool"}
    assert first["attributes"]["agent.step"] == 1

    second = steps[1]
    second_children = {c["name"] for c in collector.records if c["parent_span_id"] == second["span_id"]}
    assert second_children == {"llm_call"}
    assert second["attributes"]["agent.step"] == 2


# ---------- 失败判定 ----------


def test_recoverable_tool_error_does_not_mark_span_failed(ctx):
    """参数错误会被回灌让模型自己改，不该算进失败率。

    如果这类错误也标记成 ERROR，失败率统计会被大量"正常纠错"污染。
    """
    collector = CollectingSpanExporter()
    tracer = setup_tracing(extra_exporters=[collector])
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": "SELECT * FROM orderz"})]),
            LLMResponse(tool_calls=[ToolCall("c2", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    first_tool = collector.by_name("execute_tool")[0]
    assert first_tool["status"] != "ERROR"
    assert first_tool["attributes"]["agent.ok"] is False
    assert first_tool["attributes"]["agent.error_kind"] == "validation"
    assert any(e["name"] == "tool_error" for e in first_tool["events"])


def test_denied_tool_call_is_recorded_as_event(ctx):
    collector = CollectingSpanExporter()
    tracer = setup_tracing(extra_exporters=[collector])
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": "DELETE FROM orders"})]),
            LLMResponse(content="已改用只读查询"),
        ],
        tracer=tracer,
    )
    tool_span = collector.by_name("execute_tool")[0]
    assert tool_span["attributes"]["agent.error_kind"] == "denied"
    assert any(e["name"] == "tool_error" for e in tool_span["events"])


def test_successful_run_has_no_failure(ctx, tmp_path):
    trace = tmp_path / "t.jsonl"
    tracer = setup_tracing(json_path=trace)
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    assert find_failed_turns(load_spans(trace)) == []


def test_stopped_run_is_flagged_as_failure(ctx, tmp_path):
    """被熔断停下的任务是最该进评测集的样本，必须能被识别出来。"""
    trace = tmp_path / "t.jsonl"
    tracer = setup_tracing(json_path=trace)
    same = ToolCall("c", "run_sql", {"sql": REF_SQL})
    responses = [plan()] + [LLMResponse(tool_calls=[same]) for _ in range(5)]
    run_agent(ctx, responses, tracer=tracer, loop_guard=LoopGuard(max_repeats=1))

    failures = find_failed_turns(load_spans(trace))
    assert len(failures) == 1
    assert failures[0]["status"] == "stopped"
    assert failures[0]["question"] == "9 月的成交总额是多少？"


def test_failure_candidates_export(ctx, tmp_path):
    trace = tmp_path / "t.jsonl"
    tracer = setup_tracing(json_path=trace)
    same = ToolCall("c", "run_sql", {"sql": REF_SQL})
    run_agent(
        ctx, [plan()] + [LLMResponse(tool_calls=[same]) for _ in range(5)],
        tracer=tracer, loop_guard=LoopGuard(max_repeats=1),
    )
    failures = find_failed_turns(load_spans(trace))
    out = tmp_path / "candidates.jsonl"
    assert append_candidates(out, failures) == 1
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["question"] and rec["status"] == "stopped"


# ---------- 导出器与回放 ----------


def test_json_exporter_writes_parseable_lines(ctx, tmp_path):
    trace = tmp_path / "spans.jsonl"
    tracer = setup_tracing(json_path=trace)
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    spans = load_spans(trace)
    assert len(spans) >= 6
    assert all("span_id" in s and "trace_id" in s for s in spans)
    ids = {s["span_id"] for s in spans}
    roots = [s for s in spans if s["parent_span_id"] is None]
    assert len(roots) == 1
    assert roots[0]["name"] == "turn"
    assert all(
        s["parent_span_id"] in ids for s in spans if s["parent_span_id"]
    )


def test_replay_renders_readable_tree(ctx, tmp_path):
    trace = tmp_path / "spans.jsonl"
    tracer = setup_tracing(json_path=trace)
    run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    spans = load_spans(trace)
    trees = build_tree(list(spans))
    text = render_tree(trees)
    assert "turn" in text
    assert "execute_tool" in text
    assert "gen_ai.tool.name = run_sql" in text


def test_span_record_strips_heavy_payload():
    class FakeCtx:
        trace_id = 1
        span_id = 2

    class FakeSpan:
        context = FakeCtx()
        parent = None
        name = "llm_call"
        kind = None
        start_time = 0
        end_time = 0
        status = None
        events = []
        attributes = {"gen_ai.prompt": "很长的提示词", "gen_ai.request.model": "m"}

    rec = span_to_record(FakeSpan())
    assert "gen_ai.prompt" not in rec["attributes"]
    assert rec["attributes"]["gen_ai.request.model"] == "m"


# ---------- 关闭埋点时零开销 ----------


def test_disabled_tracer_does_not_create_spans(ctx):
    tracer = setup_tracing(enabled=False)
    assert tracer.enabled is False
    r = run_agent(
        ctx,
        [
            plan(),
            LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
            LLMResponse(content="完成"),
        ],
        tracer=tracer,
    )
    assert r.status == "answered"


def test_default_agent_has_tracing_disabled(ctx):
    """默认不埋点，避免测试和一次性脚本产生额外开销。"""
    agent = DataAgent(
        llm=ScriptedLLM([plan(), LLMResponse(content="完成")]),
        registry=default_registry(),
        tool_ctx=ctx,
    )
    assert agent.tracer.enabled is False
