"""服务层与接口层测试。

这两层之前完全没有覆盖（"用户怎么用这个项目"，测试里没验过）。
现在用**脚本化模型**驱动，不联网、不花钱、毫秒级完成，
但走的是真实的 Agent → 工具 → SQL → 存储 → 接口整条链路。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.agent import DataAgent  # noqa: E402
from data_agent.agent.policies import LoopGuard  # noqa: E402
from data_agent.api import create_app  # noqa: E402
from data_agent.infra.idempotency import IdempotencyStore  # noqa: E402
from data_agent.infra.result_store import SqliteResultStore  # noqa: E402
from data_agent.llm import ScriptedLLM  # noqa: E402
from data_agent.llm.base import LLMResponse, ToolCall  # noqa: E402
from data_agent.llm.scripted import json_response  # noqa: E402
from data_agent.observability import setup_tracing  # noqa: E402
from data_agent.semantic import load_semantic_layer  # noqa: E402
from data_agent.service import AnalysisService, TaskStore  # noqa: E402
from data_agent.service.tasks import FAILED, STOPPED, SUCCEEDED  # noqa: E402
from data_agent.tools import ToolContext, build_context, default_registry  # noqa: E402

REF_SQL = (
    "SELECT ROUND(SUM(amount), 2) AS gmv FROM orders "
    "WHERE status IN ('paid','refunded') "
    "AND pay_time >= TIMESTAMP '2026-09-01' AND pay_time < TIMESTAMP '2026-10-01'"
)


def ok_responses():
    return [
        json_response(
            {"metrics": ["成交额"], "needs_clarification": False, "plan": ["查数"]}
        ),
        LLMResponse(tool_calls=[ToolCall("c1", "run_sql", {"sql": REF_SQL})]),
        LLMResponse(content="9 月成交总额为 13,028,677.88 元。"),
    ]


def build_ctx(tmp_path, name="r"):
    base = build_context()
    return ToolContext(
        warehouse=base.warehouse,
        semantic=load_semantic_layer(),
        idempotency=IdempotencyStore(tmp_path / f"{name}_idem.sqlite"),
        results=SqliteResultStore(tmp_path / f"{name}_results.sqlite"),
        artifact_dir=tmp_path / "artifacts",
    )


def make_service(tmp_path, responses, *, trace_path=None, max_workers=2, guard=None):
    ctx = build_ctx(tmp_path)
    registry = default_registry()
    tracer = (
        setup_tracing(json_path=trace_path)
        if trace_path
        else setup_tracing(enabled=False)
    )

    def factory():
        kwargs = {"loop_guard": guard} if guard else {}
        return DataAgent(
            llm=ScriptedLLM(list(responses)),
            registry=registry,
            tool_ctx=ctx,
            tracer=tracer,
            **kwargs,
        )

    service = AnalysisService(
        agent_factory=factory,
        store=TaskStore(tmp_path / "tasks.sqlite"),
        max_workers=max_workers,
        trace_path=trace_path,
    )
    return service, ctx


def wait_terminal(service, task_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = service.get(task_id)
        if rec and rec.is_terminal:
            return rec
        time.sleep(0.02)
    raise AssertionError(f"任务 {task_id} 在 {timeout}s 内没有结束")


# ---------- 服务层 ----------


def test_submit_returns_task_id_immediately(tmp_path):
    service, _ = make_service(tmp_path, ok_responses())
    record = service.submit("9 月的成交总额是多少？", as_of="2026-10-01")
    assert record.task_id
    wait_terminal(service, record.task_id)
    service.shutdown()


def test_task_reaches_succeeded_and_carries_answer(tmp_path):
    service, _ = make_service(tmp_path, ok_responses())
    task_id = service.submit("9 月的成交总额是多少？", as_of="2026-10-01").task_id
    rec = wait_terminal(service, task_id)
    assert rec.status == SUCCEEDED
    assert "13,028,677.88" in (rec.answer or "")
    assert rec.steps == 1
    assert rec.elapsed_s is not None and rec.elapsed_s >= 0
    service.shutdown()


def test_task_stores_selection_for_debugging(tmp_path):
    """任务记录里要留语义层命中信息，出问题能回溯「它当时拿到了什么口径」。"""
    service, _ = make_service(tmp_path, ok_responses())
    task_id = service.submit("9 月的成交总额是多少？").task_id
    rec = wait_terminal(service, task_id)
    assert rec.selection.get("metrics") == ["gmv"]
    service.shutdown()


def test_clarification_is_success_not_failure(tmp_path):
    """Agent 要求澄清是正常结束，不该记成失败。"""
    responses = [
        json_response(
            {
                "metrics": ["销售额"],
                "needs_clarification": True,
                "clarification_question": "你是指含退款的成交额，还是扣除退款后的净额？",
                "plan": [],
            }
        )
    ]
    service, _ = make_service(tmp_path, responses)
    task_id = service.submit("上个月的销售额是多少？").task_id
    rec = wait_terminal(service, task_id)
    assert rec.status == SUCCEEDED
    assert "净额" in (rec.clarification or "")
    assert rec.answer, "澄清内容必须能通过接口拿到"
    service.shutdown()


def test_circuit_break_has_its_own_status(tmp_path):
    """熔断和失败是两件事，排查时要能分开统计。"""
    same = ToolCall("c", "run_sql", {"sql": REF_SQL})
    responses = [json_response({"metrics": ["成交额"], "needs_clarification": False})]
    responses += [LLMResponse(tool_calls=[same]) for _ in range(12)]
    service, _ = make_service(
        tmp_path, responses, max_workers=1, guard=LoopGuard(max_repeats=1)
    )
    task_id = service.submit("9 月的成交总额是多少？").task_id
    rec = wait_terminal(service, task_id)
    assert rec.status == STOPPED
    assert rec.answer, "熔断也要给出诚实的部分结论"
    service.shutdown()


def test_agent_exception_marks_task_failed(tmp_path):
    """后台任务抛异常不能把进程带崩，要落到任务状态里。"""

    class BoomAgent:
        def run(self, question, as_of=None):
            raise RuntimeError("模拟内部错误")

    service = AnalysisService(
        agent_factory=BoomAgent,
        store=TaskStore(tmp_path / "t.sqlite"),
        max_workers=1,
    )
    task_id = service.submit("随便问问").task_id
    rec = wait_terminal(service, task_id)
    assert rec.status == FAILED
    assert "模拟内部错误" in (rec.error or "")
    service.shutdown()


def test_list_recent_returns_newest_first(tmp_path):
    service, _ = make_service(tmp_path, ok_responses() * 3, max_workers=1)
    ids = [service.submit(f"问题 {i}").task_id for i in range(3)]
    for tid in ids:
        wait_terminal(service, tid)
    recent = service.list_recent(limit=10)
    assert len(recent) == 3
    assert [r.task_id for r in recent] == list(reversed(ids))
    service.shutdown()


def test_trace_can_be_fetched_by_task(tmp_path):
    trace_path = tmp_path / "spans.jsonl"
    service, _ = make_service(tmp_path, ok_responses(), trace_path=trace_path)
    task_id = service.submit("9 月的成交总额是多少？", as_of="2026-10-01").task_id
    rec = wait_terminal(service, task_id)
    assert rec.trace_id, "任务应该记录 trace_id"
    trace = service.get_trace(task_id)
    assert trace and trace["trace_id"] == rec.trace_id
    assert "execute_tool" in trace["tree"]
    service.shutdown()


def test_trace_returns_none_when_tracing_disabled(tmp_path):
    service, _ = make_service(tmp_path, ok_responses())
    task_id = service.submit("9 月的成交总额是多少？").task_id
    wait_terminal(service, task_id)
    assert service.get_trace(task_id) is None
    service.shutdown()


def test_results_visible_across_store_instances(tmp_path):
    """服务化必须用落盘的结果存储，否则另一个 worker 读不到中间结果。"""
    service, _ = make_service(tmp_path, ok_responses())
    task_id = service.submit("9 月的成交总额是多少？").task_id
    wait_terminal(service, task_id)
    fresh = SqliteResultStore(tmp_path / "r_results.sqlite")
    assert fresh.size() >= 1
    service.shutdown()


def test_concurrent_tasks_do_not_share_run_state(tmp_path):
    """并发提交时，每个任务必须有自己的运行期状态。

    Agent 实例上有 `_started_at` 这类运行状态；如果多个任务共用一个实例，
    预算判断会互相污染。所以服务层每个任务新建一个 Agent。
    """
    service, _ = make_service(tmp_path, ok_responses() * 4, max_workers=4)
    ids = [service.submit(f"并发问题 {i}").task_id for i in range(4)]
    records = [wait_terminal(service, tid) for tid in ids]
    assert all(r.status == SUCCEEDED for r in records)
    assert len({r.task_id for r in records}) == 4
    service.shutdown()


# ---------- 接口层 ----------


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    trace_path = tmp_path / "spans.jsonl"
    service, _ = make_service(tmp_path, ok_responses(), trace_path=trace_path)
    with TestClient(create_app(service)) as c:
        yield c
    service.shutdown()


def _poll(client, task_id, timeout=20.0):
    deadline = time.time() + timeout
    body = {}
    while time.time() < deadline:
        body = client.get(f"/analyses/{task_id}").json()
        if body["is_terminal"]:
            return body
        time.sleep(0.02)
    raise AssertionError(f"任务没有在 {timeout}s 内结束：{body}")


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_submit_returns_202_with_task_id(client):
    r = client.post(
        "/analyses", json={"question": "9 月的成交总额是多少？", "as_of": "2026-10-01"}
    )
    assert r.status_code == 202
    body = r.json()
    assert body["task_id"] and body["status"]


def test_poll_until_succeeded(client):
    task_id = client.post(
        "/analyses",
        json={"question": "9 月的成交总额是多少？", "as_of": "2026-10-01"},
    ).json()["task_id"]
    body = _poll(client, task_id)
    assert body["status"] == "succeeded"
    assert "13,028,677.88" in body["answer"]


def test_unknown_task_returns_404(client):
    assert client.get("/analyses/nope").status_code == 404
    assert client.get("/analyses/nope/trace").status_code == 404


def test_trace_endpoint(client):
    task_id = client.post(
        "/analyses", json={"question": "9 月的成交总额是多少？"}
    ).json()["task_id"]
    _poll(client, task_id)
    r = client.get(f"/analyses/{task_id}/trace")
    assert r.status_code == 200
    payload = r.json()
    assert payload["span_count"] > 0
    assert "execute_tool" in payload["tree"]


def test_empty_question_rejected_by_schema(client):
    """参数校验由 Pydantic 兜住，不该打到 Agent 层。"""
    assert client.post("/analyses", json={"question": ""}).status_code == 422


def test_list_endpoint(client):
    client.post("/analyses", json={"question": "问题一"})
    client.post("/analyses", json={"question": "问题二"})
    r = client.get("/analyses?limit=10")
    assert r.status_code == 200
    assert len(r.json()) >= 2
