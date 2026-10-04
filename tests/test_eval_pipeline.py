"""端到端打通测试：Agent → 工具 → SQL → 评测报告。

前面各层是分开测的，这个文件验证它们串起来是通的。
如果这一环断了，前面测得再好也没有意义。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.agent import DataAgent  # noqa: E402
from data_agent.llm import ScriptedLLM  # noqa: E402
from data_agent.llm.base import LLMResponse, ToolCall  # noqa: E402
from data_agent.llm.scripted import json_response  # noqa: E402
from data_agent.tools import build_context, default_registry  # noqa: E402
from eval.runner import evaluate, load_tasks  # noqa: E402
from eval.run_agent import last_successful_sql  # noqa: E402


def _plan():
    return json_response(
        {
            "metrics": ["成交额"],
            "needs_clarification": False,
            "clarification_question": None,
            "plan": ["确认口径", "查数"],
        }
    )


def test_agent_sql_flows_into_eval_report(tmp_path):
    """用脚本化模型模拟一个"每次都写对 SQL"的 Agent，
    跑通后交给评测执行器评分，应该全部通过。

    这验证的是链路，不是模型能力——模型能力要靠真实密钥去跑。
    """
    tasks = load_tasks()[:3]
    ctx = build_context()

    predictions: dict[str, str] = {}
    for t in tasks:
        llm = ScriptedLLM(
            [
                _plan(),
                LLMResponse(
                    tool_calls=[ToolCall("c1", "run_sql", {"sql": t["reference_sql"]})]
                ),
                LLMResponse(content="完成"),
            ]
        )
        agent = DataAgent(llm=llm, registry=default_registry(), tool_ctx=ctx)
        r = agent.run(t["question"], as_of=t.get("as_of"))
        assert r.status == "answered", r.stop_reason
        sql = last_successful_sql(r)
        assert sql, f"{t['id']} 没有产出可评测的 SQL"
        predictions[t["id"]] = sql

    run = evaluate(tasks, predictions, source="integration-test")
    passed = [r for r in run["results"] if r["matched"]]
    assert len(passed) == len(tasks), [
        (r["id"], r["reason"]) for r in run["results"] if not r["matched"]
    ]


def test_last_successful_sql_ignores_failed_calls():
    """只取成功的 run_sql；失败的查询不能拿去评测。"""

    class FakeResult:
        tool_calls = [
            {"name": "search_metrics", "ok": True, "arguments": {"query": "成交额"}},
            {"name": "run_sql", "ok": False, "arguments": {"sql": "SELECT bad"}},
            {"name": "run_sql", "ok": True, "arguments": {"sql": "SELECT good"}},
            {"name": "make_chart", "ok": True, "arguments": {"handle": "rs_1"}},
        ]

    assert last_successful_sql(FakeResult()) == "SELECT good"


def test_last_successful_sql_returns_none_when_no_query():
    class FakeResult:
        tool_calls = [
            {"name": "search_metrics", "ok": True, "arguments": {"query": "x"}},
        ]

    assert last_successful_sql(FakeResult()) is None


def test_missing_api_key_gives_actionable_error(monkeypatch):
    """没有密钥时必须明确报错，而不是悄悄退化成假模型。

    悄悄退化会让人以为跑出了真实结果，这种错误比直接报错危险得多。
    """
    from data_agent.llm import build_client
    from data_agent.llm.base import LLMError  # noqa: F401

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_client()
