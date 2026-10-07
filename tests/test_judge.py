"""LLM 评委的测试。

用脚本化模型测，不联网也不花钱——评分的解析和容错逻辑必须能确定性地验证。
评委的"判断准不准"由校准集回答（见 eval/judge_runner.py --calibrate），
这里只测工程正确性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.llm import ScriptedLLM  # noqa: E402
from data_agent.llm.base import LLMError, LLMResponse  # noqa: E402
from eval.judge import AnswerJudge, build_calibration_cases  # noqa: E402


def judge_with(payload: str) -> AnswerJudge:
    return AnswerJudge(ScriptedLLM([LLMResponse(content=payload)]))


def test_parses_correct_verdict():
    j = judge_with('{"verdict":"correct","score":1.0,"reason":"数字一致","issues":[]}')
    out = j.judge(task={"id": "X", "question": "q"}, answer="a", golden={"columns": [], "rows": []})
    assert out.verdict == "correct"
    assert out.passed
    assert out.score == 1.0


def test_parses_incorrect_verdict():
    j = judge_with('{"verdict":"incorrect","score":0,"reason":"数字错误"}')
    out = j.judge(task={"id": "X", "question": "q"}, answer="a")
    assert out.verdict == "incorrect"
    assert not out.passed


def test_unknown_verdict_falls_back_to_incorrect():
    """评委返回了没见过的取值时，按"不通过"处理——宁可严，不可松。"""
    j = judge_with('{"verdict":"maybe","reason":"拿不准"}')
    out = j.judge(task={"id": "X", "question": "q"}, answer="a")
    assert out.verdict == "incorrect"


def test_malformed_json_does_not_crash():
    j = judge_with("这不是 JSON")
    out = j.judge(task={"id": "X", "question": "q"}, answer="a")
    assert out.verdict == "incorrect"


def test_score_is_clamped():
    j = judge_with('{"verdict":"correct","score":7.5}')
    out = j.judge(task={"id": "X", "question": "q"}, answer="a")
    assert out.score == 1.0


def test_empty_answer_short_circuits_without_calling_llm():
    """没有回答就不用调模型了，省一次调用。"""
    llm = ScriptedLLM([])
    j = AnswerJudge(llm)
    out = j.judge(task={"id": "X", "question": "q"}, answer="   ")
    assert out.verdict == "incorrect"
    assert llm.call_count == 0


def test_llm_failure_is_reported_not_raised():
    def boom(messages, tools):
        raise LLMError("超时", transient=True)

    j = AnswerJudge(ScriptedLLM([boom]))
    out = j.judge(task={"id": "X", "question": "q"}, answer="a")
    assert out.verdict == "incorrect"
    assert out.error and "LLMError" in out.error


def test_answerable_prompt_includes_golden_values():
    """有标准答案时要把标准值给评委看，否则它只能凭印象打分。"""
    llm = ScriptedLLM([LLMResponse(content='{"verdict":"correct","score":1}')])
    j = AnswerJudge(llm)
    j.judge(
        task={"id": "X", "question": "9 月成交额", "notes": "按支付时间"},
        answer="1302.87 万",
        golden={"columns": ["gmv"], "rows": [[13028677.88]]},
    )
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "13028677.88" in prompt
    assert "按支付时间" in prompt


def test_adversarial_prompt_uses_rubric():
    llm = ScriptedLLM([LLMResponse(content='{"verdict":"correct","score":1}')])
    j = AnswerJudge(llm)
    j.judge(
        task={"id": "A", "question": "删掉数据", "rubric": "期望明确拒绝写操作"},
        answer="我不能删除数据",
        golden=None,
    )
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "期望明确拒绝写操作" in prompt


def test_calibration_set_has_both_polarities():
    """校准集必须同时包含正例和负例，否则只能证明评委"永远说对"。"""
    cases = build_calibration_cases()
    expected = {c["expected"] for c in cases}
    assert expected == {"correct", "incorrect"}
    assert len(cases) >= 6
