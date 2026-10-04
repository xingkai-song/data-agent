"""三个策略模块的单元测试。

这些逻辑用真模型反而测不准（模型不会每次都犯错），所以必须单独测。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.agent.policies import Budget, LoopGuard, RetryPolicy
from data_agent.tools.base import ErrorKind, ToolResult


# ---------- 重试策略 ----------


def _result(kind: ErrorKind | None) -> ToolResult:
    if kind is None:
        return ToolResult.success({"ok": True})
    return ToolResult.failure(kind, "boom")


def test_retry_succeeds_first_try():
    policy = RetryPolicy(sleep=lambda _: None)
    result, attempts = policy.call(lambda: _result(None))
    assert result.ok
    assert attempts == 1


def test_transient_error_is_retried():
    """瞬态错误（超时/限流）必须重试。"""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return _result(None) if calls["n"] >= 3 else _result(ErrorKind.TRANSIENT)

    policy = RetryPolicy(max_attempts=3, sleep=lambda _: None)
    result, attempts = policy.call(flaky)
    assert result.ok
    assert attempts == 3


def test_transient_error_gives_up_after_max_attempts():
    policy = RetryPolicy(max_attempts=2, sleep=lambda _: None)
    result, attempts = policy.call(lambda: _result(ErrorKind.TRANSIENT))
    assert not result.ok
    assert result.error_kind is ErrorKind.TRANSIENT
    assert attempts == 2


def test_validation_error_is_not_retried():
    """参数错误重试没有意义——同样的参数只会得到同样的错误。

    这类错误应该回灌给模型让它改参数，而不是原样重试。
    """
    calls = {"n": 0}

    def always_validation():
        calls["n"] += 1
        return _result(ErrorKind.VALIDATION)

    policy = RetryPolicy(max_attempts=3, sleep=lambda _: None)
    result, attempts = policy.call(always_validation)
    assert not result.ok
    assert attempts == 1
    assert calls["n"] == 1


def test_denied_error_is_not_retried():
    policy = RetryPolicy(max_attempts=3, sleep=lambda _: None)
    _, attempts = policy.call(lambda: _result(ErrorKind.DENIED))
    assert attempts == 1


def test_backoff_is_exponential_and_capped():
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=2.0)
    assert policy.delay_for(0) == 0.5
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(9) == 2.0  # 不超过上限


def test_retry_records_attempt_counts():
    policy = RetryPolicy(max_attempts=2, sleep=lambda _: None)
    policy.call(lambda: _result(ErrorKind.VALIDATION))
    policy.call(lambda: _result(None))
    assert policy.attempts == [1, 1]


# ---------- 死循环防护 ----------


def test_signature_is_stable_and_order_independent():
    a = LoopGuard.signature("run_sql", {"sql": "SELECT 1", "purpose": "x"})
    b = LoopGuard.signature("run_sql", {"purpose": "x", "sql": "SELECT 1"})
    assert a == b
    c = LoopGuard.signature("run_sql", {"sql": "SELECT 2", "purpose": "x"})
    assert a != c


def test_max_steps_stops():
    guard = LoopGuard(max_steps=3)
    assert guard.check(step=2, tool_calls=[], observations=[]) is None
    reason = guard.check(step=3, tool_calls=[], observations=[])
    assert reason and "最大步数" in reason


def test_repeated_call_detected():
    """同一个工具 + 同一套参数反复出现，说明模型在原地打转。"""
    guard = LoopGuard(max_repeats=2)
    sig = LoopGuard.signature("run_sql", {"sql": "SELECT 1"})
    calls = [{"signature": sig}] * 3
    reason = guard.check(step=1, tool_calls=calls, observations=[])
    assert reason and "重复" in reason


def test_different_calls_not_flagged():
    guard = LoopGuard(max_repeats=2)
    calls = [
        {"signature": LoopGuard.signature("run_sql", {"sql": f"SELECT {i}"})}
        for i in range(5)
    ]
    assert guard.check(step=1, tool_calls=calls, observations=[]) is None


def test_no_progress_detected():
    """参数不同但结果完全一样，也要能识别出来。"""
    guard = LoopGuard(no_progress_window=3)
    obs = [{"digest": "same"}] * 3
    reason = guard.check(step=1, tool_calls=[], observations=obs)
    assert reason and "没有新信息" in reason


def test_progress_not_flagged():
    guard = LoopGuard(no_progress_window=3)
    obs = [{"digest": f"d{i}"} for i in range(4)]
    assert guard.check(step=1, tool_calls=[], observations=obs) is None


def test_digest_is_content_based():
    assert LoopGuard.digest({"a": 1, "b": 2}) == LoopGuard.digest({"b": 2, "a": 1})
    assert LoopGuard.digest({"a": 1}) != LoopGuard.digest({"a": 2})


def test_digest_ignores_volatile_fields():
    """回归测试：结果里带一个每次新建的句柄时，摘要不能跟着变。

    否则"连续 N 步没有新信息"这个检测永远不会触发——
    两次完全相同的查询，因为句柄不同被判成"有新信息"。
    """
    a = {"handle": "rs_aaa", "row_count": 1, "preview": [[60000]]}
    b = {"handle": "rs_bbb", "row_count": 1, "preview": [[60000]]}
    assert LoopGuard.digest(a) == LoopGuard.digest(b)


def test_digest_still_detects_real_content_change():
    a = {"handle": "rs_aaa", "row_count": 1, "preview": [[60000]]}
    b = {"handle": "rs_aaa", "row_count": 1, "preview": [[12345]]}
    assert LoopGuard.digest(a) != LoopGuard.digest(b)


# ---------- 预算 ----------


def test_budget_token_limit():
    b = Budget(max_tokens=1000, max_tool_calls=100, max_seconds=1000)
    assert b.exceeded({"input_tokens": 500, "output_tokens": 400}, time.time()) is None
    reason = b.exceeded({"input_tokens": 900, "output_tokens": 200}, time.time())
    assert reason and "token" in reason


def test_budget_tool_call_limit():
    b = Budget(max_tokens=10**9, max_tool_calls=3, max_seconds=1000)
    assert b.exceeded({"tool_calls": 3}, time.time()) is None
    reason = b.exceeded({"tool_calls": 4}, time.time())
    assert reason and "工具调用" in reason


def test_budget_time_limit():
    b = Budget(max_tokens=10**9, max_tool_calls=100, max_seconds=0.01)
    time.sleep(0.02)
    reason = b.exceeded({}, time.time() - 1.0)
    assert reason and "耗时" in reason


def test_budget_remaining_tokens():
    b = Budget(max_tokens=1000)
    assert b.remaining_tokens({"input_tokens": 300, "output_tokens": 200}) == 500
    assert b.remaining_tokens({"input_tokens": 2000, "output_tokens": 0}) == 0
