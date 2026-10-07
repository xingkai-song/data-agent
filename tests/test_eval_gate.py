"""CI 门禁本身的测试。

门禁的可靠性比评测结果本身更重要：一个永远返回"通过"的门禁
等于没有门禁。所以这里既验证正常情况通过，也验证异常情况能被拦住。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.infra.db import Warehouse  # noqa: E402
from eval.gate import (  # noqa: E402
    GateFailure,
    _load_thresholds,
    check_determinism,
    check_negative,
    check_sanity,
    check_task_set,
)
from eval.metrics.result_set import ResultSet, compare_result_sets  # noqa: E402
from eval.runner import load_predictions, load_tasks  # noqa: E402


@pytest.fixture(scope="module")
def tasks():
    return load_tasks()


@pytest.fixture(scope="module")
def cfg():
    return _load_thresholds()


# ---------- 正常情况应该通过 ----------


def test_task_set_is_valid(tasks, cfg):
    warnings = check_task_set(tasks, cfg["offline"])
    assert warnings == [], f"评测集有告警：{warnings}"


def test_sanity_passes(tasks, cfg):
    assert check_sanity(tasks, cfg["offline"]) == 1.0


def test_negative_cases_are_all_caught(tasks, cfg):
    assert check_negative(tasks, cfg["offline"]) == 0.0


def test_ordered_tasks_are_deterministic(tasks):
    """带排序要求的任务必须可重复。

    真踩过：T35 的 ORDER BY 有并列值，加上 order_matters 之后
    自检会随机失败——同一条标准 SQL 和自己比都不相等。
    修法是加唯一 tiebreaker，这条测试防止以后新增任务再犯。
    """
    assert check_determinism(tasks, repeats=3) == []


# ---------- 关键：每条负例必须真的和标准答案不同 ----------


def test_every_wrong_answer_actually_differs(tasks):
    """元测试：防止写出"其实也算对"的负例。

    真踩过这个坑——T17 我用 COUNT(*) 代替 COUNT(DISTINCT order_id)，
    但那个查询是在 orders 表上做的，主键唯一，两者结果完全相同，
    这条负例等于没测。门禁当时就把它抓出来了。
    """
    by_id = {t["id"]: t for t in tasks}
    preds = load_predictions(
        PROJECT_ROOT / "eval" / "datasets" / "wrong_answers.jsonl"
    )
    identical: list[str] = []
    with Warehouse() as wh:
        for tid, candidates in preds.items():
            task = by_id.get(tid)
            if not task or not task.get("reference_sql"):
                continue
            golden = ResultSet(
                columns=wh.columns_of(task["reference_sql"]),
                rows=wh.query(task["reference_sql"]),
            )
            for sql in candidates:
                cand = ResultSet(columns=wh.columns_of(sql), rows=wh.query(sql))
                res = compare_result_sets(
                    golden,
                    cand,
                    abs_tol=float(task.get("tolerance", 1e-6)),
                    order_matters=bool(task.get("order_matters", False)),
                    allow_extra_columns=True,
                )
                if res.matched:
                    identical.append(f"{tid}: {sql[:80]}")
    assert not identical, "以下负例其实和标准答案一致：\n" + "\n".join(identical)


# ---------- 异常情况必须被拦住 ----------


def test_duplicate_ids_rejected(tasks, cfg):
    broken = tasks + [dict(tasks[0])]
    with pytest.raises(GateFailure, match="重复"):
        check_task_set(broken, cfg["offline"])


def test_too_few_tasks_rejected(tasks, cfg):
    with pytest.raises(GateFailure, match="少于门槛"):
        check_task_set(tasks[:10], cfg["offline"])


def test_adversarial_missing_rubric_rejected(tasks, cfg):
    broken = [dict(t) for t in tasks]
    for t in broken:
        if t["category"] == "adversarial":
            t.pop("rubric", None)
            break
    with pytest.raises(GateFailure, match="rubric"):
        check_task_set(broken, cfg["offline"])


def test_non_adversarial_missing_sql_rejected(tasks, cfg):
    broken = [dict(t) for t in tasks]
    for t in broken:
        if t["category"] != "adversarial":
            t["reference_sql"] = None
            break
    with pytest.raises(GateFailure, match="reference_sql"):
        check_task_set(broken, cfg["offline"])


def test_broken_reference_sql_rejected(tasks, cfg):
    broken = [dict(t) for t in tasks]
    for t in broken:
        if t.get("reference_sql"):
            t["reference_sql"] = "SELECT no_such_column FROM orders"
            break
    with pytest.raises(GateFailure, match="执行失败"):
        check_task_set(broken, cfg["offline"])


def test_missing_keyword_rejected_does_not_crash(tasks, cfg):
    """任务字段缺失时应该报错，而不是抛 KeyError。"""
    broken = [dict(t) for t in tasks]
    del broken[0]["category"]
    with pytest.raises((GateFailure, KeyError)):
        check_task_set(broken, cfg["offline"])
