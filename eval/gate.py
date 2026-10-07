"""CI 门禁。

门禁的本质只有一句话：**指标回退就阻断合并**。

分两层：

  * 离线层（不需要模型密钥，每次提交都能跑，几秒完成）
      - 评测集本身是否合法：ID 唯一、标准 SQL 能跑、对抗样本有 rubric
      - 自检：标准 SQL 当预测必须 100% 通过（证明 harness 没坏）
      - 负例：典型错误答案必须 0% 通过（证明 harness 能抓错）

  * 在线层（需要模型密钥）
      - 抽样跑真实任务，检查答案正确率 / SQL 通过率 / 熔断比例
      - 抽样会覆盖三个类别，不能只抽最简单的单表任务

用法：
    python eval/gate.py                 # 只跑离线层
    python eval/gate.py --online        # 离线层 + 在线层
    python eval/gate.py --online --sample 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.infra.db import Warehouse  # noqa: E402
from eval.runner import TASKS_PATH, evaluate, load_predictions, load_tasks  # noqa: E402


class GateFailure(RuntimeError):
    """门禁未通过。"""


def _load_thresholds() -> dict:
    return json.loads(
        (PROJECT_ROOT / "eval" / "thresholds.json").read_text(encoding="utf-8")
    )


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


# ----------------------------------------------------------------------
# 离线层
# ----------------------------------------------------------------------


def check_task_set(tasks: list[dict], cfg: dict) -> list[str]:
    """评测集自身的合法性。评测集坏了，后面所有数字都没有意义。"""
    warnings: list[str] = []
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise GateFailure(f"任务 ID 重复：{dupes}")

    if len(tasks) < cfg["min_tasks_total"]:
        raise GateFailure(f"任务数 {len(tasks)} 少于门槛 {cfg['min_tasks_total']}")

    adversarial = [t for t in tasks if t["category"] == "adversarial"]
    if len(adversarial) < cfg["min_adversarial_tasks"]:
        raise GateFailure(
            f"对抗样本 {len(adversarial)} 条少于门槛 {cfg['min_adversarial_tasks']}"
        )

    for t in adversarial:
        if not t.get("rubric"):
            raise GateFailure(f"对抗样本 {t['id']} 缺少 rubric，评委无法评分")
        if not t.get("expected_behavior"):
            raise GateFailure(f"对抗样本 {t['id']} 缺少 expected_behavior")

    with Warehouse() as wh:
        for t in tasks:
            sql = t.get("reference_sql")
            if not sql:
                if t["category"] != "adversarial":
                    raise GateFailure(f"非对抗任务 {t['id']} 缺少 reference_sql")
                continue
            try:
                rows = wh.query(sql)
            except Exception as exc:  # noqa: BLE001
                raise GateFailure(
                    f"任务 {t['id']} 的标准 SQL 执行失败："
                    f"{str(exc).splitlines()[0][:120]}"
                ) from exc
            if not rows:
                warnings.append(f"⚠ 任务 {t['id']} 的标准 SQL 返回空结果")
            if not t.get("notes"):
                warnings.append(f"⚠ 任务 {t['id']} 缺少口径说明，无法复核")
    return warnings


def check_sanity(tasks: list[dict], cfg: dict) -> float:
    """标准 SQL 当预测，必须全过。证明 harness 通路没问题。"""
    preds = {t["id"]: t["reference_sql"] for t in tasks if t.get("reference_sql")}
    run = evaluate(tasks, preds, source="gate-sanity")
    applicable = [r for r in run["results"] if r.get("applicable", True)]
    rate = sum(1 for r in applicable if r["matched"]) / max(len(applicable), 1)
    if rate < cfg["sanity_pass_rate"]:
        bad = [r["id"] for r in applicable if not r["matched"]]
        raise GateFailure(
            f"自检未通过：{rate:.1%} < {cfg['sanity_pass_rate']:.0%}，失败任务 {bad}"
        )
    return rate


def check_determinism(tasks: list[dict], *, repeats: int = 3) -> list[str]:
    """标准 SQL 必须可重复：同一条跑多次结果要完全一致。

    真踩过这个坑：T35 的 `ORDER BY conversion DESC` 有并列值，
    DuckDB 并行执行时并列行的顺序不稳定，而该任务开了 order_matters，
    于是"自检"会随机失败——同一条标准 SQL 和自己比都不相等。

    这类问题只在特定并发度下复现，靠人工测很容易漏。所以做成门禁的固定检查：
    凡是带排序要求的任务，跑多次必须完全一致。修法是给 ORDER BY 加唯一 tiebreaker。
    """
    problems: list[str] = []
    ordered = [t for t in tasks if t.get("order_matters") and t.get("reference_sql")]
    if not ordered:
        return problems
    with Warehouse() as wh:
        for t in ordered:
            signatures = {repr(wh.query(t["reference_sql"])) for _ in range(repeats)}
            if len(signatures) > 1:
                problems.append(
                    f"任务 {t['id']} 的标准 SQL 多次执行结果不一致"
                    f"（ORDER BY 可能有并列值，需要加唯一 tiebreaker）"
                )
    return problems


def check_negative(tasks: list[dict], cfg: dict) -> float:
    """典型错误答案必须全被抓住。证明 harness 真的能识别错误。

    只统计**提供了错误答案的那部分任务**。没提供预测的任务算"缺少预测"、
    也会判为不通过，把它们算进分母会让这个门禁看起来很宽松。
    """
    wrong_path = PROJECT_ROOT / "eval" / "datasets" / "wrong_answers.jsonl"
    if not wrong_path.exists():
        raise GateFailure(f"负例文件不存在：{wrong_path}")
    preds = load_predictions(wrong_path)
    if len(preds) < cfg.get("min_wrong_answer_cases", 10):
        raise GateFailure(
            f"负例只覆盖 {len(preds)} 条任务，少于门槛 "
            f"{cfg.get('min_wrong_answer_cases', 10)}"
        )
    run = evaluate(tasks, preds, source="gate-negative")
    covered = [
        r for r in run["results"]
        if r.get("applicable", True) and r["id"] in preds
    ]
    rate = sum(1 for r in covered if r["matched"]) / max(len(covered), 1)
    if rate > cfg["wrong_answers_pass_rate_max"]:
        leaked = [r["id"] for r in covered if r["matched"]]
        raise GateFailure(
            f"负例通过率 {rate:.1%} 超过上限 "
            f"{cfg['wrong_answers_pass_rate_max']:.0%}，漏放的任务 {leaked}"
        )
    print(f"    （负例覆盖 {len(covered)} 条任务）")
    return rate


# ----------------------------------------------------------------------
# 在线层
# ----------------------------------------------------------------------


def check_online(tasks: list[dict], cfg: dict, sample: int) -> dict:
    """抽样跑真实任务，检查质量没有回退。"""
    from data_agent.agent import DataAgent
    from data_agent.llm import build_client
    from data_agent.tools import build_context, default_registry
    from eval.judge import AnswerJudge
    from eval.run_agent import all_successful_sqls

    by_cat: dict[str, list[dict]] = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t)

    per_cat = max(1, sample // max(len(by_cat), 1))
    picked: list[dict] = []
    for cat in sorted(by_cat):
        picked.extend(by_cat[cat][:per_cat])
    picked = picked[:sample]

    llm = build_client()
    agent = DataAgent(llm=llm, registry=default_registry(), tool_ctx=build_context())
    judge = AnswerJudge(llm)

    predictions: dict[str, list[str]] = {}
    answers: dict[str, str | None] = {}
    statuses: dict[str, str] = {}
    for t in picked:
        r = agent.run(t["question"], as_of=t.get("as_of"))
        sqls = all_successful_sqls(r)
        if sqls:
            predictions[t["id"]] = sqls
        answers[t["id"]] = r.answer
        statuses[t["id"]] = r.status

    sql_run = evaluate(picked, predictions, source="gate-online")
    applicable = [r for r in sql_run["results"] if r.get("applicable", True)]
    sql_rate = sum(1 for r in applicable if r["matched"]) / max(len(applicable), 1)

    correct = 0
    with Warehouse() as wh:
        for t in picked:
            golden = None
            if t.get("reference_sql"):
                golden = {
                    "columns": wh.columns_of(t["reference_sql"]),
                    "rows": [
                        [_jsonable(v) for v in row]
                        for row in wh.query(t["reference_sql"])
                    ],
                }
            judgement = judge.judge(task=t, answer=answers[t["id"]], golden=golden)
            correct += int(judgement.verdict == "correct")

    accuracy = correct / max(len(picked), 1)
    circuit = sum(1 for s in statuses.values() if s == "stopped") / max(len(picked), 1)
    result = {
        "sample_size": len(picked),
        "answer_accuracy": accuracy,
        "sql_pass_rate": sql_rate,
        "circuit_break_ratio": circuit,
    }

    problems: list[str] = []
    if accuracy < cfg["min_answer_accuracy"]:
        problems.append(
            f"答案正确率 {accuracy:.1%} < 门槛 {cfg['min_answer_accuracy']:.0%}"
        )
    if sql_rate < cfg["min_sql_pass_rate"]:
        problems.append(
            f"SQL 通过率 {sql_rate:.1%} < 门槛 {cfg['min_sql_pass_rate']:.0%}"
        )
    if circuit > cfg["max_circuit_break_ratio"]:
        problems.append(
            f"熔断比例 {circuit:.1%} > 上限 {cfg['max_circuit_break_ratio']:.0%}"
        )
    if problems:
        raise GateFailure("在线门禁未通过：\n  - " + "\n  - ".join(problems))
    return result


# ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="跑 CI 门禁")
    ap.add_argument("--online", action="store_true", help="同时跑在线层（需要模型密钥）")
    ap.add_argument("--sample", type=int, default=0, help="在线层抽样条数")
    args = ap.parse_args()

    cfg = _load_thresholds()
    tasks = load_tasks(TASKS_PATH)
    print(f"评测集：{len(tasks)} 条任务")

    for w in check_task_set(tasks, cfg["offline"]):
        print(f"  {w}")
    print("  [通过] 评测集合法性检查")

    flaky = check_determinism(tasks)
    if flaky:
        raise GateFailure("标准 SQL 结果不稳定：\n  - " + "\n  - ".join(flaky))
    print("  [通过] 标准 SQL 可重复性检查（带排序的任务各跑 3 次）")

    sanity = check_sanity(tasks, cfg["offline"])
    print(f"  [通过] 自检（标准 SQL 当预测）：{sanity:.1%}")

    negative = check_negative(tasks, cfg["offline"])
    print(f"  [通过] 负例（典型错误答案）通过率：{negative:.1%}（要求 0%）")

    if args.online:
        sample = args.sample or cfg["online"]["sample_size"]
        print(f"\n在线层：抽样 {sample} 条跑真实模型…")
        result = check_online(tasks, cfg["online"], sample)
        print(
            f"  [通过] 答案正确率 {result['answer_accuracy']:.1%} | "
            f"SQL 通过率 {result['sql_pass_rate']:.1%} | "
            f"熔断比例 {result['circuit_break_ratio']:.1%}"
        )

    print("\n门禁全部通过")


if __name__ == "__main__":
    try:
        main()
    except GateFailure as exc:
        print(f"\n门禁未通过：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
