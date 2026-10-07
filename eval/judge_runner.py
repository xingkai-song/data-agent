"""用 LLM-as-judge 评估答案正确性。

输入是 run_agent.py 产出的预测文件（含 answer），输出是逐条判定 + 汇总报告。

用法：
    python eval/judge_runner.py                          # 评全部
    python eval/judge_runner.py --calibrate              # 只跑评委校准
    python eval/judge_runner.py --category adversarial   # 只评对抗样本
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.infra.db import Warehouse  # noqa: E402
from data_agent.llm import build_client  # noqa: E402
from eval.judge import AnswerJudge, calibrate  # noqa: E402
from eval.runner import load_tasks  # noqa: E402

PRED_DIR = PROJECT_ROOT / "eval" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def load_predictions(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[obj["id"]] = obj
    return out


def golden_for(task: dict, wh: Warehouse) -> dict | None:
    sql = task.get("reference_sql")
    if not sql:
        return None
    try:
        return {
            "columns": wh.columns_of(sql),
            "rows": [[_jsonable(v) for v in r] for r in wh.query(sql)],
        }
    except Exception:  # noqa: BLE001
        return None


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="用 LLM 评委评估答案正确性")
    ap.add_argument("--predictions", type=Path, default=PRED_DIR / "agent_latest.jsonl")
    ap.add_argument("--category", default=None, help="只评某一类（single/multi/adversarial）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true", help="只跑评委校准集")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    llm = build_client()
    judge = AnswerJudge(llm)

    if args.calibrate:
        print("跑评委校准集（6 条已知对错的样本）...\n")
        result = calibrate(judge)
        for d in result["details"]:
            mark = "OK " if d["agrees"] else "MISS"
            print(f"  [{mark}] {d['name']}: 人工={d['expected']} 评委={d['judged']}")
        print(
            f"\n评委与人工一致率：{result['agreements']}/{result['total']} "
            f"= {result['agreement_rate'] * 100:.1f}%"
        )
        path = args.out or (RESULTS_DIR / "judge_calibration.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入：{path}")
        return

    tasks = load_tasks()
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]
    if args.limit:
        tasks = tasks[: args.limit]

    preds = load_predictions(args.predictions)
    print(f"模型：{getattr(llm, 'model', '?')}")
    print(f"待评任务：{len(tasks)}\n")

    started = time.perf_counter()
    judgements = []
    with Warehouse() as wh:
        for t in tasks:
            rec = preds.get(t["id"], {})
            j = judge.judge(
                task=t,
                answer=rec.get("answer"),
                golden=golden_for(t, wh),
            )
            judgements.append(j)
            mark = {"correct": "OK  ", "partially_correct": "PART", "incorrect": "FAIL"}[
                j.verdict
            ]
            print(f"  [{mark}] {t['id']:4s} {j.reason[:80]}")

    elapsed = time.perf_counter() - started
    total = len(judgements)
    correct = sum(1 for j in judgements if j.verdict == "correct")
    partial = sum(1 for j in judgements if j.verdict == "partially_correct")
    score = sum(j.score for j in judgements) / total if total else 0.0

    by_cat: dict[str, list] = {}
    for t, j in zip(tasks, judgements):
        by_cat.setdefault(t["category"], []).append(j)

    print(f"\n{'=' * 60}")
    print(f"答案正确率（verdict=correct）：{correct}/{total} = {correct / total * 100:.1f}%")
    print(f"完全正确 + 部分正确：{(correct + partial)}/{total} = {(correct + partial) / total * 100:.1f}%")
    print(f"平均得分：{score:.3f}")
    print(f"耗时 {elapsed:.1f}s（平均 {elapsed / max(total, 1):.1f}s/条）")
    print(f"\n{'类别':<14}{'任务数':>6}{'正确':>6}{'正确率':>8}{'平均分':>9}")
    for cat, items in sorted(by_cat.items()):
        c = sum(1 for j in items if j.verdict == "correct")
        s = sum(j.score for j in items) / len(items)
        print(f"{cat:<14}{len(items):>6}{c:>6}{c / len(items) * 100:>7.1f}%{s:>9.3f}")

    out = args.out or (RESULTS_DIR / "judge_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": getattr(llm, "model", None),
                "total": total,
                "correct": correct,
                "partially_correct": partial,
                "average_score": score,
                "elapsed_s": round(elapsed, 1),
                "by_category": {
                    k: {
                        "n": len(v),
                        "correct": sum(1 for j in v if j.verdict == "correct"),
                        "avg_score": sum(j.score for j in v) / len(v),
                    }
                    for k, v in by_cat.items()
                },
                "judgements": [j.to_dict() for j in judgements],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告已写入：{out}")


if __name__ == "__main__":
    main()
