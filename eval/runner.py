"""评测执行器。

用法：
    # 1) 自检：用标准 SQL 当作预测，应该 100% 通过。用来验证 harness 本身没写错
    python eval/runner.py --sanity

    # 2) 用 Agent 产出的预测文件（每行一个 {"id": ..., "sql": ...}）
    python eval/runner.py --predictions eval/predictions/agent_v1.jsonl

    # 3) 单条调试
    python eval/runner.py --task T01 --sql "SELECT ..."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 同时保证包可导入：装了 `pip install -e .` 或直接从源码跑都能工作
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.infra.db import Warehouse  # noqa: E402
from eval.metrics.result_set import ResultSet, compare_result_sets  # noqa: E402
from eval.report import now_iso, write_report  # noqa: E402

TASKS_PATH = PROJECT_ROOT / "eval" / "datasets" / "tasks.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    tasks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_predictions(path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("sql"):
                preds[obj["id"]] = obj["sql"]
    return preds


def run_sql(wh: Warehouse, sql: str) -> ResultSet:
    columns = wh.columns_of(sql)
    rows = wh.query(sql)
    return ResultSet(columns=columns, rows=rows)


def evaluate(
    tasks: list[dict[str, Any]],
    predictions: dict[str, str],
    *,
    source: str,
    order_matters_map: dict[str, bool] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    order_matters_map = order_matters_map or {}

    with Warehouse() as wh:
        for t in tasks:
            tid = t["id"]
            started = time.perf_counter()
            record: dict[str, Any] = {
                "id": tid,
                "question": t["question"],
                "category": t["category"],
                "difficulty": t["difficulty"],
                "matched": False,
                "reason": "",
                "mismatches": [],
            }

            try:
                golden = run_sql(wh, t["reference_sql"])
            except Exception as exc:  # 标准 SQL 都跑不了，说明任务集有问题
                record["reason"] = f"标准 SQL 执行失败：{exc}"
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                results.append(record)
                continue

            sql = predictions.get(tid)
            if not sql:
                record["reason"] = "缺少预测"
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                results.append(record)
                continue

            try:
                candidate = run_sql(wh, sql)
            except Exception as exc:
                record["reason"] = f"SQL 执行失败：{type(exc).__name__}: {exc}"
                record["mismatches"].append(str(exc).strip().splitlines()[0][:200])
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                results.append(record)
                continue

            cmp_res = compare_result_sets(
                golden,
                candidate,
                abs_tol=float(t.get("tolerance", 1e-6)),
                order_matters=order_matters_map.get(tid, t.get("order_matters", False)),
            )
            record.update(
                {
                    "matched": cmp_res.matched,
                    "reason": cmp_res.reason,
                    "mismatches": cmp_res.mismatches,
                    "golden_rows": cmp_res.golden_rows,
                    "candidate_rows": cmp_res.candidate_rows,
                }
            )
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            results.append(record)

    return {
        "created_at": now_iso(),
        "source": source,
        "warehouse": str(PROJECT_ROOT / "data" / "warehouse.duckdb"),
        "results": results,
    }


def print_summary(run: dict[str, Any]) -> None:
    rows = run["results"]
    total = len(rows)
    passed = sum(1 for r in rows if r["matched"])
    print(f"\n结果集精确比对：{passed}/{total} = {passed / total * 100:.1f}%")
    for r in rows:
        flag = "PASS" if r["matched"] else "FAIL"
        print(f"  [{flag}] {r['id']} ({r['difficulty']:<6s}) {r['question']}")
        if not r["matched"]:
            print(f"         └─ {r['reason']}")
            for m in r["mismatches"][:3]:
                print(f"            {m}")


def main() -> None:
    ap = argparse.ArgumentParser(description="数据分析 Agent 评测执行器")
    ap.add_argument("--sanity", action="store_true",
                    help="用标准 SQL 当预测，验证 harness 是否正确")
    ap.add_argument("--predictions", type=Path, help="预测 JSONL 文件")
    ap.add_argument("--task", help="只跑某一条任务")
    ap.add_argument("--sql", help="配合 --task 使用，直接指定 SQL")
    ap.add_argument("--out", type=Path, default=None, help="报告输出路径")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"未找到任务 {args.task}")
            raise SystemExit(1)

    if args.sanity:
        predictions = {t["id"]: t["reference_sql"] for t in tasks}
        source = "sanity（标准 SQL 作为预测）"
    elif args.predictions:
        predictions = load_predictions(args.predictions)
        source = f"predictions:{args.predictions.name}"
    elif args.task and args.sql:
        predictions = {args.task: args.sql}
        source = f"inline:{args.task}"
    else:
        ap.error("请指定 --sanity 或 --predictions 或 (--task 与 --sql)")
        return

    try:
        run = evaluate(tasks, predictions, source=source)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

    out = args.out or (RESULTS_DIR / f"report_{run['created_at'].replace(':', '').replace(' ', '_').replace('-', '')}.md")
    write_report(run, out)
    json_path = Path(out).with_suffix(".json")
    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quiet:
        print_summary(run)
    print(f"\n报告已写入：{out}")


if __name__ == "__main__":
    main()
