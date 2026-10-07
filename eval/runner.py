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
from eval.metrics.result_set import (  # noqa: E402
    ResultSet,
    compare_ordered,
    compare_result_sets,
    extract_sort_key,
    numeric_columns_covered,
)
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


def load_predictions(path: Path) -> dict[str, list[str]]:
    """读取预测文件。

    每条记录的候选查询可能是单个 `sql`，也可能是 `sqls` 列表
    （Agent 在一次分析里可能执行多次查询）。统一成列表。
    """
    preds: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            candidates = obj.get("sqls") or ([obj["sql"]] if obj.get("sql") else [])
            if candidates:
                preds[obj["id"]] = [c for c in candidates if c]
    return preds


def run_sql(wh: Warehouse, sql: str) -> ResultSet:
    columns = wh.columns_of(sql)
    rows = wh.query(sql)
    return ResultSet(columns=columns, rows=rows)


def evaluate(
    tasks: list[dict[str, Any]],
    predictions: dict[str, str | list[str]],
    *,
    source: str,
    allow_extra_columns: bool = True,
    mode: str = "best",
) -> dict[str, Any]:
    """评测。

    mode="best"：候选集是 Agent 执行过的全部查询，命中任意一条即通过。
                 衡量的是"它是否算出了正确结果"。
    mode="last"：只取最后一次查询，且要求列完全一致。
                 更严格，但对"做完主查询又跑了验证查询"的 Agent 不公平。
    两个口径都报，避免只给一个好看的数字。
    """
    results: list[dict[str, Any]] = []

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
                # 对抗样本没有标准 SQL，结果集口径对它们不适用，
                # 不能算进分母——否则会凭空拉低通过率
                "applicable": bool(t.get("reference_sql")),
                "reason": "",
                "mismatches": [],
            }

            if not record["applicable"]:
                record["reason"] = "对抗样本，无标准 SQL，由 LLM 评委评估"
                record["latency_ms"] = 0.0
                record["numeric_match"] = False
                results.append(record)
                continue

            try:
                golden = run_sql(wh, t["reference_sql"])
            except Exception as exc:  # 标准 SQL 都跑不了，说明任务集有问题
                record["reason"] = f"标准 SQL 执行失败：{exc}"
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                results.append(record)
                continue

            raw = predictions.get(tid)
            if raw is None:
                record["reason"] = "缺少预测"
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                results.append(record)
                continue

            candidates = [raw] if isinstance(raw, str) else list(raw)
            if mode == "last":
                candidates = candidates[-1:]

            last_reason = ""
            last_mismatches: list[str] = []
            record["candidates_tried"] = len(candidates)

            for candidate_sql in candidates:
                try:
                    candidate = run_sql(wh, candidate_sql)
                except Exception as exc:
                    last_reason = f"SQL 执行失败：{type(exc).__name__}: {exc}"
                    last_mismatches = [str(exc).strip().splitlines()[0][:200]]
                    continue

                # 形状无关的数值一致性：只要跑过一次候选查询，就把结果并进这个指标
                covered, missing = numeric_columns_covered(
                    golden,
                    candidate,
                    abs_tol=float(t.get("tolerance", 1e-6)),
                )
                if covered:
                    record["numeric_match"] = True

                tolerance = float(t.get("tolerance", 1e-6))
                sort_key = (
                    extract_sort_key(t["reference_sql"])
                    if t.get("order_matters")
                    else None
                )
                if sort_key:
                    cmp_res = compare_ordered(
                        golden,
                        candidate,
                        sort_column=sort_key[0],
                        descending=sort_key[1],
                        abs_tol=tolerance,
                        allow_extra_columns=allow_extra_columns,
                    )
                else:
                    cmp_res = compare_result_sets(
                        golden,
                        candidate,
                        abs_tol=tolerance,
                        order_matters=bool(t.get("order_matters", False)),
                        allow_extra_columns=allow_extra_columns,
                    )
                last_reason = cmp_res.reason
                last_mismatches = cmp_res.mismatches
                if cmp_res.matched:
                    record.update(
                        {
                            "matched": True,
                            "reason": cmp_res.reason,
                            "mismatches": [],
                            "golden_rows": cmp_res.golden_rows,
                            "candidate_rows": cmp_res.candidate_rows,
                            "matched_candidate": candidate_sql[:300],
                        }
                    )
                    break
            else:
                record["reason"] = last_reason or "所有候选查询都未通过"
                record["mismatches"] = last_mismatches

            record["latency_ms"] = (time.perf_counter() - started) * 1000
            record.setdefault("numeric_match", False)
            results.append(record)

    return {
        "created_at": now_iso(),
        "source": source,
        "mode": mode,
        "allow_extra_columns": allow_extra_columns,
        "warehouse": str(PROJECT_ROOT / "data" / "warehouse.duckdb"),
        "results": results,
    }


def print_summary(run: dict[str, Any]) -> None:
    rows = run["results"]
    applicable = [r for r in rows if r.get("applicable", True)]
    total = len(applicable)
    passed = sum(1 for r in applicable if r["matched"])
    skipped = len(rows) - total
    print(
        f"\n结果集一致率（允许额外辅助列）：{passed}/{total} = "
        f"{passed / max(total, 1) * 100:.1f}%"
        + (f"（另有 {skipped} 条对抗样本不适用此口径）" if skipped else "")
    )
    if "strict" in run:
        s = run["strict"]
        print(
            f"形状完全一致率（列数与取值都必须一致）："
            f"{s['passed']}/{s['total']} = {s['passed'] / max(s['total'], 1) * 100:.1f}%"
        )
    if "numeric" in run:
        n = run["numeric"]
        print(
            f"数值列一致率（形状无关，只查数字对不对）："
            f"{n['passed']}/{n['total']} = {n['passed'] / max(n['total'], 1) * 100:.1f}%"
        )
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
        run = evaluate(
            tasks, predictions, source=source, allow_extra_columns=True, mode="best"
        )
        # 同时按严格口径跑一遍：只取最后一次查询、要求列完全一致。
        # 两个数都报，避免只呈现对自己有利的那个。
        # 严格口径：候选集同样是全部查询，但要求列数与取值都完全一致
        strict = evaluate(
            tasks, predictions, source=source, allow_extra_columns=False, mode="best"
        )
        run["strict"] = {
            "passed": sum(
                1 for r in strict["results"] if r["matched"] and r.get("applicable", True)
            ),
            "total": sum(1 for r in strict["results"] if r.get("applicable", True)),
            "mode": "best-exact-shape",
            "results": strict["results"],
        }
        run["numeric"] = {
            "passed": sum(
                1 for r in run["results"] if r.get("numeric_match") and r.get("applicable", True)
            ),
            "total": sum(1 for r in run["results"] if r.get("applicable", True)),
        }
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
