"""静默错误率：系统自称答完了，但结果其实是错的。

这是本项目主张"**算错比不回答更危险**"的直接度量：

  分母 = 系统给出结论的题（agent_status == "answered"）
  分子 = 其中结果集与参考 SQL 不一致的题

其余状态（clarified / stopped / failed）算"**主动未作答**"，不计入这个
指标——把它们混进来会稀释它，让数字好看但失去意义。反过来，"未作答率"
也应该单独报，否则可以通过"少答"来刷低静默错误率。

用法：
    python eval/silent_error.py eval/predictions/agent_latest.jsonl
    python eval/silent_error.py eval/predictions/agent_60.jsonl --show 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_agent.infra.db import Warehouse  # noqa: E402
from eval.metrics.result_set import ResultSet, compare_result_sets  # noqa: E402

TASKS_PATH = PROJECT_ROOT / "eval" / "datasets" / "tasks.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _result_set(wh: Warehouse, sql: str) -> ResultSet:
    return ResultSet(columns=wh.columns_of(sql), rows=wh.query(sql))


def run(tasks_path: Path, preds_path: Path, *, show: int = 10) -> dict[str, Any]:
    tasks = {t["id"]: t for t in _load_jsonl(tasks_path) if t.get("reference_sql")}
    preds = {p["id"]: p for p in _load_jsonl(preds_path)}

    answered = silent = 0
    not_answered: dict[str, int] = {}
    details: list[tuple[str, str]] = []

    with Warehouse() as wh:
        for tid, t in tasks.items():
            p = preds.get(tid)
            if p is None:
                not_answered["missing"] = not_answered.get("missing", 0) + 1
                continue
            status = p.get("agent_status") or "unknown"
            if status != "answered":
                not_answered[status] = not_answered.get(status, 0) + 1
                continue

            answered += 1
            golden = _result_set(wh, t["reference_sql"])
            candidates = p.get("sqls") or ([p["sql"]] if p.get("sql") else [])
            ok = False
            for sql in candidates:
                if not sql:
                    continue
                try:
                    cand = _result_set(wh, sql)
                except Exception:  # noqa: BLE001 - 候选跑不通就是没过
                    continue
                if compare_result_sets(
                    golden,
                    cand,
                    abs_tol=float(t.get("tolerance", 1e-6)),
                    allow_extra_columns=True,
                ).matched:
                    ok = True
                    break
            if not ok:
                silent += 1
                details.append((tid, t["question"]))

    total = answered + sum(not_answered.values())
    rate = (silent / answered) if answered else 0.0
    coverage = (answered / total) if total else 0.0
    print(f"评测集: {tasks_path.name}    预测: {preds_path.name}")
    print(f"可评测任务（有参考 SQL）      : {total}")
    print(f"系统给出结论（answered）      : {answered}")
    print(f"其中结果错误 → 静默错误        : {silent}")
    print(f"**静默错误率 = {silent}/{answered} = {rate:.1%}**")
    print(f"**覆盖率     = {answered}/{total} = {coverage:.1%}**"
          f"   主动未作答: {not_answered or '无'}")
    if details and show:
        print(f"\n静默错误明细（前 {min(show, len(details))} 条）:")
        for tid, q in details[:show]:
            print(f"  {tid}  {q}")
    return {
        "total": total,
        "answered": answered,
        "silent_errors": silent,
        "silent_error_rate": rate,
        "not_answered": not_answered,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="计算静默错误率")
    ap.add_argument("predictions", help="预测文件（JSONL）")
    ap.add_argument("--tasks", default=str(TASKS_PATH))
    ap.add_argument("--show", type=int, default=10, help="列几条明细，0 表示不列")
    args = ap.parse_args()
    run(Path(args.tasks), Path(args.predictions), show=args.show)


if __name__ == "__main__":
    main()
