"""口径交叉校验：编译出来的 SQL 必须和评测集里的参考 SQL 算出同一个数。

补的是一个具体缺口。语义层里的指标定义和评测集里的 reference_sql
是两份独立写下的口径，如果两边不一致，会出现最危险的情况——
Agent 一直按错的口径算，而评测显示"通过"（因为评委参考的也是错的那份）。

这个脚本把两边对跑：用编译出来的 SQL 和参考 SQL 查同一段时间，
结果不一致就报错。口径从此有了回归测试。

用法：
    python eval/semantic_check.py
    python eval/semantic_check.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import duckdb  # noqa: E402

from data_agent.config import warehouse_path  # noqa: E402
from data_agent.semantic.compiler import MetricCompiler  # noqa: E402
from data_agent.semantic.loader import load_semantic_layer  # noqa: E402

TASKS = PROJECT_ROOT / "eval" / "datasets" / "tasks.jsonl"
_TS = re.compile(r"TIMESTAMP\s+'(\d{4}-\d{2}-\d{2}[^']*)'")


def _window_from_sql(sql: str) -> tuple[str, str] | None:
    """从参考 SQL 里提取统计区间（取最早的时间下界和最晚的上界）。"""
    stamps = _TS.findall(sql)
    if not stamps:
        return None
    return min(stamps), max(stamps)


def _close(a, b, tol: float = 0.01) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def cross_check(*, verbose: bool = False, stream=None) -> tuple[int, list[str]]:
    """对跑编译 SQL 和参考 SQL，返回 (比对条数, 失败说明)。

    既是命令行工具，也被 CI 门禁调用。
    """
    out = stream or sys.stdout
    layer = load_semantic_layer()
    compiler = MetricCompiler(layer)
    con = duckdb.connect(str(warehouse_path()), read_only=True)
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]

    checked = 0
    failures: list[str] = []
    skipped: list[str] = []

    rows: list[tuple[str, str, object, object, bool]] = []
    for task in tasks:
        metrics = task.get("metrics") or []
        ref_sql = task.get("reference_sql")
        # 只在"单一指标的标量任务"上比对：分组任务的输出是多行的，
        # 没法直接和标量编译结果对比
        if not ref_sql or len(metrics) != 1 or "GROUP BY" in ref_sql.upper():
            continue
        # 显式标了 metric_check=false 的任务属于当前编译器表达不了的口径
        # （比如同时受两个时间条件约束的复合指标），不参与比对
        if task.get("metric_check") is False:
            skipped.append(f"{task['id']} 标记为不做单指标交叉校验")
            continue
        metric_name = metrics[0]
        if layer.metric(metric_name) is None:
            skipped.append(f"{task['id']} 指标 {metric_name} 不在语义层里")
            continue
        window = _window_from_sql(ref_sql)
        if window is None:
            skipped.append(f"{task['id']} 参考 SQL 里没有可提取的时间窗口")
            continue

        start, end = window
        try:
            compiled = compiler.compile(metric_name, start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{task['id']} {metric_name}：编译失败（{exc}）")
            continue

        try:
            got = con.execute(compiled.sql).fetchall()
            expected = con.execute(ref_sql).fetchall()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{task['id']} {metric_name}：执行失败（{exc}）")
            continue

        checked += 1
        ok = bool(got) and bool(expected) and _close(got[0][0], expected[0][0])
        rows.append(
            (
                task["id"],
                metric_name,
                got[0][0] if got else None,
                expected[0][0] if expected else None,
                ok,
            )
        )
        if not ok:
            failures.append(
                f"{task['id']} {metric_name}：编译 {got[0][0] if got else None} "
                f"≠ 参考 {expected[0][0] if expected else None}"
            )

    print("口径交叉校验：编译 SQL vs 评测集参考 SQL\n", file=out)
    print(f"{'任务':<6}{'指标':<20}{'编译结果':>18}{'参考结果':>18}   结果", file=out)
    for tid, name, got, expected, ok in rows:
        mark = "✓" if ok else "✗"
        print(f"{tid:<6}{name:<20}{str(got):>18}{str(expected):>18}   {mark}", file=out)

    if verbose and skipped:
        print("\n跳过：", file=out)
        for item in skipped:
            print("  -", item, file=out)

    print(f"\n比对 {checked} 条，失败 {len(failures)} 条", file=out)
    for f in failures:
        print("  ✗", f, file=out)
    return checked, failures


def main() -> int:
    ap = argparse.ArgumentParser(description="校验编译产物的口径")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    _, failures = cross_check(verbose=args.verbose)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
