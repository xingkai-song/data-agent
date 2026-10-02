"""把评测结果渲染成 Markdown 报告。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tabulate import tabulate


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _slice_table(title: str, rows: list[dict[str, Any]], key: str) -> str:
    buckets: dict[str, list[bool]] = {}
    for r in rows:
        buckets.setdefault(str(r[key]), []).append(bool(r["matched"]))
    if not buckets:
        return ""
    table = [
        [k, len(v), sum(v), _pct(sum(v) / len(v))]
        for k, v in sorted(buckets.items())
    ]
    return f"**{title}**\n\n" + tabulate(
        table, headers=[key, "任务数", "通过", "通过率"], tablefmt="github"
    ) + "\n"


def render_report(run: dict[str, Any]) -> str:
    rows = run["results"]
    total = len(rows)
    passed = sum(1 for r in rows if r["matched"])
    accuracy = passed / total if total else 0.0

    out: list[str] = []
    out.append("# 评测报告\n")
    out.append(f"- 生成时间：{run['created_at']}")
    out.append(f"- 预测来源：{run['source']}")
    out.append(f"- 数据快照：{run.get('warehouse', 'unknown')}")
    out.append(f"- 任务总数：{total}")
    out.append(f"- **结果集精确比对通过率：{passed}/{total} = {_pct(accuracy)}**\n")

    out.append("## 总体结果\n")
    out.append(tabulate(
        [[r["id"], r["category"], r["difficulty"], "PASS" if r["matched"] else "FAIL",
          f"{r['latency_ms']:.0f}ms", r["reason"]]
         for r in rows],
        headers=["任务", "类别", "难度", "结果", "耗时", "说明"],
        tablefmt="github",
    ))
    out.append("")

    for title, key in [("按类别", "category"), ("按难度", "difficulty")]:
        block = _slice_table(title, rows, key)
        if block:
            out.append(block)

    failures = [r for r in rows if not r["matched"]]
    if failures:
        out.append("## 失败详情\n")
        for r in failures:
            out.append(f"### {r['id']}　{r['question']}")
            out.append(f"- 说明：{r['reason']}")
            if r.get("mismatches"):
                for m in r["mismatches"]:
                    out.append(f"  - {m}")
            out.append("")

    return "\n".join(out)


def write_report(run: dict[str, Any], path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_report(run), encoding="utf-8")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
