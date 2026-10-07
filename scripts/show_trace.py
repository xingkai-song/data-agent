"""查看轨迹：把 JSONL 还原成时间线，并可选地抽取失败样本。

用法：
    python scripts/show_trace.py                          # 看最新一份轨迹
    python scripts/show_trace.py traces/spans.jsonl
    python scripts/show_trace.py --failures-only          # 只看失败轮次
    python scripts/show_trace.py --export-failures        # 导出失败样本供整理成评测任务
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_agent.observability.replay import (  # noqa: E402
    append_candidates,
    build_tree,
    find_failed_turns,
    group_by_trace,
    load_spans,
    render_tree,
)

DEFAULT_TRACE = PROJECT_ROOT / "traces" / "spans.jsonl"
CANDIDATES = PROJECT_ROOT / "eval" / "datasets" / "failure_candidates.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="查看 Agent 执行轨迹")
    ap.add_argument("path", nargs="?", default=str(DEFAULT_TRACE))
    ap.add_argument("--failures-only", action="store_true", help="只显示失败的轮次")
    ap.add_argument("--export-failures", action="store_true", help="导出失败样本")
    ap.add_argument("--no-attrs", action="store_true", help="不显示属性")
    args = ap.parse_args()

    path = Path(args.path)
    spans = load_spans(path)
    groups = group_by_trace(spans)
    print(f"轨迹文件：{path}")
    print(f"共 {len(spans)} 个 span，{len(groups)} 轮\n")

    failed_trace_ids = {f["trace_id"] for f in find_failed_turns(spans)}

    for trace_id, group in groups.items():
        turn = next((s for s in group if s.get("name") == "turn"), None)
        attrs = (turn or {}).get("attributes") or {}
        is_failed = trace_id in failed_trace_ids
        if args.failures_only and not is_failed:
            continue

        flag = "失败" if is_failed else "成功"
        question = attrs.get("agent.question", "?")
        print("=" * 72)
        print(f"[{flag}] {question}")
        print(
            f"  状态={attrs.get('agent.status')}  步数={attrs.get('agent.steps')}  "
            f"工具={attrs.get('agent.tool_calls')}  "
            f"token={int(attrs.get('agent.input_tokens', 0)) + int(attrs.get('agent.output_tokens', 0))}  "
            f"耗时={attrs.get('agent.elapsed_s')}s"
        )
        print("-" * 72)
        print(render_tree(build_tree(group), show_attributes=not args.no_attrs))
        print()

    failures = find_failed_turns(spans)
    if failures:
        print(f"共发现 {len(failures)} 个失败轮次")
        if args.export_failures:
            n = append_candidates(CANDIDATES, failures)
            print(f"已导出 {n} 条候选样本到 {CANDIDATES}")
            print("注意：这些只是候选，需要人工补上标准 SQL 和口径依据才能进评测集")


if __name__ == "__main__":
    main()
