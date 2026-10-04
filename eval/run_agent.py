"""让 Agent 跑一遍评测集，产出预测文件，交给 runner 评分。

评测对象是 **Agent 生成的 SQL 的结果集**，不是它写的结论文字。
原因：结论是自然语言，没法精确比对；而 SQL 结果集是确定的，
可以直接和标准答案比对。这也正是选数据分析场景做项目的最大好处。

用法：
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_BASE_URL=https://api.deepseek.com/v1   # 可选
    export LLM_MODEL=deepseek-chat                        # 可选

    python eval/run_agent.py                  # 跑全部任务
    python eval/run_agent.py --limit 3        # 先跑 3 条试水
    python eval/runner.py --predictions eval/predictions/agent_latest.jsonl
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

from data_agent.agent import DataAgent  # noqa: E402
from data_agent.agent.policies import Budget, LoopGuard  # noqa: E402
from data_agent.llm import build_client  # noqa: E402
from data_agent.tools import build_context, default_registry  # noqa: E402
from eval.runner import load_tasks  # noqa: E402

PRED_DIR = PROJECT_ROOT / "eval" / "predictions"


def last_successful_sql(result) -> str | None:
    """取最后一次成功的 run_sql 语句。

    这是"Agent 最终据以作答的那次查询"。多步任务里可能有多次查询，
    但结论通常来自最后一次（或最关键的那次）。
    """
    sqls = [
        c["arguments"].get("sql")
        for c in result.tool_calls
        if c["name"] == "run_sql" and c.get("ok") and c["arguments"].get("sql")
    ]
    return sqls[-1] if sqls else None


def main() -> None:
    ap = argparse.ArgumentParser(description="用 Agent 跑评测集并产出预测文件")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 表示全部）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    llm = build_client()
    print(f"模型：{getattr(llm, 'model', 'unknown')}")
    print(f"任务数：{len(tasks)}\n")

    agent = DataAgent(
        llm=llm,
        registry=default_registry(),
        tool_ctx=build_context(),
        loop_guard=LoopGuard(max_steps=args.max_steps),
        budget=Budget(),
    )

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (PRED_DIR / "agent_latest.jsonl")
    records = []
    started = time.perf_counter()

    for t in tasks:
        r = agent.run(t["question"], as_of=t.get("as_of"))
        sql = last_successful_sql(r)
        status_mark = {"answered": "OK", "clarified": "澄清", "stopped": "熔断"}.get(
            r.status, r.status
        )
        print(
            f"  [{status_mark:4s}] {t['id']}  步数={r.steps}  "
            f"工具={len(r.tool_calls)}  token={int(r.usage.get('input_tokens', 0) + r.usage.get('output_tokens', 0))}  "
            f"{r.elapsed_s:.1f}s"
        )
        if not sql:
            print(f"         └─ 未产出可评测的 SQL（status={r.status}, reason={r.stop_reason}）")
        records.append(
            {
                "id": t["id"],
                "sql": sql,
                "agent_status": r.status,
                "stop_reason": r.stop_reason,
                "steps": r.steps,
                "tool_calls": len(r.tool_calls),
                "elapsed_s": round(r.elapsed_s, 2),
                "usage": r.usage,
                "answer": (r.answer or "")[:500],
            }
        )

    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_s = time.perf_counter() - started
    answered = sum(1 for r in records if r["agent_status"] == "answered")
    with_sql = sum(1 for r in records if r["sql"])
    print(
        f"\n完成：{answered}/{len(records)} 正常作答，{with_sql}/{len(records)} 产出了可评测 SQL"
    )
    print(f"总耗时 {total_s:.1f}s，平均 {total_s / max(len(records), 1):.1f}s/条")
    print(f"预测文件：{out}")
    print(f"\n下一步：./.venv/bin/python eval/runner.py --predictions {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
