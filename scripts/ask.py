"""命令行单次提问，用于人工体验和调试。

用法：
    python scripts/ask.py "9 月的净成交额是多少？"
    python scripts/ask.py "9 月的净成交额是多少？" --trace

需要先配好模型密钥（见 README）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.agent import DataAgent  # noqa: E402
from data_agent.llm import build_client  # noqa: E402
from data_agent.tools import build_context, default_registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="向数据分析 Agent 提一个问题")
    ap.add_argument("question")
    ap.add_argument("--as-of", default=None, help="基准日期，默认今天")
    ap.add_argument("--trace", action="store_true", help="打印完整执行轨迹")
    args = ap.parse_args()

    agent = DataAgent(
        llm=build_client(),
        registry=default_registry(),
        tool_ctx=build_context(),
    )
    r = agent.run(args.question, as_of=args.as_of)

    print(f"\n状态：{r.status}")
    print(f"命中语义层：表={r.selection.get('tables')} 指标={r.selection.get('metrics')}")
    print(f"计划：{r.plan}")
    print(f"\n{'=' * 60}\n{r.answer}\n{'=' * 60}")

    print(
        f"\n步数 {r.steps} | 工具调用 {len(r.tool_calls)} | "
        f"模型调用 {int(r.usage.get('llm_calls', 0))} | "
        f"token {int(r.usage.get('input_tokens', 0) + r.usage.get('output_tokens', 0))} | "
        f"{r.elapsed_s:.1f}s"
    )

    if args.trace:
        print("\n--- 执行轨迹 ---")
        for c in r.tool_calls:
            flag = "OK " if c["ok"] else f"ERR({c['error_kind']})"
            print(f"  [{flag}] {c['name']}  {str(c['arguments'])[:140]}")


if __name__ == "__main__":
    main()
