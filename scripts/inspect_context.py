"""看一眼语义层检索效果。

面试演示和自查都能用：给一个问题，看看检索到哪些表、指标、规则，
以及最终喂给模型的上下文长什么样。

用法：
    python scripts/inspect_context.py "9 月的净成交额是多少"
    python scripts/inspect_context.py            # 跑内置的一组示例问题
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.semantic import load_semantic_layer  # noqa: E402
from data_agent.semantic.context_builder import build_context  # noqa: E402

DEMO_QUESTIONS = [
    "9 月的成交总额是多少？",
    "扣掉退款之后的净成交额是多少？",
    "订单的支付转化率有多高？",
    "9 月卖得最好的商品是哪些？",
    "9 月退款金额最高的原因是什么？",
]


def show(question: str, *, dump_context: bool) -> None:
    layer = load_semantic_layer()
    selection, context = build_context(layer, question)
    print("=" * 70)
    print(f"问题：{question}")
    print(f"命中表　：{selection.table_names()}")
    print(f"命中指标：{selection.metric_names()}")
    print(f"命中规则：{[r.id for r in selection.rules]}")
    print(f"上下文长度：{len(context)} 字符")
    if dump_context:
        print("-" * 70)
        print(context)


def main() -> None:
    args = sys.argv[1:]
    if args:
        for q in args:
            show(q, dump_context=True)
        return
    for q in DEMO_QUESTIONS:
        show(q, dump_context=False)


if __name__ == "__main__":
    main()
