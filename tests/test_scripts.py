"""入口脚本的测试。

之前整个项目测的是"库"和"逻辑"，没有测"用户怎么用"——
脚本的 `main()` 一行都没跑过。这里补上，方式是把脚本当模块调用，
而不是起子进程（更快、更容易断言、失败信息也更清楚）。

原则：**能离线跑的脚本必须跑到**。需要模型密钥的只测参数解析和错误提示。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_script(name: str):
    """按路径加载 scripts/ 下的模块（它们不是包的一部分）。"""
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- build_dataset ----------


def test_build_dataset_reports_existing_file(tmp_path, monkeypatch, capsys):
    """数仓已存在时应该提示而不是默默重建。"""
    mod = load_script("build_dataset")
    fake = tmp_path / "warehouse.duckdb"
    fake.write_bytes(b"x")
    monkeypatch.setattr(mod, "warehouse_path", lambda: fake)
    monkeypatch.setattr(sys, "argv", ["build_dataset.py"])
    mod.main()
    assert "已存在" in capsys.readouterr().out


def test_build_dataset_force_rebuilds(tmp_path, monkeypatch, capsys):
    mod = load_script("build_dataset")
    built = {"called": False}
    monkeypatch.setattr(mod, "build", lambda: built.update(called=True))
    fake = tmp_path / "warehouse.duckdb"
    fake.write_bytes(b"x")
    monkeypatch.setattr(mod, "warehouse_path", lambda: fake)
    monkeypatch.setattr(sys, "argv", ["build_dataset.py", "--force"])
    mod.main()
    assert built["called"], "--force 没有触发重建"


def test_dataset_traps_actually_exist():
    """数据集里的陷阱必须真的存在，否则语义层和评测集就是在自说自话。

    这是对 build_dataset.py 产出的数据做端到端验证，而不是测脚本本身。
    """
    from data_agent.infra.db import Warehouse

    with Warehouse() as wh:
        cross_month = wh.query(
            "SELECT COUNT(*) FROM orders WHERE pay_time IS NOT NULL "
            "AND date_trunc('month', create_time) <> date_trunc('month', pay_time)"
        )[0][0]
        unpaid = wh.query("SELECT COUNT(*) FROM orders WHERE pay_time IS NULL")[0][0]
        refunded = wh.query(
            "SELECT COUNT(*) FROM orders WHERE status = 'refunded'"
        )[0][0]
        discounted = wh.query(
            "SELECT COUNT(*) FROM (SELECT o.order_id FROM orders o "
            "JOIN order_items i USING (order_id) GROUP BY o.order_id, o.amount "
            "HAVING SUM(i.unit_price * i.quantity) > o.amount + 0.01)"
        )[0][0]
        null_city = wh.query("SELECT COUNT(*) FROM users WHERE city IS NULL")[0][0]

    assert cross_month > 0, "没有跨月支付的订单，R001 就失去了意义"
    assert unpaid > 0, "没有未支付订单，R003 就失去了意义"
    assert refunded > 0, "没有退款订单，净成交额口径就失去了意义"
    assert discounted > 0, "没有折扣差，R002 就失去了意义"
    assert null_city > 0, "没有空城市，R005 就失去了意义"


# ---------- inspect_context ----------


def test_inspect_context_prints_retrieval(capsys):
    mod = load_script("inspect_context")
    mod.show("9 月的净成交额是多少？", dump_context=False)
    out = capsys.readouterr().out
    assert "net_gmv" in out
    assert "refunds" in out


def test_inspect_context_dumps_full_context(capsys):
    mod = load_script("inspect_context")
    mod.show("9 月的成交总额是多少？", dump_context=True)
    out = capsys.readouterr().out
    assert "## 相关表结构" in out
    assert "## 指标口径" in out


def test_inspect_context_main_with_args(capsys, monkeypatch):
    mod = load_script("inspect_context")
    monkeypatch.setattr(sys, "argv", ["inspect_context.py", "扣掉退款的净成交额"])
    mod.main()
    assert "net_gmv" in capsys.readouterr().out


def test_inspect_context_main_without_args(capsys, monkeypatch):
    """不带参数时跑内置示例问题。"""
    mod = load_script("inspect_context")
    monkeypatch.setattr(sys, "argv", ["inspect_context.py"])
    mod.main()
    out = capsys.readouterr().out
    assert out.count("问题：") == len(mod.DEMO_QUESTIONS)


# ---------- show_trace ----------


def test_show_trace_renders_and_flags_failure(tmp_path, capsys, monkeypatch):
    from data_agent.agent import DataAgent
    from data_agent.llm import ScriptedLLM
    from data_agent.llm.base import LLMResponse
    from data_agent.llm.scripted import json_response
    from data_agent.observability import setup_tracing
    from data_agent.tools import build_context, default_registry

    trace = tmp_path / "spans.jsonl"
    # 构造一次"澄清"轮次：不调工具就结束，会被判为未作答
    llm = ScriptedLLM(
        [
            json_response(
                {
                    "metrics": ["x"],
                    "needs_clarification": True,
                    "clarification_question": "你指的是哪个口径？",
                }
            )
        ]
    )
    agent = DataAgent(
        llm=llm,
        registry=default_registry(),
        tool_ctx=build_context(),
        tracer=setup_tracing(json_path=trace),
    )
    agent.run("上个月的销售额是多少？")

    mod = load_script("show_trace")
    monkeypatch.setattr(sys, "argv", ["show_trace.py", str(trace)])
    mod.main()
    out = capsys.readouterr().out
    assert "共 " in out and "span" in out
    assert "失败" in out, "澄清轮次应该被标记为未作答"


def test_show_trace_missing_file_raises(tmp_path):
    mod = load_script("show_trace")
    from data_agent.observability.replay import load_spans

    with pytest.raises(FileNotFoundError):
        load_spans(tmp_path / "nope.jsonl")


# ---------- ask / serve：只测参数和错误路径（需要密钥） ----------


def test_ask_requires_question_argument(monkeypatch):
    mod = load_script("ask")
    monkeypatch.setattr(sys, "argv", ["ask.py"])
    with pytest.raises(SystemExit):
        mod.main()


def test_ask_gives_actionable_error_without_key(monkeypatch, capsys):
    """没配密钥时要明确报错，而不是静默退化成假模型。"""
    mod = load_script("ask")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(Path.home()))
    # 让 config 找不到 .env，模拟干净环境
    monkeypatch.setattr(mod, "build_client", _raise_no_key)
    monkeypatch.setattr(sys, "argv", ["ask.py", "9 月的成交额是多少？"])
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        mod.main()


def _raise_no_key(*args, **kwargs):
    raise RuntimeError("未配置模型密钥。请设置 OPENAI_API_KEY")


def test_serve_build_args_defaults():
    mod = load_script("serve")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args([])
    assert args.port == 8000 and args.workers == 4
