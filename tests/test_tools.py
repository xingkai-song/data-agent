"""工具层测试：注册、参数校验、错误分类、幂等、安全拦截。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.infra.idempotency import IdempotencyStore
from data_agent.infra.result_store import ResultStore
from data_agent.semantic import load_semantic_layer
from data_agent.tools import ErrorKind, ToolContext, build_context, default_registry


@pytest.fixture(scope="module")
def registry():
    return default_registry()


@pytest.fixture
def ctx(tmp_path):
    base = build_context()
    return ToolContext(
        warehouse=base.warehouse,
        semantic=load_semantic_layer(),
        idempotency=IdempotencyStore(tmp_path / "idem.sqlite"),
        results=ResultStore(),
        artifact_dir=tmp_path / "artifacts",
    )


def call(registry, ctx, name, **args):
    return registry.execute(name, args, ctx)


# ---------- 注册表 ----------


def test_all_ten_tools_registered(registry):
    names = registry.names()
    assert len(names) == 10
    assert {
        "list_tables", "describe_table", "search_metrics", "run_sql", "run_metric",
        "analyze_dataframe", "run_python_sandbox", "make_chart",
        "export_report", "send_notification",
    } == set(names)


def test_all_four_tool_categories_present(registry):
    cats = {s.category for s in registry.specs()}
    assert {"query", "analysis", "viz", "side_effect"} <= cats


def test_side_effect_tools_are_flagged(registry):
    """有副作用的工具必须被标记，否则幂等和权限控制无从谈起。"""
    flagged = {s.name for s in registry.specs() if s.side_effect}
    assert {"export_report", "send_notification", "make_chart"} <= flagged


def test_schema_is_generated_from_pydantic(registry):
    schema = next(s for s in registry.schemas() if s["name"] == "run_sql")
    assert "sql" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["sql"]


def test_unknown_tool_returns_validation_error_with_hint(registry, ctx):
    res = call(registry, ctx, "no_such_tool")
    assert not res.ok
    assert res.error_kind is ErrorKind.VALIDATION
    assert "run_sql" in (res.hint or "")


def test_invalid_arguments_return_validation_error(registry, ctx):
    """缺少必填参数时，提示里要写清楚哪个字段错了，模型才能自己改。"""
    res = call(registry, ctx, "run_sql")
    assert not res.ok
    assert res.error_kind is ErrorKind.VALIDATION
    assert "sql" in (res.hint or "")


# ---------- run_sql ----------


def test_run_sql_returns_handle_and_preview(registry, ctx):
    res = call(
        registry, ctx, "run_sql",
        sql="SELECT COUNT(*) AS n FROM orders WHERE status='paid'",
    )
    assert res.ok, res.error
    assert res.data["columns"] == ["n"]
    assert res.data["handle"].startswith("rs_")
    assert res.data["row_count"] == 1
    assert res.data["empty"] is False


def test_aggregate_over_no_rows_is_marked_empty(registry, ctx):
    """回归测试：聚合查询查不到数据时返回的是一行 NULL，不是空结果集。

    只判断 row_count == 0 会漏掉这种情况，Agent 会误以为查到了数据，
    然后在结论里写"成交额是 NULL"。
    """
    res = call(
        registry, ctx, "run_sql",
        sql=(
            "SELECT SUM(amount) AS gmv FROM orders "
            "WHERE pay_time >= TIMESTAMP '2019-01-01' "
            "AND pay_time < TIMESTAMP '2019-02-01'"
        ),
    )
    assert res.ok, res.error
    assert res.data["row_count"] == 1        # 确实是 1 行
    assert res.data["empty"] is True         # 但内容为空


def test_run_sql_applies_limit(registry, ctx):
    res = call(registry, ctx, "run_sql", sql="SELECT * FROM orders")
    assert res.ok, res.error
    assert res.data["limit_applied"] == 10_000


def test_small_result_is_returned_whole(registry, ctx):
    """小结果集必须全量回传，否则模型只能看到前几行。

    真实缺陷：以前固定只回传 5 行预览，结果
      * "销量前 10 的商品"只看到 5 个，模型把其余概括成"其余5个商品"
      * 某品类的订单数没落在前 5 行里，模型直接编了一个数
    给了句柄却没有读取完整结果的途径，是设计问题不是模型问题。
    """
    res = call(
        registry, ctx, "run_sql",
        sql="SELECT city, COUNT(*) AS n FROM users GROUP BY city ORDER BY n DESC",
    )
    assert res.ok, res.error
    assert res.data["row_count"] == res.data["rows_inlined"]
    assert res.data["more_rows_available"] is False
    assert len(res.data["rows"]) == res.data["row_count"]


def test_large_result_is_truncated_with_a_flag(registry, ctx):
    """大结果集截断时必须显式告诉模型，否则它会拿部分数据下结论。"""
    res = call(registry, ctx, "run_sql", sql="SELECT order_id FROM orders")
    assert res.ok, res.error
    assert res.data["row_count"] > res.data["rows_inlined"]
    assert res.data["more_rows_available"] is True
    assert len(res.data["rows"]) == 50


def test_result_row_limit_is_enough_for_ranking_tasks():
    """前 N 名的任务里 N 最大是 10，50 行的内联上限足够覆盖。"""
    from data_agent.tools.query_tools import INLINE_ROW_LIMIT

    assert INLINE_ROW_LIMIT >= 30


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE orders",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM secret_payroll",
        "SELECT 1; DROP TABLE orders",
    ],
)
def test_run_sql_blocks_dangerous_sql(registry, ctx, sql):
    res = call(registry, ctx, "run_sql", sql=sql)
    assert not res.ok
    assert res.error_kind is ErrorKind.DENIED


def test_typo_table_is_validation_not_denied(registry, ctx):
    """拼错表名应该让模型自己改，而不是当成安全事件直接放弃。

    这两种情况在语法上长得一样，处理方式必须不同——否则 Agent 遇到
    一个字母打错就卡死。
    """
    res = call(registry, ctx, "run_sql", sql="SELECT * FROM orderz")
    assert not res.ok
    assert res.error_kind is ErrorKind.VALIDATION
    assert "orders" in (res.hint or "")


def test_wrong_column_is_validation_with_db_error(registry, ctx):
    """字段名写错要回灌数据库原始报错，让模型据此修正。"""
    res = call(registry, ctx, "run_sql", sql="SELECT no_such_column FROM orders")
    assert not res.ok
    assert res.error_kind is ErrorKind.VALIDATION
    assert "describe_table" in (res.hint or "")


# ---------- 分析与画图 ----------


def test_groupby_agg(registry, ctx):
    r1 = call(
        registry, ctx, "run_sql",
        sql=(
            "SELECT channel, amount FROM orders "
            "WHERE status='paid' AND pay_time >= TIMESTAMP '2026-09-01' "
            "AND pay_time < TIMESTAMP '2026-10-01'"
        ),
    )
    assert r1.ok, r1.error
    r2 = call(
        registry, ctx, "analyze_dataframe",
        handle=r1.data["handle"], operation="groupby_agg",
        group_by=["channel"], value_column="amount", agg="sum",
    )
    assert r2.ok, r2.error
    assert "amount_sum" in r2.data["columns"]
    assert r2.data["row_count"] == 4


def test_missing_field_is_validation_with_available_fields(registry, ctx):
    r1 = call(registry, ctx, "run_sql", sql="SELECT channel FROM orders")
    r2 = call(
        registry, ctx, "analyze_dataframe",
        handle=r1.data["handle"], operation="value_counts", value_column="nope",
    )
    assert not r2.ok
    assert r2.error_kind is ErrorKind.VALIDATION
    assert "channel" in (r2.hint or "")


def test_bad_handle_is_validation(registry, ctx):
    res = call(registry, ctx, "analyze_dataframe", handle="rs_not_exist", operation="describe")
    assert not res.ok
    assert res.error_kind is ErrorKind.VALIDATION


def test_make_chart_writes_svg(registry, ctx):
    r1 = call(registry, ctx, "run_sql", sql="SELECT channel, SUM(amount) AS amt FROM orders GROUP BY channel")
    r2 = call(
        registry, ctx, "make_chart",
        handle=r1.data["handle"], chart_type="bar", x="channel", y="amt",
        title="各渠道成交额", filename="channels",
    )
    assert r2.ok, r2.error
    p = Path(r2.data["path"])
    assert p.exists() and p.suffix == ".svg"
    assert "<svg" in p.read_text(encoding="utf-8")


# ---------- 沙箱 ----------


def test_sandbox_happy_path(registry, ctx):
    r1 = call(registry, ctx, "run_sql", sql="SELECT amount FROM orders LIMIT 100")
    r2 = call(
        registry, ctx, "run_python_sandbox",
        handle=r1.data["handle"],
        code="result = pd.DataFrame({'mean':[df['amount'].mean()]})",
    )
    assert r2.ok, r2.error
    assert r2.data["columns"] == ["mean"]


@pytest.mark.parametrize(
    "code,keyword",
    [
        ("import os\nresult = df", "import"),
        ("result = open('/etc/passwd').read()", "open("),
        ("result = eval('1+1')", "eval("),
        ("result = df.__class__", "__"),
    ],
)
def test_sandbox_blocks_dangerous_code(registry, ctx, code, keyword):
    r1 = call(registry, ctx, "run_sql", sql="SELECT amount FROM orders LIMIT 10")
    r2 = call(
        registry, ctx, "run_python_sandbox", handle=r1.data["handle"], code=code
    )
    assert not r2.ok
    assert r2.error_kind is ErrorKind.DENIED


def test_sandbox_requires_result_dataframe(registry, ctx):
    r1 = call(registry, ctx, "run_sql", sql="SELECT amount FROM orders LIMIT 10")
    r2 = call(
        registry, ctx, "run_python_sandbox",
        handle=r1.data["handle"], code="x = 1",
    )
    assert not r2.ok
    assert r2.error_kind is ErrorKind.VALIDATION
    assert "result" in (r2.hint or "")


# ---------- 幂等：这一组是本项目最能体现工程能力的测试 ----------


def _log_lines(path: Path) -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_duplicate_notification_sends_only_once(registry, ctx):
    args = dict(channel="email", message="日报已生成")
    r1 = call(registry, ctx, "send_notification", **args)
    r2 = call(registry, ctx, "send_notification", **args)

    assert r1.ok and r2.ok
    assert r1.data["replayed"] is False
    assert r2.data["replayed"] is True
    lines = _log_lines(Path(r1.data["log_path"]))
    assert len(lines) == 1, f"重复发送了：{lines}"


def test_different_content_sends_twice(registry, ctx):
    r1 = call(registry, ctx, "send_notification", channel="email", message="A")
    r2 = call(registry, ctx, "send_notification", channel="email", message="B")
    assert r1.data["replayed"] is False
    assert r2.data["replayed"] is False
    assert len(_log_lines(Path(r1.data["log_path"]))) == 2


def test_explicit_idempotency_key_dedupes_across_different_payloads(registry, ctx):
    """带同一个幂等键时，即使内容不同也只执行一次（重试场景）。"""
    r1 = call(
        registry, ctx, "send_notification",
        channel="email", message="第一次尝试", idempotency_key="job-42",
    )
    r2 = call(
        registry, ctx, "send_notification",
        channel="email", message="重试时措辞变了", idempotency_key="job-42",
    )
    assert r1.data["replayed"] is False
    assert r2.data["replayed"] is True
    assert len(_log_lines(Path(r1.data["log_path"]))) == 1


def test_idempotency_survives_process_restart(tmp_path):
    """重试经常发生在进程重启之后。内存字典会失效，落盘才不会。"""
    db = tmp_path / "idem.sqlite"
    store1 = IdempotencyStore(db)
    store1.run_once("k1", lambda: {"v": 1})
    store1.close()                      # 模拟进程退出

    store2 = IdempotencyStore(db)       # 模拟重启后重新打开
    value, replayed = store2.run_once("k1", lambda: {"v": 2})
    assert replayed is True
    assert value == {"v": 1}
    store2.close()


def test_report_export_is_idempotent(registry, ctx):
    args = dict(title="9 月经营分析", content_markdown="结论：成交额环比上升。")
    r1 = call(registry, ctx, "export_report", **args)
    r2 = call(registry, ctx, "export_report", **args)
    assert r1.data["replayed"] is False
    assert r2.data["replayed"] is True

    files = list(Path(ctx.artifact_dir).glob("*.md"))
    assert len(files) == 1, f"生成了多份报告：{files}"


def test_side_effect_tool_refuses_without_idempotency_store(registry):
    """没有幂等存储时，写操作必须拒绝执行，而不是"赌它不会重复"。"""
    ctx_no_idem = ToolContext(
        warehouse=build_context().warehouse,
        semantic=load_semantic_layer(),
        idempotency=None,
        results=ResultStore(),
    )
    res = call(registry, ctx_no_idem, "send_notification", channel="email", message="x")
    assert not res.ok
    assert res.error_kind is ErrorKind.INTERNAL


# ---------- run_metric（编译优先路径） ----------


def test_run_metric_prefers_semantic_layer_sql(registry, ctx):
    """口径由编译器保证：模型只报指标名，不用自己写 SQL。"""
    res = call(
        registry, ctx, "run_metric",
        metric="客单价", dimensions=["city"],
        start="2026-09-01", end="2026-10-01",
    )
    assert res.ok, res.error
    sql = res.data["compiled_sql"]
    # 必须带上语义层声明的归期字段与过滤条件
    assert "pay_time" in sql
    assert "status IN ('paid', 'refunded')" in sql
    assert "GROUP BY" in sql
    assert res.data["metric"] == "aov"
    assert res.data["dimensions"] == ["users.city"]
    assert res.data["row_count"] > 0


def test_run_metric_total_has_no_group_by(registry, ctx):
    res = call(registry, ctx, "run_metric", metric="净成交额",
               start="2026-09-01", end="2026-10-01")
    assert res.ok, res.error
    assert "GROUP BY" not in res.data["compiled_sql"]
    assert res.data["row_count"] == 1


def test_run_metric_matches_handwritten_sql(registry, ctx):
    """编译路径与手写 SQL 在同一个口径上必须给出同一个数。"""
    compiled = call(registry, ctx, "run_metric", metric="gmv",
                    start="2026-09-01", end="2026-10-01")
    manual = call(registry, ctx, "run_sql", sql=(
        "SELECT SUM(o.amount) AS gmv FROM orders o "
        "WHERE o.status IN ('paid','refunded') AND o.pay_time IS NOT NULL "
        "AND o.pay_time >= '2026-09-01' AND o.pay_time < '2026-10-01'"
    ))
    assert compiled.ok and manual.ok
    a = compiled.data["rows"][0][0]
    b = manual.data["rows"][0][0]
    assert abs(a - b) < 0.01


def test_run_metric_rejects_unknown_metric(registry, ctx):
    res = call(registry, ctx, "run_metric", metric="毛利率")
    assert not res.ok
    assert res.error_kind is ErrorKind.VALIDATION
    assert "可用指标" in (res.hint or "")


def test_run_metric_rejects_unknown_dimension(registry, ctx):
    res = call(registry, ctx, "run_metric", metric="gmv", dimensions=["收货省份"])
    assert not res.ok
    assert res.error_kind is ErrorKind.VALIDATION


def test_run_metric_resolves_ambiguous_dimension_by_metric(registry, ctx):
    """channel 在 users 和 orders 都有，应优先指标会用到的表。"""
    res = call(registry, ctx, "run_metric", metric="new_users", dimensions=["channel"],
               start="2026-09-01", end="2026-10-01")
    assert res.ok, res.error
    assert res.data["dimensions"] == ["users.channel"]


def test_run_metric_output_uses_readable_columns(registry, ctx):
    """维度列要用字段名，不能把编译器内部的 d0 暴露出去。"""
    res = call(registry, ctx, "run_metric", metric="gmv", dimensions=["city"],
               start="2026-09-01", end="2026-10-01")
    assert res.ok, res.error
    assert res.data["columns"] == ["city", "gmv"]
    # NULL 城市按语义层声明的 null_label 归组，而不是留下 NULL
    assert "COALESCE(users.city, '未知')" in res.data["compiled_sql"]


def test_run_metric_surfaces_fanout_note(registry, ctx):
    """fan-out 风险必须透出给模型，不能静默。"""
    res = call(registry, ctx, "run_metric", metric="gmv", dimensions=["category"],
               start="2026-09-01", end="2026-10-01")
    assert res.ok, res.error
    assert any("膨胀" in n for n in res.data["notes"])
