"""SQL 安全校验的对抗测试。

这组用例的思路是"站在攻击者角度"：不是验证正常查询能过，而是验证
各种绕过手法能不能被拦住。因为一个"什么都放行"的校验器同样能让
正常查询全部通过。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.safety.sql_guard import check_readonly_sql  # noqa: E402

ALLOWED_TABLES = {"orders", "order_items", "users", "products", "refunds", "page_events"}


def guard(sql: str, **kwargs):
    return check_readonly_sql(sql, allowed_tables=ALLOWED_TABLES, **kwargs)


# ---------- 应该放行的正常查询 ----------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM orders",
        "SELECT city, SUM(amount) FROM orders GROUP BY city",
        "SELECT ROUND(AVG(amount), 2) FROM orders WHERE status = 'paid'",
        "WITH paid AS (SELECT * FROM orders WHERE status='paid') SELECT COUNT(*) FROM paid",
        "SELECT * FROM orders WHERE user_id IN (SELECT user_id FROM users)",
        "SELECT 1 UNION SELECT 2",
        "SELECT ROW_NUMBER() OVER (PARTITION BY city ORDER BY amount) FROM orders",
        "SELECT DATE_TRUNC('month', pay_time) FROM orders",
        "SELECT STRFTIME(pay_time, '%Y-%m') FROM orders",
        "SELECT CAST(amount AS DOUBLE) FROM orders",
        "SeLeCt /* 注释 */ * FROM orders",
        "SELECT COALESCE(city, '未知') FROM users",
        "SELECT COUNT(DISTINCT order_id) FROM orders",
        "SELECT amount FROM orders LIMIT 5",
        # 逻辑连接词：sqlglot 把 AND/OR 建模成 Func，容易被函数白名单误杀
        "SELECT * FROM orders WHERE status = 'paid' AND amount > 100",
        "SELECT * FROM orders WHERE status = 'paid' OR status = 'refunded'",
        "SELECT * FROM orders WHERE NOT (amount > 100)",
        "SELECT * FROM orders WHERE pay_time >= DATE '2026-09-01' AND pay_time < DATE '2026-10-01' AND status IN ('paid','refunded')",
        "SELECT city, SUM(amount) FROM orders WHERE amount BETWEEN 100 AND 200 GROUP BY city HAVING SUM(amount) > 1000",
    ],
)
def test_normal_queries_allowed(sql):
    res = guard(sql)
    assert res.allowed, f"正常查询被误拦：{sql}\n原因：{res.reason}"


def test_date_functions_are_not_false_negatives():
    """回归测试：sqlglot 把 DATE_TRUNC 规范化成 TimestampTrunc。

    如果白名单按用户写的函数名（DATE_TRUNC）来匹配，这里会被误杀。
    """
    res = guard("SELECT DATE_TRUNC('month', pay_time), STRFTIME(pay_time, '%Y') FROM orders")
    assert res.allowed, res.reason
    assert any("TRUNC" in f.upper() for f in res.functions)


def test_and_or_are_not_treated_as_functions():
    """回归测试：AND/OR 是 Connector→Binary→Func，不是真正的函数调用。

    不排除它们的话，任何带 AND 的正常查询都会被误拦。
    """
    res = guard("SELECT COUNT(*) FROM orders WHERE status = 'paid' AND amount > 100")
    assert res.allowed, res.reason
    assert "AND" not in res.functions
    assert "COUNT" in res.functions


# ---------- 应该拦住的写操作 ----------


@pytest.mark.parametrize(
    "sql,keyword",
    [
        ("DELETE FROM orders WHERE 1=1", "DELETE"),
        ("UPDATE orders SET amount = 0", "UPDATE"),
        ("INSERT INTO orders VALUES (1)", "INSERT"),
        ("DROP TABLE orders", "DROP"),
        ("CREATE TABLE t (a INT)", "CREATE"),
        ("ALTER TABLE orders ADD COLUMN x INT", "ALTER"),
        ("TRUNCATE TABLE orders", "TRUNCATE"),
    ],
)
def test_write_statements_denied(sql, keyword):
    res = guard(sql)
    assert not res.allowed, f"写操作未被拦住：{sql}"
    assert keyword in res.reason.upper()


def test_multi_statement_denied():
    """经典绕过手法：在前面放一条无害查询，后面拼接破坏性语句。"""
    res = guard("SELECT 1; DROP TABLE orders")
    assert not res.allowed
    assert "单条语句" in res.reason


def test_select_into_denied():
    """SELECT 开头但会建表写数据。"""
    res = guard("SELECT * INTO new_tbl FROM orders")
    assert not res.allowed
    assert "INTO" in res.reason.upper()


# ---------- 函数白名单：模拟面试里被问倒的那个点 ----------


def test_pg_read_file_denied():
    """PostgreSQL 的读文件函数，语法上是合法的 SELECT。"""
    res = guard("SELECT pg_read_file('/etc/passwd')")
    assert not res.allowed
    assert "pg_read_file" in res.reason


def test_unknown_udf_denied():
    """未注册的用户自定义函数可能有副作用，默认拒绝。"""
    res = guard("SELECT my_dangerous_func(amount) FROM orders")
    assert not res.allowed
    assert "my_dangerous_func" in res.reason


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT read_csv('/etc/passwd')",
        "SELECT read_parquet('s3://bucket/x')",
        "SELECT duckdb_settings()",
        "SELECT * FROM orders WHERE 1=1 OR benchmark(100000)",
    ],
)
def test_exfiltration_functions_denied(sql):
    res = guard(sql)
    assert not res.allowed, f"危险函数未被拦住：{sql}"


def test_low_risk_unknown_function_still_allowed_by_allowlist():
    """NOW() 会被 sqlglot 解析成 Anonymous，必须显式放行，否则误杀。"""
    res = guard("SELECT * FROM orders WHERE pay_time < NOW()")
    assert res.allowed, res.reason


# ---------- 表白名单 ----------


def test_table_outside_allowlist_denied():
    res = guard("SELECT * FROM secret_salaries")
    assert not res.allowed
    assert "不在允许范围" in res.reason


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM duckdb_settings()",
    ],
)
def test_system_tables_denied(sql):
    res = guard(sql)
    assert not res.allowed


def test_cte_name_is_not_treated_as_table():
    """CTE 的别名不是真实表，不能因为"不在白名单"被误杀。"""
    res = guard("WITH t AS (SELECT 1 AS x) SELECT * FROM t")
    assert res.allowed, res.reason
    assert "t" not in res.tables


# ---------- 资源限制 ----------


def test_limit_is_injected_when_missing():
    res = guard("SELECT * FROM orders", max_rows=100)
    assert res.allowed
    assert res.limit_applied == 100
    assert "LIMIT 100" in res.sql.upper()


def test_existing_limit_is_preserved():
    res = guard("SELECT * FROM orders LIMIT 5", max_rows=100)
    assert res.allowed
    assert res.limit_applied is None
    assert "LIMIT 5" in res.sql.upper()


def test_limit_injection_can_be_disabled():
    res = guard("SELECT * FROM orders", apply_limit=False)
    assert res.allowed
    assert res.limit_applied is None


# ---------- 边界情况 ----------


def test_empty_sql_denied():
    assert not guard("").allowed
    assert not guard("   ").allowed


def test_invalid_syntax_denied():
    res = guard("SELECT FROM WHERE")
    assert not res.allowed


def test_guard_reports_tables_and_functions():
    res = guard("SELECT ROUND(SUM(amount), 2) FROM orders")
    assert res.allowed
    assert res.tables == ["orders"]
    assert "ROUND" in res.functions and "SUM" in res.functions
