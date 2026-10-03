"""SQL 只读安全校验。

这一层的设计前提是：**不要相信模型生成的 SQL**。

为什么不能用字符串匹配/正则：
    注释、大小写、别名、子查询都能轻易绕过 `"delete" not in sql.lower()` 这类检查。

为什么"只允许 SELECT 开头"也不够（这一点我在模拟面试里被问倒过）：
    `SELECT pg_read_file('/etc/passwd')` 是合法的 SELECT，但会读服务器文件；
    `SELECT my_udf(x) FROM t` 里的用户自定义函数可能有副作用；
    `SELECT * INTO new_table FROM t` 也是 SELECT 开头，但会建表写数据。
    所以除了语句类型，还必须有**函数白名单**。

为什么白名单要按"表达式类型"而不是"函数名字符串"：
    sqlglot 会把函数名规范化——`DATE_TRUNC` 解析成 `TimestampTrunc`、
    `STRFTIME` 解析成 `TimeToStr`，而所有它不认识的函数（包括 `pg_read_file`
    和看起来无害的 `NOW()`）都变成 `Anonymous`。
    如果按用户写的函数名字符串去比白名单，`DATE_TRUNC` 会被误杀而
    `pg_read_file` 只要改个大小写就可能绕过。按类型判断才稳。

为什么还要排除运算符节点：
    sqlglot 把 `AND` / `OR` 建模成 `Connector -> Binary -> Func`，
    所以它们也会出现在 `find_all(exp.Func)` 的结果里。如果不排除，
    任何带 `AND` 的正常查询都会被"函数不在白名单内"误拦。
    （`=` / `>` / `LIKE` 是 `Predicate`，不是 `Func`，不受影响。）
    这也是为什么函数白名单必须写测试覆盖带 `AND` 的真实查询。

已知边界（诚实说明，不夸大）：
    * 这一层只做静态检查，拦不住"合法函数 + 合法表"组合出的错误业务语义
      （比如该用净额的地方用总额）——那要靠评测体系而不是安全校验。
    * 函数白名单是"默认拒绝"，因此不认识的合法函数会被拦。这是刻意的取舍：
      误杀会报错并提示，漏放则可能造成越权。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

# ---- 允许出现的函数（按 sqlglot 表达式类型判断）----
_ALLOWED_FUNC_CLASS_NAMES = {
    # 聚合
    "Sum", "Count", "Avg", "Min", "Max", "Stddev", "StddevPop", "StddevSamp",
    "Variance", "VariancePop", "Median", "PercentileCont", "PercentileDisc",
    "ArrayAgg", "GroupConcat",
    # 数学
    "Round", "Abs", "Floor", "Ceil", "Pow", "Sqrt", "Exp", "Ln", "Log",
    "Mod", "Sign", "Trunc", "Greatest", "Least",
    # 空值与类型
    "Coalesce", "Nullif", "If", "Case", "Cast", "TryCast", "ToChar",
    # 字符串
    "Lower", "Upper", "Length", "Substring", "Trim", "Concat", "Replace",
    "Split", "RegexpLike", "RegexpReplace", "RegexpExtract",
    # 日期时间
    "TimestampTrunc", "TimeToStr", "StrToTime", "StrToDate", "Extract",
    "Date", "DateAdd", "DateSub", "DateDiff", "DateTrunc", "Year", "Month",
    "Day", "Hour", "CurrentDate", "CurrentTimestamp", "CurrentTime",
    # 窗口
    "RowNumber", "Rank", "DenseRank", "PercentRank", "Lag", "Lead",
    "FirstValue", "LastValue", "NthValue", "Ntile",
}

_ALLOWED_FUNC_CLASSES: tuple[type, ...] = tuple(
    getattr(exp, n) for n in sorted(_ALLOWED_FUNC_CLASS_NAMES) if hasattr(exp, n)
)

# sqlglot 未能识别、但业务上确实需要的函数（小写）
_ALLOWED_ANONYMOUS = {
    "now", "ifnull", "nvl", "date_trunc", "date_diff", "strftime", "strptime",
    "epoch", "epoch_ms", "regexp_matches", "regexp_replace", "list_aggregate",
    "list_value", "unnest", "generate_series", "range",
}

# 顶层语句类型白名单
_ALLOWED_STATEMENTS = (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)

# 运算符节点（AND / OR / NOT / 比较 / 算术）不算"函数调用"，
# 它们出现在 find_all(exp.Func) 里只是 sqlglot 的建模方式
_OPERATOR_BASES: tuple[type, ...] = tuple(
    c for c in (getattr(exp, n, None) for n in ("Binary", "Unary", "Predicate")) if c
)

# 明确禁止的表（系统表/元数据）——即使误加进白名单也要拦
_DENIED_TABLE_PATTERNS = ("information_schema", "pg_catalog", "pg_", "duckdb_", "sqlite_")


@dataclass
class GuardResult:
    allowed: bool
    sql: str
    reason: str = ""
    violations: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    limit_applied: int | None = None

    def __bool__(self) -> bool:
        return self.allowed


def _func_label(node: exp.Expression) -> str:
    """给函数节点取一个可读的名字，用于报错信息。"""
    if isinstance(node, exp.Anonymous):
        return str(node.this)
    try:
        return node.sql_name()
    except Exception:  # pragma: no cover - 防御性
        return type(node).__name__


def _iter_functions(root: exp.Expression) -> list[exp.Expression]:
    """真正的函数调用，不含 AND/OR 这类被 sqlglot 建模成 Func 的运算符。"""
    return [
        node
        for node in root.find_all(exp.Func)
        if not isinstance(node, _OPERATOR_BASES)
    ]


def _cte_names(root: exp.Expression) -> set[str]:
    return {c.alias_or_name.lower() for c in root.find_all(exp.CTE) if c.alias_or_name}


def _referenced_tables(root: exp.Expression) -> list[str]:
    ctes = _cte_names(root)
    found: list[str] = []
    for t in root.find_all(exp.Table):
        name = (t.name or "").lower()
        if not name or name in ctes:
            continue
        if name not in found:
            found.append(name)
    return found


def _has_limit(root: exp.Expression) -> bool:
    if isinstance(root, exp.Select):
        return root.args.get("limit") is not None
    # UNION 等复合查询：检查最外层
    return root.args.get("limit") is not None


def check_readonly_sql(
    sql: str,
    *,
    allowed_tables: set[str] | None = None,
    dialect: str = "duckdb",
    max_rows: int = 10_000,
    apply_limit: bool = True,
) -> GuardResult:
    """校验一条 SQL 是否允许执行。

    返回的 `sql` 字段是"可以真正执行"的语句（可能补了 LIMIT）。
    校验不通过时 `allowed=False`，`violations` 里是具体原因。
    """
    violations: list[str] = []

    if not sql or not sql.strip():
        return GuardResult(False, sql, "SQL 为空")

    # 1) 解析
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except (ParseError, TokenError) as exc:
        return GuardResult(False, sql, f"SQL 解析失败：{str(exc).splitlines()[0][:160]}")
    except Exception as exc:  # noqa: BLE001
        return GuardResult(False, sql, f"SQL 解析异常：{type(exc).__name__}: {exc}")

    if not statements:
        return GuardResult(False, sql, "没有解析出可执行语句")

    # 2) 只允许单条语句（防止 `SELECT 1; DROP TABLE t` 这种拼接）
    if len(statements) != 1:
        return GuardResult(
            False, sql, f"只允许单条语句，实际解析出 {len(statements)} 条"
        )

    root = statements[0]

    # 3) 语句类型
    if not isinstance(root, _ALLOWED_STATEMENTS):
        violations.append(
            f"只允许只读查询，实际是 {type(root).__name__.upper()} 语句"
        )

    # 4) SELECT ... INTO（属于 SELECT 但会写数据）
    for sel in root.find_all(exp.Select):
        if sel.args.get("into") is not None:
            violations.append("禁止 SELECT ... INTO（会产生写操作）")
            break

    # 5) 表白名单
    tables = _referenced_tables(root)
    if allowed_tables is not None:
        allowed_lower = {t.lower() for t in allowed_tables}
        for t in tables:
            if any(pat in t for pat in _DENIED_TABLE_PATTERNS):
                violations.append(f"禁止访问系统表/元数据：{t}")
            elif t not in allowed_lower:
                violations.append(f"表不在允许范围内：{t}")

    # 6) 函数白名单（默认拒绝）
    functions: list[str] = []
    for fn in _iter_functions(root):
        label = _func_label(fn)
        functions.append(label)
        if isinstance(fn, _ALLOWED_FUNC_CLASSES):
            continue
        if isinstance(fn, exp.Anonymous):
            if str(fn.this).lower() in _ALLOWED_ANONYMOUS:
                continue
            violations.append(
                f"函数 `{label}` 不在白名单内（未知函数可能有副作用）"
            )
            continue
        violations.append(f"函数 `{label}`（{type(fn).__name__}）不在白名单内")

    if violations:
        return GuardResult(
            False,
            sql,
            "；".join(dict.fromkeys(violations)),
            violations=list(dict.fromkeys(violations)),
            tables=tables,
            functions=sorted(set(functions)),
        )

    # 7) 资源限制：给没有 LIMIT 的查询补上
    final_sql = sql
    limit_applied: int | None = None
    if apply_limit and max_rows and not _has_limit(root):
        try:
            limited = root.limit(max_rows)
            final_sql = limited.sql(dialect=dialect)
            limit_applied = max_rows
        except Exception:  # pragma: no cover - 补 LIMIT 失败不该拦住合法查询
            final_sql = sql

    return GuardResult(
        True,
        final_sql,
        "校验通过",
        tables=tables,
        functions=sorted(set(functions)),
        limit_applied=limit_applied,
    )
