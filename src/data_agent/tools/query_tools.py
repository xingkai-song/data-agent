"""查询类工具：让模型先"看懂数据"，再动手查。"""

from __future__ import annotations

import difflib
import re

from pydantic import BaseModel, Field

from data_agent.safety.sql_guard import check_readonly_sql
from data_agent.tools.base import ErrorKind, ToolContext, ToolError
from data_agent.tools.registry import tool


# ---------- list_tables ----------


class ListTablesInput(BaseModel):
    keyword: str | None = Field(
        default=None, description="可选的过滤关键词，匹配表名或表说明"
    )


class TableBrief(BaseModel):
    name: str
    description: str
    grain: str
    column_count: int


class ListTablesOutput(BaseModel):
    tables: list[TableBrief]


@tool(
    name="list_tables",
    description="列出数据仓库里有哪些表以及各自的用途，用于确定该查哪张表。",
    category="query",
    input_model=ListTablesInput,
    output_model=ListTablesOutput,
)
def list_tables(inp: ListTablesInput, ctx: ToolContext) -> ListTablesOutput:
    tables = ctx.semantic.tables
    if inp.keyword:
        kw = inp.keyword.lower()
        tables = [
            t
            for t in tables
            if kw in t.name.lower() or kw in t.description.lower()
        ]
    return ListTablesOutput(
        tables=[
            TableBrief(
                name=t.name,
                description=t.description,
                grain=t.grain,
                column_count=len(t.columns),
            )
            for t in tables
        ]
    )


# ---------- describe_table ----------


class DescribeTableInput(BaseModel):
    table: str = Field(description="表名")


class ColumnBrief(BaseModel):
    name: str
    type: str
    description: str
    notes: str | None = None
    values: dict[str, str] | None = None


class DescribeTableOutput(BaseModel):
    table: str
    description: str
    grain: str
    row_count: int
    columns: list[ColumnBrief]
    notes: list[str] = Field(default_factory=list)


@tool(
    name="describe_table",
    description=(
        "查看某张表的字段含义、取值范围和已知陷阱。"
        "写 SQL 之前应该先看这个，尤其是字段口径说明。"
    ),
    category="query",
    input_model=DescribeTableInput,
    output_model=DescribeTableOutput,
)
def describe_table(inp: DescribeTableInput, ctx: ToolContext) -> DescribeTableOutput:
    t = ctx.semantic.table(inp.table)
    if t is None:
        raise ToolError(
            ErrorKind.VALIDATION,
            f"表不存在：{inp.table}",
            hint=f"可用表：{', '.join(x.name for x in ctx.semantic.tables)}",
        )
    try:
        row_count = ctx.warehouse.query(f"SELECT COUNT(*) FROM {t.name}")[0][0]
    except Exception:  # noqa: BLE001 - 行数拿不到不该阻塞描述
        row_count = -1
    return DescribeTableOutput(
        table=t.name,
        description=t.description,
        grain=t.grain,
        row_count=row_count,
        columns=[
            ColumnBrief(
                name=c.name,
                type=c.type,
                description=c.description,
                notes=c.notes,
                values=c.values,
            )
            for c in t.columns
        ],
        notes=t.notes,
    )


# ---------- search_metrics ----------


class SearchMetricsInput(BaseModel):
    query: str = Field(description="用业务口语描述想查的指标，例如「净成交额」「转化率」")


class MetricBrief(BaseModel):
    name: str
    display_name: str
    description: str
    definition: str
    time_column: str
    filters: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class SearchMetricsOutput(BaseModel):
    metrics: list[MetricBrief]
    matched_rules: list[str] = Field(default_factory=list)


@tool(
    name="search_metrics",
    description=(
        "按业务口语查找指标的标准口径（计算公式、归期字段、必须的过滤条件）。"
        "用户说的「销售额」「流水」和指标名不一定一致，写 SQL 前应先查口径。"
    ),
    category="query",
    input_model=SearchMetricsInput,
    output_model=SearchMetricsOutput,
)
def search_metrics(inp: SearchMetricsInput, ctx: ToolContext) -> SearchMetricsOutput:
    from data_agent.semantic.compiler import CompilerError, MetricCompiler
    from data_agent.semantic.context_builder import select

    sel = select(ctx.semantic, inp.query)
    compiler = MetricCompiler(ctx.semantic)

    def compiled_sql(metric) -> str:
        try:
            return compiler.compile(metric.name).sql
        except CompilerError:
            return metric.definition or metric.expression or ""

    return SearchMetricsOutput(
        metrics=[
            MetricBrief(
                name=m.name,
                display_name=m.display_name,
                description=m.description,
                definition=compiled_sql(m),
                time_column=f"{m.time_table_name()}.{m.time_column}",
                filters=m.filters,
                caveats=m.caveats,
            )
            for m in sel.metrics
        ],
        matched_rules=[f"[{r.id}][{r.severity}] {r.description}" for r in sel.rules],
    )


# ---------- run_sql ----------


class RunSqlInput(BaseModel):
    sql: str = Field(description="只读 SQL 查询语句")
    purpose: str | None = Field(
        default=None, description="这次查询想回答什么，便于审计和调试"
    )


# 结果行数不超过这个值时全部回传给模型。
#
# 之前固定只回传 5 行预览，模型看不到剩下的行，于是出现两类失败：
#   * 漏答——"销量前 10 的商品"它只看到 5 个，就把其余概括成"其余5个商品"
#   * 编造——T37 的服饰品类订单数没落在前 5 行里，它直接编了一个约 900
# 这是设计缺陷，不是模型能力问题：给了句柄却没有读取完整结果的途径。
INLINE_ROW_LIMIT = 50


class RunSqlOutput(BaseModel):
    handle: str = Field(description="结果句柄，后续分析工具用它取数")
    columns: list[str]
    row_count: int
    rows: list[list] = Field(
        default_factory=list,
        description=f"结果行（超过 {INLINE_ROW_LIMIT} 行时只给前 {INLINE_ROW_LIMIT} 行）",
    )
    rows_inlined: int = 0
    more_rows_available: bool = Field(
        default=False,
        description="true 表示结果被截断，请把统计下推到 SQL 里，不要基于部分数据下结论",
    )
    applied_tables: list[str] = Field(default_factory=list)
    limit_applied: int | None = None
    truncated: bool = Field(
        default=False,
        description="结果是否真的被 LIMIT 截断了（行数正好等于上限时才为 true）",
    )
    empty: bool = Field(
        default=False,
        description=(
            "结果是否等于没有数据。注意：聚合查询查不到数据时返回的不是 0 行，"
            "而是一行 NULL，所以这两种情况都要算 empty。"
        ),
    )


@tool(
    name="run_sql",
    description=(
        "执行一条只读 SQL 查询。会先做安全检查（只读、表白名单、函数白名单、强制 LIMIT），"
        "通过后执行并把结果存成句柄返回，只回传前几行预览。"
    ),
    category="query",
    input_model=RunSqlInput,
    output_model=RunSqlOutput,
    timeout_s=30,
)
def run_sql(inp: RunSqlInput, ctx: ToolContext) -> RunSqlOutput:
    allowed = {t.name for t in ctx.semantic.tables}
    guard = check_readonly_sql(
        inp.sql, allowed_tables=allowed, dialect="duckdb", max_rows=10_000
    )
    if not guard.allowed:
        # 表名拼错和安全越权在语法上长得一样，但处理方式完全不同：
        # 拼错应该让模型自己改（VALIDATION），越权必须挡住（DENIED）。
        # 用相似度把这两种情况区分开。
        typo = _detect_typo_table(guard.violations, allowed)
        if typo:
            wrong, suggestion = typo
            raise ToolError(
                ErrorKind.VALIDATION,
                f"表名可能拼错了：{wrong}",
                hint=f"你是想写 `{suggestion}` 吗？可用表：{', '.join(sorted(allowed))}",
            )
        # 安全拦截不重试：需要改写 SQL 或人工确认
        raise ToolError(
            ErrorKind.DENIED,
            f"安全检查未通过：{guard.reason}",
            hint="请改写为只读查询：只允许 SELECT，禁止写操作、系统表和未知函数。",
        )

    try:
        columns = ctx.warehouse.columns_of(guard.sql)
        rows = ctx.warehouse.query(guard.sql)
    except Exception as exc:  # noqa: BLE001
        # 执行失败通常是 SQL 写错了（字段名不对等），属于可自我修正的问题，
        # 所以归类为 VALIDATION 并把数据库的原始报错回灌给模型。
        raise ToolError(
            ErrorKind.VALIDATION,
            f"SQL 执行失败：{type(exc).__name__}: {str(exc).splitlines()[0][:200]}",
            hint=(
                "请检查字段名和表名是否正确（可用 describe_table 确认），"
                "修正后重新调用。"
            ),
        ) from exc

    handle = ctx.results.put(columns, rows, source="run_sql")
    # 只有"行数正好顶到上限"才是真的被截断。
    # 否则聚合查询（返回 1 行）也会被误判为截断，白白触发一次修正循环。
    truncated = guard.limit_applied is not None and len(rows) >= guard.limit_applied
    # 关键：SQL 里 SUM/AVG/MAX 在没有匹配行时返回一行 NULL，不是空结果集。
    # 如果只判断 row_count == 0，就会把"什么都没查到"误认为"查到了"。
    empty = len(rows) == 0 or all(
        cell is None for row in rows[:50] for cell in row
    )
    inlined = rows[:INLINE_ROW_LIMIT]
    return RunSqlOutput(
        handle=handle,
        columns=list(columns),
        row_count=len(rows),
        rows=[list(r) for r in inlined],
        rows_inlined=len(inlined),
        more_rows_available=len(rows) > INLINE_ROW_LIMIT,
        applied_tables=guard.tables,
        limit_applied=guard.limit_applied,
        truncated=truncated,
        empty=empty,
    )


_UNKNOWN_TABLE = re.compile(r"表不在允许范围内：(\w+)")


def _detect_typo_table(
    violations: list[str], allowed: set[str]
) -> tuple[str, str] | None:
    """判断"表不在允许范围内"到底是拼错还是越权。

    返回 (写错的表名, 建议的表名)；如果不像拼错则返回 None。
    """
    for v in violations:
        m = _UNKNOWN_TABLE.search(v)
        if not m:
            continue
        wrong = m.group(1)
        close = difflib.get_close_matches(wrong, sorted(allowed), n=1, cutoff=0.6)
        if close:
            return wrong, close[0]
    return None
