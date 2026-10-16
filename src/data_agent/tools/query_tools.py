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
    cross_check: list[str] = Field(
        default_factory=list,
        description=(
            "多路径对算的结果：如果这条查询算的是语义层里已定义的派生指标，"
            "会用编译器再算一遍做对照。非空表示两条路径对不上，下结论前必须核对。"
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
    return _execute_readonly(inp.sql, ctx, source="run_sql")


def _execute_readonly(sql: str, ctx: ToolContext, *, source: str) -> RunSqlOutput:
    """run_sql 与 run_metric 共用的执行路径：安全检查 → 执行 → 存句柄 → 组装输出。"""
    allowed = {t.name for t in ctx.semantic.tables}
    guard = check_readonly_sql(
        sql, allowed_tables=allowed, dialect="duckdb", max_rows=10_000
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

    handle = ctx.results.put(columns, rows, source=source)
    # 只有"行数正好顶到上限"才是真的被截断。
    # 否则聚合查询（返回 1 行）也会被误判为截断，白白触发一次修正循环。
    truncated = guard.limit_applied is not None and len(rows) >= guard.limit_applied
    # 关键：SQL 里 SUM/AVG/MAX 在没有匹配行时返回一行 NULL，不是空结果集。
    # 如果只判断 row_count == 0，就会把"什么都没查到"误认为"查到了"。
    empty = len(rows) == 0 or all(
        cell is None for row in rows[:50] for cell in row
    )
    inlined = rows[:INLINE_ROW_LIMIT]
    # 多路径对算：用编译器复核标量结果（只在能可靠判定时触发）。
    # 编译路径（run_metric）的 SQL 本来就是编译器产出的，自己对自己没有意义，
    # 只有模型手写的 run_sql 才需要这条独立路径来对照。
    checks: list[str] = []
    if source == "run_sql":
        from data_agent.semantic.multipath import cross_check

        checks = cross_check(
            ctx.semantic,
            guard.sql,
            list(columns),
            [list(r) for r in rows],
            execute=lambda s: ctx.warehouse.query(s),
        )
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
        cross_check=checks,
    )


# ---------- run_metric ----------


class RunMetricInput(BaseModel):
    metric: str = Field(
        description="指标名或业务说法，例如 aov / 客单价 / 净成交额 / 退款率"
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "分组维度，用字段名，例如 city / category / channel / device；"
            "不传就是总量。"
        ),
    )
    start: str | None = Field(
        default=None, description="起始时间（含），如 2026-09-01"
    )
    end: str | None = Field(
        default=None, description="结束时间（不含），如 2026-10-01"
    )


class RunMetricOutput(RunSqlOutput):
    metric: str = Field(description="实际使用的指标名")
    dimensions: list[str] = Field(default_factory=list, description="实际使用的维度")
    compiled_sql: str = Field(description="语义层编译出的标准 SQL，便于审计")
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "编译器给出的提醒，例如「从父表走向子表会让行数膨胀、聚合结果偏高」。"
            "有内容时不要直接下结论。"
        ),
    )


@tool(
    name="run_metric",
    description=(
        "按语义层口径直接取一个已定义的指标，SQL 由编译器生成，不需要自己写。"
        "问题就是要成交额 / 净成交额 / 客单价 / 退款率这类已定义指标时**优先用它**——"
        "归期字段、过滤条件、join 都由语义层保证，不会写错口径。"
        "只有问题超出已定义指标（临时组合、窗口函数、统计计算、探索性分析）时，"
        "才改用 run_sql 自己写。"
    ),
    category="query",
    input_model=RunMetricInput,
    output_model=RunMetricOutput,
    timeout_s=30,
)
def run_metric(inp: RunMetricInput, ctx: ToolContext) -> RunMetricOutput:
    layer = ctx.semantic
    metric = _resolve_metric(layer, inp.metric)
    if metric is None:
        raise ToolError(
            ErrorKind.VALIDATION,
            f"语义层里没有指标 `{inp.metric}`",
            hint="可用指标：" + ", ".join(sorted(m.name for m in layer.metrics)),
        )

    dims: list[str] = []
    for raw in inp.dimensions:
        resolved, options = _resolve_dimension(layer, raw, metric)
        if resolved is None:
            raise ToolError(
                ErrorKind.VALIDATION,
                f"维度 `{raw}` 无法解析",
                hint="可用维度：" + ("；".join(options) if options else "无"),
            )
        if resolved not in dims:
            dims.append(resolved)

    from data_agent.semantic.compiler import CompilerError, MetricCompiler

    try:
        compiled = MetricCompiler(layer).compile(
            metric.name,
            dimensions=dims,
            start=inp.start,
            end=inp.end,
            friendly=True,  # 维度列用字段原名，NULL 按语义层声明归组
        )
    except CompilerError as exc:
        raise ToolError(
            ErrorKind.VALIDATION,
            f"指标 `{metric.name}` 编译失败：{exc}",
            hint="换一个指标，或改用 run_sql 自己写查询。",
        ) from exc

    base = _execute_readonly(compiled.sql, ctx, source="run_metric")
    return RunMetricOutput(
        **base.model_dump(),
        metric=metric.name,
        dimensions=dims,
        compiled_sql=compiled.sql,
        notes=list(compiled.notes),
    )


def _resolve_metric(layer, text: str):
    """把用户/模型给的说法解析成语义层里的指标。"""
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    for m in layer.metrics:  # 1) 指标名精确匹配
        if m.name.lower() == low:
            return m
    for m in layer.metrics:  # 2) 展示名精确匹配
        if m.display_name == t:
            return m
    best, best_len = None, 0  # 3) 别名/名字的最长包含匹配
    for m in layer.metrics:
        for alias in (*m.aliases, m.display_name, m.name):
            a = alias.lower()
            if len(a) < 2:
                continue
            if a in low or (len(low) >= 2 and low in a):
                if len(alias) > best_len:
                    best, best_len = m, len(alias)
    return best


def _resolve_dimension(layer, text: str, metric) -> tuple[str | None, list[str]]:
    """把字段名解析成 `表.字段`。返回 (解析结果, 候选列表)。"""
    name = (text or "").strip()
    if not name:
        return None, []
    if "." in name:  # 已经是限定名
        tbl, col = (p.strip() for p in name.split(".", 1))
        t = layer.table(tbl)
        if t is not None and col in t.column_names():
            return f"{t.name}.{col}", []
        return None, [f"`{name}` 不存在"]

    cands = [(t.name, c.name) for t in layer.tables for c in t.columns if c.name == name]
    if not cands:
        cols = sorted({c.name for t in layer.tables for c in t.columns})
        return None, [", ".join(cols)]
    if len(cands) == 1:
        tbl, col = cands[0]
        return f"{tbl}.{col}", []

    # 重名列（例如 channel 在 users 和 orders 都有）：优先该指标会用到的表
    from data_agent.semantic.compiler import MetricCompiler

    preferred = set(MetricCompiler(layer).tables_for(metric))
    for tbl, col in cands:
        if tbl in preferred:
            return f"{tbl}.{col}", []
    return None, [f"{t}.{c}" for t, c in cands]


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
