"""计算类工具。

设计上刻意让"取数"和"算数"分开：run_sql 只负责把数据取回来，
统计和建模交给这两个工具。这样出错时能定位到底是取数错了还是算错了
——这一点直接对应评测里的分层指标。
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from data_agent.tools.base import ErrorKind, ToolContext, ToolError
from data_agent.tools.registry import tool


def _load_df(ctx: ToolContext, handle: str) -> pd.DataFrame:
    try:
        entry = ctx.results.get(handle)
    except KeyError as exc:
        raise ToolError(
            ErrorKind.VALIDATION,
            str(exc),
            hint="句柄来自 run_sql 或上一个分析工具的输出，请确认没有写错。",
        ) from exc
    return pd.DataFrame(entry["rows"], columns=entry["columns"])


def _store_df(ctx: ToolContext, df: pd.DataFrame, *, source: str) -> str:
    return ctx.results.put(list(df.columns), list(df.itertuples(index=False, name=None)), source=source)


# ---------- analyze_dataframe ----------


class AnalyzeInput(BaseModel):
    handle: str = Field(description="上一步结果的句柄")
    operation: Literal["describe", "groupby_agg", "value_counts", "corr", "top_n"] = Field(
        description="describe=描述统计；groupby_agg=分组聚合；value_counts=频次；corr=相关系数；top_n=取前 N 行"
    )
    group_by: list[str] | None = Field(default=None, description="分组字段（groupby_agg 用）")
    value_column: str | None = Field(default=None, description="被统计的数值字段")
    agg: Literal["sum", "mean", "count", "min", "max", "median"] | None = Field(
        default=None, description="聚合方式（groupby_agg 用）"
    )
    sort_desc: bool = True
    top_n: int = Field(default=10, ge=1, le=1000)


class AnalyzeOutput(BaseModel):
    handle: str
    columns: list[str]
    row_count: int
    preview: list[list]


@tool(
    name="analyze_dataframe",
    description=(
        "对已取回的结果做统计计算：描述统计、分组聚合、频次统计、相关系数、取前 N 行。"
        "输入是 run_sql 返回的句柄，输出是新结果的句柄。"
    ),
    category="analysis",
    input_model=AnalyzeInput,
    output_model=AnalyzeOutput,
)
def analyze_dataframe(inp: AnalyzeInput, ctx: ToolContext) -> AnalyzeOutput:
    df = _load_df(ctx, inp.handle)

    if inp.operation == "describe":
        out = df.describe(include="all").reset_index().rename(columns={"index": "stat"})
        out = out.astype(object).where(pd.notna(out), None)
    elif inp.operation == "groupby_agg":
        if not inp.group_by or not inp.value_column or not inp.agg:
            raise ToolError(
                ErrorKind.VALIDATION,
                "groupby_agg 需要同时提供 group_by、value_column 和 agg",
            )
        missing = [c for c in inp.group_by + [inp.value_column] if c not in df.columns]
        if missing:
            raise ToolError(
                ErrorKind.VALIDATION,
                f"字段不存在：{missing}",
                hint=f"可用字段：{list(df.columns)}",
            )
        out = (
            df.groupby(inp.group_by, dropna=False)[inp.value_column]
            .agg(inp.agg)
            .reset_index()
        )
        out.columns = [*inp.group_by, f"{inp.value_column}_{inp.agg}"]
        out = out.sort_values(out.columns[-1], ascending=not inp.sort_desc)
    elif inp.operation == "value_counts":
        if not inp.value_column or inp.value_column not in df.columns:
            raise ToolError(
                ErrorKind.VALIDATION,
                f"字段不存在：{inp.value_column}",
                hint=f"可用字段：{list(df.columns)}",
            )
        out = (
            df[inp.value_column]
            .value_counts(dropna=False)
            .rename_axis(inp.value_column)
            .reset_index(name="count")
        )
    elif inp.operation == "corr":
        out = df.select_dtypes(include="number").corr(numeric_only=True).reset_index()
        out = out.rename(columns={"index": "column"})
    else:  # top_n
        out = df.head(inp.top_n)

    out = out.astype(object).where(pd.notna(out), None)
    handle = _store_df(ctx, out, source=f"analyze:{inp.operation}")
    rows = list(out.itertuples(index=False, name=None))
    return AnalyzeOutput(
        handle=handle,
        columns=[str(c) for c in out.columns],
        row_count=len(rows),
        preview=[[_jsonable(v) for v in r] for r in rows[:5]],
    )


def _jsonable(v: object) -> object:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return str(v)


# ---------- run_python_sandbox ----------


class SandboxInput(BaseModel):
    handle: str = Field(description="输入数据句柄，代码里通过变量 df 访问")
    code: str = Field(
        description=(
            "要执行的 Python 片段。可用的变量是 df（pandas DataFrame），"
            "需要把最终结果赋值给 result（必须是 DataFrame）。"
        )
    )


class SandboxOutput(BaseModel):
    handle: str
    stdout: str
    columns: list[str]
    row_count: int


@tool(
    name="run_python_sandbox",
    description=(
        "在受限环境里对数据执行 Python 片段，适合 SQL 不好表达的统计计算"
        "（如相关性、回归、自定义指标）。结果需赋值给 result。"
    ),
    category="analysis",
    input_model=SandboxInput,
    output_model=SandboxOutput,
    permission="read",
    timeout_s=15,
)
def run_python_sandbox(inp: SandboxInput, ctx: ToolContext) -> SandboxOutput:
    """受限执行。

    ⚠ 诚实说明：这**不是**一个真正的安全沙箱。它只做了三件很浅的事：
      1. 去掉 import 语句
      2. 只注入白名单内的内置函数
      3. 不提供文件与网络相关的名字
    真正要做隔离必须用子进程 + 容器 + 资源限制（seccomp / cgroup），
    并且不共享文件系统。这里保留这个实现是为了演示接口，
    生产环境不能这么用。
    """
    df = _load_df(ctx, inp.handle)

    code = inp.code
    if "import" in code:
        raise ToolError(
            ErrorKind.DENIED,
            "沙箱内禁止 import",
            hint="可用对象：df（DataFrame）、pd（pandas）、len/round/sum 等内置函数。",
        )
    forbidden = ("__", "open(", "eval(", "exec(", "compile(", "globals(", "locals(")
    for bad in forbidden:
        if bad in code:
            raise ToolError(ErrorKind.DENIED, f"代码包含被禁止的内容：{bad}")

    safe_builtins = {
        "len": len, "round": round, "sum": sum, "min": min, "max": max,
        "abs": abs, "sorted": sorted, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "float": float, "int": int, "str": str, "bool": bool,
        "range": range, "enumerate": enumerate, "zip": zip, "print": print,
    }
    captured: list[str] = []

    def _capture(*args: object) -> None:
        captured.append(" ".join(str(a) for a in args))

    safe_builtins["print"] = _capture

    scope: dict[str, object] = {
        "__builtins__": safe_builtins,
        "df": df.copy(),
        "pd": pd,
        "result": None,
    }
    try:
        exec(compile(code, "<sandbox>", "exec"), scope)  # noqa: S102
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolError(
            ErrorKind.VALIDATION,
            f"代码执行出错：{type(exc).__name__}: {exc}",
            hint="请检查字段名是否正确，并把最终结果赋值给 result。",
        ) from exc

    result = scope.get("result")
    if not isinstance(result, pd.DataFrame):
        raise ToolError(
            ErrorKind.VALIDATION,
            "代码没有把 DataFrame 赋值给变量 result",
            hint="请在代码结尾写上 result = <你的 DataFrame>",
        )

    result = result.astype(object).where(pd.notna(result), None)
    handle = _store_df(ctx, result, source="sandbox")
    rows = list(result.itertuples(index=False, name=None))
    return SandboxOutput(
        handle=handle,
        stdout="\n".join(captured)[:2000],
        columns=[str(c) for c in result.columns],
        row_count=len(rows),
    )
