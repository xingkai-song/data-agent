"""结果集比对。

这是整个评测体系里最硬的一层：同一个问题，标准 SQL 的结果集是确定的，
所以可以直接比对，不需要依赖 LLM 打分。

比对要处理三件容易被忽略的事：
  1. 浮点误差（金额和比率都不能用 == 比较）
  2. 行的顺序（除非任务明确要求排序，否则顺序不应影响判定）
  3. 整数与 Decimal 的混用（DuckDB 会返回 Decimal）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

Number = int | float | Decimal


@dataclass
class ResultSet:
    columns: list[str]
    rows: list[tuple]

    def __post_init__(self) -> None:
        self.rows = [tuple(r) for r in self.rows]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.columns)


@dataclass
class CompareResult:
    matched: bool
    reason: str = ""
    mismatches: list[str] = field(default_factory=list)
    golden_rows: int = 0
    candidate_rows: int = 0

    def __bool__(self) -> bool:
        return self.matched


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def _to_float(v: Number) -> float:
    return float(v)


def _num_close(a: Any, b: Any, abs_tol: float, rel_tol: float) -> bool:
    fa, fb = _to_float(a), _to_float(b)
    if fa == fb:
        return True
    diff = abs(fa - fb)
    if diff <= abs_tol:
        return True
    scale = max(abs(fa), abs(fb))
    return scale > 0 and diff / scale <= rel_tol


def _cell_close(a: Any, b: Any, abs_tol: float, rel_tol: float) -> bool:
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    if _is_number(a) and _is_number(b):
        return _num_close(a, b, abs_tol, rel_tol)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    # 日期 / 布尔等：直接比较其字符串形式，避免类型差异造成误判
    return str(a) == str(b)


def _row_label(golden_cols: list[str], row: tuple) -> str:
    """用第一个文本列作为行标识，方便定位是哪一行出的问题。"""
    for j, v in enumerate(row):
        if isinstance(v, str) and j < len(golden_cols):
            return f"（{golden_cols[j]}={v}）"
    return ""


def _row_key(row: tuple, abs_tol: float, rel_tol: float) -> tuple:
    """把一行转成可排序、可比较的键。数值保留 6 位小数参与排序。"""
    key = []
    for v in row:
        if _is_number(v):
            key.append(round(_to_float(v), 6))
        elif v is None:
            key.append((0, ""))
        else:
            key.append((1, str(v)))
    return tuple(key)


def compare_result_sets(
    golden: ResultSet,
    candidate: ResultSet,
    *,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-9,
    order_matters: bool = False,
    allow_extra_columns: bool = False,
    max_reported: int = 5,
) -> CompareResult:
    """比对两个结果集。

    order_matters=False 时按行排序后比较，避免因为排序不同误判。

    容差默认收到 1e-6（近乎精确）。这里刻意不放松：
      * 相对容差一旦放松，大额金额的实质性差异就会被放过
        ——400 万的 GMV 只要设 0.5% 容差就能容忍 2 万的误差；
      * 绝对容差一旦统一放宽，又会对小量纲指标失去意义
        ——0.01 的容差放在 0.86 这样的比率上就是 1.2% 的误差。

    不同量纲的指标允许的误差本来就不同，所以容差应该是**任务级配置**：
    金额类任务在 tasks.jsonl 里声明 abs=0.01（一分钱），
    比率类任务声明 abs=1e-4。全局默认保持严格。

    allow_extra_columns=True 时，允许候选结果多带辅助列（例如回答成交额时
    顺便带上订单数）。做法是先把标准列按"取值的多重集"匹配到候选列，
    再投影出来逐行比对。
    这**不等于放宽正确性**：匹配上的列仍要逐行一致；如果主指标算错了
    （比如用 create_time 归期），第一阶段的取值匹配就会失败。
    """
    res = CompareResult(
        matched=False,
        golden_rows=golden.n_rows,
        candidate_rows=candidate.n_rows,
    )

    if golden.n_cols != candidate.n_cols and not allow_extra_columns:
        res.reason = f"列数不一致：标准 {golden.n_cols} 列，实际 {candidate.n_cols} 列"
        res.mismatches.append(f"标准列：{golden.columns}")
        res.mismatches.append(f"实际列：{candidate.columns}")
        return res

    if allow_extra_columns and candidate.n_cols != golden.n_cols:
        projected = _project_candidate(golden, candidate, abs_tol, rel_tol)
        if projected is None:
            res.reason = (
                f"候选结果的 {candidate.n_cols} 列里找不到与标准 {golden.n_cols} "
                f"列取值一致的列"
            )
            res.mismatches.append(f"标准列：{golden.columns}")
            res.mismatches.append(f"候选列：{candidate.columns}")
            return res
        candidate = projected

    if golden.n_rows != candidate.n_rows:
        res.reason = f"行数不一致：标准 {golden.n_rows} 行，实际 {candidate.n_rows} 行"
        return res

    g_rows = list(golden.rows)
    c_rows = list(candidate.rows)
    if not order_matters:
        g_rows = sorted(g_rows, key=lambda r: _row_key(r, abs_tol, rel_tol))
        c_rows = sorted(c_rows, key=lambda r: _row_key(r, abs_tol, rel_tol))

    mismatch_count = 0
    for i, (gr, cr) in enumerate(zip(g_rows, c_rows)):
        label = _row_label(golden.columns, gr)
        for j, (gv, cv) in enumerate(zip(gr, cr)):
            if not _cell_close(gv, cv, abs_tol, rel_tol):
                mismatch_count += 1
                if len(res.mismatches) < max_reported:
                    col = golden.columns[j] if j < golden.n_cols else f"col{j}"
                    res.mismatches.append(
                        f"第 {i + 1} 行{label} `{col}`：标准={gv!r}，实际={cv!r}"
                    )

    if mismatch_count:
        res.reason = f"{mismatch_count} 个单元格不一致"
        return res

    res.matched = True
    res.reason = "完全一致"
    return res


def _project_candidate(
    golden: ResultSet,
    candidate: ResultSet,
    abs_tol: float,
    rel_tol: float,
) -> ResultSet | None:
    """把候选结果投影成与标准列对应的子集。

    列名不可靠（模型起的别名五花八门），所以按**取值的多重集**匹配：
    排序后逐个比较，一致就认为这两列是同一个东西。
    """
    used: set[int] = set()
    chosen: list[int] = []

    for j in range(golden.n_cols):
        g_vals = sorted((row[j] for row in golden.rows), key=_sort_token)
        found: int | None = None
        for k in range(candidate.n_cols):
            if k in used:
                continue
            c_vals = sorted((row[k] for row in candidate.rows), key=_sort_token)
            if len(c_vals) != len(g_vals):
                continue
            if all(
                _cell_close(a, b, abs_tol, rel_tol) for a, b in zip(g_vals, c_vals)
            ):
                found = k
                break
        if found is None:
            return None
        used.add(found)
        chosen.append(found)

    return ResultSet(
        columns=[golden.columns[i] for i in range(golden.n_cols)],
        rows=[tuple(row[k] for k in chosen) for row in candidate.rows],
    )


def _sort_token(value: Any) -> tuple:
    if _is_number(value):
        return (0, _to_float(value))
    if value is None:
        return (1, 0.0)
    return (2, str(value))


# ----------------------------------------------------------------------
# 排序检查
# ----------------------------------------------------------------------


def extract_sort_key(sql: str, dialect: str = "duckdb") -> tuple[str, bool] | None:
    """从 SQL 里解析出第一个 ORDER BY 表达式和方向。

    返回 (列名或别名, 是否降序)。解析不出来就返回 None。

    为什么要解析而不是硬编码：排序要求应该由标准 SQL 自己声明，
    不然任务集和评测逻辑会各说各话。
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:  # pragma: no cover
        return None
    try:
        root = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001
        return None
    order = root.args.get("order")
    if not order or not getattr(order, "expressions", None):
        return None
    first = order.expressions[0]
    target = first.this
    if isinstance(target, exp.Column):
        name = target.name
    elif isinstance(target, exp.Literal):
        # ORDER BY 1 这种位置引用拿不到列名，用第一个投影列兜底
        name = ""
        if isinstance(root, exp.Select) and root.expressions:
            projection = root.expressions[0]
            name = getattr(projection, "alias_or_name", "") or ""
    else:
        name = getattr(target, "alias_or_name", "") or ""
    if not name:
        return None
    return name, bool(first.args.get("desc"))


def compare_ordered(
    golden: ResultSet,
    candidate: ResultSet,
    *,
    sort_column: str,
    descending: bool,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-9,
    allow_extra_columns: bool = True,
) -> CompareResult:
    """带排序要求的比对。

    关键取舍：**不按整行位置比对，而是「行集合一致 + 候选按排序键单调有序」**。

    原因是并列值。比如"各城市成交额排名"里两个城市成交额相同，
    它们在标准答案里的先后是任意的；如果按位置严格比对，
    Agent 只要把这两个城市换个顺序就会被判错——而它的排名其实是对的。
    第一版就是这么写的，自检时同一条标准 SQL 和自己比都会随机失败。
    """
    # 先做形状无关的行集合比对
    base = compare_result_sets(
        golden,
        candidate,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        order_matters=False,
        allow_extra_columns=allow_extra_columns,
    )
    if not base.matched:
        return base

    if sort_column not in golden.columns:
        base.reason = f"{base.reason}（未找到排序键列 {sort_column}，跳过排序检查）"
        return base

    projected = _project_candidate(golden, candidate, abs_tol, rel_tol)
    if projected is None:
        return base
    idx = golden.columns.index(sort_column)
    values = [row[idx] for row in projected.rows]

    for a, b in zip(values, values[1:]):
        if a is None or b is None:
            continue
        if not _is_number(a) or not _is_number(b):
            continue
        fa, fb = _to_float(a), _to_float(b)
        if descending and fb > fa + abs_tol:
            base.matched = False
            base.reason = (
                f"排序不正确：{sort_column} 应降序，但出现 {fa} 在 {fb} 之前"
            )
            return base
        if not descending and fb < fa - abs_tol:
            base.matched = False
            base.reason = (
                f"排序不正确：{sort_column} 应升序，但出现 {fa} 在 {fb} 之前"
            )
            return base

    base.reason = "行集合一致且排序正确"
    return base


def numeric_columns_covered(
    golden: ResultSet,
    candidate: ResultSet,
    *,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-9,
) -> tuple[bool, list[str]]:
    """检查标准里的每个数值列，能否在候选里找到取值一致的列。

    这是"形状无关"的补充指标，用来区分两种失败：
      * 数字算错了 —— 数值列找不到对应，这个指标也会失败
      * 只是标签写法不同（布尔值 vs "VIP/普通"）或表格形状不同
        —— 数值列仍能找到，这个指标通过

    只对数值列做检查，因为标签的表示方式本来就没有唯一正确答案，
    强行比对只会制造假失败。返回 (是否全部覆盖, 缺失的列名)。
    """
    missing: list[str] = []
    used: set[int] = set()

    for j in range(golden.n_cols):
        g_vals = [row[j] for row in golden.rows]
        if not g_vals or not all(_is_number(v) or v is None for v in g_vals):
            continue  # 非数值列不参与
        if all(v is None for v in g_vals):
            continue

        g_sorted = sorted(g_vals, key=_sort_token)
        found = False
        for k in range(candidate.n_cols):
            if k in used:
                continue
            c_sorted = sorted((row[k] for row in candidate.rows), key=_sort_token)
            if len(c_sorted) != len(g_sorted):
                continue
            if all(
                _cell_close(a, b, abs_tol, rel_tol)
                for a, b in zip(g_sorted, c_sorted)
            ):
                used.add(k)
                found = True
                break
        if not found:
            missing.append(golden.columns[j])

    return (not missing), missing
