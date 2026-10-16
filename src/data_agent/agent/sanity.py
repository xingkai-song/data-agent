"""结果体检：对工具返回的结果集做一组确定性不变量检查。

`observe` 原本只检查"空结果 / 被截断"，抓不出"跑通了但数明显不对"。
这里补的就是第二类——**这是项目主张"算错比不回答更危险"的具体落地**。

代价为零：只看已经拿到的结果集，不额外查库、不调模型。检查项都是业务
恒等式，破了说明口径写错了：

  1. 金额 / 数量不应该为负
  2. 率、占比类应该落在 [0, 1]（规则 R008 规定返回小数而不是百分数）
  3. 净额不应该超过总额（net_gmv ≤ gmv）
  4. 退款额不应该超过成交额（refund_amount ≤ gmv）

**已知边界（不夸大）**：这些检查只能抓"数值自相矛盾"，抓不住"数值自洽
但口径被替换"——例如把客单价的分子从 GMV 悄悄换成净成交额，四个等式
全都不破。那类需要多路径对算，不在这里解决。
"""

from __future__ import annotations

from typing import Any

_TOL = 1e-6
_MAX_WARNINGS = 3

_RATIO_EN = ("rate", "ratio", "share", "percent", "pct", "conversion")
_RATIO_ZH = ("率", "占比")
_GROSS_EXACT = ("gmv", "gross", "amount", "total_amount")
_GROSS_ZH = ("成交额", "销售额", "总额")
_REFUND = ("refund", "退款", "退货")
# 这些列天然可能为负（增量、差额、变化率），不做"负值"检查，避免误报
_CAN_BE_NEGATIVE = ("增长", "变化", "差", "同比", "环比", "delta", "diff", "growth", "change")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _extract(data: Any) -> tuple[list[str], list[list[Any]]] | None:
    """只认"列名是字符串 + 行是列表"的结果集。

    `describe_table` 的 columns 是字典列表（字段元数据），不是结果集，
    这里必须排除掉，否则会对元数据做数值检查、产生噪音。
    """
    if not isinstance(data, dict):
        return None
    cols = data.get("columns")
    rows = data.get("rows")
    if rows is None:
        rows = data.get("preview")
    if not isinstance(cols, list) or not cols:
        return None
    if not all(isinstance(c, str) for c in cols):
        return None
    if not isinstance(rows, list) or not rows:
        return None
    if not all(isinstance(r, (list, tuple)) for r in rows):
        return None
    return list(cols), [list(r) for r in rows]


def _flags(name: str) -> dict[str, bool]:
    low = name.lower()
    ratio = any(h in low for h in _RATIO_EN) or any(h in name for h in _RATIO_ZH)
    net = "net" in low or "净" in name
    gross = (
        low in _GROSS_EXACT or any(h in name for h in _GROSS_ZH)
    ) and not net
    money = (
        "amount" in low or "amt" in low or "金额" in name or "额" in name
    ) and not ratio
    count = any(h in low for h in ("count", "cnt", "qty", "quantity")) or any(
        h in name for h in ("数", "量", "笔")
    )
    refund = any(h in low for h in _REFUND)
    return {
        "ratio": ratio,
        "net": net,
        "gross": gross,
        "money": money,
        "count": count,
        "refund": refund,
    }


def _column_values(rows: list[list[Any]], i: int) -> list[float]:
    out = []
    for r in rows:
        if i < len(r) and _is_num(r[i]):
            out.append(float(r[i]))
    return out


def check_result(data: Any) -> list[str]:
    """对一次工具结果做体检，返回可疑点（最多 3 条）。"""
    parsed = _extract(data)
    if parsed is None:
        return []
    cols, rows = parsed
    flags = {c: _flags(c) for c in cols}
    warnings: list[str] = []

    # 1) 金额 / 数量不该为负
    for i, c in enumerate(cols):
        f = flags[c]
        if not (f["money"] or f["count"] or f["gross"]):
            continue
        if any(h in c.lower() or h in c for h in _CAN_BE_NEGATIVE):
            continue
        neg = [v for v in _column_values(rows, i) if v < -_TOL]
        if neg:
            warnings.append(
                f"列 `{c}` 出现了负值（最小 {min(neg):.4g}），金额/数量不应为负"
            )

    # 2) 率 / 占比应在 [0, 1]
    for i, c in enumerate(cols):
        if not flags[c]["ratio"]:
            continue
        vals = _column_values(rows, i)
        bad = [v for v in vals if v < -_TOL or v > 1 + _TOL]
        if bad:
            warnings.append(
                f"列 `{c}` 是比率类，但取值超出 [0,1]（最大 {max(bad):.4g}）。"
                "按规则 R008 应返回 0-1 的小数，除非确实要按百分数展示"
            )

    # 3) 净额 ≤ 总额
    gross_i = next((i for i, c in enumerate(cols) if flags[c]["gross"]), None)
    net_i = next((i for i, c in enumerate(cols) if flags[c]["net"]), None)
    if gross_i is not None and net_i is not None and gross_i != net_i:
        for r in rows:
            if gross_i < len(r) and net_i < len(r) and _is_num(r[gross_i]) and _is_num(r[net_i]):
                if float(r[net_i]) > float(r[gross_i]) + _TOL:
                    warnings.append(
                        f"列 `{cols[net_i]}`（{r[net_i]:.4g}）大于 `{cols[gross_i]}`"
                        f"（{r[gross_i]:.4g}）：净额不应超过总额，退款可能被加而不是被减"
                    )
                    break

    # 4) 退款额 ≤ 成交额
    refund_i = next(
        (i for i, c in enumerate(cols) if flags[c]["refund"] and (flags[c]["money"] or flags[c]["count"])),
        None,
    )
    if refund_i is not None and gross_i is not None and refund_i != gross_i:
        for r in rows:
            if (
                gross_i < len(r)
                and refund_i < len(r)
                and _is_num(r[gross_i])
                and _is_num(r[refund_i])
            ):
                if float(r[refund_i]) > float(r[gross_i]) + _TOL:
                    warnings.append(
                        f"列 `{cols[refund_i]}`（{r[refund_i]:.4g}）大于 `{cols[gross_i]}`"
                        f"（{r[gross_i]:.4g}）：退款不应超过成交额，多表 join 可能造成膨胀"
                    )
                    break

    # 去重（同一条消息可能被多个规则命中）
    seen: list[str] = []
    for w in warnings:
        if w not in seen:
            seen.append(w)
    return seen[:_MAX_WARNINGS]
