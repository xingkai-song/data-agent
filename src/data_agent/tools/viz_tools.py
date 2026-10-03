"""展示类工具：把结果画成图。

刻意不依赖 matplotlib——手写 SVG 足够表达，
而且避免引入一堆绘图依赖和字体问题（中文字体在容器里经常缺失）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from data_agent.config import PROJECT_ROOT
from data_agent.tools.base import ErrorKind, ToolContext, ToolError
from data_agent.tools.registry import tool


class ChartInput(BaseModel):
    handle: str = Field(description="数据句柄")
    chart_type: Literal["bar", "line"] = Field(description="图表类型")
    x: str = Field(description="X 轴字段名")
    y: str = Field(description="Y 轴字段名")
    title: str | None = Field(default=None, description="图表标题")
    filename: str | None = Field(default=None, description="输出文件名（可选）")


class ChartOutput(BaseModel):
    path: str
    chart_type: str
    points: int


@tool(
    name="make_chart",
    description="把已取回的数据渲染成柱状图或折线图（SVG）。",
    category="viz",
    input_model=ChartInput,
    output_model=ChartOutput,
    side_effect=True,
    permission="write",
)
def make_chart(inp: ChartInput, ctx: ToolContext) -> ChartOutput:
    try:
        entry = ctx.results.get(inp.handle)
    except KeyError as exc:
        raise ToolError(ErrorKind.VALIDATION, str(exc)) from exc

    columns = entry["columns"]
    rows = entry["rows"]
    for col in (inp.x, inp.y):
        if col not in columns:
            raise ToolError(
                ErrorKind.VALIDATION,
                f"字段不存在：{col}",
                hint=f"可用字段：{columns}",
            )
    xi, yi = columns.index(inp.x), columns.index(inp.y)
    points = [(str(r[xi]), float(r[yi])) for r in rows if r[yi] is not None]
    if not points:
        raise ToolError(ErrorKind.BUSINESS, "没有可绘制的数据点")

    out_dir = Path(ctx.artifact_dir or (PROJECT_ROOT / "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = inp.filename or f"chart_{inp.chart_type}_{inp.x}_{inp.y}"
    path = out_dir / (name if name.endswith(".svg") else f"{name}.svg")
    path.write_text(
        _render_svg(points, inp.chart_type, inp.title or f"{inp.y} by {inp.x}", inp.x, inp.y),
        encoding="utf-8",
    )
    return ChartOutput(path=str(path), chart_type=inp.chart_type, points=len(points))


def _render_svg(
    points: list[tuple[str, float]], kind: str, title: str, xlabel: str, ylabel: str
) -> str:
    w, h = 720, 400
    pad_l, pad_r, pad_t, pad_b = 70, 20, 50, 90
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    max_v = max(v for _, v in points) or 1.0
    min_v = min(0.0, min(v for _, v in points))
    span = (max_v - min_v) or 1.0
    n = len(points)
    slot = plot_w / max(n, 1)

    def sx(i: int) -> float:
        return pad_l + slot * (i + 0.5)

    def sy(v: float) -> float:
        return pad_t + plot_h * (1 - (v - min_v) / span)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="{w / 2}" y="28" text-anchor="middle" font-size="16">{_esc(title)}</text>',
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - pad_r}" y2="{pad_t + plot_h}" stroke="#888"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#888"/>',
        f'<text x="{w / 2}" y="{h - 50}" text-anchor="middle" font-size="12">{_esc(xlabel)}</text>',
        f'<text x="18" y="{pad_t + plot_h / 2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 18 {pad_t + plot_h / 2})">{_esc(ylabel)}</text>',
        f'<text x="{pad_l - 8}" y="{pad_t + 4}" text-anchor="end" font-size="10">{max_v:,.0f}</text>',
        f'<text x="{pad_l - 8}" y="{pad_t + plot_h + 4}" text-anchor="end" font-size="10">{min_v:,.0f}</text>',
    ]

    if kind == "bar":
        bw = slot * 0.6
        for i, (_, v) in enumerate(points):
            y = sy(v)
            parts.append(
                f'<rect x="{sx(i) - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                f'height="{pad_t + plot_h - y:.1f}" fill="#4C8BF5"/>'
            )
    else:
        d = " ".join(
            f'{"M" if i == 0 else "L"}{sx(i):.1f},{sy(v):.1f}'
            for i, (_, v) in enumerate(points)
        )
        parts.append(f'<path d="{d}" fill="none" stroke="#4C8BF5" stroke-width="2"/>')
        for i, (_, v) in enumerate(points):
            parts.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="3" fill="#4C8BF5"/>')

    step = max(1, n // 12)
    for i, (label, _) in enumerate(points):
        if i % step:
            continue
        parts.append(
            f'<text x="{sx(i):.1f}" y="{pad_t + plot_h + 16}" text-anchor="end" '
            f'font-size="10" transform="rotate(-35 {sx(i):.1f} {pad_t + plot_h + 16})">'
            f"{_esc(label[:18])}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
