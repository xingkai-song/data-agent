"""有副作用的工具。

这类工具是这个项目里最能体现工程能力的部分：**它们会改变外部状态，
所以重试必须是安全的**。没有这两个工具，"幂等键设计"就无从谈起。

两个工具都遵守同一套约定：
  * 优先使用调用方传入的 idempotency_key
  * 没传时按 (工具名, 参数) 生成稳定的键
  * 命中已有记录时不重复执行，直接返回上次的结果并标记 replayed=True
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from data_agent.config import PROJECT_ROOT
from data_agent.infra.idempotency import make_key
from data_agent.tools.base import ErrorKind, ToolContext, ToolError
from data_agent.tools.registry import tool


def _artifact_dir(ctx: ToolContext) -> Path:
    d = Path(ctx.artifact_dir or (PROJECT_ROOT / "artifacts"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- export_report ----------


class ExportReportInput(BaseModel):
    title: str = Field(description="报告标题")
    content_markdown: str = Field(description="报告正文（Markdown）")
    filename: str | None = Field(default=None, description="输出文件名，可省略")
    idempotency_key: str | None = Field(
        default=None,
        description="幂等键。重试时必须传同一个值，否则会生成两份报告。",
    )


class ExportReportOutput(BaseModel):
    path: str
    replayed: bool = Field(description="true 表示这次没有真正写文件，命中了已有结果")
    bytes_written: int


@tool(
    name="export_report",
    description=(
        "把分析结论导出成 Markdown 报告文件。这是写操作，重复调用不会产生多份文件。"
    ),
    category="side_effect",
    input_model=ExportReportInput,
    output_model=ExportReportOutput,
    side_effect=True,
    permission="write",
)
def export_report(inp: ExportReportInput, ctx: ToolContext) -> ExportReportOutput:
    key = inp.idempotency_key or make_key("export_report", inp.title, inp.content_markdown)

    def _write() -> dict:
        name = inp.filename or f"report_{_ts()}.md"
        path = _artifact_dir(ctx) / (name if name.endswith(".md") else f"{name}.md")
        body = f"# {inp.title}\n\n_生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n{inp.content_markdown}\n"
        path.write_text(body, encoding="utf-8")
        return {
            "path": str(path),
            "replayed": False,
            "bytes_written": len(body.encode("utf-8")),
        }

    if ctx.idempotency is None:
        raise ToolError(
            ErrorKind.INTERNAL, "缺少幂等存储，写操作不允许执行"
        )
    value, replayed = ctx.idempotency.run_once(key, _write)
    return ExportReportOutput(**{**value, "replayed": replayed})


# ---------- send_notification ----------


class NotifyInput(BaseModel):
    channel: Literal["log", "email", "webhook"] = Field(description="通知渠道")
    message: str = Field(description="通知内容")
    idempotency_key: str | None = Field(
        default=None,
        description="幂等键。重试时必须传同一个值，否则会重复发送。",
    )


class NotifyOutput(BaseModel):
    delivered: bool
    replayed: bool
    channel: str
    log_path: str


@tool(
    name="send_notification",
    description=(
        "发送分析结果通知。这是写操作，同样的内容重复调用只会发送一次。"
    ),
    category="side_effect",
    input_model=NotifyInput,
    output_model=NotifyOutput,
    side_effect=True,
    permission="write",
)
def send_notification(inp: NotifyInput, ctx: ToolContext) -> NotifyOutput:
    key = inp.idempotency_key or make_key("send_notification", inp.channel, inp.message)
    log_path = _artifact_dir(ctx) / "notifications.log"

    def _send() -> dict:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{inp.channel}] {inp.message}\n"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
        return {
            "delivered": True,
            "replayed": False,
            "channel": inp.channel,
            "log_path": str(log_path),
        }

    if ctx.idempotency is None:
        raise ToolError(ErrorKind.INTERNAL, "缺少幂等存储，写操作不允许执行")
    value, replayed = ctx.idempotency.run_once(key, _send)
    return NotifyOutput(**{**value, "replayed": replayed})


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
