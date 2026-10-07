"""启动 HTTP 服务。

用法：
    python scripts/serve.py                     # 默认 8000 端口
    python scripts/serve.py --port 9000 --workers 4

注意：这是本地开发用的单进程服务。多 worker 部署时（uvicorn --workers N）
任务存储和结果存储都走 SQLite 落盘，所以是安全的；内存版结果存储不行，
见 README 的说明。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_agent.agent import DataAgent  # noqa: E402
from data_agent.api import create_app  # noqa: E402
from data_agent.config import PROJECT_ROOT as ROOT  # noqa: E402
from data_agent.infra.result_store import build_result_store  # noqa: E402
from data_agent.llm import build_client  # noqa: E402
from data_agent.observability import setup_tracing  # noqa: E402
from data_agent.service import AnalysisService, TaskStore  # noqa: E402
from data_agent.tools import build_context, default_registry  # noqa: E402


def build_service(*, workers: int, trace_path: Path) -> AnalysisService:
    llm = build_client()
    tool_ctx = build_context()
    # 服务化必须用落盘的结果存储：任务可能被另一个 worker 查询
    tool_ctx.results = build_result_store(
        "sqlite", path=ROOT / "data" / "results.sqlite", ttl_seconds=3600
    )
    registry = default_registry()

    def factory() -> DataAgent:
        return DataAgent(
            llm=llm,
            registry=registry,
            tool_ctx=tool_ctx,
            tracer=setup_tracing(json_path=trace_path),
        )

    return AnalysisService(
        agent_factory=factory,
        store=TaskStore(ROOT / "data" / "tasks.sqlite"),
        max_workers=workers,
        trace_path=trace_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="启动数据分析 Agent 的 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=4, help="后台并发执行的分析任务数")
    args = ap.parse_args()

    import uvicorn

    trace_path = ROOT / "traces" / "api_spans.jsonl"
    service = build_service(workers=args.workers, trace_path=trace_path)
    app = create_app(service)
    print(f"启动服务 http://{args.host}:{args.port}")
    print(f"接口文档 http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
