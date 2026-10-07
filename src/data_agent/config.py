"""项目级配置。所有路径都以项目根目录为基准，避免相对路径踩坑。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DEFINITIONS_DIR = PROJECT_ROOT / "src" / "data_agent" / "semantic" / "definitions"
EVAL_DIR = PROJECT_ROOT / "eval"


def load_env(path: Path | None = None) -> bool:
    """加载项目根目录的 .env。

    .env 已在 .gitignore 里，密钥不会被提交。没装 python-dotenv 时静默跳过，
    不因为一个可选依赖就让整个包导入失败。
    """
    env_file = path or (PROJECT_ROOT / ".env")
    if not env_file.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(env_file, override=False))


# 导入本模块即加载，保证所有入口脚本都能拿到配置
load_env()


def warehouse_path() -> Path:
    """本地 DuckDB 数仓文件路径。"""
    raw = os.getenv("DUCKDB_PATH")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return DATA_DIR / "warehouse.duckdb"
