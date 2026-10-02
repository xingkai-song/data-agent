"""项目级配置。所有路径都以项目根目录为基准，避免相对路径踩坑。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DEFINITIONS_DIR = PROJECT_ROOT / "src" / "data_agent" / "semantic" / "definitions"
EVAL_DIR = PROJECT_ROOT / "eval"


def warehouse_path() -> Path:
    """本地 DuckDB 数仓文件路径。"""
    raw = os.getenv("DUCKDB_PATH")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return DATA_DIR / "warehouse.duckdb"
