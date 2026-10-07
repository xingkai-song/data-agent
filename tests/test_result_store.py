"""结果存储的两个实现必须行为一致。

内存版用于本地开发和测试，落盘版用于服务化。如果两者的行为不一致，
就会出现"本地跑得好好的、上了多 worker 就出问题"这类最难查的 bug。
所以同一组行为用同一批测试覆盖两个实现。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.infra.result_store import (  # noqa: E402
    ResultStore,
    SqliteResultStore,
    build_result_store,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return ResultStore()
    return SqliteResultStore(tmp_path / "results.sqlite", ttl_seconds=3600)


# ---------- 两个实现共同的行为 ----------


def test_put_and_get_roundtrip(store):
    handle = store.put(["city", "gmv"], [("上海", 100.0), ("北京", 200.0)])
    entry = store.get(handle)
    assert entry["columns"] == ["city", "gmv"]
    assert entry["rows"] == [("上海", 100.0), ("北京", 200.0)]


def test_handle_is_prefixed_and_unique(store):
    a = store.put(["x"], [(1,)])
    b = store.put(["x"], [(1,)])
    assert a.startswith("rs_") and b.startswith("rs_") and a != b


def test_preview_truncates_rows(store):
    handle = store.put(["n"], [(i,) for i in range(10)])
    prev = store.preview(handle, n=3)
    assert prev["row_count"] == 10
    assert len(prev["preview"]) == 3


def test_missing_handle_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.get("rs_not_exist")


def test_clear_empties_store(store):
    store.put(["x"], [(1,)])
    store.clear()
    assert store.size() == 0


def test_non_json_types_are_coerced(store):
    """日期、Decimal 这类值要能存进去，读出来是可用的字符串。"""
    from datetime import date
    from decimal import Decimal

    handle = store.put(["d", "amount"], [(date(2026, 9, 1), Decimal("12.34"))])
    row = store.get(handle)["rows"][0]
    assert "2026-09-01" in str(row[0])
    assert "12.34" in str(row[1])


# ---------- 落盘版特有的行为 ----------


def test_sqlite_store_is_shared_across_instances(tmp_path):
    """这是做落盘版本的全部理由：另一个进程/线程创建的句柄要能读到。

    内存版做不到这一点，多 worker 部署时就会"查不到结果"。
    """
    path = tmp_path / "results.sqlite"
    writer = SqliteResultStore(path, ttl_seconds=3600)
    handle = writer.put(["x"], [(42,)])

    reader = SqliteResultStore(path, ttl_seconds=3600)
    assert reader.get(handle)["rows"] == [(42,)]


def test_sqlite_store_survives_reopen(tmp_path):
    """进程重启后句柄仍然有效（在 TTL 内）。"""
    path = tmp_path / "results.sqlite"
    handle = SqliteResultStore(path, ttl_seconds=3600).put(["x"], [(7,)])
    # 模拟进程重启：重新构造一个实例
    assert SqliteResultStore(path, ttl_seconds=3600).get(handle)["rows"] == [(7,)]


def test_sqlite_store_evicts_expired_entries(tmp_path):
    """TTL 到期后句柄应该失效，而不是无限堆积。"""
    path = tmp_path / "results.sqlite"
    store = SqliteResultStore(path, ttl_seconds=0.05)
    handle = store.put(["x"], [(1,)])
    assert store.size() == 1
    time.sleep(0.1)
    with pytest.raises(KeyError):
        store.get(handle)
    assert store.size() == 0, "过期记录没有被清理"


def test_expired_handle_error_mentions_ttl(tmp_path):
    """报错要能让人看懂是过期了，而不是"不存在"。"""
    store = SqliteResultStore(tmp_path / "r.sqlite", ttl_seconds=0.05)
    handle = store.put(["x"], [(1,)])
    time.sleep(0.1)
    with pytest.raises(KeyError, match="过期"):
        store.get(handle)


# ---------- 工厂 ----------


def test_build_result_store_by_name(tmp_path):
    assert isinstance(build_result_store("memory"), ResultStore)
    assert isinstance(
        build_result_store("sqlite", path=tmp_path / "x.sqlite"), SqliteResultStore
    )


def test_build_result_store_rejects_unknown_backend():
    with pytest.raises(ValueError, match="未知"):
        build_result_store("redis")
