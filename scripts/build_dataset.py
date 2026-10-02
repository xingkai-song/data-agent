"""生成确定性的本地数仓（DuckDB）。

设计原则：数据里必须埋入真实业务会遇到的"陷阱"，否则语义层和评测集
就失去了意义。本脚本刻意构造了以下口径歧义：

  1. 下单时间(create_time) 与 支付时间(pay_time) 会跨月
     —— "上个月销售额"按哪个时间算，结论不同
  2. 未支付订单的 pay_time 为 NULL
     —— 统计成交类指标不能只按 status 过滤而忽略时间字段
  3. 退款单独一张表，status='refunded' 的订单仍有支付时间
     —— GMV（含退款）与净成交额（扣退款）是两个口径
  4. orders.amount 已扣折扣，明细表 unit_price*quantity 是折扣前金额
     —— 从明细表算 GMV 会系统性偏高
  5. 少量用户 city 为空

固定随机种子，保证任何人 clone 后生成的数据完全一致。

用法：
    python scripts/build_dataset.py
    python scripts/build_dataset.py --force      # 覆盖已有文件
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent.config import DATA_DIR, warehouse_path  # noqa: E402

SEED = 20261003
START = datetime(2025, 1, 1)
END = datetime(2026, 10, 1)  # 不含

N_USERS = 20_000
N_PRODUCTS = 800
N_ORDERS = 60_000
N_EVENTS = 150_000

CITIES = ["上海", "北京", "杭州", "深圳", "成都", "广州", "武汉", "南京"]
CITY_WEIGHTS = [22, 20, 14, 14, 8, 8, 7, 7]
CHANNELS = ["organic", "ads", "referral", "social"]
CHANNEL_WEIGHTS = [40, 30, 15, 15]
DEVICES = ["ios", "android", "web"]

CATEGORIES = ["数码", "家电", "服饰", "美妆", "食品", "家居"]
CATEGORY_PRICE = {
    "数码": (800, 9000),
    "家电": (500, 6000),
    "服饰": (80, 1200),
    "美妆": (60, 900),
    "食品": (20, 300),
    "家居": (100, 2500),
}

REFUND_REASONS = ["七天无理由", "质量问题", "发错货", "不想要了", "价格保护"]
PAGES = ["home", "search", "detail", "cart", "pay"]


def month_starts(start: datetime, end: datetime) -> list[datetime]:
    months, cur = [], start
    while cur < end:
        months.append(cur)
        cur = datetime(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
    return months


def build() -> None:
    rng = random.Random(SEED)
    months = month_starts(START, END)

    # 按月递增的下单量，模拟业务增长
    month_weights = [1.0 + 0.055 * i for i in range(len(months))]
    total_w = sum(month_weights)
    month_probs = [w / total_w for w in month_weights]

    # ---------- users ----------
    users = []
    for i in range(1, N_USERS + 1):
        city = None if rng.random() < 0.02 else rng.choices(CITIES, CITY_WEIGHTS)[0]
        users.append(
            {
                "user_id": f"U{i:06d}",
                "register_time": START + timedelta(seconds=rng.random() * (END - START).total_seconds()),
                "city": city,
                "channel": rng.choices(CHANNELS, CHANNEL_WEIGHTS)[0],
                "is_vip": rng.random() < 0.18,
            }
        )
    users_df = pd.DataFrame(users)

    # ---------- products ----------
    products = []
    for i in range(1, N_PRODUCTS + 1):
        cat = rng.choice(CATEGORIES)
        lo, hi = CATEGORY_PRICE[cat]
        price = round(rng.uniform(lo, hi), 2)
        products.append(
            {
                "product_id": f"P{i:04d}",
                "product_name": f"{cat}-商品{i:04d}",
                "category": cat,
                "list_price": price,
            }
        )
    products_df = pd.DataFrame(products)
    product_by_id = {p["product_id"]: p for p in products}

    # ---------- orders + order_items + refunds ----------
    orders, items, refunds = [], [], []
    refund_seq = 0
    item_seq = 0

    for n in range(1, N_ORDERS + 1):
        order_id = f"O{n:08d}"
        # 先挑月份，再在该月内取一个时间点
        mi = rng.choices(range(len(months)), month_probs)[0]
        m_start = months[mi]
        m_end = months[mi + 1] if mi + 1 < len(months) else END
        span = (m_end - m_start).total_seconds()
        create_time = m_start + timedelta(seconds=rng.random() * span)

        roll = rng.random()
        if roll < 0.80:
            status = "paid"
        elif roll < 0.86:
            status = "refunded"
        elif roll < 0.95:
            status = "pending"
        else:
            status = "cancelled"

        # 未支付订单没有支付时间
        if status in ("pending", "cancelled"):
            pay_time = None
        else:
            # 3.5% 的订单在次月才支付（跨月支付陷阱）
            if rng.random() < 0.035:
                delay = timedelta(days=rng.uniform(1, 6))
            else:
                delay = timedelta(seconds=rng.uniform(60, 3 * 86400))
            pay_time = create_time + delay
            if pay_time >= END:
                pay_time = END - timedelta(hours=1)

        user = users_df.iloc[rng.randrange(N_USERS)]
        n_items = rng.choices([1, 2, 3, 4], [55, 28, 12, 5])[0]
        gross, discount_total, order_items = 0.0, 0.0, []

        for _ in range(n_items):
            item_seq += 1
            prod = product_by_id[f"P{rng.randint(1, N_PRODUCTS):04d}"]
            qty = rng.choices([1, 2, 3], [75, 20, 5])[0]
            unit_price = prod["list_price"]
            disc = round(unit_price * qty * rng.uniform(0.0, 0.18), 2)
            gross += unit_price * qty
            discount_total += disc
            order_items.append(
                {
                    "order_item_id": f"I{item_seq:09d}",
                    "order_id": order_id,
                    "product_id": prod["product_id"],
                    "quantity": qty,
                    "unit_price": unit_price,
                    "discount": disc,
                }
            )
        items.extend(order_items)

        # 关键：orders.amount 是折扣后应付金额，明细表是折扣前金额
        amount = round(gross - discount_total, 2)

        orders.append(
            {
                "order_id": order_id,
                "user_id": user["user_id"],
                "create_time": create_time,
                "pay_time": pay_time,
                "status": status,
                "amount": amount,
                "channel": user["channel"],
            }
        )

        if status == "refunded":
            refund_seq += 1
            refunds.append(
                {
                    "refund_id": f"R{refund_seq:07d}",
                    "order_id": order_id,
                    "refund_time": pay_time + timedelta(days=rng.uniform(0.5, 20)),
                    "refund_amount": round(amount * rng.uniform(0.3, 1.0), 2),
                    "reason": rng.choice(REFUND_REASONS),
                }
            )

    orders_df = pd.DataFrame(orders)
    items_df = pd.DataFrame(items)
    refunds_df = pd.DataFrame(refunds)

    # ---------- page_events ----------
    events = []
    for i in range(1, N_EVENTS + 1):
        ev_time = START + timedelta(seconds=rng.random() * (END - START).total_seconds())
        events.append(
            {
                "event_id": f"E{i:09d}",
                "user_id": f"U{rng.randint(1, N_USERS):06d}",
                "event_time": ev_time,
                "page": rng.choices(PAGES, [35, 25, 22, 10, 8])[0],
                "device": rng.choices(DEVICES, [42, 43, 15])[0],
            }
        )
    events_df = pd.DataFrame(events)

    # ---------- 写入 DuckDB ----------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(warehouse_path()))
    con.execute("BEGIN")
    for name, df in [
        ("users", users_df),
        ("products", products_df),
        ("orders", orders_df),
        ("order_items", items_df),
        ("refunds", refunds_df),
        ("page_events", events_df),
    ]:
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.register(f"_tmp_{name}", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp_{name}")
        con.unregister(f"_tmp_{name}")
    con.execute("COMMIT")

    print("已生成数仓:", warehouse_path())
    for t in ["users", "products", "orders", "order_items", "refunds", "page_events"]:
        cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:14s} {cnt:>8,} 行")

    # 自检：确认陷阱数据真的存在
    checks = {
        "跨月支付的订单数": """
            SELECT COUNT(*) FROM orders
            WHERE pay_time IS NOT NULL
              AND date_trunc('month', create_time) <> date_trunc('month', pay_time)
        """,
        "pay_time 为 NULL 的订单数": "SELECT COUNT(*) FROM orders WHERE pay_time IS NULL",
        "已退款订单数": "SELECT COUNT(*) FROM orders WHERE status = 'refunded'",
        "明细折扣前金额 > 订单金额的订单数": """
            SELECT COUNT(*) FROM (
                SELECT o.order_id
                FROM orders o JOIN order_items i USING (order_id)
                GROUP BY o.order_id, o.amount
                HAVING SUM(i.unit_price * i.quantity) > o.amount + 0.01
            )
        """,
        "city 为空的用户数": "SELECT COUNT(*) FROM users WHERE city IS NULL",
    }
    print("\n陷阱数据自检：")
    for label, sql in checks.items():
        print(f"  {label}: {con.execute(sql).fetchone()[0]:,}")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="生成本地数仓")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的数仓文件")
    args = ap.parse_args()

    path = warehouse_path()
    if path.exists() and not args.force:
        print(f"数仓已存在：{path}\n如需重建请加 --force")
        return
    build()


if __name__ == "__main__":
    main()
