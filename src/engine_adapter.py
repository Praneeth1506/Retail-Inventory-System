"""Adapter between the SQLite app (src/) and Person 2's decision_engine package.

Everything the agent needs from the engine goes through this module, so the rest of the app does not
depend on the engine's interface (see decision_engine/interface.md).

Schema mapping: the app's Products table uses text ids ("P001"); the engine uses integer ids. Both
the products and the sales history are mapped with one id map (sorted text ids -> 1, 2, 3, ...).
Every other Products column already matches the engine's schema.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from decision_engine import generate_sales_history, recommend_order
from src.db import get_sales_history, insert_sales_rows

ORDER_COST = 30.0          # fixed cost per order, passed to recommend_order
HISTORY_DAYS = 180         # days of synthetic history to seed an empty SalesHistory table
HISTORY_SEED = 42
PROMO_DAYS = 7             # "Active Promotion?" = promotion on the next 7 days


def engine_id_map(product_ids) -> dict[str, int]:
    """Text product ids -> integer engine ids (stable: sorted order, starting at 1)."""
    return {pid: i + 1 for i, pid in enumerate(sorted(product_ids))}


def to_engine_products(df_products: pd.DataFrame, ids: dict[str, int]) -> pd.DataFrame:
    """The single Products mapping: same columns, integer product_id."""
    engine = df_products.copy()
    engine["product_id"] = engine["product_id"].map(ids).astype(int)
    return engine


def to_engine_history(df_sales: pd.DataFrame, ids: dict[str, int]) -> pd.DataFrame:
    """SalesHistory rows in the engine's SalesHistory schema (integer ids, date objects, bools)."""
    columns = ["date", "product_id", "units_sold", "is_weekend", "has_promo"]
    if df_sales.empty:
        return pd.DataFrame(columns=columns)
    history = df_sales[df_sales["product_id"].isin(ids)].copy()
    history["product_id"] = history["product_id"].map(ids).astype(int)
    history["date"] = pd.to_datetime(history["date"]).dt.date
    history["is_weekend"] = history["is_weekend"].astype(bool)
    history["has_promo"] = history["has_promo"].astype(bool)
    return history[columns].reset_index(drop=True)


def ensure_sales_history(df_products: pd.DataFrame, today: date | None = None) -> int:
    """Seed SalesHistory with synthetic history ending yesterday if it is empty. Returns rows added."""
    if not get_sales_history().empty:
        return 0
    today = today or date.today()
    ids = engine_id_map(df_products["product_id"])
    text_ids = {v: k for k, v in ids.items()}
    history = generate_sales_history(
        to_engine_products(df_products, ids), HISTORY_DAYS, HISTORY_SEED,
        start_date=today - timedelta(days=HISTORY_DAYS),
    )
    rows = pd.DataFrame({
        "date": [d.isoformat() for d in history["date"]],
        "product_id": history["product_id"].map(text_ids),
        "units_sold": history["units_sold"].astype(int),
        "is_weekend": history["is_weekend"].astype(int),
        "has_promo": history["has_promo"].astype(int),
    })
    insert_sales_rows(rows)
    return len(rows)


def promo_window(current_date: date, active: bool) -> list[date]:
    """The app has no per-product promotion calendar; the sidebar switch means a promotion on
    every product for the next PROMO_DAYS days."""
    return [current_date + timedelta(days=i) for i in range(PROMO_DAYS)] if active else []


def recommend(
    product: pd.Series,
    ids: dict[str, int],
    current_stock: int,
    current_date: date,
    history: pd.DataFrame,
    promo_dates: list[date],
    spoilage_cost: float = 0.0,
) -> dict:
    """One recommend_order call for one app product. current_stock = on hand + on order."""
    engine_product = product.copy()
    engine_product["product_id"] = ids[product["product_id"]]
    return recommend_order(
        engine_product, int(current_stock), current_date, history,
        promo_dates=promo_dates, order_cost=ORDER_COST, spoilage_cost=float(spoilage_cost),
    )
