"""One-call daily order recommendation: how much to order (utility-optimal Q*) and whether to order today.

Two Utility Theory decisions
----------------------------
1. How much. Q* = calculate_optimal_order_quantity over the order horizon lead_time_days +
   review_period_days, where review_period_days is the economic (EOQ) cycle by default.

2. Whether to order today. Re-optimizing every day with a rolling horizon would top the order up
   almost daily (each day reveals one more day of demand worth covering), paying order_cost each
   time. Instead, compare the two options for today:
     - order now: the Q* units arrive one day earlier than if ordered tomorrow, so they are held one
       extra day: early_holding_cost = holding_cost * Q*.
     - wait one day: an order placed tomorrow arrives on day t + L + 1 instead of t + L, so demand on
       day t + L must be met from the current position alone. The extra expected shortage is
           extra_shortage = E[max(0, D_(L+1) - pos)] - E[max(0, D_L - pos)]
       where D_L is demand over days t .. t+L-1 and D_(L+1) over days t .. t+L (both from
       calculate_stockout_risk with review_period_days = 1), and pos = current_stock. Each unit
       short loses its margin and incurs the stockout penalty:
           wait_cost = (price - unit_cost + stockout_penalty) * extra_shortage
   Order Q* today only if Q* > 0 and wait_cost > early_holding_cost; otherwise order 0.
   With lead_time_days = 0, D_L is zero demand.
   This is a one-day look-ahead: it compares ordering today with ordering tomorrow, not with every
   later day, which keeps it cheap and transparent but not provably optimal over many periods.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import date

import pandas as pd

from decision_engine.bayes_risk import (
    _REQUIRED_HISTORY_COLUMNS,
    DEFAULT_DAILY_DEMAND,
    _fit_product,
    _ProductFit,
    _stockout_risk,
)
from decision_engine.optimizer import calculate_optimal_order_quantity, economic_review_period

REQUIRED_PRODUCT_FIELDS = (
    "product_id",
    "unit_cost",
    "selling_price",
    "holding_cost_per_day",
    "stockout_penalty",
    "shelf_life_days",
    "lead_time_days",
)


def _shelf_life(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _mean_daily_demand(history_df: pd.DataFrame, product_id: int) -> float:
    sales = history_df.loc[history_df["product_id"] == product_id, "units_sold"]
    return float(sales.mean()) if len(sales) > 0 else DEFAULT_DAILY_DEMAND


def recommend_order(
    product: Mapping,
    current_stock: int,
    current_date: date,
    history_df: pd.DataFrame,
    *,
    promo_dates: Collection[date] = (),
    order_cost: float = 30.0,
    spoilage_cost: float = 0.0,
    risk_tolerance: float | None = None,
    review_period_days: int | None = None,
) -> dict:
    """Recommend today's order for one product: how much (Q*) and whether to order it today.

    Steps (see the module docstring for the full rule):
        1. If review_period_days is None, choose it with economic_review_period from the product's
           mean daily units_sold in history_df, its holding cost, order_cost and shelf life.
        2. calculate_stockout_risk over the lead time and the order horizon
           (lead_time_days + review_period_days), starting on current_date.
        3. calculate_optimal_order_quantity on the order-horizon distribution, with the same
           review_period_days: this gives Q*.
        4. calculate_stockout_risk with review_period_days = 1 (wait_risk), and order Q* today only
           if Q* > 0 and the expected cost of waiting one day exceeds the cost of holding Q* one
           extra day. recommended_qty is Q* or 0.

    Assumptions:
        - Person 1's agent calls this once per product per day. The two underlying functions stay
          public for charts and testing.
        - current_stock is the inventory position (on hand + on order).
        - promo_dates are the dates on which this product has a promotion.
        - With no history for the product, the mean daily demand used for the review period is
          DEFAULT_DAILY_DEMAND (8 units per day), matching the risk model's prior.
        - The agent still runs daily. On most days the when-to-order rule says wait, so orders
          happen roughly once per review period.
        - The agent should act on recommended_qty, not order["optimal_qty"].

    Args:
        product: One Products row (dict or pandas Series) with product_id, unit_cost,
            selling_price, holding_cost_per_day, stockout_penalty, shelf_life_days (None/NaN if
            not perishable) and lead_time_days.
        current_stock: Inventory position. Must be >= 0.
        current_date: Decision day (start of day, before that day's sales).
        history_df: Sales history in the SalesHistory schema.
        promo_dates: Dates on which this product has a promotion.
        order_cost: Fixed cost per order. Must be >= 0.
        spoilage_cost: Extra disposal cost per spoiled unit. Must be >= 0.
        risk_tolerance: None for linear utility, otherwise R > 0 for exponential utility.
        review_period_days: Order cycle in days, or None for the economic review period.

    Returns:
        {"product_id": int, "review_period_days": int,
         "recommended_qty": int (Q* or 0), "should_order": bool,
         "wait_cost": float, "early_holding_cost": float,
         "risk": <calculate_stockout_risk result over lead time + review period>,
         "order": <calculate_optimal_order_quantity result>,
         "wait_risk": <calculate_stockout_risk result with review_period_days = 1>}

    Raises:
        ValueError: If product is missing a field, or any underlying function rejects its inputs.
    """
    return _recommend(
        product, current_stock, current_date, history_df, promo_dates=promo_dates,
        order_cost=order_cost, spoilage_cost=spoilage_cost, risk_tolerance=risk_tolerance,
        review_period_days=review_period_days,
    )


def _recommend(
    product: Mapping,
    current_stock: int,
    current_date: date,
    history_df: pd.DataFrame,
    *,
    promo_dates: Collection[date] = (),
    order_cost: float = 30.0,
    spoilage_cost: float = 0.0,
    risk_tolerance: float | None = None,
    review_period_days: int | None = None,
    fit: _ProductFit | None = None,
    pooled: bool = False,
    mean_daily_demand: float | None = None,
) -> dict:
    """recommend_order with private options for the evaluation, which holds the history fixed:
    fit (cached posterior, see bayes_risk._stockout_risk), pooled (no-evidence ablation) and
    mean_daily_demand (cached mean daily units_sold for the economic review period). With the
    defaults it is exactly recommend_order.
    """
    missing = [f for f in REQUIRED_PRODUCT_FIELDS if f not in product]
    if missing:
        raise ValueError(f"product is missing fields: {missing}")
    missing = [c for c in _REQUIRED_HISTORY_COLUMNS if c not in history_df.columns]
    if missing:
        raise ValueError(f"history_df is missing columns: {missing}")
    product_id = int(product["product_id"])
    if fit is None:
        fit = _fit_product(history_df, product_id, pooled=pooled)
    shelf_life_days = _shelf_life(product["shelf_life_days"])
    holding_cost = float(product["holding_cost_per_day"])

    if review_period_days is None:
        if mean_daily_demand is None:
            mean_daily_demand = _mean_daily_demand(history_df, product_id)
        review_period_days = economic_review_period(
            mean_daily_demand, holding_cost, order_cost, shelf_life_days=shelf_life_days
        )

    risk = _stockout_risk(
        product_id,
        current_stock,
        int(product["lead_time_days"]),
        current_date,
        history_df,
        review_period_days=review_period_days,
        promo_dates=promo_dates,
        fit=fit,
        pooled=pooled,
    )
    order = calculate_optimal_order_quantity(
        current_stock,
        risk["order_horizon_distribution"],
        float(product["unit_cost"]),
        float(product["selling_price"]),
        holding_cost,
        float(product["stockout_penalty"]),
        risk["order_horizon_days"],
        spoilage_cost=spoilage_cost,
        order_cost=order_cost,
        shelf_life_days=shelf_life_days,
        review_period_days=review_period_days,
        risk_tolerance=risk_tolerance,
    )
    wait_risk = _stockout_risk(
        product_id,
        current_stock,
        int(product["lead_time_days"]),
        current_date,
        history_df,
        review_period_days=1,
        promo_dates=promo_dates,
        fit=fit,
        pooled=pooled,
    )
    extra_shortage = _expected_shortage(wait_risk["order_horizon_distribution"], current_stock) - (
        _expected_shortage(wait_risk["lead_time_distribution"], current_stock)
    )
    unit_loss = float(product["selling_price"]) - float(product["unit_cost"]) + float(product["stockout_penalty"])
    wait_cost = unit_loss * extra_shortage
    optimal_qty = order["optimal_qty"]
    early_holding_cost = holding_cost * optimal_qty
    should_order = bool(optimal_qty > 0 and wait_cost > early_holding_cost)

    return {
        "product_id": product_id,
        "review_period_days": int(review_period_days),
        "recommended_qty": int(optimal_qty if should_order else 0),
        "should_order": should_order,
        "wait_cost": float(wait_cost),
        "early_holding_cost": float(early_holding_cost),
        "risk": risk,
        "order": order,
        "wait_risk": wait_risk,
    }


def _expected_shortage(distribution: dict[int, float], position: int) -> float:
    """E[max(0, D - position)] for a demand distribution {units: probability}."""
    return float(sum(p * (d - position) for d, p in distribution.items() if d > position))
