"""Evaluation simulation comparing the decision engine against a static reorder policy.

Simulation
----------
1. Demand. generate_sales_history produces history_days + days of daily demand from the seed, starting
   2025-01-01, and the result is split by date. The first history_days form the FIXED history that
   both policies learn from (it is never updated during the simulation: on stockout days a store only
   observes sales, not demand, so appending sales would bias estimates down, and appending true demand
   would use information a real store cannot see). The remaining days are the true daily demand and
   promotion flags of the simulated period. Promotion dates are known in advance to the DSS (stores
   plan promotions); the baseline ignores them. Every policy faces exactly the same daily demand.

2. State. For each policy and product the simulation keeps on-hand stock as first-in-first-out
   batches (quantity + arrival day) and in-transit orders (quantity + arrival day).

3. Each simulated day t, in this order:
   a. Receive: orders arriving on day t are added to on-hand stock as a new batch (age 0).
   b. Expire: for perishables, any batch whose age (t - arrival day) has reached shelf_life_days is
      discarded and counted as spoiled; each spoiled unit is charged spoilage_cost.
   c. Decide: the inventory position is the EFFECTIVE inventory position
      (inventory.effective_inventory_position): on hand + in transit, except that perishable batches
      that will expire before a new order could arrive count only for the demand they are expected
      to meet first. The baseline uses its mu as the expected daily demand; the DSS uses the
      history's mean daily demand (the same number). The policy chooses Q. If Q > 0 the store pays
      unit_cost * Q + order_cost now, and the order arrives at the start of day t + lead_time_days
      (a lead time of 0 means it arrives immediately, before today's sales).
   d. Sell: demand is served first in, first out up to the stock on hand. Revenue = price * sold.
      Each unit of unmet demand is charged stockout_penalty. A stockout day is any day with unmet demand.
   e. Hold: holding_cost is charged per unit on hand at the end of the day.

4. Starting state. Both policies start each product with on-hand stock equal to the baseline's
   order-up-to level S (one fresh batch arriving on day 0) and nothing in transit.

5. Accounting.
       profit = revenue - purchases - order costs - holding - stockout penalties - spoilage costs
                + value of ending inventory (on hand + in transit, at unit_cost)
                - value of starting inventory (at unit_cost)
   The inventory adjustment means a policy is neither rewarded nor punished for the stock it happens
   to hold on the last day; stock bought but not yet sold is valued at what it cost.

6. Policies.
   - baseline: the static (s, S) policy. mu = mean daily units_sold of the product over the history
     (all days; 8 if the product has no history), D = cover_days (default 7) for non-perishables and
     max(1, min(cover_days, shelf_life_days - 1)) for perishables (so a batch can sell before it
     expires), s = ceil(mu * lead_time_days * safety_factor) with safety_factor default 1.2,
     S = s + ceil(mu * D). safety_factor and cover_days can be set per product for robustness checks.
   - weekend-aware baseline (baseline_weekend_aware=True, robustness check): separate mean daily
     demand on weekdays and weekends from the history (promotion days included, not separated). On
     day t, s_t = ceil(safety_factor * sum of the day-type means over the lead-time days t .. t+L-1)
     and S_t = s_t + ceil(sum of the day-type means over the D cover days t+L .. t+L+D-1), with D
     and the perishable cap as above. It still never adjusts for promotions. Starting stock for every
     policy is S on the first simulated day. Each day, if the effective inventory
     position <= s, order S - position. It never adjusts for weekends or promotions.
   - dss: each day, recommend_order with current_stock = effective inventory position, the actual
     date, the fixed history, the product's promotion dates, order_cost, risk_tolerance,
     spoilage_cost from the policies argument (default 0), and the review period override if given
     (otherwise the economic review period). The order placed is recommended_qty, which is Q* only
     when waiting one more day would cost more than holding Q* one extra day.
   - dss_no_evidence (ablation): identical to dss, except the demand model pools all history days into
     one group, so weekends and promotions carry no information. The difference between dss and
     dss_no_evidence measures the value of the weekend/promotion evidence in the Bayes model.

7. Robustness options. baseline_safety_factor and baseline_cover_days tune the baseline (one value
   or per product); include_dss=False runs the baseline alone (for tuning it); generator_options
   (overdispersion, month_start_mult) make the true demand differ from the DSS's model, which is not
   told about them. All defaults reproduce the standard evaluation exactly.

8. Caching. The history is fixed, so each product's posterior (and its mean daily demand for the
   economic review period) is computed once per policy and reused every day. This gives results
   identical to calling recommend_order afresh each day.

Assumptions and limitations (for the report):
  - Demand is the synthetic generator's true demand; there is no model misspecification beyond what
    the generator's structure implies (Poisson demand with weekend and promotion multipliers).
  - Unmet demand is lost, not backordered.
  - Orders always arrive exactly lead_time_days later and in full; there is no supplier uncertainty.
  - Purchases are paid when ordered; the ending-inventory adjustment values stock at unit_cost.
  - Spoilage in the simulation tracks real batch ages, while the DSS's optimizer approximates spoilage
    (it treats stock on hand as fresh). The simulation is the ground truth for both policies.
  - The effective inventory position is a deterministic approximation (demand at its mean). Both
    policies use it, so neither gets an advantage from knowing batch ages.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from decision_engine.bayes_risk import DEFAULT_DAILY_DEMAND, _fit_product
from decision_engine.data_generator import DEFAULT_START_DATE, generate_sales_history
from decision_engine.inventory import effective_inventory_position
from decision_engine.recommend import _mean_daily_demand, _recommend

EVALUATION_COLUMNS = [
    "policy",
    "product_id",
    "total_profit",
    "stockout_days",
    "units_lost",
    "units_spoiled",
    "avg_stock_held",
    "orders_placed",
    "fill_rate",
    "revenue",
    "units_demanded",
]
REQUIRED_PRODUCT_COLUMNS = [
    "product_id",
    "name",
    "category",
    "unit_cost",
    "selling_price",
    "holding_cost_per_day",
    "stockout_penalty",
    "shelf_life_days",
    "lead_time_days",
]
BASELINE_CYCLE_DAYS = 7
BASELINE_SAFETY_FACTOR = 1.2
GENERATOR_OPTIONS = {"overdispersion", "month_start_mult"}


def _values(value) -> list:
    return list(value.values()) if isinstance(value, dict) else [value]


@dataclass(frozen=True)
class _Product:
    """A product with its effective perishability and spoilage cost for this evaluation."""

    product_id: int
    unit_cost: float
    price: float
    holding_cost: float
    stockout_penalty: float
    lead_time_days: int
    shelf_life_days: int | None
    spoilage_cost: float
    baseline_safety_factor: float = 1.2
    baseline_cover_days: int = 7

    def as_row(self) -> dict:
        """The Products-row form recommend_order expects."""
        return {
            "product_id": self.product_id,
            "unit_cost": self.unit_cost,
            "selling_price": self.price,
            "holding_cost_per_day": self.holding_cost,
            "stockout_penalty": self.stockout_penalty,
            "shelf_life_days": self.shelf_life_days,
            "lead_time_days": self.lead_time_days,
        }


@dataclass
class _Ledger:
    """Everything that happened to one product under one policy (used by the tests and the output)."""

    product_id: int
    starting_stock: int
    starting_value: float
    received: int = 0
    sold: int = 0
    spoiled: int = 0
    lost: int = 0
    demanded: int = 0
    stockout_days: int = 0
    revenue: float = 0.0
    purchases: float = 0.0
    order_costs: float = 0.0
    holding_costs: float = 0.0
    stockout_penalties: float = 0.0
    spoilage_costs: float = 0.0
    ending_on_hand: int = 0
    ending_in_transit: int = 0
    ending_value: float = 0.0
    orders: list[tuple[int, int, int]] = field(default_factory=list)  # (day placed, qty, arrival day)
    receipts: list[tuple[int, int]] = field(default_factory=list)  # (day, qty)
    spoilage_events: list[tuple[int, int]] = field(default_factory=list)  # (day, qty)
    decisions: list[tuple[int, int, int]] = field(default_factory=list)  # (day, inventory position, qty)
    on_hand_end: list[int] = field(default_factory=list)

    @property
    def profit(self) -> float:
        return (
            self.revenue
            - self.purchases
            - self.order_costs
            - self.holding_costs
            - self.stockout_penalties
            - self.spoilage_costs
            + self.ending_value
            - self.starting_value
        )


def _per_product(value, product_id: int, default):
    if isinstance(value, dict):
        return value.get(product_id, default)
    return value


def _effective_products(
    products_df: pd.DataFrame,
    policies: dict[int, dict],
    safety_factor: float | dict[int, float] = 1.2,
    cover_days: int | dict[int, int] = 7,
) -> list[_Product]:
    products = []
    for row in products_df.sort_values("product_id").itertuples(index=False):
        product_id = int(row.product_id)
        policy = policies.get(product_id, {})
        shelf = policy.get("shelf_life_days", row.shelf_life_days)
        shelf = None if shelf is None or pd.isna(shelf) else int(shelf)
        perishable = bool(policy.get("is_perishable", shelf is not None))
        products.append(
            _Product(
                product_id=product_id,
                unit_cost=float(row.unit_cost),
                price=float(row.selling_price),
                holding_cost=float(row.holding_cost_per_day),
                stockout_penalty=float(row.stockout_penalty),
                lead_time_days=int(row.lead_time_days),
                shelf_life_days=shelf if perishable else None,
                spoilage_cost=float(policy.get("spoilage_cost", 0.0)),
                baseline_safety_factor=float(_per_product(safety_factor, product_id, BASELINE_SAFETY_FACTOR)),
                baseline_cover_days=int(_per_product(cover_days, product_id, BASELINE_CYCLE_DAYS)),
            )
        )
    return products


def _baseline_mu(history_df: pd.DataFrame, product: _Product) -> float:
    sales = history_df.loc[history_df["product_id"] == product.product_id, "units_sold"]
    return float(sales.mean()) if len(sales) > 0 else DEFAULT_DAILY_DEMAND


def _baseline_cycle_days(product: _Product) -> int:
    """D: cover_days of cycle stock (default 7), capped for perishables at shelf_life_days - 1."""
    if product.shelf_life_days is None:
        return product.baseline_cover_days
    return max(1, min(product.baseline_cover_days, product.shelf_life_days - 1))


def _baseline_levels(history_df: pd.DataFrame, product: _Product) -> tuple[int, int]:
    """Reorder point s and order-up-to level S of the static baseline policy."""
    mu = _baseline_mu(history_df, product)
    s = math.ceil(mu * product.lead_time_days * product.baseline_safety_factor)
    return s, s + math.ceil(mu * _baseline_cycle_days(product))


def _weekday_weekend_mu(history_df: pd.DataFrame, product: _Product) -> tuple[float, float]:
    """Mean daily units_sold on weekdays and on weekends (promotion days included, not separated).
    Falls back to the overall mean for a missing day type, and to DEFAULT_DAILY_DEMAND without history."""
    rows = history_df[history_df["product_id"] == product.product_id]
    overall = float(rows["units_sold"].mean()) if len(rows) > 0 else DEFAULT_DAILY_DEMAND
    weekend = rows["is_weekend"].astype(bool)
    weekday_mu = float(rows.loc[~weekend, "units_sold"].mean()) if (~weekend).any() else overall
    weekend_mu = float(rows.loc[weekend, "units_sold"].mean()) if weekend.any() else overall
    return weekday_mu, weekend_mu


def _weekend_aware_levels(mu: tuple[float, float], product: _Product, start: date) -> tuple[int, int]:
    """s and S of the weekend-aware baseline for a decision on date `start`."""
    weekday_mu, weekend_mu = mu

    def expected(first: int, days: int) -> float:
        dates = (start + timedelta(days=first + i) for i in range(days))
        return sum(weekend_mu if d.weekday() >= 5 else weekday_mu for d in dates)

    lead = product.lead_time_days
    s = math.ceil(expected(0, lead) * product.baseline_safety_factor)
    return s, s + math.ceil(expected(lead, _baseline_cycle_days(product)))


def _inventory_position(
    batches, in_transit: list[tuple[int, int]], day: int, sim_start: date, product: _Product, mean_daily: float
) -> int:
    """Effective inventory position from the simulation's day-indexed batches."""
    return effective_inventory_position(
        [(sim_start + timedelta(days=arrival), qty) for arrival, qty in batches],
        sum(qty for _, qty in in_transit),
        sim_start + timedelta(days=day),
        product.lead_time_days,
        product.shelf_life_days,
        mean_daily,
    )


def _simulate(
    products: list[_Product],
    history_df: pd.DataFrame,
    demand: dict[int, np.ndarray],
    promo_dates: dict[int, list[date]],
    sim_start: date,
    policy: str,
    *,
    order_cost: float,
    risk_tolerance: float | None = None,
    review_period_overrides: dict[int, int] | None = None,
    baseline_weekend_aware: bool = False,
) -> dict[int, _Ledger]:
    """Run one policy over the simulated period (steps 2-5 of the module docstring)."""
    overrides = review_period_overrides or {}
    ledgers = {}
    for product in products:
        pid = product.product_id
        days = len(demand[pid])
        if baseline_weekend_aware:
            week_mu = _weekday_weekend_mu(history_df, product)
            s_level, S_level = _weekend_aware_levels(week_mu, product, sim_start)
        else:
            s_level, S_level = _baseline_levels(history_df, product)
        mean_daily = _baseline_mu(history_df, product)  # equals _mean_daily_demand(history_df, pid)

        if policy == "baseline" and baseline_weekend_aware:
            def decide(day: int, position: int) -> int:
                s_t, S_t = _weekend_aware_levels(week_mu, product, sim_start + timedelta(days=day))
                return S_t - position if position <= s_t else 0
        elif policy == "baseline":
            def decide(day: int, position: int) -> int:
                return S_level - position if position <= s_level else 0
        elif policy in ("dss", "dss_no_evidence"):
            pooled = policy == "dss_no_evidence"
            fit = _fit_product(history_df, pid, pooled=pooled)
            history_mean = _mean_daily_demand(history_df, pid)
            row = product.as_row()

            def decide(day: int, position: int) -> int:
                result = _recommend(
                    row, position, sim_start + timedelta(days=day), history_df,
                    promo_dates=promo_dates[pid], order_cost=order_cost,
                    spoilage_cost=product.spoilage_cost, risk_tolerance=risk_tolerance,
                    review_period_days=overrides.get(pid), fit=fit, pooled=pooled,
                    mean_daily_demand=history_mean,
                )
                return result["recommended_qty"]
        else:
            raise ValueError(f"unknown policy {policy!r}")

        ledger = _Ledger(pid, starting_stock=S_level, starting_value=product.unit_cost * S_level)
        batches: deque[list[int]] = deque([[0, S_level]])  # [arrival day, qty], oldest first
        in_transit: list[tuple[int, int]] = []  # (arrival day, qty)

        for t in range(days):
            # a. Receive.
            arriving = [qty for arrival, qty in in_transit if arrival == t]
            in_transit = [(arrival, qty) for arrival, qty in in_transit if arrival != t]
            for qty in arriving:
                batches.append([t, qty])
                ledger.received += qty
                ledger.receipts.append((t, qty))

            # b. Expire.
            if product.shelf_life_days is not None:
                expired = 0
                while batches and t - batches[0][0] >= product.shelf_life_days:
                    expired += batches.popleft()[1]
                if expired:
                    ledger.spoiled += expired
                    ledger.spoilage_costs += product.spoilage_cost * expired
                    ledger.spoilage_events.append((t, expired))

            # c. Decide on the effective inventory position.
            position = _inventory_position(batches, in_transit, t, sim_start, product, mean_daily)
            qty = int(decide(t, position))
            ledger.decisions.append((t, position, qty))
            if qty > 0:
                ledger.purchases += product.unit_cost * qty
                ledger.order_costs += order_cost
                arrival = t + product.lead_time_days
                ledger.orders.append((t, qty, arrival))
                if arrival == t:
                    batches.append([t, qty])
                    ledger.received += qty
                    ledger.receipts.append((t, qty))
                else:
                    in_transit.append((arrival, qty))

            # d. Sell first in, first out.
            wanted = int(demand[pid][t])
            remaining = wanted
            while remaining > 0 and batches:
                take = min(remaining, batches[0][1])
                batches[0][1] -= take
                remaining -= take
                if batches[0][1] == 0:
                    batches.popleft()
            sold = wanted - remaining
            ledger.demanded += wanted
            ledger.sold += sold
            ledger.lost += remaining
            ledger.revenue += product.price * sold
            ledger.stockout_penalties += product.stockout_penalty * remaining
            if remaining > 0:
                ledger.stockout_days += 1

            # e. Hold.
            on_hand = sum(qty for _, qty in batches)
            ledger.holding_costs += product.holding_cost * on_hand
            ledger.on_hand_end.append(on_hand)

        ledger.ending_on_hand = sum(qty for _, qty in batches)
        ledger.ending_in_transit = sum(qty for _, qty in in_transit)
        ledger.ending_value = product.unit_cost * (ledger.ending_on_hand + ledger.ending_in_transit)
        ledgers[pid] = ledger
    return ledgers


def _summary_row(policy: str, product_id: int | None, ledgers: list[_Ledger]) -> dict:
    demanded = sum(l.demanded for l in ledgers)
    sold = sum(l.sold for l in ledgers)
    return {
        "policy": policy,
        "product_id": product_id,
        "total_profit": float(sum(l.profit for l in ledgers)),
        "stockout_days": int(sum(l.stockout_days for l in ledgers)),
        "units_lost": int(sum(l.lost for l in ledgers)),
        "units_spoiled": int(sum(l.spoiled for l in ledgers)),
        "avg_stock_held": float(sum(np.mean(l.on_hand_end) for l in ledgers)),
        "orders_placed": int(sum(len(l.orders) for l in ledgers)),
        "fill_rate": float(sold / demanded) if demanded else 1.0,
        "revenue": float(sum(l.revenue for l in ledgers)),
        "units_demanded": int(demanded),
    }


def _evaluate(
    products_df: pd.DataFrame,
    days: int,
    seed: int,
    *,
    history_days: int = 180,
    policies: dict[int, dict] | None = None,
    order_cost: float = 30.0,
    risk_tolerance: float | None = None,
    review_period_overrides: dict[int, int] | None = None,
    include_ablation: bool = True,
    baseline_safety_factor: float | dict[int, float] = BASELINE_SAFETY_FACTOR,
    baseline_cover_days: int | dict[int, int] = BASELINE_CYCLE_DAYS,
    include_dss: bool = True,
    generator_options: dict | None = None,
    baseline_weekend_aware: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[int, _Ledger]]]:
    """run_evaluation, also returning the per-policy ledgers for testing and reporting."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError(f"days must be a positive int, got {days!r}")
    if isinstance(history_days, bool) or not isinstance(history_days, int) or history_days < 1:
        raise ValueError(f"history_days must be a positive int, got {history_days!r}")
    if not order_cost >= 0:
        raise ValueError(f"order_cost must be >= 0, got {order_cost}")
    if risk_tolerance is not None and not risk_tolerance > 0:
        raise ValueError(f"risk_tolerance must be positive or None, got {risk_tolerance}")
    for pid, review in (review_period_overrides or {}).items():
        if isinstance(review, bool) or not isinstance(review, int) or review < 1:
            raise ValueError(f"review_period_overrides[{pid}] must be a positive int, got {review!r}")
    missing = [c for c in REQUIRED_PRODUCT_COLUMNS if c not in products_df.columns]
    if missing:
        raise ValueError(f"products_df is missing columns: {missing}")
    for value in _values(baseline_safety_factor):
        if isinstance(value, bool) or not value > 0:
            raise ValueError(f"baseline_safety_factor values must be > 0, got {value!r}")
    for value in _values(baseline_cover_days):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"baseline_cover_days values must be positive ints, got {value!r}")
    if not isinstance(baseline_weekend_aware, bool):
        raise ValueError(f"baseline_weekend_aware must be a bool, got {baseline_weekend_aware!r}")
    unknown = set(generator_options or {}) - GENERATOR_OPTIONS
    if unknown:
        raise ValueError(f"unknown generator_options: {sorted(unknown)}; allowed: {sorted(GENERATOR_OPTIONS)}")

    products = _effective_products(products_df, policies or {}, baseline_safety_factor, baseline_cover_days)
    sim_start = DEFAULT_START_DATE + timedelta(days=history_days)
    full = generate_sales_history(
        products_df, history_days + days, seed, start_date=DEFAULT_START_DATE, **(generator_options or {})
    )
    history = full[full["date"] < sim_start].reset_index(drop=True)
    future = full[full["date"] >= sim_start]
    demand = {}
    promo_dates = {}
    for product in products:
        rows = future[future["product_id"] == product.product_id].sort_values("date")
        demand[product.product_id] = rows["units_sold"].to_numpy()
        promo_dates[product.product_id] = list(rows.loc[rows["has_promo"], "date"])

    names = ["baseline"]
    if include_dss:
        names += ["dss"] + (["dss_no_evidence"] if include_ablation else [])
    ledgers = {
        name: _simulate(
            products, history, demand, promo_dates, sim_start, name, order_cost=order_cost,
            risk_tolerance=risk_tolerance, review_period_overrides=review_period_overrides,
            baseline_weekend_aware=baseline_weekend_aware,
        )
        for name in names
    }

    rows = []
    for name in names:
        per_product = ledgers[name]
        rows.extend(_summary_row(name, pid, [ledger]) for pid, ledger in per_product.items())
        rows.append(_summary_row(name, None, list(per_product.values())))
    result = pd.DataFrame(rows, columns=EVALUATION_COLUMNS)
    # Built separately so pandas keeps ints and None instead of inferring float64 with NaN.
    result["product_id"] = pd.Series([r["product_id"] for r in rows], dtype=object)
    return result, ledgers


def run_evaluation(
    products_df: pd.DataFrame,
    days: int,
    seed: int,
    *,
    history_days: int = 180,
    policies: dict[int, dict] | None = None,
    order_cost: float = 30.0,
    risk_tolerance: float | None = None,
    review_period_overrides: dict[int, int] | None = None,
    include_ablation: bool = True,
    baseline_safety_factor: float | dict[int, float] = 1.2,
    baseline_cover_days: int | dict[int, int] = 7,
    include_dss: bool = True,
    generator_options: dict | None = None,
    baseline_weekend_aware: bool = False,
) -> pd.DataFrame:
    """Simulate the static baseline, the DSS and (optionally) the no-evidence ablation on identical demand.

    See the module docstring for the full simulation, accounting and policy definitions.

    Assumptions:
        - history_days of generated demand (from seed) form a fixed history; the next `days` are the
          simulated period. Promotions are known in advance; only the DSS uses them.
        - Both policies start each product with the baseline's order-up-to level S on hand.
        - Unmet demand is lost. Orders arrive exactly lead_time_days later.
        - Profit includes an ending-minus-starting inventory adjustment at unit_cost.
        - If policies is None, perishability comes from shelf_life_days and spoilage_cost is 0.
        - generator_options change the true demand only; the DSS's model is not told about them.

    Args:
        products_df: Products table in the schema from interface.md.
        days: Days to simulate. Must be a positive int.
        seed: Random seed for the history and the simulated demand.
        history_days: Days of history before the simulated period. Must be a positive int.
        policies: product_id -> {"is_perishable", "spoilage_cost", "shelf_life_days"}, or None.
        order_cost: Fixed cost per order placed for one product. Must be >= 0.
        risk_tolerance: Passed to recommend_order (None = risk-neutral).
        review_period_overrides: product_id -> review period for recommend_order; products not
            listed use the economic review period.
        include_ablation: Also run the "dss_no_evidence" policy (ignored if include_dss is False).
        baseline_safety_factor: Baseline reorder-point safety factor, one value or
            {product_id: value}; missing products use 1.2. Must be > 0.
        baseline_cover_days: Baseline cycle cover D in days, one value or {product_id: value};
            missing products use 7. Perishables are capped at shelf_life_days - 1. Positive ints.
        include_dss: If False, only baseline rows are produced.
        generator_options: Passed to generate_sales_history: "overdispersion" and/or
            "month_start_mult". None means standard demand.
        baseline_weekend_aware: Use the weekend-aware baseline (separate weekday/weekend means
            applied to the actual upcoming days). Default False.

    Returns:
        DataFrame with columns policy, product_id, total_profit, stockout_days, units_lost,
        units_spoiled, avg_stock_held, orders_placed, fill_rate, revenue, units_demanded: one row
        per (policy, product) plus a total row per policy with product_id None.

    Raises:
        ValueError: On invalid arguments or a products_df missing columns.
    """
    result, _ = _evaluate(
        products_df, days, seed, history_days=history_days, policies=policies,
        order_cost=order_cost, risk_tolerance=risk_tolerance,
        review_period_overrides=review_period_overrides, include_ablation=include_ablation,
        baseline_safety_factor=baseline_safety_factor, baseline_cover_days=baseline_cover_days,
        include_dss=include_dss, generator_options=generator_options,
        baseline_weekend_aware=baseline_weekend_aware,
    )
    return result
