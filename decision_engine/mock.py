"""Mock implementations with the same signatures as the real functions, for Person 1 to integrate against.

Every function here matches the contract in interface.md: same arguments, same return
structure, same types. The numbers are simple heuristics, not the real Bayes or Utility
Theory models, but they respond to inputs in the expected direction (promotions and
weekends raise risk, more stock lowers it, higher demand raises the order quantity).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

# effective_inventory_position is a small deterministic helper with no model behind it, so the mock
# re-exports the real implementation instead of duplicating it.
from decision_engine.inventory import effective_inventory_position  # noqa: F401

SALES_COLUMNS = ["date", "product_id", "units_sold", "is_weekend", "has_promo"]
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

_DEFAULT_START_DATE = date(2025, 1, 1)
_BASE_DAILY_DEMAND = {
    "Dairy": 18.0,
    "Bakery": 12.0,
    "Staples": 6.0,
    "Personal Care": 4.0,
    "Snacks": 15.0,
}
_FALLBACK_DAILY_DEMAND = 8.0
_WEEKEND_FACTOR = 1.2
_PROMO_FACTOR = 1.4
_PROMO_PROBABILITY = 0.1
_TAIL_PROBABILITY = 1e-6
_GROUP_EVIDENCE = {
    "weekday": (False, False),
    "weekend": (True, False),
    "weekday_promo": (False, True),
    "weekend_promo": (True, True),
}


def _base_demand(category: str) -> float:
    return _BASE_DAILY_DEMAND.get(category, _FALLBACK_DAILY_DEMAND)


def _demand_mean(base: float, is_weekend: bool, has_promo: bool) -> float:
    mean = base
    if is_weekend:
        mean *= _WEEKEND_FACTOR
    if has_promo:
        mean *= _PROMO_FACTOR
    return mean


def _poisson_distribution(mean: float) -> dict[int, float]:
    """Poisson pmf with the negligible upper tail cut off and renormalized."""
    if mean <= 0:
        return {0: 1.0}
    upper = int(poisson.ppf(1 - _TAIL_PROBABILITY, mean))
    support = np.arange(upper + 1)
    pmf = poisson.pmf(support, mean)
    pmf = pmf / pmf.sum()
    return {int(k): float(p) for k, p in zip(support, pmf)}


def _validate_generator_options(overdispersion: float | None, month_start_mult: float) -> None:
    if overdispersion is not None and not overdispersion > 0:
        raise ValueError(f"overdispersion must be > 0 or None, got {overdispersion}")
    if not month_start_mult > 0:
        raise ValueError(f"month_start_mult must be > 0, got {month_start_mult}")


def _draw(rng: np.random.Generator, mean: float, overdispersion: float | None) -> int:
    if overdispersion is None:
        return int(rng.poisson(mean))
    return int(rng.negative_binomial(overdispersion, overdispersion / (overdispersion + mean)))


def _is_perishable(row: pd.Series, policy: dict | None) -> bool:
    if policy is not None and "is_perishable" in policy:
        return bool(policy["is_perishable"])
    return not pd.isna(row["shelf_life_days"])


def generate_sales_history(
    products_df: pd.DataFrame,
    days: int,
    seed: int,
    *,
    start_date: date | None = None,
    overdispersion: float | None = None,
    month_start_mult: float = 1.0,
) -> pd.DataFrame:
    """Mock: generate synthetic daily sales history for every product.

    Assumptions:
        - units_sold is true demand; stock limits are not applied.
        - Mock demand is Poisson with a per-category mean, scaled up on weekends and promo days.
        - Each product is on promotion on a given day with probability 0.1.
        - start_date defaults to 2025-01-01 so output is reproducible.
        - overdispersion = k gives negative binomial demand (same mean, variance mean + mean^2/k);
          month_start_mult multiplies demand on days 1-5 of each month.

    Args:
        products_df: Products table (uses product_id and category).
        days: Number of consecutive days to generate. Must be >= 1.
        seed: Random seed for numpy.random.default_rng.
        start_date: First day of the history.
        overdispersion: None for Poisson demand, or k > 0 for negative binomial demand.
        month_start_mult: Demand multiplier on days 1-5 of each month (> 0).

    Returns:
        DataFrame in the SalesHistory schema, sorted by date then product_id.

    Raises:
        ValueError: If days < 1.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    _validate_generator_options(overdispersion, month_start_mult)
    rng = np.random.default_rng(seed)
    start = start_date if start_date is not None else _DEFAULT_START_DATE
    products = products_df.sort_values("product_id")

    rows = []
    for offset in range(days):
        current = start + timedelta(days=offset)
        is_weekend = current.weekday() >= 5
        for _, product in products.iterrows():
            has_promo = bool(rng.random() < _PROMO_PROBABILITY)
            mean = _demand_mean(_base_demand(product["category"]), is_weekend, has_promo)
            mean *= month_start_mult if current.day <= 5 else 1.0
            rows.append(
                {
                    "date": current,
                    "product_id": int(product["product_id"]),
                    "units_sold": _draw(rng, mean, overdispersion),
                    "is_weekend": is_weekend,
                    "has_promo": has_promo,
                }
            )
    return pd.DataFrame(rows, columns=SALES_COLUMNS)


def simulate_one_day(
    products_df: pd.DataFrame,
    current_date: date,
    has_promo_map: dict[int, bool],
    seed: int,
    *,
    overdispersion: float | None = None,
    month_start_mult: float = 1.0,
) -> pd.DataFrame:
    """Mock: generate one day of sales rows for every product.

    Assumptions:
        - units_sold is true demand; stock limits are not applied.
        - Products missing from has_promo_map are not on promotion.
        - The caller varies seed from day to day; the same seed gives the same draws.
        - overdispersion and month_start_mult behave as in generate_sales_history.

    Args:
        products_df: Products table (uses product_id and category).
        current_date: The day being simulated; sets is_weekend.
        has_promo_map: product_id -> whether that product is on promotion today.
        seed: Random seed for numpy.random.default_rng.
        overdispersion: None for Poisson demand, or k > 0 for negative binomial demand.
        month_start_mult: Demand multiplier on days 1-5 of each month (> 0).

    Returns:
        DataFrame in the SalesHistory schema with one row per product.
    """
    _validate_generator_options(overdispersion, month_start_mult)
    rng = np.random.default_rng(seed)
    is_weekend = current_date.weekday() >= 5
    rows = []
    for _, product in products_df.sort_values("product_id").iterrows():
        product_id = int(product["product_id"])
        has_promo = bool(has_promo_map.get(product_id, False))
        mean = _demand_mean(_base_demand(product["category"]), is_weekend, has_promo)
        mean *= month_start_mult if current_date.day <= 5 else 1.0
        rows.append(
            {
                "date": current_date,
                "product_id": product_id,
                "units_sold": _draw(rng, mean, overdispersion),
                "is_weekend": is_weekend,
                "has_promo": has_promo,
            }
        )
    return pd.DataFrame(rows, columns=SALES_COLUMNS)


def calculate_stockout_risk(
    product_id: int,
    current_stock: int,
    lead_time_days: int,
    current_date: date,
    history_df: pd.DataFrame,
    *,
    review_period_days: int = 1,
    promo_dates: Collection[date] = (),
) -> dict:
    """Mock: estimate demand distributions and the probability of a stockout.

    Assumptions:
        - The decision is made at the start of current_date. Lead-time window: current_date ..
          current_date + lead_time_days - 1; order-horizon window adds review_period_days more.
        - current_stock is the inventory position (on hand + on order).
        - Mock base daily demand is the product's historical mean units_sold (8.0 if there is
          no history). Each window day is scaled up if it is a Saturday/Sunday or in promo_dates.
        - Window demand is Poisson with mean = sum of the window's daily means.
        - group_rates are the base scaled by the mock factors, with a fixed +/-10% interval.
        - stockout_risk covers the lead-time window only.

    Args:
        product_id: Product to assess.
        current_stock: Inventory position. Must be >= 0.
        lead_time_days: Lead time in days. Must be >= 0.
        current_date: Decision day; the first day of both windows.
        history_df: Sales history in the SalesHistory schema.
        review_period_days: Days between decisions. Must be >= 1.
        promo_dates: Dates on which this product has a promotion.

    Returns:
        Dict with product_id, stockout_risk, group_rates, lead_time_day_groups,
        lead_time_distribution, order_horizon_days, order_horizon_distribution,
        expected_lead_time_demand and expected_order_horizon_demand.

    Raises:
        ValueError: If current_stock < 0, lead_time_days < 0 or review_period_days < 1.
    """
    if current_stock < 0:
        raise ValueError(f"current_stock must be >= 0, got {current_stock}")
    if lead_time_days < 0:
        raise ValueError(f"lead_time_days must be >= 0, got {lead_time_days}")
    if review_period_days < 1:
        raise ValueError(f"review_period_days must be >= 1, got {review_period_days}")

    product_history = history_df[history_df["product_id"] == product_id]
    base = (
        float(product_history["units_sold"].mean())
        if len(product_history) > 0
        else _FALLBACK_DAILY_DEMAND
    )
    promos = {d.date() if isinstance(d, datetime) else d for d in promo_dates}
    start = current_date.date() if isinstance(current_date, datetime) else current_date

    group_rates = {}
    for group, (is_weekend, has_promo) in _GROUP_EVIDENCE.items():
        rate = _demand_mean(base, is_weekend, has_promo)
        n_days = int(
            (
                (product_history["is_weekend"].astype(bool) == is_weekend)
                & (product_history["has_promo"].astype(bool) == has_promo)
            ).sum()
        )
        group_rates[group] = {
            "posterior_mean": float(rate),
            "ci_low": float(rate * 0.9),
            "ci_high": float(rate * 1.1),
            "n_days": n_days,
        }

    def window(days: int) -> tuple[dict[str, int], float]:
        counts = {g: 0 for g in _GROUP_EVIDENCE}
        mean = 0.0
        for offset in range(days):
            day = start + timedelta(days=offset)
            is_weekend, has_promo = day.weekday() >= 5, day in promos
            counts[("weekend" if is_weekend else "weekday") + ("_promo" if has_promo else "")] += 1
            mean += _demand_mean(base, is_weekend, has_promo)
        return counts, mean

    order_horizon_days = lead_time_days + review_period_days
    lead_time_day_groups, lead_mean = window(lead_time_days)
    _, horizon_mean = window(order_horizon_days)
    lead_time_distribution = _poisson_distribution(lead_mean)
    order_horizon_distribution = _poisson_distribution(horizon_mean)

    stockout_risk = sum(p for d, p in lead_time_distribution.items() if d > current_stock)

    return {
        "product_id": int(product_id),
        "stockout_risk": float(min(1.0, max(0.0, stockout_risk))),
        "group_rates": group_rates,
        "lead_time_day_groups": lead_time_day_groups,
        "lead_time_distribution": lead_time_distribution,
        "order_horizon_days": int(order_horizon_days),
        "order_horizon_distribution": order_horizon_distribution,
        "expected_lead_time_demand": float(sum(d * p for d, p in lead_time_distribution.items())),
        "expected_order_horizon_demand": float(
            sum(d * p for d, p in order_horizon_distribution.items())
        ),
    }


def calculate_optimal_order_quantity(
    current_stock: int,
    demand_distribution: dict[int, float],
    unit_cost: float,
    price: float,
    holding_cost: float,
    stockout_penalty: float,
    horizon_days: int,
    *,
    spoilage_cost: float = 0.0,
    order_cost: float = 0.0,
    shelf_life_days: int | None = None,
    review_period_days: int = 1,
    risk_tolerance: float | None = None,
    max_qty: int | None = None,
    step: int = 1,
) -> dict:
    """Mock: choose an order quantity on the grid 0, step, 2*step, ... up to the upper bound.

    Assumptions:
        - current_stock is the inventory position (on hand + on order).
        - demand_distribution is order_horizon_distribution from calculate_stockout_risk.
        - Mock candidate profits form a parabola peaking near expected demand * 1.1 minus
          current_stock, minus order_cost for every Q > 0; they do not follow the real
          profit model in interface.md. optimal_qty is the best candidate (ties -> smaller Q).
        - Mock profits are treated as certain, so certainty_equivalent equals expected_profit.
        - The automatic upper bound (max_qty None) follows the interface: the demand value with
          cumulative probability >= 1 - 1e-4, minus current_stock.

    Args:
        current_stock: Inventory position. Must be >= 0.
        demand_distribution: Units demanded over the horizon -> probability. Non-empty.
        unit_cost: Purchase cost per unit.
        price: Selling price per unit.
        holding_cost: Holding cost per unit per day.
        stockout_penalty: Goodwill cost per unit short, on top of lost margin.
        horizon_days: Order horizon in days. Must be >= 1.
        spoilage_cost: Extra disposal cost per spoiled unit.
        order_cost: Fixed cost of placing an order, charged only when Q > 0. Must be >= 0.
        shelf_life_days: Shelf life, or None if not perishable.
        review_period_days: Days until the next order can arrive. Must be between 1 and horizon_days.
        risk_tolerance: None for linear utility, otherwise a positive value for exponential utility.
        max_qty: Largest order considered, or None for the automatic bound. Must be >= 0.
        step: Grid spacing. Must be >= 1.

    Returns:
        Dict with optimal_qty, expected_utility, expected_profit, certainty_equivalent,
        no_stockout_probability, at_upper_bound, utility_type and candidates.

    Raises:
        ValueError: On invalid arguments (see the checks below).
    """
    if current_stock < 0:
        raise ValueError(f"current_stock must be >= 0, got {current_stock}")
    if not demand_distribution:
        raise ValueError("demand_distribution must be non-empty")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if not 1 <= review_period_days <= horizon_days:
        raise ValueError(f"review_period_days must be between 1 and horizon_days, got {review_period_days}")
    if max_qty is not None and max_qty < 0:
        raise ValueError(f"max_qty must be >= 0, got {max_qty}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if order_cost < 0:
        raise ValueError(f"order_cost must be >= 0, got {order_cost}")
    if risk_tolerance is not None and risk_tolerance <= 0:
        raise ValueError(f"risk_tolerance must be positive or None, got {risk_tolerance}")

    distribution = sorted((int(d), float(p)) for d, p in demand_distribution.items())
    expected_demand = sum(d * p for d, p in distribution)
    if max_qty is None:
        cumulative = 0.0
        for demand, prob in distribution:
            cumulative += prob
            if cumulative >= 1 - 1e-4:
                break
        upper = max(0, demand - current_stock)
    else:
        upper = max_qty
    target = max(0.0, expected_demand * 1.1 - current_stock)
    grid = list(range(0, upper + 1, step))
    peak_qty = min(grid, key=lambda q: (abs(q - target), q))

    margin = price - unit_cost
    peak_profit = margin * min(expected_demand, current_stock + peak_qty) - (
        holding_cost * horizon_days * (current_stock + peak_qty) / 2
    )
    curvature = max(0.05, (margin + stockout_penalty) / max(expected_demand, 1.0))

    if risk_tolerance is None:
        utility_type = "linear"

        def utility(profit: float) -> float:
            return profit

    else:
        utility_type = "exponential"

        def utility(profit: float) -> float:
            return 1.0 - math.exp(-profit / risk_tolerance)

    candidates = {}
    for q in grid:
        profit = peak_profit - curvature * (q - peak_qty) ** 2 - (order_cost if q > 0 else 0.0)
        candidates[int(q)] = {
            "expected_utility": float(utility(profit)),
            "expected_profit": float(profit),
            "certainty_equivalent": float(profit),
        }

    optimal_qty = max(grid, key=lambda q: (candidates[q]["expected_utility"], -q))
    best = candidates[optimal_qty]
    no_stockout = sum(p for d, p in distribution if d <= current_stock + optimal_qty)
    is_largest = optimal_qty == grid[-1]
    return {
        "optimal_qty": int(optimal_qty),
        "expected_utility": best["expected_utility"],
        "expected_profit": best["expected_profit"],
        "certainty_equivalent": best["certainty_equivalent"],
        "no_stockout_probability": float(min(1.0, no_stockout)),
        "at_upper_bound": bool(is_largest and (max_qty is not None or optimal_qty > 0)),
        "utility_type": utility_type,
        "candidates": candidates,
    }


def economic_review_period(
    mean_daily_demand: float,
    holding_cost: float,
    order_cost: float,
    *,
    shelf_life_days: int | None = None,
    min_days: int = 1,
    max_days: int = 30,
) -> int:
    """Mock: EOQ cycle length, round(sqrt(2 * order_cost / (holding_cost * mean_daily_demand))).

    Same rules as the real function (simple enough that the mock implements them fully):
    clipped to [min_days, max_days]; order_cost == 0 -> min_days; holding_cost == 0 or
    mean_daily_demand == 0 -> max_days; perishables capped at max(1, shelf_life_days - 1).

    Args:
        mean_daily_demand: Average units demanded per day. Must be >= 0.
        holding_cost: Holding cost per unit per day. Must be >= 0.
        order_cost: Fixed cost per order. Must be >= 0.
        shelf_life_days: Shelf life, or None if not perishable.
        min_days: Shortest allowed cycle. Must be >= 1.
        max_days: Longest allowed cycle. Must be >= min_days.

    Returns:
        Review period in days (int).

    Raises:
        ValueError: On negative inputs or min_days < 1 or max_days < min_days.
    """
    if mean_daily_demand < 0 or holding_cost < 0 or order_cost < 0:
        raise ValueError("mean_daily_demand, holding_cost and order_cost must be >= 0")
    if min_days < 1 or max_days < min_days:
        raise ValueError(f"need 1 <= min_days <= max_days, got {min_days}, {max_days}")
    if order_cost == 0:
        days = min_days
    elif holding_cost == 0 or mean_daily_demand == 0:
        days = max_days
    else:
        days = min(max_days, max(min_days, round(math.sqrt(2 * order_cost / (holding_cost * mean_daily_demand)))))
    if shelf_life_days is not None:
        days = min(days, max(1, int(shelf_life_days) - 1))
    return int(days)


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
    """Mock: recommend today's order for one product by chaining the mock risk and order functions.

    Applies the same when-to-order rule as the real function (see interface.md): order Q* today
    only if Q* > 0 and the expected cost of waiting one day exceeds holding Q* one extra day.

    Assumptions:
        - The review period (if None) comes from economic_review_period with the product's mean
          daily units_sold (8.0 without history).

    Args:
        product: One Products row (dict or pandas Series).
        current_stock: Inventory position. Must be >= 0.
        current_date: Decision day.
        history_df: Sales history in the SalesHistory schema.
        promo_dates: Dates on which this product has a promotion.
        order_cost: Fixed cost per order.
        spoilage_cost: Extra disposal cost per spoiled unit.
        risk_tolerance: None for linear utility, otherwise R > 0.
        review_period_days: Order cycle in days, or None for the economic review period.

    Returns:
        {"product_id", "review_period_days", "recommended_qty", "should_order", "wait_cost",
         "early_holding_cost", "risk", "order", "wait_risk"}
    """
    product_id = int(product["product_id"])
    shelf = product["shelf_life_days"]
    shelf_life_days = None if shelf is None or pd.isna(shelf) else int(shelf)
    holding_cost = float(product["holding_cost_per_day"])

    if review_period_days is None:
        sales = history_df.loc[history_df["product_id"] == product_id, "units_sold"]
        mean_daily = float(sales.mean()) if len(sales) > 0 else _FALLBACK_DAILY_DEMAND
        review_period_days = economic_review_period(
            mean_daily, holding_cost, order_cost, shelf_life_days=shelf_life_days
        )

    risk = calculate_stockout_risk(
        product_id, current_stock, int(product["lead_time_days"]), current_date, history_df,
        review_period_days=review_period_days, promo_dates=promo_dates,
    )
    order = calculate_optimal_order_quantity(
        current_stock, risk["order_horizon_distribution"], float(product["unit_cost"]),
        float(product["selling_price"]), holding_cost, float(product["stockout_penalty"]),
        risk["order_horizon_days"], spoilage_cost=spoilage_cost, order_cost=order_cost,
        shelf_life_days=shelf_life_days, review_period_days=review_period_days,
        risk_tolerance=risk_tolerance,
    )
    wait_risk = calculate_stockout_risk(
        product_id, current_stock, int(product["lead_time_days"]), current_date, history_df,
        review_period_days=1, promo_dates=promo_dates,
    )

    def shortage(distribution: dict[int, float]) -> float:
        return sum(p * (d - current_stock) for d, p in distribution.items() if d > current_stock)

    extra_shortage = shortage(wait_risk["order_horizon_distribution"]) - shortage(wait_risk["lead_time_distribution"])
    unit_loss = float(product["selling_price"]) - float(product["unit_cost"]) + float(product["stockout_penalty"])
    wait_cost = unit_loss * extra_shortage
    early_holding_cost = holding_cost * order["optimal_qty"]
    should_order = bool(order["optimal_qty"] > 0 and wait_cost > early_holding_cost)
    return {
        "product_id": product_id,
        "review_period_days": int(review_period_days),
        "recommended_qty": int(order["optimal_qty"] if should_order else 0),
        "should_order": should_order,
        "wait_cost": float(wait_cost),
        "early_holding_cost": float(early_holding_cost),
        "risk": risk,
        "order": order,
        "wait_risk": wait_risk,
    }


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
    """Mock: compare "baseline", "dss" and (optionally) "dss_no_evidence".

    Assumptions:
        - Mock results are derived from per-category base demand and the product's margin;
          no simulation is run. "dss" is always somewhat better than "dss_no_evidence", which
          is better than "baseline".
        - All policies face the same demand per product (units_demanded is identical).
        - Both policies pay order_cost on every order; total_profit includes it.
        - risk_tolerance, review_period_overrides, baseline_safety_factor, baseline_cover_days,
          generator_options and baseline_weekend_aware are validated but unused by the mock.
        - If policies is None, perishability comes from shelf_life_days (null = not perishable).
        - Total rows have product_id None; fill_rate on total rows is total sold / total demanded.

    Args:
        products_df: Products table in the schema from interface.md.
        days: Days to simulate. Must be >= 1.
        seed: Random seed for numpy.random.default_rng.
        history_days: Days of history before the simulated period. Must be a positive int.
        policies: product_id -> {"is_perishable", "spoilage_cost", "shelf_life_days"}, or None.
        order_cost: Fixed cost per order placed for one product. Must be >= 0.
        risk_tolerance: None or R > 0.
        review_period_overrides: product_id -> positive int review period, or None.
        include_ablation: Include the "dss_no_evidence" rows (ignored if include_dss is False).
        baseline_safety_factor: One value or {product_id: value}, > 0.
        baseline_cover_days: One value or {product_id: value}, positive ints.
        include_dss: If False, only baseline rows are produced.
        generator_options: None or a dict with "overdispersion" and/or "month_start_mult".
        baseline_weekend_aware: Use the weekend-aware baseline (bool).

    Returns:
        DataFrame with columns policy, product_id, total_profit, stockout_days, units_lost,
        units_spoiled, avg_stock_held, orders_placed, fill_rate, revenue, units_demanded: one row
        per (policy, product) plus a total row per policy.

    Raises:
        ValueError: On invalid arguments.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    if isinstance(history_days, bool) or not isinstance(history_days, int) or history_days < 1:
        raise ValueError(f"history_days must be a positive int, got {history_days!r}")
    if order_cost < 0:
        raise ValueError(f"order_cost must be >= 0, got {order_cost}")
    if risk_tolerance is not None and risk_tolerance <= 0:
        raise ValueError(f"risk_tolerance must be positive or None, got {risk_tolerance}")
    for pid, review in (review_period_overrides or {}).items():
        if isinstance(review, bool) or not isinstance(review, int) or review < 1:
            raise ValueError(f"review_period_overrides[{pid}] must be a positive int, got {review!r}")
    factors = baseline_safety_factor.values() if isinstance(baseline_safety_factor, dict) else [baseline_safety_factor]
    if any(isinstance(v, bool) or not v > 0 for v in factors):
        raise ValueError("baseline_safety_factor values must be > 0")
    covers = baseline_cover_days.values() if isinstance(baseline_cover_days, dict) else [baseline_cover_days]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in covers):
        raise ValueError("baseline_cover_days values must be positive ints")
    if not isinstance(baseline_weekend_aware, bool):
        raise ValueError(f"baseline_weekend_aware must be a bool, got {baseline_weekend_aware!r}")
    unknown = set(generator_options or {}) - {"overdispersion", "month_start_mult"}
    if unknown:
        raise ValueError(f"unknown generator_options: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    policies = policies or {}

    # PLACEHOLDER numbers: these fixed rates make "dss" beat "baseline" so Person 1 has
    # something plausible to display. They are not a result. The real simulation in
    # evaluation.py may show a smaller gain, no gain, or a loss for some products.
    settings = {
        "baseline": {"service": 0.90, "stock_days": 2.0, "spoil_rate": 0.06, "cycle_shrink": 0},
        "dss": {"service": 0.97, "stock_days": 1.5, "spoil_rate": 0.02, "cycle_shrink": 2},
    }
    if not include_dss:
        del settings["dss"]
    elif include_ablation:
        settings["dss_no_evidence"] = {"service": 0.95, "stock_days": 1.7, "spoil_rate": 0.03, "cycle_shrink": 2}

    products = products_df.sort_values("product_id")
    daily_demand = {
        int(pid): _base_demand(cat) * float(rng.uniform(0.9, 1.1))
        for pid, cat in zip(products["product_id"], products["category"])
    }

    rows = []
    for policy_name, s in settings.items():
        policy_rows = []
        for _, product in products.iterrows():
            product_id = int(product["product_id"])
            policy = policies.get(product_id)
            perishable = _is_perishable(product, policy)
            spoilage_cost = float((policy or {}).get("spoilage_cost", 0.0))
            shelf_life = (policy or {}).get("shelf_life_days", product["shelf_life_days"])
            cycle_days = 7
            if perishable and shelf_life is not None and not pd.isna(shelf_life):
                cycle_days = min(7, int(shelf_life))
            cycle_days = max(1, cycle_days - s["cycle_shrink"])
            orders_placed = math.ceil(days / cycle_days)

            daily = daily_demand[product_id]
            units_demanded = int(round(daily * days))
            units_lost = int(round(units_demanded * (1 - s["service"])))
            units_sold = units_demanded - units_lost
            units_spoiled = int(round(units_demanded * s["spoil_rate"])) if perishable else 0
            avg_stock_held = daily * s["stock_days"]
            price = float(product["selling_price"])
            margin = price - float(product["unit_cost"])
            total_profit = (
                margin * units_sold
                - float(product["holding_cost_per_day"]) * avg_stock_held * days
                - float(product["stockout_penalty"]) * units_lost
                - (float(product["unit_cost"]) + spoilage_cost) * units_spoiled
                - order_cost * orders_placed
            )
            policy_rows.append(
                {
                    "policy": policy_name,
                    "product_id": product_id,
                    "total_profit": round(float(total_profit), 2),
                    "stockout_days": int(round(days * (1 - s["service"]) * 1.5)),
                    "units_lost": units_lost,
                    "units_spoiled": units_spoiled,
                    "avg_stock_held": round(float(avg_stock_held), 2),
                    "orders_placed": int(orders_placed),
                    "fill_rate": float(units_sold / units_demanded) if units_demanded else 1.0,
                    "revenue": float(price * units_sold),
                    "units_demanded": units_demanded,
                }
            )
        rows.extend(policy_rows)
        demanded = sum(r["units_demanded"] for r in policy_rows)
        sold = demanded - sum(r["units_lost"] for r in policy_rows)
        rows.append(
            {
                "policy": policy_name,
                "product_id": None,
                "total_profit": round(sum(r["total_profit"] for r in policy_rows), 2),
                "stockout_days": sum(r["stockout_days"] for r in policy_rows),
                "units_lost": sum(r["units_lost"] for r in policy_rows),
                "units_spoiled": sum(r["units_spoiled"] for r in policy_rows),
                "avg_stock_held": round(sum(r["avg_stock_held"] for r in policy_rows), 2),
                "orders_placed": sum(r["orders_placed"] for r in policy_rows),
                "fill_rate": float(sold / demanded) if demanded else 1.0,
                "revenue": float(sum(r["revenue"] for r in policy_rows)),
                "units_demanded": demanded,
            }
        )

    result = pd.DataFrame(rows, columns=EVALUATION_COLUMNS)
    # Built separately so pandas keeps ints and None instead of inferring float64 with NaN.
    result["product_id"] = pd.Series([r["product_id"] for r in rows], dtype=object)
    return result
