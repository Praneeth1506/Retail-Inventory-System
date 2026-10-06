"""Order quantity optimization using Utility Theory.

Model
-----
The owner chooses an order quantity Q now. Demand d over the order horizon (lead time + review
period) is uncertain, with the distribution P(d) produced by calculate_stockout_risk. For each
candidate Q and each demand value d, profit over the horizon is

    available = current_stock + Q
    sold      = min(available, d)
    leftover  = max(0, available - d)
    shortage  = max(0, d - available)
    spoiled   = see "Spoilage" below (0 for non-perishables)

    profit = (price - unit_cost) * sold
             - holding_cost * horizon_days * (available + leftover) / 2
             - stockout_penalty * shortage
             - (unit_cost + spoilage_cost) * spoiled
             - (order_cost if Q > 0 else 0)

Leftover units that do not spoil keep their purchase value (they can be sold later), so only
their holding cost counts. Holding cost uses average inventory over the horizon, (start + end) / 2.

Spoilage. The order arrives at the end of the lead time, so at the end of the horizon the newest
stock has been on the shelf for review_period_days, not horizon_days. For a perishable product:
    remaining_life = shelf_life_days - review_period_days
    if remaining_life <= 0:  spoiled = leftover
    else:                    spoiled = max(0, leftover - ceil(mean_daily * remaining_life))
where mean_daily = expected horizon demand / horizon_days. In words: leftover stock keeps selling
for its remaining life at the average daily rate, and whatever is still unsold then spoils.
Assumptions: stock is sold first in, first out; units already on hand are treated as fresh (their
age is not tracked); and future demand after the horizon is approximated by its average.

Utility. The owner's attitude to risk is a utility function of profit:
  - linear (risk-neutral):        U(profit) = profit
  - exponential (risk-averse):    U(profit) = 1 - exp(-profit / R), with risk tolerance R > 0 in rupees.
    Smaller R means more risk-averse.
The chosen Q maximizes expected utility E[U(profit)] = sum_d P(d) * U(profit(Q, d)). Ties go to
the smaller Q.

Certainty equivalent. The guaranteed profit the owner would consider exactly as good as the risky
outcome of ordering Q: CE = U^-1(E[U]). For linear utility CE = E[profit]. For exponential utility
CE = -R * log(E[exp(-profit / R)]), which is at most E[profit] (Jensen's inequality); the gap is the
risk premium. CE is a monotone transform of expected utility, so ranking candidates by CE gives the
same choice. It is computed with logsumexp so that large losses cannot overflow.

Candidates. Q runs over 0, step, 2*step, ... up to an upper bound. If max_qty is None the bound is
max(0, ceil(q - current_stock)), where q is the smallest demand value with cumulative probability
>= 1 - 1e-4. Beyond that point extra units cannot measurably add sales (demand almost never reaches
them) and only add holding or spoilage cost, so the optimum cannot lie beyond it. (Exception: with
holding_cost = 0 and no spoilage, extra units cost nothing; then the optimum can sit at the bound,
which at_upper_bound reports.)

Economic review period. economic_review_period gives the order cycle length of the Economic Order
Quantity model: ordering every R days, the fixed order cost per day is order_cost / R and the
average holding cost per day is holding_cost * mean_daily_demand * R / 2. Their sum is smallest at
    R* = sqrt(2 * order_cost / (holding_cost * mean_daily_demand)).
R is rounded, clipped to [min_days, max_days], and for perishables capped at shelf_life_days - 1
(at least 1) so new stock has at least one day left to sell after the horizon.

Computation is vectorized with numpy over a (candidate Q) x (demand value) matrix.

Assumptions and limitations (for the report):
  - This is a SINGLE-DECISION model, applied again every day by the agent. With order_cost > 0 the
    best multi-period policy would batch orders by looking ahead to future order costs; this model
    only charges the cost of today's order. It is a sensible heuristic, not a proven optimal
    multi-period policy.
  - Spoilage assumes first in, first out, treats units already on hand as fresh (ages are not
    tracked), and approximates demand after the horizon by its average (see "Spoilage" above).
  - The order is assumed to arrive within the horizon and to cover the whole horizon.
  - Demand above the distribution's largest value is treated as impossible.
  - With exponential utility, a risk-averse owner may order MORE or LESS than a risk-neutral one.
    Risk aversion penalizes the spread of profit. When large shortage penalties dominate the bad
    outcomes, ordering more reduces the spread and a risk-averse owner orders more. When leftover
    costs (spoilage, holding) dominate, ordering less reduces the spread and they order less.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

UPPER_BOUND_QUANTILE = 1 - 1e-4
PROBABILITY_SUM_TOLERANCE = 1e-6
_TIE_TOLERANCE = 1e-12


def _parse_distribution(demand_distribution: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sorted integer demand values and their probabilities. Accepts JSON-style string keys."""
    if not demand_distribution:
        raise ValueError("demand_distribution must be non-empty")
    parsed: dict[int, float] = {}
    for key, prob in demand_distribution.items():
        try:
            demand = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"demand_distribution key {key!r} is not an integer") from None
        if isinstance(key, float) and key != demand:
            raise ValueError(f"demand_distribution key {key!r} is not an integer")
        if demand < 0:
            raise ValueError(f"demand_distribution key {key!r} is negative")
        if demand in parsed:
            raise ValueError(f"demand_distribution has duplicate key {demand} after conversion to int")
        prob = float(prob)
        if not prob >= 0:
            raise ValueError(f"probability for demand {demand} must be >= 0, got {prob}")
        parsed[demand] = prob
    total = sum(parsed.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"demand_distribution probabilities must sum to 1, got {total}")
    demands = np.array(sorted(parsed), dtype=np.int64)
    probs = np.array([parsed[d] for d in demands], dtype=float)
    return demands, probs


def _auto_upper_bound(demands: np.ndarray, probs: np.ndarray, current_stock: int) -> int:
    cdf = np.cumsum(probs) / probs.sum()
    index = min(int(np.searchsorted(cdf, UPPER_BOUND_QUANTILE - 1e-12)), len(demands) - 1)
    return max(0, math.ceil(int(demands[index]) - current_stock))


def _spoiled(
    leftover: np.ndarray, shelf_life_days: int | None, review_period_days: int, mean_daily: float
) -> np.ndarray:
    """Units of leftover that spoil (see "Spoilage" in the module docstring)."""
    if shelf_life_days is None:
        return np.zeros_like(leftover)
    remaining_life = shelf_life_days - review_period_days
    if remaining_life <= 0:
        return leftover
    sellable = math.ceil(mean_daily * remaining_life)
    return np.maximum(0.0, leftover - sellable)


def _profit_matrix(
    quantities: np.ndarray,
    demands: np.ndarray,
    current_stock: int,
    unit_cost: float,
    price: float,
    holding_cost: float,
    stockout_penalty: float,
    horizon_days: int,
    spoilage_cost: float,
    order_cost: float,
    shelf_life_days: int | None,
    review_period_days: int,
    mean_daily: float,
) -> np.ndarray:
    """profit[i, j] for order quantities[i] and demand demands[j] (the profit model above)."""
    available = (current_stock + quantities)[:, None].astype(float)
    d = demands[None, :].astype(float)
    sold = np.minimum(available, d)
    leftover = np.maximum(0.0, available - d)
    shortage = np.maximum(0.0, d - available)
    spoiled = _spoiled(leftover, shelf_life_days, review_period_days, mean_daily)
    fixed = np.where(quantities > 0, order_cost, 0.0)[:, None]
    return (
        (price - unit_cost) * sold
        - holding_cost * horizon_days * (available + leftover) / 2
        - stockout_penalty * shortage
        - (unit_cost + spoilage_cost) * spoiled
        - fixed
    )


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
    """Choose the order quantity that maximizes expected utility of profit over the order horizon.

    See the module docstring for the full profit model, utility functions and limitations.

    Assumptions:
        - current_stock is the inventory position (on hand + on order).
        - demand_distribution is order_horizon_distribution from calculate_stockout_risk and
          horizon_days is order_horizon_days. Keys may be ints or strings of ints (from JSON) and
          need not start at 0. Demand above the largest key is treated as impossible.
        - Single-decision model applied daily: with order_cost > 0 this is a heuristic, not an
          optimal multi-period policy.
        - Spoilage: the new order has been on the shelf for review_period_days at the end of the
          horizon. If shelf_life_days - review_period_days <= 0 all leftover spoils; otherwise
          leftover beyond ceil(mean_daily * remaining_life) spoils. First in, first out; units on
          hand treated as fresh; future demand approximated by its average.
        - With max_qty None, candidates stop at the demand value with cumulative probability
          >= 1 - 1e-4 (minus current_stock); ordering beyond it only adds cost.

    Args:
        current_stock: Inventory position. Must be >= 0.
        demand_distribution: Demand over the horizon -> probability. Non-empty, probabilities
            non-negative and summing to 1 within 1e-6.
        unit_cost: Purchase cost per unit. Must be >= 0.
        price: Selling price per unit. Must be >= 0.
        holding_cost: Holding cost per unit per day. Must be >= 0.
        stockout_penalty: Goodwill cost per unit short, on top of the lost margin. Must be >= 0.
        horizon_days: Order horizon in days. Must be >= 1.
        spoilage_cost: Extra disposal cost per spoiled unit. Must be >= 0.
        order_cost: Fixed cost of placing an order, charged only when Q > 0. Must be >= 0.
        shelf_life_days: Shelf life, or None if not perishable.
        review_period_days: Days until the next order can arrive (use the same value given to
            calculate_stockout_risk). Must be >= 1 and <= horizon_days.
        risk_tolerance: None for linear utility; otherwise R > 0 for U = 1 - exp(-profit / R).
        max_qty: Largest order considered, or None for the automatic bound. Must be >= 0.
        step: Candidate spacing; candidates are 0, step, 2*step, ... Must be >= 1.

    Returns:
        Dict with optimal_qty (int), expected_utility (float), expected_profit (float),
        certainty_equivalent (float), no_stockout_probability (float), at_upper_bound (bool),
        utility_type ("linear" or "exponential") and candidates ({Q: {"expected_utility",
        "expected_profit", "certainty_equivalent"}}).

    Raises:
        ValueError: On invalid arguments or an invalid demand distribution.
    """
    if current_stock < 0:
        raise ValueError(f"current_stock must be >= 0, got {current_stock}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if not 1 <= review_period_days <= horizon_days:
        raise ValueError(
            f"review_period_days must be between 1 and horizon_days ({horizon_days}), got {review_period_days}"
        )
    for name, value in (
        ("unit_cost", unit_cost),
        ("price", price),
        ("holding_cost", holding_cost),
        ("stockout_penalty", stockout_penalty),
        ("spoilage_cost", spoilage_cost),
        ("order_cost", order_cost),
    ):
        if not value >= 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if risk_tolerance is not None and not risk_tolerance > 0:
        raise ValueError(f"risk_tolerance must be positive or None, got {risk_tolerance}")
    if max_qty is not None and max_qty < 0:
        raise ValueError(f"max_qty must be >= 0, got {max_qty}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    demands, probs = _parse_distribution(demand_distribution)
    upper = int(max_qty) if max_qty is not None else _auto_upper_bound(demands, probs, current_stock)
    quantities = np.arange(0, upper + 1, step, dtype=np.int64)

    mean_daily = float(demands @ probs) / horizon_days
    profit = _profit_matrix(
        quantities, demands, current_stock, unit_cost, price, holding_cost, stockout_penalty,
        horizon_days, spoilage_cost, order_cost, shelf_life_days, review_period_days, mean_daily,
    )
    expected_profit = profit @ probs

    if risk_tolerance is None:
        utility_type = "linear"
        expected_utility = expected_profit
        certainty_equivalent = expected_profit
    else:
        utility_type = "exponential"
        # log E[exp(-profit / R)], stable for large losses.
        log_mean = logsumexp(-profit / risk_tolerance, b=probs[None, :], axis=1)
        certainty_equivalent = -risk_tolerance * log_mean
        with np.errstate(over="ignore"):
            expected_utility = 1.0 - np.exp(log_mean)

    # Rank by certainty equivalent (same order as expected utility, never overflows); ties -> smaller Q.
    best_value = certainty_equivalent.max()
    best = int(np.argmax(certainty_equivalent >= best_value - _TIE_TOLERANCE * max(1.0, abs(best_value))))
    optimal_qty = int(quantities[best])

    no_stockout_probability = float(probs[demands <= current_stock + optimal_qty].sum())
    is_largest = best == len(quantities) - 1
    at_upper_bound = bool(is_largest and (max_qty is not None or optimal_qty > 0))

    candidates = {
        int(q): {
            "expected_utility": float(expected_utility[i]),
            "expected_profit": float(expected_profit[i]),
            "certainty_equivalent": float(certainty_equivalent[i]),
        }
        for i, q in enumerate(quantities)
    }

    return {
        "optimal_qty": optimal_qty,
        "expected_utility": float(expected_utility[best]),
        "expected_profit": float(expected_profit[best]),
        "certainty_equivalent": float(certainty_equivalent[best]),
        "no_stockout_probability": min(1.0, no_stockout_probability),
        "at_upper_bound": at_upper_bound,
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
    """Economic order cycle length in days (the Economic Order Quantity cycle).

    Ordering every R days costs order_cost / R per day in fixed order costs and about
    holding_cost * mean_daily_demand * R / 2 per day in holding. The total is smallest at
        R* = sqrt(2 * order_cost / (holding_cost * mean_daily_demand)).

    Assumptions:
        - Demand is steady at mean_daily_demand; the classic EOQ trade-off ignores stockouts and
          demand variability, which calculate_optimal_order_quantity handles for each order.
        - The result is rounded to whole days and clipped to [min_days, max_days].
        - order_cost == 0 gives min_days (ordering is free, so order as often as allowed).
          holding_cost == 0 or mean_daily_demand == 0 gives max_days (holding is free, or there is
          nothing to hold).
        - For perishables the result is also capped at max(1, shelf_life_days - 1), so stock that
          arrives has at least one day left to sell after the horizon. This cap wins over min_days.

    Args:
        mean_daily_demand: Average units demanded per day. Must be >= 0.
        holding_cost: Holding cost per unit per day. Must be >= 0.
        order_cost: Fixed cost per order. Must be >= 0.
        shelf_life_days: Shelf life, or None if not perishable. Must be >= 1 if given.
        min_days: Shortest allowed cycle. Must be >= 1.
        max_days: Longest allowed cycle. Must be >= min_days.

    Returns:
        Review period in days (int).

    Raises:
        ValueError: On negative inputs, min_days < 1, max_days < min_days or shelf_life_days < 1.
    """
    for name, value in (
        ("mean_daily_demand", mean_daily_demand),
        ("holding_cost", holding_cost),
        ("order_cost", order_cost),
    ):
        if not value >= 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if min_days < 1:
        raise ValueError(f"min_days must be >= 1, got {min_days}")
    if max_days < min_days:
        raise ValueError(f"max_days must be >= min_days, got {max_days} < {min_days}")
    if shelf_life_days is not None and shelf_life_days < 1:
        raise ValueError(f"shelf_life_days must be >= 1, got {shelf_life_days}")

    if order_cost == 0:
        days = min_days
    elif holding_cost == 0 or mean_daily_demand == 0:
        days = max_days
    else:
        raw = math.sqrt(2 * order_cost / (holding_cost * mean_daily_demand))
        days = min(max_days, max(min_days, round(raw)))
    if shelf_life_days is not None:
        days = min(days, max(1, int(shelf_life_days) - 1))
    return int(days)
