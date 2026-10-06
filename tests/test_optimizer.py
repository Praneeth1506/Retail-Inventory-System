"""Tests for decision_engine.optimizer."""

import inspect
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_engine import mock, optimizer
from decision_engine.bayes_risk import calculate_stockout_risk
from decision_engine.data_generator import generate_sales_history
from decision_engine.optimizer import calculate_optimal_order_quantity, economic_review_period
from tests.contract import MONDAY, SATURDAY, assert_order_structure

PRODUCTS_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"
PRODUCT_IDS = range(1, 11)


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


@pytest.fixture(scope="module")
def history(products_df) -> pd.DataFrame:
    return generate_sales_history(products_df, 180, 42)


def product(products_df: pd.DataFrame, product_id: int) -> dict:
    row = products_df.set_index("product_id").loc[product_id]
    shelf = row["shelf_life_days"]
    return {
        "lead_time_days": int(row["lead_time_days"]),
        "costs": {
            "unit_cost": float(row["unit_cost"]),
            "price": float(row["selling_price"]),
            "holding_cost": float(row["holding_cost_per_day"]),
            "stockout_penalty": float(row["stockout_penalty"]),
        },
        "shelf_life_days": None if pd.isna(shelf) else int(shelf),
    }


def window(history, products_df, product_id, start=MONDAY, promo=False) -> dict:
    lead = product(products_df, product_id)["lead_time_days"]
    promo_dates = [start + timedelta(days=i) for i in range(lead + 1)] if promo else ()
    return calculate_stockout_risk(product_id, 0, lead, start, history, promo_dates=promo_dates)


def order(risk_result: dict, costs: dict, current_stock: int = 0, **kwargs) -> dict:
    return calculate_optimal_order_quantity(
        current_stock,
        risk_result["order_horizon_distribution"],
        horizon_days=risk_result["order_horizon_days"],
        **costs,
        **kwargs,
    )


def quantile(dist: dict[int, float], q: float) -> int:
    keys = sorted(dist)
    cdf = np.cumsum([dist[k] for k in keys])
    return int(keys[np.searchsorted(cdf, q)])


# Structure and interface


@pytest.mark.parametrize("risk_tolerance, utility_type", [(None, "linear"), (300.0, "exponential")])
@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_structure(products_df, history, product_id, risk_tolerance, utility_type):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    result = order(r, p["costs"], current_stock=5, order_cost=30.0,
                   shelf_life_days=p["shelf_life_days"], risk_tolerance=risk_tolerance)
    assert_order_structure(result, utility_type)
    assert list(result["candidates"]) == list(range(0, max(result["candidates"]) + 1))


def test_matches_mock_signature():
    assert inspect.signature(optimizer.calculate_optimal_order_quantity) == inspect.signature(
        mock.calculate_optimal_order_quantity
    )


def test_defaulted_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        calculate_optimal_order_quantity(0, {5: 1.0}, 10.0, 12.0, 0.1, 3.0, 2, 1.0)


def test_step_controls_candidates():
    result = calculate_optimal_order_quantity(0, {20: 1.0}, 10.0, 12.0, 0.1, 3.0, 2, step=5, max_qty=23)
    assert list(result["candidates"]) == [0, 5, 10, 15, 20]


def test_string_keys_accepted(products_df, history):
    p = product(products_df, 1)
    r = window(history, products_df, 1)
    as_ints = order(r, p["costs"], order_cost=30.0)
    as_strings = calculate_optimal_order_quantity(
        0, {str(k): v for k, v in r["order_horizon_distribution"].items()}, **p["costs"],
        horizon_days=r["order_horizon_days"], order_cost=30.0,
    )
    assert as_strings == as_ints


@pytest.mark.parametrize(
    "overrides",
    [
        {"current_stock": -1},
        {"horizon_days": 0},
        {"unit_cost": -1.0},
        {"price": -1.0},
        {"holding_cost": -0.1},
        {"stockout_penalty": -1.0},
        {"spoilage_cost": -1.0},
        {"order_cost": -1.0},
        {"risk_tolerance": 0.0},
        {"risk_tolerance": -5.0},
        {"step": 0},
        {"review_period_days": 0},
        {"review_period_days": 3},
        {"max_qty": -1},
        {"demand_distribution": {}},
        {"demand_distribution": {5: 0.5, 6: 0.4}},
        {"demand_distribution": {5: 1.2, 6: -0.2}},
        {"demand_distribution": {-1: 0.5, 6: 0.5}},
        {"demand_distribution": {"five": 1.0}},
        {"demand_distribution": {5: 0.5, "5": 0.5}},
    ],
)
def test_rejects_invalid_arguments(overrides):
    args = dict(current_stock=0, demand_distribution={5: 0.5, 6: 0.5}, unit_cost=10.0, price=12.0,
                holding_cost=0.1, stockout_penalty=3.0, horizon_days=2)
    args.update(overrides)
    with pytest.raises(ValueError):
        calculate_optimal_order_quantity(**args)


# Correctness


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_newsvendor_critical_fractile(products_df, history, product_id):
    """With no holding/order cost and all leftover spoiling (shelf life = review period, so no
    life remains after the horizon), the optimum is the critical fractile."""
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    spoilage_cost = 2.0
    costs = {**p["costs"], "holding_cost": 0.0}
    result = order(r, costs, current_stock=0, order_cost=0.0, step=1, spoilage_cost=spoilage_cost,
                   shelf_life_days=1, review_period_days=1)

    cu = (costs["price"] - costs["unit_cost"]) + costs["stockout_penalty"]
    co = costs["unit_cost"] + spoilage_cost
    expected = quantile(r["order_horizon_distribution"], cu / (cu + co))
    assert result["optimal_qty"] == expected
    assert not result["at_upper_bound"]


@pytest.mark.parametrize("shelf_life_days", [None, 1, 2, 3, 5])
@pytest.mark.parametrize("risk_tolerance", [None, 15.0])
def test_matches_brute_force(shelf_life_days, risk_tolerance):
    dist = {3: 0.2, 5: 0.45, 9: 0.3, 12: 0.05}
    params = dict(unit_cost=10.0, price=14.0, holding_cost=0.3, stockout_penalty=6.0)
    stock, horizon, review, spoilage_cost, order_cost = 2, 3, 2, 1.5, 4.0
    result = calculate_optimal_order_quantity(
        stock, dist, **params, horizon_days=horizon, spoilage_cost=spoilage_cost,
        order_cost=order_cost, shelf_life_days=shelf_life_days, review_period_days=review,
        risk_tolerance=risk_tolerance,
    )
    mean_daily = sum(d * prob for d, prob in dist.items()) / horizon
    assert list(result["candidates"]) == list(range(0, 12 - stock + 1))

    for q, c in result["candidates"].items():
        profits = []
        for d, prob in dist.items():
            available = stock + q
            sold = min(available, d)
            leftover = max(0, available - d)
            shortage = max(0, d - available)
            if shelf_life_days is None:
                spoiled = 0
            elif shelf_life_days - review <= 0:
                spoiled = leftover
            else:
                spoiled = max(0, leftover - math.ceil(mean_daily * (shelf_life_days - review)))
            profit = ((params["price"] - params["unit_cost"]) * sold
                      - params["holding_cost"] * horizon * (available + leftover) / 2
                      - params["stockout_penalty"] * shortage
                      - (params["unit_cost"] + spoilage_cost) * spoiled
                      - (order_cost if q > 0 else 0))
            profits.append((prob, profit))
        expected = sum(prob * profit for prob, profit in profits)
        assert c["expected_profit"] == pytest.approx(expected, abs=1e-9)
        if risk_tolerance is not None:
            mean_exp = sum(prob * math.exp(-profit / risk_tolerance) for prob, profit in profits)
            assert c["expected_utility"] == pytest.approx(1 - mean_exp, abs=1e-9)
            assert c["certainty_equivalent"] == pytest.approx(-risk_tolerance * math.log(mean_exp), abs=1e-9)

    best = max(result["candidates"], key=lambda q: (result["candidates"][q]["expected_utility"], -q))
    assert result["optimal_qty"] == best
    no_stockout = sum(prob for d, prob in dist.items() if d <= stock + best)
    assert result["no_stockout_probability"] == pytest.approx(no_stockout)


def test_ties_choose_smaller_quantity():
    """Zero holding cost and no spoilage: every Q covering the largest demand is equally good."""
    dist = {4: 0.5, 7: 0.5}
    result = calculate_optimal_order_quantity(1, dist, 10.0, 15.0, 0.0, 5.0, 2, max_qty=50)
    assert result["optimal_qty"] == 6
    assert not result["at_upper_bound"]
    auto = calculate_optimal_order_quantity(1, dist, 10.0, 15.0, 0.0, 5.0, 2)
    assert auto["optimal_qty"] == 6 and auto["at_upper_bound"]


# Monotonicity


MONO_PRODUCTS = [1, 4, 5, 8]


@pytest.mark.parametrize("product_id", MONO_PRODUCTS)
def test_demand_shift_up_does_not_reduce_order(products_df, history, product_id):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    qs = []
    for shift in range(0, 25, 3):
        shifted = {k + shift: v for k, v in r["order_horizon_distribution"].items()}
        qs.append(calculate_optimal_order_quantity(
            10, shifted, **p["costs"], horizon_days=r["order_horizon_days"], order_cost=30.0,
            shelf_life_days=p["shelf_life_days"])["optimal_qty"])
    assert qs == sorted(qs) and qs[-1] > qs[0]


@pytest.mark.parametrize("product_id", MONO_PRODUCTS)
def test_higher_stockout_penalty_does_not_reduce_order(products_df, history, product_id):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    qs = [order(r, {**p["costs"], "stockout_penalty": pen}, current_stock=5, order_cost=30.0,
                shelf_life_days=p["shelf_life_days"])["optimal_qty"]
          for pen in (0.0, 2.0, 5.0, 10.0, 25.0, 60.0, 150.0)]
    assert qs == sorted(qs)


@pytest.mark.parametrize("product_id", MONO_PRODUCTS)
def test_lower_holding_cost_does_not_reduce_order(products_df, history, product_id):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    qs = [order(r, {**p["costs"], "holding_cost": h}, current_stock=5, order_cost=30.0)["optimal_qty"]
          for h in (5.0, 2.0, 1.0, 0.5, 0.2, 0.05, 0.01)]
    assert qs == sorted(qs)


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_spoiling_orders_no_more_than_non_perishable(products_df, history, product_id):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    spoiling = order(r, p["costs"], order_cost=30.0, spoilage_cost=1.0,
                     shelf_life_days=r["order_horizon_days"])
    lasting = order(r, p["costs"], order_cost=30.0, spoilage_cost=1.0, shelf_life_days=None)
    assert spoiling["optimal_qty"] <= lasting["optimal_qty"]


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_long_shelf_life_equals_non_perishable(products_df, history, product_id):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    lasting = order(r, p["costs"], current_stock=5, order_cost=30.0, spoilage_cost=2.0)
    long_life = order(r, p["costs"], current_stock=5, order_cost=30.0, spoilage_cost=2.0,
                      shelf_life_days=1000)
    assert long_life == lasting


def test_spoilage_partial_sell_through():
    """remaining_life = 3 - 1 = 2 days at mean 3/day: up to 6 leftover units sell, the rest spoil."""
    dist = {4: 0.5, 8: 0.5}  # horizon 2 days -> mean_daily = 3; ceil(3 * 2) = 6 sellable
    kwargs = dict(unit_cost=10.0, price=12.0, holding_cost=0.0, stockout_penalty=0.0, horizon_days=2,
                  spoilage_cost=1.0, max_qty=0)
    fresh = calculate_optimal_order_quantity(20, dist, **kwargs)
    perishable = calculate_optimal_order_quantity(20, dist, **kwargs, shelf_life_days=3, review_period_days=1)
    # leftover 16 or 12; spoiled 10 or 6 -> expected 8 spoiled units at Rs 11 each.
    assert fresh["expected_profit"] - perishable["expected_profit"] == pytest.approx(8 * 11.0)


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_large_order_cost_with_ample_stock_orders_nothing(products_df, history, product_id):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    ample = quantile(r["order_horizon_distribution"], 0.999)
    result = order(r, p["costs"], current_stock=ample, order_cost=1000.0)
    assert result["optimal_qty"] == 0
    assert not result["at_upper_bound"]


# Exponential utility


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
@pytest.mark.parametrize("risk_tolerance", [20.0, 200.0, 1000.0])
def test_certainty_equivalent_below_expected_profit(products_df, history, product_id, risk_tolerance):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id, start=SATURDAY, promo=True)
    result = order(r, p["costs"], current_stock=5, order_cost=30.0,
                   shelf_life_days=p["shelf_life_days"], risk_tolerance=risk_tolerance)
    assert result["certainty_equivalent"] <= result["expected_profit"] + 1e-9
    for c in result["candidates"].values():
        assert c["certainty_equivalent"] <= c["expected_profit"] + 1e-9 * max(1.0, abs(c["expected_profit"]))


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_large_risk_tolerance_converges_to_linear(products_df, history, product_id):
    p = product(products_df, product_id)
    r = window(history, products_df, product_id)
    kwargs = dict(current_stock=5, order_cost=30.0, shelf_life_days=p["shelf_life_days"])
    linear = order(r, p["costs"], **kwargs)
    nearly = order(r, p["costs"], risk_tolerance=1e7, **kwargs)
    assert nearly["optimal_qty"] == linear["optimal_qty"]
    assert nearly["certainty_equivalent"] == pytest.approx(linear["expected_profit"], rel=1e-4, abs=1e-3)


def test_huge_losses_do_not_overflow():
    result = calculate_optimal_order_quantity(0, {5000: 1.0}, 10.0, 12.0, 0.1, 500.0, 2,
                                              risk_tolerance=1.0, max_qty=10)
    assert math.isfinite(result["certainty_equivalent"])
    assert result["optimal_qty"] == 10 and result["at_upper_bound"]


# Upper bound


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
@pytest.mark.parametrize("promo_weekend", [False, True])
def test_auto_upper_bound_not_reached(products_df, history, product_id, promo_weekend):
    p = product(products_df, product_id)
    start = SATURDAY if promo_weekend else MONDAY
    r = window(history, products_df, product_id, start=start, promo=promo_weekend)
    stock = round(r["expected_lead_time_demand"])
    result = order(r, p["costs"], current_stock=stock, order_cost=30.0,
                   shelf_life_days=p["shelf_life_days"])
    assert not result["at_upper_bound"]
    q = quantile(r["order_horizon_distribution"], 1 - 1e-4)
    assert max(result["candidates"]) == max(0, q - stock)


def test_explicit_max_qty_can_bind(products_df, history):
    p = product(products_df, 1)
    r = window(history, products_df, 1)
    result = order(r, p["costs"], max_qty=3)
    assert result["optimal_qty"] == 3 and result["at_upper_bound"]


# Economic review period


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_economic_review_period_formula(products_df, history, product_id):
    row = products_df.set_index("product_id").loc[product_id]
    mean_daily = float(history.loc[history["product_id"] == product_id, "units_sold"].mean())
    holding = float(row["holding_cost_per_day"])
    expected = min(30, max(1, round(math.sqrt(2 * 30.0 / (holding * mean_daily)))))
    assert economic_review_period(mean_daily, holding, 30.0) == expected
    shelf = None if pd.isna(row["shelf_life_days"]) else int(row["shelf_life_days"])
    capped = economic_review_period(mean_daily, holding, 30.0, shelf_life_days=shelf)
    assert capped == (expected if shelf is None else min(expected, max(1, shelf - 1)))


def test_economic_review_period_perishable_cap_for_milk(products_df, history):
    mean_daily = float(history.loc[history["product_id"] == 1, "units_sold"].mean())
    uncapped = economic_review_period(mean_daily, 0.10, 30.0)
    assert uncapped > 1
    assert economic_review_period(mean_daily, 0.10, 30.0, shelf_life_days=2) == 1
    assert economic_review_period(mean_daily, 0.10, 30.0, shelf_life_days=1) == 1


def test_economic_review_period_edge_cases():
    assert economic_review_period(10.0, 0.5, 0.0) == 1
    assert economic_review_period(10.0, 0.5, 0.0, min_days=3) == 3
    assert economic_review_period(10.0, 0.0, 30.0) == 30
    assert economic_review_period(0.0, 0.5, 30.0, max_days=14) == 14
    assert economic_review_period(1e6, 1e6, 30.0) == 1
    assert economic_review_period(1e-6, 1e-6, 30.0) == 30
    assert type(economic_review_period(10.0, 0.5, 30.0)) is int


@pytest.mark.parametrize(
    "args, kwargs",
    [((-1.0, 0.1, 30.0), {}), ((10.0, -0.1, 30.0), {}), ((10.0, 0.1, -1.0), {}),
     ((10.0, 0.1, 30.0), {"min_days": 0}), ((10.0, 0.1, 30.0), {"min_days": 5, "max_days": 4}),
     ((10.0, 0.1, 30.0), {"shelf_life_days": 0})],
)
def test_economic_review_period_rejects_invalid(args, kwargs):
    with pytest.raises(ValueError):
        economic_review_period(*args, **kwargs)
