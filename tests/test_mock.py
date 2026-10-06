"""Contract tests for decision_engine.mock: documented keys, types and valid distributions."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from decision_engine import mock as engine
from tests.contract import (
    MONDAY,
    SATURDAY,
    assert_distribution,
    assert_evaluation_structure,
    assert_order_structure,
    assert_risk_structure,
)

PRODUCTS_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"
TOL = 1e-6


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


@pytest.fixture(scope="module")
def history_df(products_df: pd.DataFrame) -> pd.DataFrame:
    return engine.generate_sales_history(products_df, days=30, seed=42)


def risk(history_df: pd.DataFrame, **overrides) -> dict:
    args = dict(
        product_id=1,
        current_stock=20,
        lead_time_days=1,
        current_date=MONDAY,
        history_df=history_df,
    )
    args.update(overrides)
    return engine.calculate_stockout_risk(**args)


def order(demand_distribution: dict[int, float], **overrides) -> dict:
    args = dict(
        current_stock=10,
        demand_distribution=demand_distribution,
        unit_cost=25.0,
        price=28.0,
        holding_cost=0.10,
        stockout_penalty=5.0,
        horizon_days=2,
    )
    args.update(overrides)
    return engine.calculate_optimal_order_quantity(**args)


def assert_sales_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == ["date", "product_id", "units_sold", "is_weekend", "has_promo"]
    for row in df.itertuples(index=False):
        assert type(row.date) is date
        assert isinstance(row.product_id, int)
        assert isinstance(row.units_sold, int) and row.units_sold >= 0
        assert isinstance(row.is_weekend, bool)
        assert isinstance(row.has_promo, bool)


# generate_sales_history


def test_generate_sales_history_schema(products_df, history_df):
    assert len(history_df) == 30 * len(products_df)
    assert_sales_schema(history_df)
    assert history_df["date"].min() == date(2025, 1, 1)


def test_generate_sales_history_is_reproducible(products_df):
    a = engine.generate_sales_history(products_df, days=7, seed=1, start_date=date(2025, 6, 1))
    b = engine.generate_sales_history(products_df, days=7, seed=1, start_date=date(2025, 6, 1))
    pd.testing.assert_frame_equal(a, b)
    assert a["date"].min() == date(2025, 6, 1)


# simulate_one_day


def test_simulate_one_day_schema(products_df):
    today = date(2025, 3, 8)  # Saturday
    df = engine.simulate_one_day(products_df, today, {1: True}, seed=7)
    assert len(df) == len(products_df)
    assert_sales_schema(df)
    assert (df["date"] == today).all()
    assert df["is_weekend"].all()
    promo = dict(zip(df["product_id"], df["has_promo"]))
    assert promo[1]
    assert not any(v for pid, v in promo.items() if pid != 1)


# calculate_stockout_risk


def test_stockout_risk_keys_and_types(history_df):
    result = risk(history_df, lead_time_days=3, review_period_days=2)
    assert_risk_structure(result, product_id=1, lead_time_days=3, horizon_days=5)


@pytest.mark.parametrize("start", [MONDAY, SATURDAY])
@pytest.mark.parametrize("with_promo", [False, True])
def test_stockout_risk_distributions_sum_to_one(history_df, start, with_promo):
    promo_dates = [start] if with_promo else []
    result = risk(history_df, current_date=start, lead_time_days=2, promo_dates=promo_dates)
    assert_risk_structure(result, product_id=1, lead_time_days=2, horizon_days=3)


def test_stockout_risk_responds_to_inputs(history_df):
    base = risk(history_df)["stockout_risk"]
    assert risk(history_df, promo_dates=[MONDAY])["stockout_risk"] > base
    assert risk(history_df, current_date=SATURDAY)["stockout_risk"] > base
    assert risk(history_df, current_stock=60)["stockout_risk"] < base


def test_stockout_risk_without_history(history_df):
    result = risk(history_df.iloc[0:0])
    assert_distribution(result["order_horizon_distribution"])


def test_stockout_risk_rejects_old_evidence_arguments(history_df):
    with pytest.raises(TypeError):
        risk(history_df, is_weekend=True)


# calculate_optimal_order_quantity


@pytest.mark.parametrize("risk_tolerance, utility_type", [(None, "linear"), (500.0, "exponential")])
def test_order_quantity_keys_and_types(history_df, risk_tolerance, utility_type):
    r = risk(history_df)
    result = order(
        r["order_horizon_distribution"],
        horizon_days=r["order_horizon_days"],
        risk_tolerance=risk_tolerance,
        shelf_life_days=2,
    )
    assert_order_structure(result, utility_type)


@pytest.mark.parametrize("risk_tolerance, utility_type", [(None, "linear"), (500.0, "exponential")])
def test_order_quantity_with_order_cost(history_df, risk_tolerance, utility_type):
    r = risk(history_df)
    dist = r["order_horizon_distribution"]
    free = order(dist, horizon_days=r["order_horizon_days"], risk_tolerance=risk_tolerance)
    costly = order(
        dist, horizon_days=r["order_horizon_days"], risk_tolerance=risk_tolerance, order_cost=40.0
    )
    assert_order_structure(costly, utility_type)
    # The fixed cost applies to every Q > 0 and never to Q = 0.
    assert costly["candidates"][0] == free["candidates"][0]
    for q in free["candidates"]:
        if q > 0:
            diff = free["candidates"][q]["expected_profit"] - costly["candidates"][q]["expected_profit"]
            assert diff == pytest.approx(40.0)


def test_order_quantity_rejects_negative_order_cost():
    with pytest.raises(ValueError):
        order({10: 1.0}, order_cost=-1.0)


def test_order_quantity_rises_with_demand(history_df):
    low = risk(history_df)
    high = risk(history_df, current_date=SATURDAY, promo_dates=[SATURDAY], lead_time_days=3)
    q_low = order(low["order_horizon_distribution"])["optimal_qty"]
    q_high = order(high["order_horizon_distribution"])["optimal_qty"]
    assert q_high > q_low


# run_evaluation


@pytest.mark.parametrize("use_policies", [False, True])
def test_run_evaluation_structure(products_df, use_policies):
    policies = None
    if use_policies:
        policies = {
            int(pid): {
                "is_perishable": not pd.isna(shelf),
                "spoilage_cost": 1.0,
                "shelf_life_days": None if pd.isna(shelf) else int(shelf),
            }
            for pid, shelf in zip(products_df["product_id"], products_df["shelf_life_days"])
        }
    df = engine.run_evaluation(products_df, days=30, seed=3, policies=policies)
    assert_evaluation_structure(df, list(products_df["product_id"]), ["baseline", "dss", "dss_no_evidence"])

    non_perishable = products_df.loc[products_df["shelf_life_days"].isna(), "product_id"]
    spoiled = df[df["product_id"].isin(list(non_perishable))]["units_spoiled"]
    assert (spoiled == 0).all()


def test_run_evaluation_order_cost_reduces_profit(products_df):
    free = engine.run_evaluation(products_df, days=30, seed=3, order_cost=0.0)
    costly = engine.run_evaluation(products_df, days=30, seed=3, order_cost=30.0)
    expected_drop = 30.0 * free["orders_placed"]
    actual_drop = free["total_profit"] - costly["total_profit"]
    assert actual_drop.to_numpy() == pytest.approx(expected_drop.to_numpy().astype(float), abs=0.02)
    with pytest.raises(ValueError):
        engine.run_evaluation(products_df, days=30, seed=3, order_cost=-1.0)


@pytest.mark.parametrize("history_days", [0, -5])
def test_run_evaluation_rejects_non_positive_history_days(products_df, history_days):
    with pytest.raises(ValueError):
        engine.run_evaluation(products_df, days=30, seed=3, history_days=history_days)


def test_run_evaluation_accepts_history_days(products_df):
    df = engine.run_evaluation(products_df, days=30, seed=3, history_days=60)
    assert len(df) == 3 * (len(products_df) + 1)


def test_run_evaluation_without_ablation(products_df):
    df = engine.run_evaluation(products_df, days=30, seed=3, include_ablation=False)
    assert_evaluation_structure(df, list(products_df["product_id"]), ["baseline", "dss"])


@pytest.mark.parametrize(
    "kwargs",
    [{"risk_tolerance": 0.0}, {"review_period_overrides": {1: 0}}, {"review_period_overrides": {1: 2.5}}],
)
def test_run_evaluation_rejects_invalid_new_arguments(products_df, kwargs):
    with pytest.raises(ValueError):
        engine.run_evaluation(products_df, days=30, seed=3, **kwargs)


def test_run_evaluation_defaulted_args_are_keyword_only(products_df):
    with pytest.raises(TypeError):
        engine.run_evaluation(products_df, 30, 1, {})
