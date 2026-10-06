"""Tests for decision_engine.recommend and the economic review period."""

import inspect
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from decision_engine import mock, optimizer, recommend
from decision_engine.bayes_risk import calculate_stockout_risk
from decision_engine.data_generator import generate_sales_history
from decision_engine.optimizer import calculate_optimal_order_quantity, economic_review_period
from decision_engine.recommend import recommend_order
from tests.contract import MONDAY, SATURDAY, assert_order_structure, assert_risk_structure

PRODUCTS_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"
PRODUCT_IDS = range(1, 11)


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


@pytest.fixture(scope="module")
def history(products_df) -> pd.DataFrame:
    return generate_sales_history(products_df, 180, 42)


def row(products_df, product_id) -> pd.Series:
    return products_df.set_index("product_id", drop=False).loc[product_id]


def manual(product, current_stock, current_date, history, review, promo_dates=(), order_cost=30.0,
           spoilage_cost=0.0, risk_tolerance=None) -> dict:
    shelf = None if pd.isna(product["shelf_life_days"]) else int(product["shelf_life_days"])
    risk = calculate_stockout_risk(
        int(product["product_id"]), current_stock, int(product["lead_time_days"]), current_date,
        history, review_period_days=review, promo_dates=promo_dates,
    )
    order = calculate_optimal_order_quantity(
        current_stock, risk["order_horizon_distribution"], float(product["unit_cost"]),
        float(product["selling_price"]), float(product["holding_cost_per_day"]),
        float(product["stockout_penalty"]), risk["order_horizon_days"], spoilage_cost=spoilage_cost,
        order_cost=order_cost, shelf_life_days=shelf, review_period_days=review,
        risk_tolerance=risk_tolerance,
    )
    wait_risk = calculate_stockout_risk(
        int(product["product_id"]), current_stock, int(product["lead_time_days"]), current_date,
        history, review_period_days=1, promo_dates=promo_dates,
    )

    def shortage(dist):
        return sum(p * (d - current_stock) for d, p in dist.items() if d > current_stock)

    extra = shortage(wait_risk["order_horizon_distribution"]) - shortage(wait_risk["lead_time_distribution"])
    unit_loss = float(product["selling_price"]) - float(product["unit_cost"]) + float(product["stockout_penalty"])
    early = float(product["holding_cost_per_day"]) * order["optimal_qty"]
    should = order["optimal_qty"] > 0 and unit_loss * extra > early
    return {
        "product_id": int(product["product_id"]),
        "review_period_days": review,
        "recommended_qty": order["optimal_qty"] if should else 0,
        "should_order": should,
        "wait_cost": unit_loss * extra,
        "early_holding_cost": early,
        "risk": risk,
        "order": order,
        "wait_risk": wait_risk,
    }


RECOMMEND_KEYS = {
    "product_id", "review_period_days", "recommended_qty", "should_order", "wait_cost",
    "early_holding_cost", "risk", "order", "wait_risk",
}


def assert_matches(result: dict, expected: dict) -> None:
    assert set(result) == RECOMMEND_KEYS
    for key in ("product_id", "review_period_days", "recommended_qty", "should_order", "risk", "order", "wait_risk"):
        assert result[key] == expected[key], key
    assert result["wait_cost"] == pytest.approx(expected["wait_cost"], abs=1e-9)
    assert result["early_holding_cost"] == pytest.approx(expected["early_holding_cost"], abs=1e-12)


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_matches_manual_calls_with_economic_review_period(products_df, history, product_id):
    product = row(products_df, product_id)
    promo = [SATURDAY, SATURDAY + timedelta(days=1)]
    result = recommend_order(product, 10, SATURDAY, history, promo_dates=promo, spoilage_cost=1.0,
                             risk_tolerance=1000.0)

    mean_daily = float(history.loc[history["product_id"] == product_id, "units_sold"].mean())
    shelf = None if pd.isna(product["shelf_life_days"]) else int(product["shelf_life_days"])
    review = economic_review_period(mean_daily, float(product["holding_cost_per_day"]), 30.0,
                                    shelf_life_days=shelf)
    assert result["review_period_days"] == review
    assert_matches(result, manual(product, 10, SATURDAY, history, review, promo_dates=promo,
                                  spoilage_cost=1.0, risk_tolerance=1000.0))
    assert type(result["product_id"]) is int and type(result["review_period_days"]) is int
    assert type(result["recommended_qty"]) is int and type(result["should_order"]) is bool
    assert type(result["wait_cost"]) is float and type(result["early_holding_cost"]) is float
    assert_risk_structure(result["wait_risk"], product_id, int(product["lead_time_days"]),
                          int(product["lead_time_days"]) + 1)
    lead = int(product["lead_time_days"])
    assert_risk_structure(result["risk"], product_id, lead, lead + review)
    assert_order_structure(result["order"], "exponential")


@pytest.mark.parametrize("product_id", [1, 5])
def test_explicit_review_period_and_dict_product(products_df, history, product_id):
    product = row(products_df, product_id).to_dict()
    result = recommend_order(product, 0, MONDAY, history, review_period_days=3, order_cost=0.0)
    assert result["review_period_days"] == 3
    assert_matches(result, manual(product, 0, MONDAY, history, 3, order_cost=0.0))


def test_no_history_uses_default_demand(products_df, history):
    product = row(products_df, 5)
    result = recommend_order(product, 0, MONDAY, history.iloc[0:0])
    expected = economic_review_period(8.0, float(product["holding_cost_per_day"]), 30.0)
    assert result["review_period_days"] == expected


def test_missing_product_field(products_df, history):
    product = row(products_df, 1).drop("lead_time_days")
    with pytest.raises(ValueError):
        recommend_order(product, 0, MONDAY, history)


def test_milk_no_longer_understocked(products_df, history):
    """Fix 1: with the new spoilage rule, milk's leftover can still sell on its last shelf day,
    so the recommended order covers far more outcomes than the old ~0.25."""
    product = row(products_df, 1)
    lead = int(product["lead_time_days"])
    stock = round(calculate_stockout_risk(1, 0, lead, MONDAY, history)["expected_lead_time_demand"])
    result = recommend_order(product, stock, MONDAY, history)
    assert result["review_period_days"] == 1
    assert result["order"]["no_stockout_probability"] > 0.6


# When to order


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_no_order_when_stock_far_exceeds_near_term_demand(products_df, history, product_id):
    product = row(products_df, product_id)
    lead = int(product["lead_time_days"])
    near_term = calculate_stockout_risk(product_id, 0, lead, MONDAY, history, review_period_days=1)
    stock = 3 * max(near_term["order_horizon_distribution"]) + 10
    result = recommend_order(product, stock, MONDAY, history)
    assert not result["should_order"] and result["recommended_qty"] == 0
    assert result["wait_cost"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_orders_full_quantity_when_out_of_stock(products_df, history, product_id):
    result = recommend_order(row(products_df, product_id), 0, MONDAY, history)
    assert result["order"]["optimal_qty"] > 0
    assert result["should_order"]
    assert result["recommended_qty"] == result["order"]["optimal_qty"]
    assert result["wait_cost"] > result["early_holding_cost"]


@pytest.mark.parametrize("product_id", [1, 4, 5, 9])
def test_recommended_qty_is_zero_whenever_not_ordering(products_df, history, product_id):
    product = row(products_df, product_id)
    seen = set()
    for stock in range(0, 400, 7):
        result = recommend_order(product, stock, SATURDAY, history, promo_dates=[SATURDAY])
        seen.add(result["should_order"])
        if result["should_order"]:
            assert result["recommended_qty"] == result["order"]["optimal_qty"] > 0
            assert result["wait_cost"] > result["early_holding_cost"]
        else:
            assert result["recommended_qty"] == 0
    assert seen == {True, False}


def test_defaulted_arguments_are_keyword_only(products_df, history):
    with pytest.raises(TypeError):
        recommend_order(row(products_df, 1), 0, MONDAY, history, ())


@pytest.mark.parametrize(
    "real, fake",
    [
        (recommend.recommend_order, mock.recommend_order),
        (optimizer.economic_review_period, mock.economic_review_period),
    ],
)
def test_matches_mock_signatures(real, fake):
    assert inspect.signature(real) == inspect.signature(fake)


def test_mock_recommend_order_structure(products_df):
    mock_history = mock.generate_sales_history(products_df, days=30, seed=1)
    product = row(products_df, 2)
    result = mock.recommend_order(product, 5, SATURDAY, mock_history, promo_dates=[SATURDAY])
    assert set(result) == RECOMMEND_KEYS
    lead = int(product["lead_time_days"])
    assert_risk_structure(result["risk"], 2, lead, lead + result["review_period_days"])
    assert_risk_structure(result["wait_risk"], 2, lead, lead + 1)
    assert_order_structure(result["order"], "linear")
    assert result["recommended_qty"] == (result["order"]["optimal_qty"] if result["should_order"] else 0)
    empty = mock.recommend_order(product, 0, SATURDAY, mock_history)
    assert empty["should_order"] and empty["recommended_qty"] > 0


@pytest.mark.parametrize("args, kwargs", [((10.0, 0.1, 30.0), {}), ((10.0, 0.0, 30.0), {}),
                                          ((10.0, 0.1, 0.0), {}), ((30.0, 0.1, 30.0), {"shelf_life_days": 2})])
def test_mock_economic_review_period_matches_real(args, kwargs):
    assert mock.economic_review_period(*args, **kwargs) == economic_review_period(*args, **kwargs)
