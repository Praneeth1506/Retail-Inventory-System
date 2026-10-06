"""Tests for decision_engine.data_generator."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_engine import data_generator
from decision_engine.data_generator import generate_sales_history, simulate_one_day

PRODUCTS_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"
SALES_COLUMNS = ["date", "product_id", "units_sold", "is_weekend", "has_promo"]


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


@pytest.fixture(scope="module")
def history(products_df: pd.DataFrame) -> pd.DataFrame:
    return generate_sales_history(products_df, 180, 42)


def assert_sales_schema(df: pd.DataFrame) -> None:
    assert list(df.columns) == SALES_COLUMNS
    assert df["date"].dtype == object
    assert all(type(d) is date for d in df["date"])
    assert df["product_id"].dtype == np.int64
    assert df["units_sold"].dtype == np.int64
    assert df["is_weekend"].dtype == bool
    assert df["has_promo"].dtype == bool
    assert (df["is_weekend"] == df["date"].map(lambda d: d.weekday() >= 5)).all()


# Schema and validity


def test_history_schema(products_df, history):
    assert len(history) == 180 * len(products_df)
    assert_sales_schema(history)
    assert history["date"].min() == date(2025, 1, 1)
    assert history["date"].max() == date(2025, 1, 1) + timedelta(days=179)
    expected_order = history.sort_values(["date", "product_id"]).index
    assert (history.index == expected_order).all()


def test_simulate_one_day_schema(products_df):
    df = simulate_one_day(products_df, date(2025, 3, 8), {1: True}, 7)
    assert len(df) == len(products_df)
    assert_sales_schema(df)
    assert (df["date"] == date(2025, 3, 8)).all()


def test_no_negative_values(products_df, history):
    assert (history["units_sold"] >= 0).all()
    day = simulate_one_day(products_df, date(2025, 3, 8), {}, 7)
    assert (day["units_sold"] >= 0).all()


# Reproducibility and slicing


def test_same_seed_identical(products_df):
    a = generate_sales_history(products_df, 60, 5)
    b = generate_sales_history(products_df, 60, 5)
    pd.testing.assert_frame_equal(a, b)
    one = simulate_one_day(products_df, date(2025, 3, 8), {2: True}, 9)
    two = simulate_one_day(products_df, date(2025, 3, 8), {2: True}, 9)
    pd.testing.assert_frame_equal(one, two)


def test_different_seed_differs(products_df):
    a = generate_sales_history(products_df, 60, 5)
    b = generate_sales_history(products_df, 60, 6)
    assert not a.equals(b)
    assert not a["has_promo"].equals(b["has_promo"])


def test_split_matches_separate_history(products_df):
    """run_evaluation generates history_days + days once and splits; the history half must
    equal a standalone history, and the simulated half must be a valid frame on its own."""
    history_days, sim_days = 120, 30
    start = date(2025, 1, 1)
    full = generate_sales_history(products_df, history_days + sim_days, 11, start_date=start)
    split = start + timedelta(days=history_days)

    head = full[full["date"] < split].reset_index(drop=True)
    tail = full[full["date"] >= split].reset_index(drop=True)
    alone = generate_sales_history(products_df, history_days, 11, start_date=start)

    pd.testing.assert_frame_equal(head, alone)
    assert_sales_schema(tail)
    assert len(tail) == sim_days * len(products_df)
    assert tail["date"].min() == split


def test_product_rows_independent_of_other_products(products_df):
    full = generate_sales_history(products_df, 60, 3)
    subset = generate_sales_history(products_df[products_df["product_id"].isin([4, 7])], 60, 3)
    expected = full[full["product_id"].isin([4, 7])].reset_index(drop=True)
    pd.testing.assert_frame_equal(subset, expected)


# Demand model


def test_weekend_demand_higher(history):
    for product_id, rows in history[~history["has_promo"]].groupby("product_id"):
        weekend = rows.loc[rows["is_weekend"], "units_sold"].mean()
        weekday = rows.loc[~rows["is_weekend"], "units_sold"].mean()
        assert weekend > weekday, f"product {product_id}: weekend {weekend} <= weekday {weekday}"


def test_promo_demand_higher(history):
    for product_id, rows in history.groupby("product_id"):
        promo = rows.loc[rows["has_promo"], "units_sold"].mean()
        no_promo = rows.loc[~rows["has_promo"], "units_sold"].mean()
        assert promo > no_promo, f"product {product_id}: promo {promo} <= no promo {no_promo}"


@pytest.mark.parametrize("seed", [0, 1, 42, 2024])
def test_promo_fraction(products_df, seed):
    df = generate_sales_history(products_df, 180, seed)
    fractions = df.groupby("product_id")["has_promo"].mean()
    assert ((fractions >= 0.05) & (fractions <= 0.30)).all(), fractions.to_dict()


def test_promo_periods_are_3_to_7_days(products_df):
    """Interior promotions last 3 to 7 days; runs touching either end may be cut off."""
    df = generate_sales_history(products_df, 365, 8)
    for _, rows in df.groupby("product_id"):
        flags = rows["has_promo"].to_numpy()
        edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(int), [0]])))
        for start, end in zip(edges[::2], edges[1::2]):
            if start > 0 and end < len(flags):
                assert 3 <= end - start <= 7


def test_simulate_one_day_promo_raises_demand(products_df):
    day = date(2025, 3, 5)  # Wednesday
    all_promo = {int(pid): True for pid in products_df["product_id"]}
    on = [simulate_one_day(products_df, day, all_promo, s) for s in range(300)]
    off = [simulate_one_day(products_df, day, {}, s) for s in range(300)]
    mean_on = pd.concat(on).groupby("product_id")["units_sold"].mean()
    mean_off = pd.concat(off).groupby("product_id")["units_sold"].mean()
    assert (mean_on > mean_off).all()
    assert pd.concat(on)["has_promo"].all() and not pd.concat(off)["has_promo"].any()


def test_simulate_one_day_missing_products_default_to_no_promo(products_df):
    df = simulate_one_day(products_df, date(2025, 3, 5), {3: True}, 1)
    assert df.set_index("product_id")["has_promo"].to_dict() == {
        int(pid): pid == 3 for pid in products_df["product_id"]
    }


def test_base_demand_fallbacks():
    assert data_generator._base_demand("Toned Milk 500ml", "Dairy") == 30.0
    assert data_generator._base_demand("Paneer 200g", "Dairy") == data_generator.CATEGORY_BASE_DEMAND["Dairy"]
    assert data_generator._base_demand("Phone Charger", "Electronics") == data_generator.GLOBAL_BASE_DEMAND


def test_category_override_applies(monkeypatch):
    monkeypatch.setitem(data_generator.CATEGORY_MULT_OVERRIDES, "Staples", {"weekend": 1.0})
    rice = ("Sona Masoori Rice 5kg", "Staples")
    assert data_generator._demand_rate(*rice, True, False) == data_generator._demand_rate(*rice, False, False)
    assert data_generator._demand_rate(*rice, False, True) == pytest.approx(4.0 * data_generator.PROMO_MULT)


# Validation and interface


def test_start_date_is_keyword_only(products_df):
    with pytest.raises(TypeError):
        generate_sales_history(products_df, 10, 1, date(2025, 1, 1))


@pytest.mark.parametrize("days", [0, -3])
def test_rejects_non_positive_days(products_df, days):
    with pytest.raises(ValueError):
        generate_sales_history(products_df, days, 1)


def test_rejects_bad_products(products_df):
    with pytest.raises(ValueError):
        generate_sales_history(products_df.drop(columns=["name"]), 10, 1)
    with pytest.raises(ValueError):
        simulate_one_day(pd.concat([products_df, products_df]), date(2025, 1, 1), {}, 1)


def test_matches_mock_signatures():
    import inspect

    from decision_engine import mock

    for name in ("generate_sales_history", "simulate_one_day"):
        real = inspect.signature(getattr(data_generator, name))
        fake = inspect.signature(getattr(mock, name))
        assert real == fake, f"{name}: {real} != {fake}"


def test_pooled_multipliers_match_constants(products_df):
    """Over 20 seeds x 365 days, demand normalized by each product's base rate shows the
    weekend (1.4), promo (1.8) and combined (2.52) multipliers within 3 percent."""
    frames = [generate_sales_history(products_df, 365, seed) for seed in range(20)]
    df = pd.concat(frames, ignore_index=True)
    base = {
        int(p.product_id): data_generator._base_demand(p.name, p.category)
        for p in products_df.itertuples(index=False)
    }
    df["normalized"] = df["units_sold"] / df["product_id"].map(base)
    means = df.groupby(["is_weekend", "has_promo"])["normalized"].mean()
    neither = means[(False, False)]

    assert means[(True, False)] / neither == pytest.approx(data_generator.WEEKEND_MULT, rel=0.03)
    assert means[(False, True)] / neither == pytest.approx(data_generator.PROMO_MULT, rel=0.03)
    assert means[(True, True)] / neither == pytest.approx(
        data_generator.WEEKEND_MULT * data_generator.PROMO_MULT, rel=0.03
    )
