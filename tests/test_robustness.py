"""Tests for the robustness options: baseline parameters, include_dss, generator misspecification."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_engine.data_generator import _demand_rate, generate_sales_history, simulate_one_day
from decision_engine.evaluation import _evaluate, run_evaluation

ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_CSV = ROOT / "data" / "sample_products.csv"
SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "pre_robustness_snapshot.json"


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text())


def expected_rates(products_df: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
    info = products_df.set_index("product_id")
    return np.array([
        _demand_rate(info.loc[pid, "name"], info.loc[pid, "category"], bool(w), bool(p))
        for pid, w, p in zip(history["product_id"], history["is_weekend"], history["has_promo"])
    ])


# Defaults reproduce the pre-change behaviour exactly


def test_run_evaluation_defaults_match_snapshot(products_df, snapshot):
    """Snapshot saved from the code before the robustness options existed."""
    args = snapshot["run_evaluation"]["args"]
    expected = snapshot["run_evaluation"]["rows"]
    for df in (
        run_evaluation(products_df, args["days"], args["seed"], history_days=args["history_days"]),
        run_evaluation(products_df, args["days"], args["seed"], history_days=args["history_days"],
                       baseline_safety_factor=1.2, baseline_cover_days=7, include_dss=True,
                       generator_options=None),
    ):
        assert df.to_dict("records") == expected


def test_generator_defaults_match_snapshot(products_df, snapshot):
    args = snapshot["generate_sales_history"]["args"]
    for history in (
        generate_sales_history(products_df, args["days"], args["seed"]),
        generate_sales_history(products_df, args["days"], args["seed"], overdispersion=None,
                               month_start_mult=1.0),
    ):
        assert history["units_sold"].tolist() == snapshot["generate_sales_history"]["units_sold"]
        assert history["has_promo"].tolist() == snapshot["generate_sales_history"]["has_promo"]


# Baseline parameters and include_dss


def baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["product_id"].notna()].set_index("product_id")


def test_include_dss_false_gives_only_baseline_rows(products_df):
    alone = run_evaluation(products_df, 21, 3, history_days=90, include_dss=False, include_ablation=True)
    assert list(alone["policy"].unique()) == ["baseline"]
    assert len(alone) == len(products_df) + 1
    full = run_evaluation(products_df, 21, 3, history_days=90)
    pd.testing.assert_frame_equal(alone, full[full["policy"] == "baseline"].reset_index(drop=True))


def test_per_product_parameters_apply_only_to_named_products(products_df):
    default = baseline_rows(run_evaluation(products_df, 28, 4, history_days=90, include_dss=False))
    tuned = baseline_rows(run_evaluation(
        products_df, 28, 4, history_days=90, include_dss=False,
        baseline_safety_factor={5: 2.5}, baseline_cover_days={9: 14, 1: 14},
    ))
    changed = {pid for pid in default.index if not default.loc[pid].equals(tuned.loc[pid])}
    # Milk (1) is perishable with shelf life 2, so cover 14 is capped back at 2 - 1 = 1: unchanged.
    assert changed == {5, 9}


def test_per_product_parameters_set_levels(products_df):
    _, default = _evaluate(products_df, 7, 4, history_days=90, include_dss=False)
    _, tuned = _evaluate(products_df, 7, 4, history_days=90, include_dss=False,
                         baseline_safety_factor={5: 2.5}, baseline_cover_days={9: 14, 2: 2})
    assert tuned["baseline"][5].starting_stock > default["baseline"][5].starting_stock
    assert tuned["baseline"][9].starting_stock > default["baseline"][9].starting_stock
    assert tuned["baseline"][2].starting_stock < default["baseline"][2].starting_stock  # curd: 4 -> 2
    assert tuned["baseline"][7].starting_stock == default["baseline"][7].starting_stock


def test_scalar_parameters_apply_to_every_product(products_df):
    default = baseline_rows(run_evaluation(products_df, 14, 4, history_days=90, include_dss=False))
    tuned = baseline_rows(run_evaluation(products_df, 14, 4, history_days=90, include_dss=False,
                                         baseline_safety_factor=2.0, baseline_cover_days=3))
    non_perishable = products_df.loc[products_df["shelf_life_days"].isna(), "product_id"]
    assert all(not default.loc[pid].equals(tuned.loc[pid]) for pid in non_perishable)


@pytest.mark.parametrize(
    "kwargs",
    [{"baseline_safety_factor": 0.0}, {"baseline_safety_factor": {3: -1.0}}, {"baseline_cover_days": 0},
     {"baseline_cover_days": {3: 2.5}}, {"generator_options": {"seasonality": 2}},
     {"generator_options": {"overdispersion": 0}}, {"generator_options": {"month_start_mult": 0}}],
)
def test_rejects_invalid_options(products_df, kwargs):
    with pytest.raises(ValueError):
        run_evaluation(products_df, 7, 1, history_days=30, include_dss=False, **kwargs)


# Generator misspecification


def test_overdispersion_keeps_mean_and_raises_variance(products_df):
    history = generate_sales_history(products_df, 365, 21, overdispersion=10)
    rates = expected_rates(products_df, history)
    assert history["units_sold"].sum() / rates.sum() == pytest.approx(1.0, abs=0.03)

    plain = history[~history["is_weekend"] & ~history["has_promo"]]
    for pid, rows in plain.groupby("product_id"):
        units = rows["units_sold"]
        assert units.var() > units.mean(), pid
        mean = rows.pipe(lambda r: expected_rates(products_df, r)).mean()
        assert units.mean() == pytest.approx(mean, rel=0.10), pid  # per product, looser

    poisson = generate_sales_history(products_df, 365, 21)
    p_plain = poisson[~poisson["is_weekend"] & ~poisson["has_promo"]]
    dispersion = plain.groupby("product_id")["units_sold"].var() / plain.groupby("product_id")["units_sold"].mean()
    p_dispersion = p_plain.groupby("product_id")["units_sold"].var() / p_plain.groupby("product_id")["units_sold"].mean()
    assert (dispersion > p_dispersion).all()


def test_month_start_mult_raises_days_1_to_5_only(products_df):
    normalized = {True: [], False: []}
    for seed in range(5):
        history = generate_sales_history(products_df, 365, seed, month_start_mult=1.3)
        ratio = history["units_sold"].to_numpy() / expected_rates(products_df, history)
        month_start = history["date"].map(lambda d: d.day <= 5).to_numpy()
        normalized[True].extend(ratio[month_start])
        normalized[False].extend(ratio[~month_start])
        # Promotions come from their own random stream and are unaffected.
        assert history["has_promo"].equals(generate_sales_history(products_df, 365, seed)["has_promo"])
    assert np.mean(normalized[True]) == pytest.approx(1.3, rel=0.03)
    assert np.mean(normalized[False]) == pytest.approx(1.0, rel=0.02)


def test_simulate_one_day_month_start_only_on_days_1_to_5(products_df):
    later = date(2025, 3, 12)
    for seed in range(20):
        assert simulate_one_day(products_df, later, {}, seed, month_start_mult=1.3).equals(
            simulate_one_day(products_df, later, {}, seed))
    early = date(2025, 3, 3)
    boosted = sum(simulate_one_day(products_df, early, {}, s, month_start_mult=1.3)["units_sold"].sum()
                  for s in range(300))
    plain = sum(simulate_one_day(products_df, early, {}, s)["units_sold"].sum() for s in range(300))
    assert boosted / plain == pytest.approx(1.3, rel=0.03)


def test_simulate_one_day_overdispersion(products_df):
    draws = pd.concat([simulate_one_day(products_df, date(2025, 3, 12), {}, s, overdispersion=5)
                       for s in range(400)])
    milk = draws[draws["product_id"] == 1]["units_sold"]
    assert milk.mean() == pytest.approx(30.0, rel=0.05)
    assert milk.var() > 2 * milk.mean()  # NB variance 30 + 900/5 = 210


def test_reproducible_with_new_options(products_df):
    options = {"overdispersion": 10, "month_start_mult": 1.3}
    a = generate_sales_history(products_df, 90, 8, **options)
    b = generate_sales_history(products_df, 90, 8, **options)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generate_sales_history(products_df, 90, 8))

    kwargs = dict(history_days=60, generator_options=options, baseline_safety_factor={5: 2.0},
                  baseline_cover_days=14)
    first = run_evaluation(products_df, 14, 8, **kwargs)
    pd.testing.assert_frame_equal(first, run_evaluation(products_df, 14, 8, **kwargs))
    default = run_evaluation(products_df, 14, 8, history_days=60)
    assert first["units_demanded"].sum() != default["units_demanded"].sum()


@pytest.mark.parametrize("kwargs", [{"overdispersion": 0.0}, {"overdispersion": -2}, {"month_start_mult": 0.0}])
def test_generator_rejects_invalid_options(products_df, kwargs):
    with pytest.raises(ValueError):
        generate_sales_history(products_df, 10, 1, **kwargs)
    with pytest.raises(ValueError):
        simulate_one_day(products_df, date(2025, 3, 3), {}, 1, **kwargs)
