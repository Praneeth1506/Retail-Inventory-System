"""Tests for decision_engine.bayes_risk."""

import inspect
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_engine import bayes_risk, mock
from decision_engine.bayes_risk import GROUPS, calculate_stockout_risk
from decision_engine.data_generator import _demand_rate, generate_sales_history
from tests.contract import MONDAY, SATURDAY, THURSDAY, assert_distribution, assert_risk_structure

PRODUCTS_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"
GROUP_EVIDENCE = {
    "weekday": (False, False),
    "weekend": (True, False),
    "weekday_promo": (False, True),
    "weekend_promo": (True, True),
}
MILK, RICE = 1, 5
PRODUCT_IDS = range(1, 11)


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


@pytest.fixture(scope="module")
def history_180(products_df) -> pd.DataFrame:
    return generate_sales_history(products_df, 180, 42)


@pytest.fixture(scope="module")
def history_365(products_df) -> pd.DataFrame:
    return generate_sales_history(products_df, 365, 42)


def true_rate(products_df: pd.DataFrame, product_id: int, group: str) -> float:
    product = products_df.set_index("product_id").loc[product_id]
    return _demand_rate(product["name"], product["category"], *GROUP_EVIDENCE[group])


def lead_time(products_df: pd.DataFrame, product_id: int) -> int:
    return int(products_df.set_index("product_id").loc[product_id, "lead_time_days"])


def risk(history_df, product_id=MILK, current_stock=30, lead_time_days=1, current_date=MONDAY,
         **kwargs) -> dict:
    return calculate_stockout_risk(
        product_id, current_stock, lead_time_days, current_date, history_df, **kwargs
    )


def days(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def percentile_stock(dist: dict[int, float], q: float) -> int:
    keys = np.array(list(dist))
    cdf = np.cumsum(list(dist.values()))
    return int(keys[np.searchsorted(cdf, q)])


# Structure, types and sums


def test_keys_and_types(history_180):
    result = risk(history_180, lead_time_days=3, review_period_days=2)
    assert_risk_structure(result, product_id=MILK, lead_time_days=3, horizon_days=5)


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
@pytest.mark.parametrize("start", [MONDAY, THURSDAY, SATURDAY])
@pytest.mark.parametrize("with_promo", [False, True])
def test_structure_for_every_window(history_180, product_id, start, with_promo):
    promo_dates = days(start, 2) if with_promo else ()
    result = risk(history_180, product_id, lead_time_days=4, current_date=start,
                  promo_dates=promo_dates, review_period_days=2)
    assert_risk_structure(result, product_id=product_id, lead_time_days=4, horizon_days=6)


def test_zero_lead_time(history_180):
    result = risk(history_180, current_stock=0, lead_time_days=0)
    assert result["lead_time_distribution"] == {0: 1.0}
    assert result["lead_time_day_groups"] == {g: 0 for g in GROUPS}
    assert result["stockout_risk"] == 0.0
    assert result["order_horizon_days"] == 1


def test_no_history_uses_default_prior(history_180):
    result = risk(history_180.iloc[0:0], lead_time_days=2)
    assert_risk_structure(result, product_id=MILK, lead_time_days=2, horizon_days=3)
    assert all(r["n_days"] == 0 for r in result["group_rates"].values())
    assert result["expected_lead_time_demand"] == pytest.approx(
        2 * bayes_risk.DEFAULT_DAILY_DEMAND, rel=0.2
    )


def test_unknown_product_uses_default_prior(history_180):
    result = risk(history_180, product_id=999)
    assert_distribution(result["order_horizon_distribution"])


# Window dates


def test_lead_time_day_groups_spanning_weekend(history_180):
    # Thursday + 4 days = Thu, Fri, Sat, Sun
    result = risk(history_180, lead_time_days=4, current_date=THURSDAY)
    assert result["lead_time_day_groups"] == {
        "weekday": 2, "weekend": 2, "weekday_promo": 0, "weekend_promo": 0,
    }


def test_lead_time_day_groups_with_promo_dates(history_180):
    # Thu, Fri(promo), Sat(promo), Sun; promo dates outside the window are ignored.
    promo = [THURSDAY + timedelta(days=1), SATURDAY, date(2025, 3, 20), THURSDAY - timedelta(days=1)]
    result = risk(history_180, lead_time_days=4, current_date=THURSDAY, promo_dates=promo)
    assert result["lead_time_day_groups"] == {
        "weekday": 1, "weekend": 1, "weekday_promo": 1, "weekend_promo": 1,
    }


def test_promo_dates_accept_timestamps(history_180):
    as_dates = risk(history_180, lead_time_days=3, promo_dates=[MONDAY, MONDAY + timedelta(days=1)])
    as_stamps = risk(history_180, lead_time_days=3, current_date=pd.Timestamp(MONDAY),
                     promo_dates=pd.to_datetime([MONDAY, MONDAY + timedelta(days=1)]))
    assert as_stamps == as_dates


def test_window_demand_uses_per_day_groups(history_180):
    """Step 6: expected window demand = sum over groups of k_g * posterior mean_g."""
    promo = [THURSDAY + timedelta(days=1), SATURDAY]
    result = risk(history_180, lead_time_days=4, current_date=THURSDAY, promo_dates=promo,
                  review_period_days=2)
    rates = result["group_rates"]
    expected = sum(k * rates[g]["posterior_mean"] for g, k in result["lead_time_day_groups"].items())
    assert result["expected_lead_time_demand"] == pytest.approx(expected, rel=1e-4)
    # Order horizon adds Mon and Tue, both plain weekdays.
    expected_horizon = expected + 2 * rates["weekday"]["posterior_mean"]
    assert result["expected_order_horizon_demand"] == pytest.approx(expected_horizon, rel=1e-4)


# Recovery and coverage


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_recovers_true_rates(products_df, history_365, product_id):
    rates = risk(history_365, product_id)["group_rates"]
    for group in GROUPS:
        n = rates[group]["n_days"]
        if n == 0:
            continue
        truth = true_rate(products_df, product_id, group)
        error = abs(rates[group]["posterior_mean"] - truth)
        bound = 3.5 * np.sqrt(truth / n)
        assert error < bound, f"{group}: |{rates[group]['posterior_mean']:.2f} - {truth:.2f}| >= {bound:.2f} (n={n})"


@pytest.mark.parametrize("product_id", PRODUCT_IDS)
def test_sparse_group_borrows_from_prior(products_df, history_365, product_id):
    history = history_365[~(history_365["is_weekend"] & history_365["has_promo"])]
    result = risk(history, product_id, current_date=SATURDAY, promo_dates=[SATURDAY])
    assert_distribution(result["lead_time_distribution"])
    wp = result["group_rates"]["weekend_promo"]
    assert wp["n_days"] == 0
    assert wp["posterior_mean"] == pytest.approx(true_rate(products_df, product_id, "weekend_promo"), rel=0.25)


def test_credible_interval_coverage(products_df):
    """50 histories x 10 products x 4 groups: the 90% intervals should contain the true rate
    about 90% of the time."""
    hits = {g: [] for g in GROUPS}
    for seed in range(50):
        history = generate_sales_history(products_df, 180, seed)
        for product_id in PRODUCT_IDS:
            fit = bayes_risk._fit_product(history, product_id)
            for group, r in bayes_risk._group_rates(fit).items():
                truth = true_rate(products_df, product_id, group)
                hits[group].append(r["ci_low"] <= truth <= r["ci_high"])

    print("\n90% credible interval coverage (50 seeds x 10 products, 180 days):")
    for group in GROUPS:
        print(f"  {group:<14} {np.mean(hits[group]):.3f}  ({len(hits[group])} intervals)")
    overall = np.mean([h for g in GROUPS for h in hits[g]])
    print(f"  {'overall':<14} {overall:.3f}")
    assert 0.86 <= overall <= 0.94


# Monotonicity


def test_risk_decreases_with_stock(history_180):
    risks = [risk(history_180, lead_time_days=2, current_stock=s)["stockout_risk"] for s in range(0, 121, 5)]
    assert all(a >= b for a, b in zip(risks, risks[1:]))
    assert risks[0] > 0.99 and risks[-1] < 0.01


@pytest.mark.parametrize("product_id", [MILK, RICE])
def test_promo_in_window_raises_risk(products_df, history_180, product_id):
    stock = round(2 * true_rate(products_df, product_id, "weekday"))
    base = risk(history_180, product_id, current_stock=stock, lead_time_days=2)
    promo = risk(history_180, product_id, current_stock=stock, lead_time_days=2, promo_dates=[MONDAY])
    assert promo["stockout_risk"] > base["stockout_risk"]


@pytest.mark.parametrize("product_id", [MILK, RICE])
def test_weekend_in_window_raises_risk(products_df, history_180, product_id):
    stock = round(4 * true_rate(products_df, product_id, "weekday"))
    weekdays = risk(history_180, product_id, current_stock=stock, lead_time_days=4, current_date=MONDAY)
    with_weekend = risk(history_180, product_id, current_stock=stock, lead_time_days=4,
                        current_date=THURSDAY)
    assert weekdays["lead_time_day_groups"]["weekend"] == 0
    assert with_weekend["lead_time_day_groups"]["weekend"] == 2
    assert with_weekend["stockout_risk"] > weekdays["stockout_risk"]


# Calibration


@pytest.mark.parametrize("product_id", [MILK, RICE])
def test_calibration_over_histories(products_df, product_id):
    """Predictive calibration: averaged over 200 histories, the predicted risk at each chosen
    stock level matches how often true demand exceeds it. Windows start on Monday, so a lead
    time of up to 5 days is all weekdays."""
    product_df = products_df[products_df["product_id"] == product_id]
    lead = lead_time(products_df, product_id)
    truth = true_rate(products_df, product_id, "weekday")
    rng = np.random.default_rng(7)
    quantiles = (0.5, 0.8, 0.95)
    predicted = {q: [] for q in quantiles}
    empirical = {q: [] for q in quantiles}

    for history_seed in range(200):
        history = generate_sales_history(product_df, 180, history_seed)
        dist = risk(history, product_id, current_stock=0, lead_time_days=lead)["lead_time_distribution"]
        future = rng.poisson(lead * truth, 200)
        for q in quantiles:
            stock = percentile_stock(dist, q)
            predicted[q].append(sum(p for d, p in dist.items() if d > stock))
            empirical[q].append(float((future > stock).mean()))

    for q in quantiles:
        assert np.mean(empirical[q]) == pytest.approx(np.mean(predicted[q]), abs=0.03), q


# Grid convergence


def test_grid_convergence(products_df, history_180, monkeypatch):
    """Doubling the grid from 4000 to 8000 points barely changes any output."""
    assert bayes_risk.GRID_SIZE == 4000

    def outputs():
        results = {}
        for product_id in PRODUCT_IDS:
            lead = lead_time(products_df, product_id)
            promo = [THURSDAY + timedelta(days=1)]
            probe = risk(history_180, product_id, current_stock=0, lead_time_days=lead,
                         current_date=THURSDAY, promo_dates=promo)
            stock = round(probe["expected_lead_time_demand"])
            results[product_id] = risk(history_180, product_id, current_stock=stock,
                                       lead_time_days=lead, current_date=THURSDAY, promo_dates=promo)
        return results

    base = outputs()
    monkeypatch.setattr(bayes_risk, "GRID_SIZE", 8000)
    fine = outputs()

    worst_rate, worst_risk = 0.0, 0.0
    for product_id in PRODUCT_IDS:
        for group in GROUPS:
            for key in ("posterior_mean", "ci_low", "ci_high"):
                a = base[product_id]["group_rates"][group][key]
                b = fine[product_id]["group_rates"][group][key]
                worst_rate = max(worst_rate, abs(a - b) / b)
        worst_risk = max(worst_risk, abs(base[product_id]["stockout_risk"] - fine[product_id]["stockout_risk"]))

    print(f"\nGrid 4000 -> 8000: max relative change in posterior mean / CI bounds {100 * worst_rate:.4f}%,"
          f" max change in stockout_risk {worst_risk:.5f}")
    assert worst_rate < 0.001
    assert worst_risk < 0.002


# Validation and interface


@pytest.mark.parametrize(
    "overrides",
    [{"current_stock": -1}, {"lead_time_days": -1}, {"review_period_days": 0},
     {"current_date": "2025-03-03"}, {"promo_dates": ["2025-03-03"]}],
)
def test_rejects_invalid_arguments(history_180, overrides):
    with pytest.raises(ValueError):
        risk(history_180, **overrides)


def test_rejects_history_missing_columns(history_180):
    with pytest.raises(ValueError):
        risk(history_180.drop(columns=["has_promo"]))


def test_defaulted_arguments_are_keyword_only(history_180):
    with pytest.raises(TypeError):
        calculate_stockout_risk(MILK, 30, 1, MONDAY, history_180, 1)
    with pytest.raises(TypeError):
        calculate_stockout_risk(MILK, 30, 1, MONDAY, history_180, 1, [MONDAY])


def test_old_evidence_arguments_rejected(history_180):
    with pytest.raises(TypeError):
        risk(history_180, is_weekend=True)
    with pytest.raises(TypeError):
        risk(history_180, has_promo=True)


def test_matches_mock_signature():
    assert inspect.signature(bayes_risk.calculate_stockout_risk) == inspect.signature(
        mock.calculate_stockout_risk
    )
