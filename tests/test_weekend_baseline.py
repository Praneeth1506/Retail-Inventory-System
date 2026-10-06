"""Tests for the weekend-aware baseline (a robustness check; off by default)."""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_engine import evaluation
from decision_engine.evaluation import (
    _Product,
    _evaluate,
    _simulate,
    _weekday_weekend_mu,
    _weekend_aware_levels,
    run_evaluation,
)

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "pre_robustness_snapshot.json"
MONDAY, FRIDAY = date(2025, 3, 3), date(2025, 3, 7)


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "sample_products.csv")


def weekly_history(weekday_units: int, weekend_units: int, days: int = 56, promo_every: int = 0) -> pd.DataFrame:
    start = date(2025, 1, 6)  # a Monday
    dates = [start + timedelta(days=i) for i in range(days)]
    weekend = [d.weekday() >= 5 for d in dates]
    return pd.DataFrame({
        "date": dates,
        "product_id": 99,
        "units_sold": [weekend_units if w else weekday_units for w in weekend],
        "is_weekend": weekend,
        "has_promo": [bool(promo_every) and i % promo_every == 0 for i in range(days)],
    })


def product(cover_days: int = 3, lead: int = 2, shelf=None) -> _Product:
    return _Product(product_id=99, unit_cost=10.0, price=15.0, holding_cost=0.1, stockout_penalty=5.0,
                    lead_time_days=lead, shelf_life_days=shelf, spoilage_cost=0.0,
                    baseline_safety_factor=1.2, baseline_cover_days=cover_days)


def test_default_false_reproduces_snapshot(products_df):
    snap = json.loads(SNAPSHOT.read_text())["run_evaluation"]
    args = snap["args"]
    df = run_evaluation(products_df, args["days"], args["seed"], history_days=args["history_days"],
                        baseline_weekend_aware=False)
    assert df.to_dict("records") == snap["rows"]


def test_weekday_weekend_means():
    assert _weekday_weekend_mu(weekly_history(10, 20), product()) == (10.0, 20.0)


def test_levels_higher_when_window_contains_weekend():
    mu = _weekday_weekend_mu(weekly_history(10, 20), product())
    p = product(cover_days=3, lead=2)
    # Monday: lead Mon+Tue = 20 -> s = ceil(24.0) = 24; cover Wed-Fri = 30 -> S = 54.
    assert _weekend_aware_levels(mu, p, MONDAY) == (24, 54)
    # Friday: lead Fri+Sat = 30 -> s = 36; cover Sun+Mon+Tue = 40 -> S = 76.
    assert _weekend_aware_levels(mu, p, FRIDAY) == (36, 76)


def test_levels_respect_perishable_cap():
    mu = _weekday_weekend_mu(weekly_history(10, 20), product())
    p = product(cover_days=14, lead=1, shelf=3)  # D = max(1, min(14, 3 - 1)) = 2
    s, S = _weekend_aware_levels(mu, p, FRIDAY)  # lead Fri = 10; cover Sat+Sun = 40
    assert (s, S) == (12, 52)


def test_equal_means_match_static_baseline():
    """With the same demand on every day of the week, it reduces to the static baseline."""
    history = weekly_history(10, 10)
    p = product(cover_days=7, lead=2)
    assert _weekend_aware_levels(_weekday_weekend_mu(history, p), p, FRIDAY) == evaluation._baseline_levels(history, p)


def test_decisions_follow_daily_levels():
    history = weekly_history(10, 20)
    p = product(cover_days=3, lead=2)
    ledgers = _simulate([p], history, {99: np.full(28, 13)}, {99: []}, MONDAY, "baseline",
                        order_cost=30.0, baseline_weekend_aware=True)
    mu = _weekday_weekend_mu(history, p)
    l = ledgers[99]
    assert l.starting_stock == _weekend_aware_levels(mu, p, MONDAY)[1]
    assert l.orders
    for day, position, qty in l.decisions:
        s, S = _weekend_aware_levels(mu, p, MONDAY + timedelta(days=day))
        assert qty == (S - position if position <= s else 0)


def test_promotions_never_change_decisions():
    p = product(cover_days=3, lead=2)
    demand = {99: np.full(28, 13)}
    # Same units, different promo flags in the history: identical means and decisions.
    plain = _simulate([p], weekly_history(10, 20), demand, {99: []}, MONDAY, "baseline",
                      order_cost=30.0, baseline_weekend_aware=True)[99]
    flagged = _simulate([p], weekly_history(10, 20, promo_every=3), demand,
                        {99: [MONDAY + timedelta(days=i) for i in range(28)]}, MONDAY, "baseline",
                        order_cost=30.0, baseline_weekend_aware=True)[99]
    assert plain.decisions == flagged.decisions
    assert plain.orders == flagged.orders


def test_end_to_end_runs_and_all_policies_share_starting_stock(products_df):
    df, ledgers = _evaluate(products_df, 21, 2, history_days=90, baseline_weekend_aware=True)
    assert set(df["policy"]) == {"baseline", "dss", "dss_no_evidence"}
    for pid in ledgers["baseline"]:
        assert len({ledgers[name][pid].starting_stock for name in ledgers}) == 1
    default = run_evaluation(products_df, 21, 2, history_days=90)
    base = lambda d: d[d["policy"] == "baseline"].reset_index(drop=True)  # noqa: E731
    assert not base(df).equals(base(default))


def test_rejects_non_bool(products_df):
    with pytest.raises(ValueError):
        run_evaluation(products_df, 7, 1, history_days=30, include_dss=False, baseline_weekend_aware=1)
