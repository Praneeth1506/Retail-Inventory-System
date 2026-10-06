"""Tests for decision_engine.evaluation (short runs so the suite stays fast)."""

import inspect
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decision_engine import evaluation, mock
from decision_engine.bayes_risk import _fit_product
from decision_engine.data_generator import generate_sales_history
from decision_engine.evaluation import _Product, _baseline_levels, _evaluate, _simulate, run_evaluation
from decision_engine.optimizer import economic_review_period
from decision_engine.recommend import _recommend, recommend_order
from tests.contract import assert_evaluation_structure

PRODUCTS_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"
ALL_POLICIES = ["baseline", "dss", "dss_no_evidence"]
DAYS = 21


@pytest.fixture(scope="module")
def products_df() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


@pytest.fixture(scope="module")
def run(products_df):
    """One 21-day evaluation with every policy, reused by the read-only tests."""
    policies = {
        int(r.product_id): {
            "is_perishable": not pd.isna(r.shelf_life_days),
            "spoilage_cost": 2.0,
            "shelf_life_days": None if pd.isna(r.shelf_life_days) else int(r.shelf_life_days),
        }
        for r in products_df.itertuples(index=False)
    }
    df, ledgers = _evaluate(products_df, DAYS, 5, history_days=120, policies=policies)
    return df, ledgers


def lead_times(products_df) -> dict[int, int]:
    return dict(zip(products_df["product_id"].astype(int), products_df["lead_time_days"].astype(int)))


# Structure and interface


def test_output_structure(products_df, run):
    df, _ = run
    assert_evaluation_structure(df, list(products_df["product_id"]), ALL_POLICIES)
    totals = df[df["product_id"].isna()].set_index("policy")
    per_product = df[df["product_id"].notna()]
    for policy in ALL_POLICIES:
        rows = per_product[per_product["policy"] == policy]
        assert totals.loc[policy, "total_profit"] == pytest.approx(rows["total_profit"].sum())
        sold = (rows["fill_rate"] * rows["units_demanded"]).sum()
        assert totals.loc[policy, "fill_rate"] == pytest.approx(sold / rows["units_demanded"].sum())


def test_without_ablation(products_df):
    df = run_evaluation(products_df, 14, 1, history_days=60, include_ablation=False)
    assert_evaluation_structure(df, list(products_df["product_id"]), ["baseline", "dss"])


def test_matches_mock_signature():
    assert inspect.signature(evaluation.run_evaluation) == inspect.signature(mock.run_evaluation)


def test_defaulted_args_are_keyword_only(products_df):
    with pytest.raises(TypeError):
        run_evaluation(products_df, 14, 1, 60)


@pytest.mark.parametrize(
    "kwargs",
    [{"days": 0}, {"history_days": 0}, {"order_cost": -1.0}, {"risk_tolerance": 0.0},
     {"review_period_overrides": {1: 0}}, {"review_period_overrides": {1: 1.5}}],
)
def test_rejects_invalid_arguments(products_df, kwargs):
    args = {"days": 14, "history_days": 60, **kwargs}
    days = args.pop("days")
    with pytest.raises(ValueError):
        run_evaluation(products_df, days, 1, **args)


def test_same_seed_identical(products_df):
    a = run_evaluation(products_df, 14, 3, history_days=60)
    b = run_evaluation(products_df, 14, 3, history_days=60)
    pd.testing.assert_frame_equal(a, b)


# Bookkeeping


def test_unit_conservation(run):
    _, ledgers = run
    for policy, per_product in ledgers.items():
        for pid, l in per_product.items():
            assert l.starting_stock + l.received == l.sold + l.spoiled + l.ending_on_hand, (policy, pid)
            assert l.sold + l.lost == l.demanded
            assert l.received == sum(q for _, q in l.receipts)
            assert l.spoiled == sum(q for _, q in l.spoilage_events)
            assert sum(q for _, q, _ in l.orders) == l.received + l.ending_in_transit


def test_profit_identity(products_df, run):
    df, ledgers = run
    prices = products_df.set_index("product_id")
    for policy, per_product in ledgers.items():
        for pid, l in per_product.items():
            p = prices.loc[pid]
            ending = l.ending_on_hand + l.ending_in_transit
            recomputed = (
                float(p["selling_price"]) * l.sold
                - float(p["unit_cost"]) * sum(q for _, q, _ in l.orders)
                - 30.0 * len(l.orders)
                - float(p["holding_cost_per_day"]) * sum(l.on_hand_end)
                - float(p["stockout_penalty"]) * l.lost
                - 2.0 * l.spoiled
                + float(p["unit_cost"]) * ending
                - float(p["unit_cost"]) * l.starting_stock
            )
            row = df[(df["policy"] == policy) & (df["product_id"] == pid)].iloc[0]
            assert row["total_profit"] == pytest.approx(recomputed, abs=1e-6)
            assert row["revenue"] == pytest.approx(float(p["selling_price"]) * l.sold, abs=1e-6)


def test_identical_demand_across_policies(run):
    _, ledgers = run
    for pid in ledgers["baseline"]:
        assert len({ledgers[p][pid].demanded for p in ALL_POLICIES}) == 1


def test_orders_arrive_after_lead_time(products_df, run):
    _, ledgers = run
    leads = lead_times(products_df)
    for per_product in ledgers.values():
        for pid, l in per_product.items():
            for placed, qty, arrival in l.orders:
                assert arrival == placed + leads[pid]
                if arrival < DAYS:
                    assert (arrival, qty) in l.receipts
            assert sorted(l.receipts) == sorted((a, q) for _, q, a in l.orders if a < DAYS)


def test_stock_never_negative_and_non_perishables_never_spoil(products_df, run):
    _, ledgers = run
    non_perishable = set(products_df.loc[products_df["shelf_life_days"].isna(), "product_id"].astype(int))
    for per_product in ledgers.values():
        for pid, l in per_product.items():
            assert min(l.on_hand_end) >= 0
            if pid in non_perishable:
                assert l.spoiled == 0


def test_starting_stock_is_baseline_order_up_to_level(run):
    _, ledgers = run
    for pid in ledgers["baseline"]:
        assert len({ledgers[p][pid].starting_stock for p in ALL_POLICIES}) == 1


# Hand-made products


def constant_history(product_id: int, units: int, days: int = 30) -> pd.DataFrame:
    start = date(2025, 1, 1)
    return pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(days)],
        "product_id": product_id,
        "units_sold": units,
        "is_weekend": [(start + timedelta(days=i)).weekday() >= 5 for i in range(days)],
        "has_promo": False,
    })


def hand_product(shelf_life_days=None, lead=2) -> _Product:
    return _Product(product_id=99, unit_cost=10.0, price=15.0, holding_cost=0.1, stockout_penalty=5.0,
                    lead_time_days=lead, shelf_life_days=shelf_life_days, spoilage_cost=1.0)


def test_baseline_orders_up_to_S_when_position_at_or_below_s():
    product = hand_product()
    history = constant_history(99, 10)
    s, S = _baseline_levels(history, product)
    assert (s, S) == (24, 94)  # mu 10, lead 2: s = ceil(10*2*1.2), S = s + 10*7
    ledgers = _simulate([product], history, {99: np.full(30, 10)}, {99: []}, date(2025, 2, 1),
                        "baseline", order_cost=30.0)
    l = ledgers[99]
    assert l.orders
    for day, position, qty in l.decisions:
        assert qty == (S - position if position <= s else 0)


def test_perishable_batch_expires_on_correct_day():
    product = hand_product(shelf_life_days=3, lead=2)
    history = constant_history(99, 10)
    s, S = _baseline_levels(history, product)
    assert (s, S) == (24, 24 + 20)  # D = max(1, min(7, 3 - 1)) = 2
    ledgers = _simulate([product], history, {99: np.zeros(15, dtype=int)}, {99: []}, date(2025, 2, 1),
                        "baseline", order_cost=30.0)
    l = ledgers[99]
    # With zero demand every batch expires untouched exactly shelf_life_days after it arrived
    # (age 0 on arrival day, discarded at the start of the day its age reaches 3).
    arrivals = [(0, S)] + l.receipts
    expected = [(day + 3, qty) for day, qty in arrivals if day + 3 < 15]
    assert l.spoilage_events == expected
    assert l.spoilage_events[0] == (3, S)
    assert l.spoilage_costs == pytest.approx(1.0 * l.spoiled)
    # The effective position stops crediting the day-0 batch before it expires, so the first
    # reorder is placed on day 1 (credit 2 days x 10 = 20 <= s), arriving day 3 as it expires.
    assert l.orders[0] == (1, S - 20, 3)


def test_zero_lead_time_order_arrives_before_sales():
    product = hand_product(lead=0)
    history = constant_history(99, 10)
    ledgers = _simulate([product], history, {99: np.full(10, 10)}, {99: []}, date(2025, 2, 1),
                        "baseline", order_cost=30.0)
    l = ledgers[99]
    assert l.lost == 0
    assert all(arrival == placed for placed, _, arrival in l.orders)


# Caching and the ablation


@pytest.mark.parametrize("product_id", [1, 5, 8])
def test_cached_posterior_matches_uncached(products_df, product_id):
    history = generate_sales_history(products_df, 90, 2)
    row = products_df[products_df["product_id"] == product_id].iloc[0]
    day = date(2025, 4, 5)
    promos = [day + timedelta(days=1), day + timedelta(days=2)]
    for pooled in (False, True):
        fit = _fit_product(history, product_id, pooled=pooled)
        cached = _recommend(row, 7, day, history, promo_dates=promos, fit=fit, pooled=pooled)
        fresh = _recommend(row, 7, day, history, promo_dates=promos, pooled=pooled)
        assert cached == fresh
    assert _recommend(row, 7, day, history, promo_dates=promos) == recommend_order(row, 7, day, history,
                                                                                  promo_dates=promos)


def test_pooled_model_ignores_evidence(products_df):
    history = generate_sales_history(products_df, 90, 2)
    row = products_df[products_df["product_id"] == 1].iloc[0]
    monday, saturday = date(2025, 4, 7), date(2025, 4, 5)
    plain = _recommend(row, 0, monday, history, pooled=True)
    busy = _recommend(row, 0, saturday, history, promo_dates=[saturday], pooled=True)
    assert plain["risk"]["order_horizon_distribution"] == busy["risk"]["order_horizon_distribution"]
    full_plain = _recommend(row, 0, monday, history)
    full_busy = _recommend(row, 0, saturday, history, promo_dates=[saturday])
    assert full_busy["risk"]["expected_order_horizon_demand"] > full_plain["risk"]["expected_order_horizon_demand"]


def test_review_period_override_is_used(products_df):
    df_default, _ = _evaluate(products_df, 14, 1, history_days=60, include_ablation=False)
    df_override, ledgers = _evaluate(products_df, 14, 1, history_days=60, include_ablation=False,
                                     review_period_overrides={5: 2})
    assert ledgers["dss"][5].orders  # sanity: product 5 was managed
    dss = lambda df: df[(df["policy"] == "dss") & (df["product_id"] == 5)].iloc[0]  # noqa: E731
    assert dss(df_default)["total_profit"] != dss(df_override)["total_profit"]
    other = lambda df: df[(df["policy"] == "dss") & (df["product_id"] == 1)].iloc[0]  # noqa: E731
    assert other(df_default).equals(other(df_override))


# Fixes to the baseline and the DSS ordering frequency


def test_milk_baseline_levels_use_shelf_life_minus_one(products_df):
    history = generate_sales_history(products_df, 120, 3)
    milk = evaluation._effective_products(products_df[products_df["product_id"] == 1], {})[0]
    mu = history.loc[history["product_id"] == 1, "units_sold"].mean()
    s, S = _baseline_levels(history, milk)
    assert s == int(np.ceil(mu * 1 * 1.2))
    assert S == s + int(np.ceil(mu * 1))  # D = max(1, min(7, 2 - 1)) = 1


def test_new_baseline_beats_old_rule_on_milk(products_df, monkeypatch):
    full = generate_sales_history(products_df, 120 + 28, 4)
    sim_start = date(2025, 1, 1) + timedelta(days=120)
    history = full[full["date"] < sim_start]
    future = full[(full["date"] >= sim_start) & (full["product_id"] == 1)].sort_values("date")
    milk = evaluation._effective_products(products_df[products_df["product_id"] == 1], {})[0]
    args = ([milk], history, {1: future["units_sold"].to_numpy()}, {1: []}, sim_start, "baseline")

    new = _simulate(*args, order_cost=30.0)[1]

    def old_cycle_days(product):
        shelf = product.shelf_life_days
        return 7 if shelf is None else min(7, shelf)

    def raw_position(batches, in_transit, day, sim_start, product, mean_daily):
        return sum(q for _, q in batches) + sum(q for _, q in in_transit)

    monkeypatch.setattr(evaluation, "_baseline_cycle_days", old_cycle_days)
    monkeypatch.setattr(evaluation, "_inventory_position", raw_position)
    old = _simulate(*args, order_cost=30.0)[1]

    assert new.sold / new.demanded > old.sold / old.demanded
    assert new.spoiled < old.spoiled


def test_dss_orders_about_once_per_review_period(products_df):
    days = 90
    _, ledgers = _evaluate(products_df, days, 0, include_ablation=False)
    history = generate_sales_history(products_df, 180, 0)  # the run's history: same seed and start
    for r in products_df[products_df["shelf_life_days"].isna()].itertuples(index=False):
        mean = history.loc[history["product_id"] == r.product_id, "units_sold"].mean()
        review = economic_review_period(mean, r.holding_cost_per_day, 30.0)
        per_month = len(ledgers["dss"][r.product_id].orders) * 30 / days
        assert per_month <= 1.5 * 30 / review, (r.product_id, per_month, review)
