"""Shared contract checks for results of the public functions (mock and real)."""

from datetime import date

TOL = 1e-6
GROUPS = ("weekday", "weekend", "weekday_promo", "weekend_promo")
RISK_KEYS = {
    "product_id",
    "stockout_risk",
    "group_rates",
    "lead_time_day_groups",
    "lead_time_distribution",
    "order_horizon_days",
    "order_horizon_distribution",
    "expected_lead_time_demand",
    "expected_order_horizon_demand",
}
MONDAY = date(2025, 3, 3)
THURSDAY = date(2025, 3, 6)
SATURDAY = date(2025, 3, 8)


def assert_distribution(dist: dict) -> None:
    assert isinstance(dist, dict) and dist
    for k, v in dist.items():
        assert type(k) is int and k >= 0
        assert type(v) is float and v >= 0
    assert abs(sum(dist.values()) - 1.0) < TOL


def assert_risk_structure(result: dict, product_id: int, lead_time_days: int, horizon_days: int) -> None:
    assert set(result) == RISK_KEYS
    assert type(result["product_id"]) is int and result["product_id"] == product_id
    assert type(result["stockout_risk"]) is float and 0.0 <= result["stockout_risk"] <= 1.0
    assert type(result["order_horizon_days"]) is int and result["order_horizon_days"] == horizon_days

    rates = result["group_rates"]
    assert set(rates) == set(GROUPS)
    for r in rates.values():
        assert set(r) == {"posterior_mean", "ci_low", "ci_high", "n_days"}
        assert all(type(r[k]) is float for k in ("posterior_mean", "ci_low", "ci_high"))
        assert type(r["n_days"]) is int and r["n_days"] >= 0
        assert r["ci_low"] <= r["posterior_mean"] <= r["ci_high"]

    groups = result["lead_time_day_groups"]
    assert set(groups) == set(GROUPS)
    assert all(type(v) is int and v >= 0 for v in groups.values())
    assert sum(groups.values()) == lead_time_days

    assert_distribution(result["lead_time_distribution"])
    assert_distribution(result["order_horizon_distribution"])
    for key, dist in (
        ("expected_lead_time_demand", result["lead_time_distribution"]),
        ("expected_order_horizon_demand", result["order_horizon_distribution"]),
    ):
        assert type(result[key]) is float
        assert abs(result[key] - sum(k * p for k, p in dist.items())) < 1e-6 * max(1.0, result[key])


ORDER_KEYS = {
    "optimal_qty",
    "expected_utility",
    "expected_profit",
    "certainty_equivalent",
    "no_stockout_probability",
    "at_upper_bound",
    "utility_type",
    "candidates",
}


def assert_order_structure(result: dict, utility_type: str) -> None:
    assert set(result) == ORDER_KEYS
    assert type(result["optimal_qty"]) is int and result["optimal_qty"] >= 0
    for key in ("expected_utility", "expected_profit", "certainty_equivalent", "no_stockout_probability"):
        assert type(result[key]) is float, key
    assert 0.0 <= result["no_stockout_probability"] <= 1.0
    assert type(result["at_upper_bound"]) is bool
    assert result["utility_type"] == utility_type

    candidates = result["candidates"]
    assert candidates and 0 in candidates
    for q, c in candidates.items():
        assert type(q) is int and q >= 0
        assert set(c) == {"expected_utility", "expected_profit", "certainty_equivalent"}
        assert all(type(v) is float for v in c.values())

    assert result["optimal_qty"] in candidates
    chosen = candidates[result["optimal_qty"]]
    assert chosen["expected_utility"] == result["expected_utility"]
    assert chosen["expected_profit"] == result["expected_profit"]
    assert chosen["certainty_equivalent"] == result["certainty_equivalent"]
    best_utility = max(c["expected_utility"] for c in candidates.values())
    assert result["expected_utility"] >= best_utility - 1e-9 * max(1.0, abs(best_utility))
    smaller_ties = [q for q, c in candidates.items() if q < result["optimal_qty"] and c["expected_utility"] == best_utility]
    assert not smaller_ties
    if result["at_upper_bound"]:
        assert result["optimal_qty"] == max(candidates)
    if utility_type == "linear":
        assert result["expected_utility"] == result["expected_profit"] == result["certainty_equivalent"]
    else:
        assert result["certainty_equivalent"] <= result["expected_profit"] + 1e-9 * max(1.0, abs(result["expected_profit"]))


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


def assert_evaluation_structure(df, product_ids, policies) -> None:
    """Columns, row layout, types, None on total rows, identical demand across policies."""
    assert list(df.columns) == EVALUATION_COLUMNS
    assert df["product_id"].dtype == object
    assert len(df) == len(policies) * (len(product_ids) + 1)
    assert list(dict.fromkeys(df["policy"])) == list(policies)

    for policy in policies:
        rows = df[df["policy"] == policy]
        totals = rows[rows["product_id"].isna()]
        assert len(totals) == 1 and totals["product_id"].iloc[0] is None
        per_product = rows[rows["product_id"].notna()]
        assert sorted(per_product["product_id"]) == sorted(product_ids)
        for column in ("stockout_days", "units_lost", "units_spoiled", "orders_placed", "units_demanded"):
            assert totals[column].iloc[0] == per_product[column].sum(), column

    for row in df.itertuples(index=False):
        assert row.product_id is None or type(row.product_id) is int
        for column in ("total_profit", "avg_stock_held", "fill_rate", "revenue"):
            assert isinstance(getattr(row, column), float), column
        for column in ("stockout_days", "units_lost", "units_spoiled", "orders_placed", "units_demanded"):
            value = getattr(row, column)
            assert isinstance(value, int) and value >= 0, column
        assert 0.0 <= row.fill_rate <= 1.0

    demanded = df[df["product_id"].notna()].pivot(index="product_id", columns="policy", values="units_demanded")
    assert (demanded.nunique(axis=1) == 1).all()
