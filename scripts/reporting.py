"""Shared analyses for the evaluation and robustness reports and for export_results.py.

Every function returns a DataFrame (no printing), so the report scripts format the same numbers that
export_results.py writes to CSV. Seeds run in parallel worker processes; every run is deterministic,
so results do not depend on scheduling.
"""

from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision_engine import bayes_risk  # noqa: E402
from decision_engine.data_generator import _demand_rate, generate_sales_history  # noqa: E402
from decision_engine.evaluation import run_evaluation  # noqa: E402

PRODUCTS_CSV = ROOT / "data" / "sample_products.csv"
DAYS = 90
TUNING_SEEDS = range(100, 120)
TEST_SEEDS = range(20)
COST_SEEDS = range(10)
COVERAGE_SEEDS = range(50)
SAFETY_FACTORS = (1.0, 1.2, 1.5, 2.0, 2.5)
COVER_DAYS = (3, 7, 14, 21, 28)
PENALTY_MULTS = (0.5, 1.0, 2.0)
ORDER_COSTS = (0.0, 30.0, 100.0)
MISSPECIFICATION = {"overdispersion": 10, "month_start_mult": 1.3}
POLICIES = ["baseline", "dss", "dss_no_evidence"]
STANDARD_PAIRS = [("dss", "baseline"), ("dss_no_evidence", "baseline"), ("dss", "dss_no_evidence")]


def load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS_CSV)


def make_pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=os.cpu_count() or 1)


def _run(job: tuple[pd.DataFrame, int, dict, dict]) -> pd.DataFrame:
    products, seed, kwargs, tags = job
    return run_evaluation(products, DAYS, seed, **kwargs).assign(seed=seed, **tags)


def run_all(pool: ProcessPoolExecutor, jobs: list) -> pd.DataFrame:
    return pd.concat(pool.map(_run, jobs), ignore_index=True)


def evaluate_seeds(pool, products: pd.DataFrame, seeds, kwargs: dict | None = None) -> pd.DataFrame:
    return run_all(pool, [(products, seed, kwargs or {}, {}) for seed in seeds])


def t_interval(values) -> tuple[float, float, float]:
    """Mean and 95% t confidence interval."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def _totals(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["product_id"].isna()]


def policy_totals(results: pd.DataFrame) -> pd.DataFrame:
    """Per policy, totals over all products averaged over seeds."""
    totals = _totals(results)
    rows = []
    for policy in [p for p in POLICIES if p in set(totals["policy"])]:
        t = totals[totals["policy"] == policy]
        rows.append({
            "policy": policy,
            "seeds": len(t),
            "profit": t["total_profit"].mean(),
            "fill_rate": t["fill_rate"].mean(),
            "stockout_days": t["stockout_days"].mean(),
            "units_spoiled": t["units_spoiled"].mean(),
            "orders_placed": t["orders_placed"].mean(),
            "orders_per_month": t["orders_placed"].mean() * 30 / DAYS,
        })
    return pd.DataFrame(rows)


def paired_comparisons(results: pd.DataFrame, pairs=STANDARD_PAIRS) -> pd.DataFrame:
    """Per-seed paired profit differences a - b with mean and 95% t CI."""
    profit = _totals(results).pivot(index="seed", columns="policy", values="total_profit")
    rows = []
    for a, b in pairs:
        if a in profit and b in profit:
            mean, low, high = t_interval(profit[a] - profit[b])
            rows.append({"comparison": f"{a} - {b}", "mean": mean, "ci_low": low, "ci_high": high,
                         "excludes_zero": bool(low > 0 or high < 0), "seeds": len(profit),
                         "relative_to_b": mean / profit[b].mean()})
    return pd.DataFrame(rows)


def per_product(results: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Per product: dss - baseline profit (mean, CI) and fill rate, spoilage, orders/month per policy."""
    names = products.set_index("product_id")["name"]
    rows_all = results[results["product_id"].notna()].astype({"product_id": int})
    present = [p for p in POLICIES if p in set(rows_all["policy"])]
    rows = []
    for pid in sorted(rows_all["product_id"].unique()):
        r = rows_all[rows_all["product_id"] == pid]
        by = {p: r[r["policy"] == p].set_index("seed") for p in present}
        mean, low, high = t_interval(by["dss"]["total_profit"] - by["baseline"]["total_profit"])
        row = {"product_id": pid, "name": names[pid], "dss_minus_baseline": mean, "ci_low": low,
               "ci_high": high, "excludes_zero": bool(low > 0 or high < 0)}
        for p in present:
            row[f"fill_rate_{p}"] = by[p]["fill_rate"].mean()
            row[f"units_spoiled_{p}"] = by[p]["units_spoiled"].mean()
            row[f"orders_per_month_{p}"] = by[p]["orders_placed"].mean() * 30 / DAYS
        rows.append(row)
    return pd.DataFrame(rows)


def tune_baseline(pool, products: pd.DataFrame, extra: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Grid-search safety factor x cover days per product on the tuning seeds (baseline only).
    Returns the chosen parameters per product and the dicts to pass to run_evaluation."""
    extra = extra or {}
    jobs = [
        (products, seed, {"include_dss": False, "baseline_safety_factor": sf, "baseline_cover_days": cd, **extra},
         {"sf": sf, "cd": cd})
        for sf, cd in itertools.product(SAFETY_FACTORS, COVER_DAYS)
        for seed in TUNING_SEEDS
    ]
    tuning = run_all(pool, jobs)
    rows_all = tuning[tuning["product_id"].notna()].astype({"product_id": int})
    mean_profit = rows_all.groupby(["product_id", "sf", "cd"])["total_profit"].mean()
    names = products.set_index("product_id")
    rows, tuned_sf, tuned_cd = [], {}, {}
    for pid in sorted(rows_all["product_id"].unique()):
        grid = mean_profit.loc[pid]
        sf, cd = grid.idxmax()
        tuned_sf[pid], tuned_cd[pid] = float(sf), int(cd)
        shelf = names.loc[pid, "shelf_life_days"]
        rows.append({"product_id": pid, "name": names.loc[pid, "name"], "safety_factor": sf, "cover_days": cd,
                     "effective_cover_days": cd if pd.isna(shelf) else max(1, min(cd, int(shelf) - 1)),
                     "tuning_mean_profit": grid.max(), "default_mean_profit": grid.loc[(1.2, 7)]})
    return pd.DataFrame(rows), tuned_sf, tuned_cd


def cost_sensitivity(pool, products: pd.DataFrame) -> pd.DataFrame:
    """dss - default baseline profit for stockout_penalty x {0.5,1,2} crossed with order_cost {0,30,100}."""
    jobs = []
    for mult, cost in itertools.product(PENALTY_MULTS, ORDER_COSTS):
        scaled = products.assign(stockout_penalty=products["stockout_penalty"] * mult)
        jobs += [(scaled, seed, {"order_cost": cost, "include_ablation": False}, {"mult": mult, "cost": cost})
                 for seed in COST_SEEDS]
    totals = _totals(run_all(pool, jobs))
    rows = []
    for mult, cost in itertools.product(PENALTY_MULTS, ORDER_COSTS):
        cell = totals[(totals["mult"] == mult) & (totals["cost"] == cost)]
        p = cell.pivot(index="seed", columns="policy", values="total_profit")
        mean, low, high = t_interval(p["dss"] - p["baseline"])
        rows.append({"stockout_penalty_mult": mult, "order_cost": cost, "dss_minus_baseline": mean,
                     "ci_low": low, "ci_high": high, "excludes_zero": bool(low > 0 or high < 0), "seeds": len(p)})
    return pd.DataFrame(rows)


def credible_interval_coverage(products: pd.DataFrame, history_days: int = 180) -> pd.DataFrame:
    """Share of the Bayes model's 90% credible intervals containing the true daily rate
    (same procedure as tests/test_bayes_risk.py::test_credible_interval_coverage)."""
    evidence = {"weekday": (False, False), "weekend": (True, False),
                "weekday_promo": (False, True), "weekend_promo": (True, True)}
    info = products.set_index("product_id")
    hits = {g: [] for g in bayes_risk.GROUPS}
    for seed in COVERAGE_SEEDS:
        history = generate_sales_history(products, history_days, seed)
        for pid in info.index:
            rates = bayes_risk._group_rates(bayes_risk._fit_product(history, int(pid)))
            for group, r in rates.items():
                truth = _demand_rate(info.loc[pid, "name"], info.loc[pid, "category"], *evidence[group])
                hits[group].append(r["ci_low"] <= truth <= r["ci_high"])
    rows = [{"group": g, "coverage": float(np.mean(h)), "intervals": len(h)} for g, h in hits.items()]
    all_hits = [x for h in hits.values() for x in h]
    rows.append({"group": "overall", "coverage": float(np.mean(all_hits)), "intervals": len(all_hits)})
    return pd.DataFrame(rows)
