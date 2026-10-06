"""Robustness report: tuned baseline, cost sensitivity and demand misspecification.

Sections
  a. Tune the baseline per product (safety factor x cover days) on tuning seeds 100-119.
  b. Test seeds 0-19: tuned baseline vs dss vs dss_no_evidence.
  c. Cost sensitivity (test seeds 0-9): stockout_penalty x {0.5, 1, 2} crossed with order_cost {0, 30, 100},
     default baseline vs dss.
  d. Misspecified demand (test seeds 0-19): overdispersion 10 and month-start multiplier 1.3, which the
     DSS's model does not know about; default baseline vs dss vs dss_no_evidence.
  e. Weekend-aware baseline: tuned per product like section a on seeds 100-119, then compared on test
     seeds 0-19 against dss and dss_no_evidence.
Seeds run in parallel worker processes; every run is deterministic, so results do not depend on scheduling.
"""

import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision_engine.evaluation import run_evaluation  # noqa: E402

DAYS = 90
TUNING_SEEDS = range(100, 120)
TEST_SEEDS = range(20)
COST_SEEDS = range(10)
SAFETY_FACTORS = (1.0, 1.2, 1.5, 2.0, 2.5)
COVER_DAYS = (3, 7, 14, 21, 28)
PENALTY_MULTS = (0.5, 1.0, 2.0)
ORDER_COSTS = (0.0, 30.0, 100.0)
MISSPECIFICATION = {"overdispersion": 10, "month_start_mult": 1.3}


def _run(job: tuple[pd.DataFrame, int, dict, dict]) -> pd.DataFrame:
    products, seed, kwargs, tags = job
    return run_evaluation(products, DAYS, seed, **kwargs).assign(seed=seed, **tags)


def run_all(pool: ProcessPoolExecutor, jobs: list) -> pd.DataFrame:
    return pd.concat(pool.map(_run, jobs), ignore_index=True)


def t_interval(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def fmt_ci(values) -> str:
    mean, low, high = t_interval(values)
    flag = "excludes 0" if low > 0 or high < 0 else "INCLUDES 0"
    return f"{mean:>9,.0f}  [{low:>9,.0f}, {high:>9,.0f}]  {flag}"


def totals_of(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["product_id"].isna()]


def print_policy_totals(results: pd.DataFrame, labels: dict[str, str]) -> None:
    totals = totals_of(results)
    print(f"  {'policy':<18}{'profit (Rs)':>13}{'fill rate':>11}{'stockout days':>15}{'units spoiled':>15}{'orders/month':>14}")
    for policy, label in labels.items():
        rows = totals[totals["policy"] == policy]
        print(f"  {label:<18}{rows['total_profit'].mean():>13,.0f}{rows['fill_rate'].mean():>11.3f}"
              f"{rows['stockout_days'].mean():>15.1f}{rows['units_spoiled'].mean():>15.1f}"
              f"{rows['orders_placed'].mean() * 30 / DAYS:>14.1f}")


def print_paired(results: pd.DataFrame, pairs: list[tuple[str, str, str]]) -> None:
    profit = totals_of(results).pivot(index="seed", columns="policy", values="total_profit")
    print(f"  {'comparison':<34}{'mean':>9}   {'95% CI':^22}")
    for a, b, label in pairs:
        print(f"  {label:<34}{fmt_ci(profit[a] - profit[b])}")


def tune_baseline(pool, products: pd.DataFrame, names: pd.Series, extra: dict, title: str) -> tuple[dict, dict]:
    """Grid-search safety factor x cover days per product on the tuning seeds (baseline only)."""
    jobs = [
        (products, seed, {"include_dss": False, "baseline_safety_factor": sf, "baseline_cover_days": cd, **extra},
         {"sf": sf, "cd": cd})
        for sf, cd in itertools.product(SAFETY_FACTORS, COVER_DAYS)
        for seed in TUNING_SEEDS
    ]
    tuning = run_all(pool, jobs)
    per_product = tuning[tuning["product_id"].notna()].astype({"product_id": int})
    mean_profit = per_product.groupby(["product_id", "sf", "cd"])["total_profit"].mean()
    tuned_sf, tuned_cd = {}, {}
    print(f"{title} (tuning seeds 100-119, baseline only; highest mean profit over "
          f"{len(SAFETY_FACTORS) * len(COVER_DAYS)} combinations)")
    print(f"  {'id':>2} {'product':<24}{'safety':>8}{'cover':>7}{'mean profit':>13}{'default (1.2, 7)':>18}")
    for pid in sorted(per_product["product_id"].unique()):
        grid = mean_profit.loc[pid]
        sf, cd = grid.idxmax()
        tuned_sf[pid], tuned_cd[pid] = float(sf), int(cd)
        shelf = products.set_index("product_id").loc[pid, "shelf_life_days"]
        note = f"  (cover capped at {max(1, min(cd, int(shelf) - 1))})" if not pd.isna(shelf) else ""
        print(f"  {pid:>2} {names[pid]:<24}{sf:>8.1f}{cd:>7d}{grid.max():>13,.0f}{grid.loc[(1.2, 7)]:>18,.0f}{note}")
    return tuned_sf, tuned_cd


def print_per_product(results: pd.DataFrame, names: pd.Series, baseline_label: str) -> None:
    print(f"\n  Per product, dss - {baseline_label} profit (Rs):")
    rows = results[results["product_id"].notna()].astype({"product_id": int})
    profit = rows.pivot_table(index=["product_id", "seed"], columns="policy", values="total_profit")
    fill = rows.pivot_table(index="product_id", columns="policy", values="fill_rate")
    print(f"  {'id':>2} {'product':<24}{'mean':>9}   {'95% CI':^22}{'fill base':>12}{'fill dss':>10}{'fill no-ev':>12}")
    for pid in sorted(rows["product_id"].unique()):
        diff = profit.loc[pid, "dss"] - profit.loc[pid, "baseline"]
        print(f"  {pid:>2} {names[pid]:<24}{fmt_ci(diff)}{fill.loc[pid, 'baseline']:>12.3f}"
              f"{fill.loc[pid, 'dss']:>10.3f}{fill.loc[pid, 'dss_no_evidence']:>12.3f}")


def main() -> None:
    started = time.perf_counter()
    products = pd.read_csv(ROOT / "data" / "sample_products.csv")
    names = products.set_index("product_id")["name"]
    workers = os.cpu_count() or 1
    print(f"Robustness report: {DAYS}-day runs, {workers} worker processes.\n")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # a. Tune the baseline.
        t0 = time.perf_counter()
        tuned_sf, tuned_cd = tune_baseline(pool, products, names, {}, "a. Tuned baseline")
        print(f"  section a runtime: {time.perf_counter() - t0:.1f} s\n")

        # b. Tuned baseline vs DSS on the test seeds.
        t0 = time.perf_counter()
        tuned_kwargs = {"baseline_safety_factor": tuned_sf, "baseline_cover_days": tuned_cd}
        test = run_all(pool, [(products, seed, tuned_kwargs, {}) for seed in TEST_SEEDS])
        print("b. Test seeds 0-19: tuned baseline vs dss vs dss_no_evidence")
        print_policy_totals(test, {"baseline": "tuned baseline", "dss": "dss", "dss_no_evidence": "dss_no_evidence"})
        print()
        print_paired(test, [("dss", "baseline", "dss - tuned baseline"),
                            ("dss_no_evidence", "baseline", "dss_no_evidence - tuned baseline"),
                            ("dss", "dss_no_evidence", "dss - dss_no_evidence")])
        print_per_product(test, names, "tuned baseline")
        print(f"  section b runtime: {time.perf_counter() - t0:.1f} s\n")

        # c. Cost sensitivity.
        t0 = time.perf_counter()
        jobs = []
        for mult, cost in itertools.product(PENALTY_MULTS, ORDER_COSTS):
            scaled = products.assign(stockout_penalty=products["stockout_penalty"] * mult)
            jobs += [(scaled, seed, {"order_cost": cost, "include_ablation": False}, {"mult": mult, "cost": cost})
                     for seed in COST_SEEDS]
        costs = run_all(pool, jobs)
        totals = totals_of(costs)
        print("c. Cost sensitivity (test seeds 0-9, default baseline): mean dss - baseline profit (Rs) [95% CI]")
        print("   * = CI includes 0")
        print(f"  {'penalty x  /  order cost':<26}" + "".join(f"{f'Rs {cost:.0f}':^34}" for cost in ORDER_COSTS))
        for mult in PENALTY_MULTS:
            cells = []
            for cost in ORDER_COSTS:
                cell = totals[(totals["mult"] == mult) & (totals["cost"] == cost)]
                p = cell.pivot(index="seed", columns="policy", values="total_profit")
                mean, low, high = t_interval(p["dss"] - p["baseline"])
                star = "*" if low <= 0 <= high else " "
                cells.append(f"{mean:>8,.0f} [{low:>8,.0f}, {high:>8,.0f}]{star}".center(34))
            print(f"  {f'{mult:g}':<26}" + "".join(cells))
        print(f"  section c runtime: {time.perf_counter() - t0:.1f} s\n")

        # d. Misspecified demand.
        t0 = time.perf_counter()
        misspec = run_all(pool, [(products, seed, {"generator_options": MISSPECIFICATION}, {}) for seed in TEST_SEEDS])
        print(f"d. Misspecified demand {MISSPECIFICATION} (test seeds 0-19, default baseline);")
        print("   the DSS's model still assumes Poisson demand with no month-start effect")
        print_policy_totals(misspec, {"baseline": "default baseline", "dss": "dss", "dss_no_evidence": "dss_no_evidence"})
        print()
        print_paired(misspec, [("dss", "baseline", "dss - baseline"),
                               ("dss_no_evidence", "baseline", "dss_no_evidence - baseline"),
                               ("dss", "dss_no_evidence", "dss - dss_no_evidence")])
        print(f"  section d runtime: {time.perf_counter() - t0:.1f} s\n")

        # e. Weekend-aware baseline, tuned, vs DSS on the test seeds.
        t0 = time.perf_counter()
        aware = {"baseline_weekend_aware": True}
        aware_sf, aware_cd = tune_baseline(pool, products, names, aware, "e. Tuned weekend-aware baseline")
        aware_kwargs = {"baseline_safety_factor": aware_sf, "baseline_cover_days": aware_cd, **aware}
        weekend_test = run_all(pool, [(products, seed, aware_kwargs, {}) for seed in TEST_SEEDS])
        print("\n  Test seeds 0-19: tuned weekend-aware baseline vs dss vs dss_no_evidence")
        print_policy_totals(weekend_test, {"baseline": "weekend-aware base", "dss": "dss",
                                           "dss_no_evidence": "dss_no_evidence"})
        print()
        print_paired(weekend_test, [("dss", "baseline", "dss - weekend-aware baseline"),
                                    ("dss_no_evidence", "baseline", "dss_no_evidence - weekend-aware"),
                                    ("dss", "dss_no_evidence", "dss - dss_no_evidence")])
        print_per_product(weekend_test, names, "weekend-aware baseline")
        print(f"  section e runtime: {time.perf_counter() - t0:.1f} s\n")

    print(f"Total runtime: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
