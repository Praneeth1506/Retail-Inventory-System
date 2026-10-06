"""Evaluation report: baseline vs. DSS vs. no-evidence ablation over test seeds 0-19, 90 days each."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision_engine.evaluation import run_evaluation  # noqa: E402

DAYS = 90
SEEDS = range(20)
POLICIES = ["baseline", "dss", "dss_no_evidence"]


def t_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def main() -> None:
    started = time.perf_counter()
    products = pd.read_csv(ROOT / "data" / "sample_products.csv")
    names = products.set_index("product_id")["name"]

    frames = []
    for seed in SEEDS:
        df = run_evaluation(products, DAYS, seed)
        frames.append(df.assign(seed=seed))
        print(f"  seed {seed:>2} done ({time.perf_counter() - started:.0f} s)", flush=True)
    results = pd.concat(frames, ignore_index=True)
    totals = results[results["product_id"].isna()]
    per_product = results[results["product_id"].notna()].astype({"product_id": int})

    print(f"\nEvaluation: {len(SEEDS)} test seeds (0-{max(SEEDS)}), {DAYS} days each, defaults "
          "(history 180 days, order_cost Rs 30, risk-neutral, economic review period,\n"
          "effective inventory position for both policies, DSS orders only when waiting costs more).\n")

    print("Per policy (total over 10 products, averaged over seeds):")
    print(f"  {'policy':<16}{'profit (Rs)':>13}{'fill rate':>11}{'stockout days':>15}{'units spoiled':>15}"
          f"{'orders':>9}{'orders/month':>14}")
    for policy in POLICIES:
        rows = totals[totals["policy"] == policy]
        print(f"  {policy:<16}{rows['total_profit'].mean():>13,.0f}{rows['fill_rate'].mean():>11.3f}"
              f"{rows['stockout_days'].mean():>15.1f}{rows['units_spoiled'].mean():>15.1f}"
              f"{rows['orders_placed'].mean():>9.1f}{rows['orders_placed'].mean() * 30 / DAYS:>14.1f}")

    profit = totals.pivot(index="seed", columns="policy", values="total_profit")
    print("\nPaired profit differences per seed (mean, 95% t confidence interval):")
    for a, b in (("dss", "baseline"), ("dss", "dss_no_evidence")):
        mean, low, high = t_interval((profit[a] - profit[b]).to_numpy())
        excludes = "excludes 0" if low > 0 or high < 0 else "includes 0"
        print(f"  {a} - {b:<16} {mean:>10,.0f}   [{low:>10,.0f}, {high:>10,.0f}]   {excludes}")

    print("\nPer product (averaged over seeds):")
    print(f"  {'':>2} {'':<22}{'profit diff':>12} |{'fill rate':^22}|{'units spoiled':^22}|{'orders / month':^22}")
    header = "".join(f"{label:>7}" for label in ("base", "dss", "no-ev"))
    print(f"  {'id':>2} {'product':<22}{'dss - base':>12} |{header} |{header} |{header}")
    for pid in sorted(per_product["product_id"].unique()):
        rows = per_product[per_product["product_id"] == pid]
        by_policy = {p: rows[rows["policy"] == p].set_index("seed") for p in POLICIES}
        diff = (by_policy["dss"]["total_profit"] - by_policy["baseline"]["total_profit"]).mean()
        fill = "".join(f"{by_policy[p]['fill_rate'].mean():>7.3f}" for p in POLICIES)
        spoiled = "".join(f"{by_policy[p]['units_spoiled'].mean():>7.1f}" for p in POLICIES)
        per_month = "".join(f"{by_policy[p]['orders_placed'].mean() * 30 / DAYS:>7.1f}" for p in POLICIES)
        print(f"  {pid:>2} {names[pid]:<22}{diff:>12,.0f} |{fill} |{spoiled} |{per_month}")

    print(f"\nTotal runtime: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
