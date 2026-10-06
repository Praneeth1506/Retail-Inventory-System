"""Evaluation report: default baseline vs. DSS vs. no-evidence ablation over test seeds 0-19, 90 days each.

Analyses come from scripts/reporting.py (shared with export_results.py); this script only prints them.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reporting as rp  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    products = rp.load_products()
    with rp.make_pool() as pool:
        results = rp.evaluate_seeds(pool, products, rp.TEST_SEEDS)
    print(f"Evaluation: {len(rp.TEST_SEEDS)} test seeds, {rp.DAYS} days each, default settings.\n")
    with rp.pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
        print("Per policy (total over 10 products, averaged over seeds):")
        print(rp.policy_totals(results).to_string(index=False))
        print("\nPaired profit differences per seed (mean, 95% t CI):")
        print(rp.paired_comparisons(results).to_string(index=False))
        print("\nPer product (averaged over seeds):")
        print(rp.per_product(results, products).to_string(index=False))
    print(f"\nTotal runtime: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
