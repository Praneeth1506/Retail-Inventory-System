"""Robustness report: tuned baseline, cost sensitivity, misspecified demand, weekend-aware baseline.

Sections
  a. Tune the static baseline per product (safety factor x cover days) on tuning seeds 100-119.
  b. Test seeds 0-19: tuned baseline vs dss vs dss_no_evidence.
  c. Cost sensitivity (test seeds 0-9): stockout_penalty x {0.5, 1, 2} crossed with order_cost {0, 30, 100}.
  d. Misspecified demand (test seeds 0-19): overdispersion 10 and month-start multiplier 1.3.
  e. Weekend-aware baseline tuned like section a, then tested on seeds 0-19.
Analyses come from scripts/reporting.py (shared with export_results.py); this script only prints them.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reporting as rp  # noqa: E402


def show(title: str, df) -> None:
    print(title)
    print(df.to_string(index=False))
    print()


def main() -> None:
    started = time.perf_counter()
    products = rp.load_products()
    rp.pd.set_option("display.width", 220)
    rp.pd.set_option("display.float_format", "{:,.3f}".format)

    with rp.make_pool() as pool:
        t0 = time.perf_counter()
        params, sf, cd = rp.tune_baseline(pool, products)
        show("a. Tuned baseline parameters (tuning seeds 100-119)", params)
        print(f"  section a runtime: {time.perf_counter() - t0:.1f} s\n")

        t0 = time.perf_counter()
        tuned = rp.evaluate_seeds(pool, products, rp.TEST_SEEDS, {"baseline_safety_factor": sf, "baseline_cover_days": cd})
        show("b. Tuned baseline: per-policy totals", rp.policy_totals(tuned))
        show("   Paired comparisons", rp.paired_comparisons(tuned))
        show("   Per product", rp.per_product(tuned, products))
        print(f"  section b runtime: {time.perf_counter() - t0:.1f} s\n")

        t0 = time.perf_counter()
        show("c. Cost sensitivity (dss - default baseline)", rp.cost_sensitivity(pool, products))
        print(f"  section c runtime: {time.perf_counter() - t0:.1f} s\n")

        t0 = time.perf_counter()
        misspec = rp.evaluate_seeds(pool, products, rp.TEST_SEEDS, {"generator_options": rp.MISSPECIFICATION})
        show(f"d. Misspecified demand {rp.MISSPECIFICATION}: per-policy totals", rp.policy_totals(misspec))
        show("   Paired comparisons", rp.paired_comparisons(misspec))
        print(f"  section d runtime: {time.perf_counter() - t0:.1f} s\n")

        t0 = time.perf_counter()
        aware = {"baseline_weekend_aware": True}
        params, sf, cd = rp.tune_baseline(pool, products, aware)
        show("e. Tuned weekend-aware baseline parameters", params)
        weekend = rp.evaluate_seeds(pool, products, rp.TEST_SEEDS,
                                    {"baseline_safety_factor": sf, "baseline_cover_days": cd, **aware})
        show("   Per-policy totals", rp.policy_totals(weekend))
        show("   Paired comparisons", rp.paired_comparisons(weekend))
        show("   Per product", rp.per_product(weekend, products))
        print(f"  section e runtime: {time.perf_counter() - t0:.1f} s\n")

    print(f"Total runtime: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
