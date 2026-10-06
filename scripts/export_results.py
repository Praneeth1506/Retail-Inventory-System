"""Run every evaluation analysis once and write the results as CSV files to results/.

The Streamlit evaluation page and README read these files; nothing is run live.
See results/README.md for what each file contains.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reporting as rp  # noqa: E402

OUT = rp.ROOT / "results"


def write(name: str, df) -> None:
    df.to_csv(OUT / name, index=False)
    print(f"  wrote results/{name} ({len(df)} rows)")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    products = rp.load_products()
    timings = []

    def section(label: str, t0: float) -> None:
        seconds = time.perf_counter() - t0
        timings.append({"section": label, "seconds": round(seconds, 1)})
        print(f"  [{label}: {seconds:.1f} s]")

    with rp.make_pool() as pool:
        t0 = time.perf_counter()
        print("Default baseline vs dss vs dss_no_evidence (test seeds 0-19)")
        default = rp.evaluate_seeds(pool, products, rp.TEST_SEEDS)
        write("default_policy_totals.csv", rp.policy_totals(default))
        write("default_paired.csv", rp.paired_comparisons(default))
        write("default_per_product.csv", rp.per_product(default, products))
        section("default", t0)

        t0 = time.perf_counter()
        print("Tuned baseline (tuned on seeds 100-119, tested on 0-19)")
        params, sf, cd = rp.tune_baseline(pool, products)
        write("tuned_baseline_params.csv", params)
        tuned = rp.evaluate_seeds(pool, products, rp.TEST_SEEDS,
                                  {"baseline_safety_factor": sf, "baseline_cover_days": cd})
        write("tuned_policy_totals.csv", rp.policy_totals(tuned))
        write("tuned_paired.csv", rp.paired_comparisons(tuned))
        write("tuned_per_product.csv", rp.per_product(tuned, products))
        section("tuned", t0)

        t0 = time.perf_counter()
        print("Cost sensitivity (default baseline, test seeds 0-9)")
        write("cost_sensitivity.csv", rp.cost_sensitivity(pool, products))
        section("cost_sensitivity", t0)

        t0 = time.perf_counter()
        print(f"Misspecified demand {rp.MISSPECIFICATION} (default baseline, test seeds 0-19)")
        misspec = rp.evaluate_seeds(pool, products, rp.TEST_SEEDS, {"generator_options": rp.MISSPECIFICATION})
        write("misspec_policy_totals.csv", rp.policy_totals(misspec))
        write("misspec_paired.csv", rp.paired_comparisons(misspec))
        section("misspecification", t0)

        t0 = time.perf_counter()
        print("Weekend-aware baseline (tuned on seeds 100-119, tested on 0-19)")
        aware = {"baseline_weekend_aware": True}
        params, sf, cd = rp.tune_baseline(pool, products, aware)
        write("weekend_baseline_params.csv", params)
        weekend = rp.evaluate_seeds(pool, products, rp.TEST_SEEDS,
                                    {"baseline_safety_factor": sf, "baseline_cover_days": cd, **aware})
        write("weekend_policy_totals.csv", rp.policy_totals(weekend))
        write("weekend_paired.csv", rp.paired_comparisons(weekend))
        write("weekend_per_product.csv", rp.per_product(weekend, products))
        section("weekend_aware", t0)

    t0 = time.perf_counter()
    print("Credible interval coverage (50 seeds x 10 products x 4 groups)")
    write("coverage.csv", rp.credible_interval_coverage(products))
    section("coverage", t0)

    total = time.perf_counter() - started
    timings.append({"section": "total", "seconds": round(total, 1)})
    write("runtime.csv", rp.pd.DataFrame(timings))
    print(f"\nTotal runtime: {total:.1f} s")


if __name__ == "__main__":
    main()
