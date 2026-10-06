"""Preview the Bayes stockout risk model: posterior group rates vs. true rates, and risk by scenario."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision_engine.bayes_risk import GROUPS, calculate_stockout_risk  # noqa: E402
from decision_engine.data_generator import _demand_rate, generate_sales_history  # noqa: E402

DAYS = 180
SEED = 42
HISTORY_START = date(2025, 1, 1)
EVIDENCE = {
    "weekday": (False, False),
    "weekend": (True, False),
    "weekday_promo": (False, True),
    "weekend_promo": (True, True),
}


def scenarios(lead: int, first_day: date) -> list[tuple[str, date, list[date]]]:
    """Weekday window starting on a Monday; weekend window covering a Saturday, promo every day."""
    monday = first_day + timedelta(days=(7 - first_day.weekday()) % 7)
    saturday = monday + timedelta(days=5)
    weekend_start = saturday - timedelta(days=max(0, lead - 2) // 2)
    promo = [weekend_start + timedelta(days=i) for i in range(lead + 1)]
    return [
        ("weekday, no promo", monday, []),
        ("weekend + promo", weekend_start, promo),
    ]


def main() -> None:
    products = pd.read_csv(ROOT / "data" / "sample_products.csv")
    history = generate_sales_history(products, DAYS, SEED, start_date=HISTORY_START)
    first_day = HISTORY_START + timedelta(days=DAYS)
    print(f"History: {DAYS} days from {HISTORY_START}, seed {SEED}. Rates are units per day.\n")

    for p in products.itertuples(index=False):
        lead = int(p.lead_time_days)
        cases = scenarios(lead, first_day)
        base = calculate_stockout_risk(p.product_id, 0, lead, cases[0][1], history)

        print(f"[{p.product_id}] {p.name}  (lead time {lead} d)")
        print(f"    {'group':<14}{'n_days':>7}{'posterior mean':>16}{'90% interval':>20}{'true':>8}{'error':>8}")
        for group in GROUPS:
            r = base["group_rates"][group]
            truth = _demand_rate(p.name, p.category, *EVIDENCE[group])
            interval = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
            error = f"{100 * (r['posterior_mean'] / truth - 1):+.1f}%"
            print(f"    {group:<14}{r['n_days']:>7}{r['posterior_mean']:>16.2f}{interval:>20}{truth:>8.2f}{error:>8}")

        expected = base["expected_lead_time_demand"]
        print(f"    Stockout risk over the lead-time window "
              f"(expected weekday lead-time demand {expected:.1f}):")
        for label, start, promo in cases:
            groups = calculate_stockout_risk(p.product_id, 0, lead, start, history, promo_dates=promo)
            day_groups = ", ".join(f"{g} {k}" for g, k in groups["lead_time_day_groups"].items() if k)
            print(f"      {label:<18} starts {start:%a %d %b}: {day_groups}")
        for multiple in (1.0, 1.5):
            stock = round(multiple * expected)
            cells = []
            for label, start, promo in cases:
                result = calculate_stockout_risk(p.product_id, stock, lead, start, history, promo_dates=promo)
                cells.append(f"{label}: {result['stockout_risk']:.3f}")
            print(f"      stock {stock:>4} ({multiple:.1f}x)   " + "   ".join(cells))
        print()


if __name__ == "__main__":
    main()
