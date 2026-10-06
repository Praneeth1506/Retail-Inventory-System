"""Preview daily order recommendations: linear vs. risk-averse, weekday vs. weekend + promo."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision_engine.bayes_risk import calculate_stockout_risk  # noqa: E402
from decision_engine.data_generator import generate_sales_history  # noqa: E402
from decision_engine.recommend import recommend_order  # noqa: E402
from preview_risk import scenarios  # noqa: E402

DAYS = 180
SEED = 42
HISTORY_START = date(2025, 1, 1)
ORDER_COST = 30.0
UTILITIES = [("linear", None), ("R=1000", 1000.0), ("R=5000", 5000.0)]
PROMO_SPAN_DAYS = 60  # "promotion on every day": covers any horizon; dates outside it are ignored


def main() -> None:
    products = pd.read_csv(ROOT / "data" / "sample_products.csv")
    history = generate_sales_history(products, DAYS, SEED, start_date=HISTORY_START)
    first_day = HISTORY_START + timedelta(days=DAYS)
    print(f"History: {DAYS} days, seed {SEED}. order_cost Rs {ORDER_COST:.0f}. "
          "Review period = economic (EOQ) cycle from recommend_order.")
    print("current_stock = expected lead-time demand of the window.")
    print("Risk aversion vs. linear: ^ raised the order, v lowered it, = same.\n")

    for p in products.itertuples(index=False):
        product = products[products["product_id"] == p.product_id].iloc[0]
        lead = int(p.lead_time_days)
        shelf_text = f"shelf life {int(p.shelf_life_days)} d" if not pd.isna(p.shelf_life_days) else "non-perishable"
        print(f"[{p.product_id}] {p.name}  (lead {lead} d, {shelf_text})")

        for label, start, promo in scenarios(lead, first_day):
            if promo:
                promo = [start + timedelta(days=i) for i in range(PROMO_SPAN_DAYS)]
            lead_risk = calculate_stockout_risk(p.product_id, 0, lead, start, history, promo_dates=promo)
            stock = round(lead_risk["expected_lead_time_demand"])

            rows, linear_qty = [], None
            for name, tolerance in UTILITIES:
                rec = recommend_order(product, stock, start, history, promo_dates=promo,
                                      order_cost=ORDER_COST, risk_tolerance=tolerance)
                order = rec["order"]
                qty = order["optimal_qty"]
                if linear_qty is None:
                    linear_qty, mark = qty, ""
                else:
                    mark = "^" if qty > linear_qty else "v" if qty < linear_qty else "="
                bound = "  (at upper bound!)" if order["at_upper_bound"] else ""
                rows.append(f"      {name:<8}{qty:>5}{order['expected_profit']:>12.1f}"
                            f"{order['certainty_equivalent']:>13.1f}{order['no_stockout_probability']:>16.3f}  {mark}{bound}")

            review = rec["review_period_days"]
            risk = rec["risk"]
            print(f"    {label} (starts {start:%a %d %b}, stock {stock}): review period {review} d "
                  f"(~{30 / review:.1f} orders/month), horizon {risk['order_horizon_days']} d, "
                  f"expected horizon demand {risk['expected_order_horizon_demand']:.1f}")
            print(f"      {'utility':<8}{'qty':>5}{'E[profit]':>12}{'cert. equiv':>13}{'P(no stockout)':>16}  vs linear")
            print("\n".join(rows))
        print()


if __name__ == "__main__":
    main()
