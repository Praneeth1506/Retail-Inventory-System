"""Preview synthetic sales: mean units sold per product by weekend/promo group over 180 days."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision_engine.data_generator import generate_sales_history  # noqa: E402

DAYS = 180
SEED = 42
GROUPS = [
    ("weekday", False, False),
    ("weekend", True, False),
    ("wkday+promo", False, True),
    ("wkend+promo", True, True),
]


def main() -> None:
    products = pd.read_csv(ROOT / "data" / "sample_products.csv")
    history = generate_sales_history(products, DAYS, SEED)
    names = products.set_index("product_id")["name"]

    print(f"{DAYS} days from {history['date'].min()} to {history['date'].max()}, seed {SEED}")
    print("Each cell: mean units sold per day (number of days)\n")
    header = f"{'id':>2}  {'product':<24}" + "".join(f"{label:>16}" for label, _, _ in GROUPS)
    print(header + f"{'promo %':>9}")
    print("-" * (len(header) + 9))

    for product_id, rows in history.groupby("product_id"):
        cells = []
        for _, weekend, promo in GROUPS:
            group = rows[(rows["is_weekend"] == weekend) & (rows["has_promo"] == promo)]
            mean = f"{group['units_sold'].mean():.1f}" if len(group) else "-"
            cells.append(f"{mean} ({len(group)})".rjust(16))
        promo_pct = f"{100 * rows['has_promo'].mean():.1f}%"
        print(f"{product_id:>2}  {names[product_id]:<24}" + "".join(cells) + f"{promo_pct:>9}")


if __name__ == "__main__":
    main()
