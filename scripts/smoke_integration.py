"""Integration smoke test without the UI: run the app's agent on the real engine for 3 simulated days.

Works on a temporary copy of data/inventory.db, so the committed database is not modified.
Between days, demand comes from decision_engine.simulate_one_day. StockLevels is treated as the
inventory position (on hand + on order), so recommended orders are added when placed.
"""

import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.db as db  # noqa: E402
from decision_engine import simulate_one_day  # noqa: E402
from src import engine_adapter  # noqa: E402
from src.agent import InventoryAgent  # noqa: E402

DAYS = 3


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db.DB_PATH = Path(tmp) / "inventory.db"
        shutil.copy(ROOT / "data" / "inventory.db", db.DB_PATH)
        db.init_db()

        start = date.today()
        agent = InventoryAgent()
        products = db.get_products()
        ids = engine_adapter.engine_id_map(products["product_id"])
        text_ids = {v: k for k, v in ids.items()}

        for offset in range(DAYS):
            day = start + timedelta(days=offset)
            results = agent.run_agent_cycle(current_date=day)
            if offset == 0:
                print(f"SalesHistory rows after seeding: {len(db.get_sales_history())}")
            print(f"\nDay {offset + 1}: {day:%a %d %b %Y}   agent state after cycle: {agent.state}")
            print(f"  {'id':<5}{'name':<26}{'stock':>6}{'stockout_risk':>15}{'recommended_qty':>17}{'review d':>10}")
            for r in results:
                print(f"  {r['product_id']:<5}{r['name']:<26}{r['stock']:>6}{r['stockout_risk']:>15.3f}"
                      f"{r['recommended_qty']:>17}{r['review_period_days']:>10}")

            demand = simulate_one_day(engine_adapter.to_engine_products(products, ids), day, {}, 1000 + offset)
            sold = {text_ids[pid]: int(units) for pid, units in zip(demand["product_id"], demand["units_sold"])}
            for r in results:
                pid = r["product_id"]
                db.update_stock(pid, max(0, r["stock"] - sold[pid]) + r["recommended_qty"])

        print(f"\nReorderLogs rows written in the copy: {len(db.get_logs())}")


if __name__ == "__main__":
    main()
