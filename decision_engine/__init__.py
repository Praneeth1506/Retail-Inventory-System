"""Decision engine: pure-Python decision backend for the Retail Inventory Decision Support System.

The real public functions are exported here with the same signatures as decision_engine.mock:
    import decision_engine as engine
    rec = engine.recommend_order(product_row, current_stock, today, history_df, promo_dates=promos)
See decision_engine/interface.md for the contract.
"""

from decision_engine.bayes_risk import calculate_stockout_risk
from decision_engine.data_generator import generate_sales_history, simulate_one_day
from decision_engine.evaluation import run_evaluation
from decision_engine.inventory import effective_inventory_position
from decision_engine.optimizer import calculate_optimal_order_quantity, economic_review_period
from decision_engine.recommend import recommend_order

__all__ = [
    "generate_sales_history",
    "simulate_one_day",
    "calculate_stockout_risk",
    "calculate_optimal_order_quantity",
    "economic_review_period",
    "effective_inventory_position",
    "recommend_order",
    "run_evaluation",
]
