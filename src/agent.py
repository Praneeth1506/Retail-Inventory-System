from datetime import date, timedelta
from src.db import get_products, get_sales_history, get_stock, log_decision
from src.rules import get_category_policy, evaluate_alert_rules

# Person 2's decision engine, through one adapter (src/mock_person2.py is no longer used).
from src import engine_adapter


class InventoryAgent:
    def __init__(self):
        self.state = "Idle"

    def transition(self, new_state: str):
        """Transitions the agent to a new state."""
        self.state = new_state

    def run_agent_cycle(self, is_weekend: bool = False, has_promo: bool = False,
                        current_date: date | None = None) -> list[dict]:
        """
        Executes a full cycle across all inventory products:
        Idle -> Checking Stock -> Classifying -> Calculating Risk -> Optimizing Order -> Alerting & Logging -> Idle

        current_date: decision day (default today). The engine derives weekends from dates, so
        is_weekend=True on a weekday moves the decision to the next Saturday. has_promo=True means
        a promotion on every product for the next 7 days.
        """
        results = []
        df_products = get_products()
        current_date = current_date or date.today()
        if is_weekend and current_date.weekday() < 5:
            current_date += timedelta(days=5 - current_date.weekday())
        engine_adapter.ensure_sales_history(df_products, today=current_date)
        ids = engine_adapter.engine_id_map(df_products["product_id"])
        history = engine_adapter.to_engine_history(get_sales_history(), ids)
        promo_dates = engine_adapter.promo_window(current_date, has_promo)
        on_order = engine_adapter.on_order_quantities(df_products, current_date)

        for _, product in df_products.iterrows():
            p_id = product["product_id"]

            # Step 1: Checking Stock
            self.transition("Checking Stock")
            current_stock = get_stock(p_id)

            # Step 2: Classifying (Rule Engine)
            self.transition("Classifying")
            policy = get_category_policy(p_id)

            # Steps 3-4: Calculating Risk and Optimizing Order (Person 2 engine, one call).
            # current_stock = on hand + on order. On order comes from ReorderLogs and assumes every
            # logged recommendation was actually placed (see src/engine_adapter.py).
            self.transition("Calculating Risk")
            rec = engine_adapter.recommend(
                product, ids, current_stock + on_order[p_id], current_date, history, promo_dates,
                spoilage_cost=policy.get("spoilage_cost", 0.0),
            )
            self.transition("Optimizing Order")
            stockout_risk = rec["risk"]["stockout_risk"]
            opt_result = {
                "recommended_qty": rec["recommended_qty"],
                "expected_utility": rec["order"]["expected_utility"],
            }

            # Step 5: Alerting & Logging
            self.transition("Alerting & Logging")
            alerts = evaluate_alert_rules(p_id, stockout_risk)
            alert_reason = "; ".join([a["message"] for a in alerts]) if alerts else "Normal operation"

            # Log decision to SQLite
            log_decision(
                product_id=p_id,
                agent_state=self.state,
                stockout_risk=stockout_risk,
                recommended_qty=opt_result["recommended_qty"],
                expected_utility=opt_result["expected_utility"],
                alert_reason=alert_reason
            )

            results.append({
                "product_id": p_id,
                "name": policy["name"],
                "stock": current_stock,
                "on_order": on_order[p_id],
                "stockout_risk": stockout_risk,
                "recommended_qty": opt_result["recommended_qty"],
                "expected_utility": opt_result["expected_utility"],
                "alerts": alerts,
                "policy": policy,
                "decision_date": current_date,
                "should_order": rec["should_order"],
                "optimal_qty": rec["order"]["optimal_qty"],
                "review_period_days": rec["review_period_days"],
                "group_rates": rec["risk"]["group_rates"],
                "profit_by_qty": {q: c["expected_profit"] for q, c in rec["order"]["candidates"].items()},
            })

        # Return to Idle
        self.transition("Idle")
        return results