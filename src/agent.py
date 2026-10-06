from datetime import datetime
from src.db import get_products, get_stock, log_decision
from src.rules import get_category_policy, evaluate_alert_rules

# Import Mock backend initially. Swap to Person 2's module once ready.
try:
    import src.person2_engine as p2_engine
except ImportError:
    import src.mock_person2 as p2_engine


class InventoryAgent:
    def __init__(self):
        self.state = "Idle"

    def transition(self, new_state: str):
        """Transitions the agent to a new state."""
        self.state = new_state

    def run_agent_cycle(self, is_weekend: bool = False, has_promo: bool = False) -> list[dict]:
        """
        Executes a full cycle across all inventory products:
        Idle -> Checking Stock -> Classifying -> Calculating Risk -> Optimizing Order -> Alerting & Logging -> Idle
        """
        results = []
        df_products = get_products()

        for _, product in df_products.iterrows():
            p_id = product["product_id"]

            # Step 1: Checking Stock
            self.transition("Checking Stock")
            current_stock = get_stock(p_id)

            # Step 2: Classifying (Rule Engine)
            self.transition("Classifying")
            policy = get_category_policy(p_id)

            # Step 3: Calculating Risk (Person 2 AI Engine)
            self.transition("Calculating Risk")
            stockout_risk = p2_engine.calculate_stockout_risk(
                product_id=p_id, 
                current_stock=current_stock, 
                is_weekend=is_weekend, 
                has_promo=has_promo
            )

            # Step 4: Optimizing Order (Person 2 Utility Engine)
            self.transition("Optimizing Order")
            opt_result = p2_engine.calculate_optimal_order_quantity(
                product_id=p_id, 
                stockout_risk=stockout_risk, 
                category_policy=policy
            )

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
                "stockout_risk": stockout_risk,
                "recommended_qty": opt_result["recommended_qty"],
                "expected_utility": opt_result["expected_utility"],
                "alerts": alerts,
                "policy": policy,
                "posterior": opt_result.get("posterior_distribution", {}),
                "utility_candidates": opt_result.get("utility_by_candidate_qty", {})
            })

        # Return to Idle
        self.transition("Idle")
        return results