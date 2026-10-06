"""
Mock implementation of Person 2's AI Engine (Bayes Risk & Utility Optimizer).
This allows Person 1 to develop and test the Agent & Streamlit UI independently.
"""

def calculate_stockout_risk(product_id: str, current_stock: int, is_weekend: bool = False, has_promo: bool = False) -> float:
    """
    Mock function for Person 2's Bayes Risk Model.
    Returns a stockout probability between 0.0 and 1.0 based on simple heuristics.
    """
    base_risk = 0.2
    if current_stock < 15:
        base_risk += 0.4
    if is_weekend:
        base_risk += 0.15
    if has_promo:
        base_risk += 0.2
    
    return min(1.0, max(0.0, base_risk))


def calculate_optimal_order_quantity(product_id: str, stockout_risk: float, category_policy: dict) -> dict:
    """
    Mock function for Person 2's Utility Theory Optimizer.
    Returns recommended reorder quantity and expected monetary utility.
    """
    if stockout_risk > 0.4:
        recommended_qty = 50 if category_policy.get("spoilage_applies") else 100
        expected_utility = 250.0 * (1 - stockout_risk)
    else:
        recommended_qty = 0
        expected_utility = 500.0

    return {
        "recommended_qty": recommended_qty,
        "expected_utility": round(expected_utility, 2),
        "posterior_distribution": {
            "Low Demand": round(max(0.1, 1.0 - stockout_risk), 2),
            "Normal Demand": 0.2,
            "High Demand": round(min(0.7, stockout_risk), 2)
        },
        "utility_by_candidate_qty": {
            0: 100.0,
            25: 180.0,
            50: expected_utility,
            75: expected_utility - 20.0,
            100: expected_utility - 50.0
        }
    }


def simulate_one_day(current_date: str) -> dict:
    """
    Mock function for day simulation.
    """
    return {
        "date": current_date,
        "status": "Simulated 1 day progression."
    }