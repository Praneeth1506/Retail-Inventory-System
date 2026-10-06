from src.db import get_isa_relations, get_products, get_stock


def isa_closure(category: str, isa_relations: list[tuple[str, str]] = None) -> set[str]:
    """
    Computes the transitive closure of categories using BFS to avoid infinite loops.
    Example: 'Milk' -> {'Milk', 'Dairy', 'Perishable'}
    """
    if isa_relations is None:
        isa_relations = get_isa_relations()

    visited = set()
    queue = [category]

    while queue:
        curr = queue.pop(0)
        if curr not in visited:
            visited.add(curr)
            # Find all parents where child == curr
            parents = [parent for child, parent in isa_relations if child == curr]
            queue.extend(parents)

    return visited


def get_category_policy(product_id: str) -> dict:
    """
    Evaluates First Order Predicate Logic rules on a product to determine its policy.
    Rules:
    - ∀x IsA(x, Perishable) → Policy(x, spoilage_applies = True)
    - ∀x IsA(x, Perishable) → Policy(x, planning_horizon = shelf_life_days)
    - ∀x IsA(x, Seasonal) → Policy(x, promo_sensitive = True)
    """
    df_products = get_products()
    product_row = df_products[df_products["product_id"] == product_id]

    if product_row.empty:
        raise ValueError(f"Product ID '{product_id}' not found.")

    product = product_row.iloc[0]
    category = product["category"]
    
    # Get all categories this item belongs to via ISA hierarchy
    all_categories = isa_closure(category)

    # Base policy defaults
    policy = {
        "product_id": product_id,
        "name": product["name"],
        "category": category,
        "all_categories": list(all_categories),
        "spoilage_applies": False,
        "promo_sensitive": False,
        "shelf_life_days": product["shelf_life_days"],
        "holding_cost_per_day": product["holding_cost_per_day"],
        "unit_cost": product["unit_cost"],
        "selling_price": product["selling_price"],
        "stockout_penalty": product["stockout_penalty"],
        "lead_time_days": product["lead_time_days"],
        "applied_rules": []
    }

    # Rule 1 & 2: Perishable Items
    if "Perishable" in all_categories:
        policy["spoilage_applies"] = True
        policy["applied_rules"].append(
            f"Rule Fired: ∀x IsA(x, Perishable) → Spoilage applies & Shelf Life ({product['shelf_life_days']} days) enforced."
        )

    # Rule 3: Seasonal Items
    if "Seasonal" in all_categories:
        policy["promo_sensitive"] = True
        policy["applied_rules"].append(
            "Rule Fired: ∀x IsA(x, Seasonal) → Marked as Promo Sensitive."
        )

    return policy


def evaluate_alert_rules(product_id: str, stockout_risk: float) -> list[dict]:
    """
    Evaluates threshold & risk rules for alerting.
    Rules:
    - ∀x StockoutRisk(x) > 0.7 → Alert(x, "High stockout risk")
    - ∀x IsA(x, Perishable) ∧ StockoutRisk(x) > 0.5 → Alert(x, "Perishable at risk")
    """
    policy = get_category_policy(product_id)
    all_categories = set(policy["all_categories"])
    alerts = []

    # Rule: High Stockout Risk (> 70%)
    if stockout_risk > 0.7:
        alerts.append({
            "product_id": product_id,
            "severity": "HIGH",
            "message": f"High stockout risk ({stockout_risk*100:.1f}%) detected.",
            "rule_triggered": "∀x StockoutRisk(x) > 0.7 → Alert(x, 'High stockout risk')"
        })

    # Rule: Perishable at Risk (> 50%)
    if "Perishable" in all_categories and stockout_risk > 0.5:
        alerts.append({
            "product_id": product_id,
            "severity": "CRITICAL",
            "message": f"Perishable item '{policy['name']}' has elevated stockout risk ({stockout_risk*100:.1f}%).",
            "rule_triggered": "∀x IsA(x, Perishable) ∧ StockoutRisk(x) > 0.5 → Alert(x, 'Perishable at risk')"
        })

    return alerts