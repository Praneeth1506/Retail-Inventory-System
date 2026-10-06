from src.rules import isa_closure, get_category_policy, evaluate_alert_rules

if __name__ == "__main__":
    print("--- 1. Testing ISA Category Closure ---")
    milk_closure = isa_closure("Milk")
    print(f"Categories for 'Milk': {milk_closure}")

    print("\n--- 2. Testing Category Policy for P001 (Whole Milk) ---")
    policy = get_category_policy("P001")
    print(f"Product: {policy['name']}")
    print(f"Spoilage Applies: {policy['spoilage_applies']}")
    print(f"Applied Rules: {policy['applied_rules']}")

    print("\n--- 3. Testing Alert Rules for P001 at 60% Stockout Risk ---")
    alerts = evaluate_alert_rules("P001", stockout_risk=0.6)
    for alert in alerts:
        print(f"[{alert['severity']}] {alert['message']}")
        print(f" Triggered by: {alert['rule_triggered']}")