import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path
from src import engine_adapter
from src.agent import InventoryAgent
from src.db import get_products, get_logs, get_isa_relations, get_sales_history, get_stock, update_stock
from src.rules import isa_closure, get_category_policy

st.set_page_config(page_title="Retail Inventory AI Agent", layout="wide")

st.title("📦 Intelligent Retail Inventory Agent")
st.caption("First-Order Logic Rules + Bayesian Risk & Decision Theory Optimizer")

# Sidebar - Agent Controls
st.sidebar.header("🕹️ Agent Simulation Controls")

is_weekend = st.sidebar.checkbox("Is Weekend?", value=False)
has_promo = st.sidebar.checkbox("Active Promotion?", value=False)

if st.sidebar.button("🚀 Run Agent Decision Cycle", type="primary"):
    agent = InventoryAgent()
    with st.spinner("Agent running cycle across all products..."):
        results = agent.run_agent_cycle(is_weekend=is_weekend, has_promo=has_promo)
    st.sidebar.success(f"Cycle completed! State: {agent.state}")
    st.session_state["latest_results"] = results
    st.cache_data.clear()  # the cycle may have seeded SalesHistory

# Main Layout Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Inventory Dashboard", "🔍 Rules & ISA Reasoning", "📈 AI Risk & Utility", "📜 Decision Logs", "🧪 Evaluation Results", "🔮 What-If Explorer"])

# TAB 1: Dashboard Overview
with tab1:
    st.subheader("Current Inventory & Risk Summary")
    
    if "latest_results" in st.session_state:
        results = st.session_state["latest_results"]
        df_res = pd.DataFrame(results)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Products Tracked", len(df_res))
        high_risk_count = len(df_res[df_res["stockout_risk"] > 0.5])
        col2.metric("High/Critical Risk Items", high_risk_count)
        reorder_count = len(df_res[df_res["recommended_qty"] > 0])
        col3.metric("Products Needing Reorder", reorder_count)

        # Overview Table
        display_df = df_res[["product_id", "name", "stock", "stockout_risk", "recommended_qty", "expected_utility"]].copy()
        display_df["stockout_risk"] = display_df["stockout_risk"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Click 'Run Agent Decision Cycle' in the sidebar to populate live predictions.")

    st.markdown("#### Update stock level")
    with st.form("update_stock_form"):
        stock_products = get_products()
        stock_choice = st.selectbox("Product", stock_products["product_id"] + " - " + stock_products["name"],
                                    key="stock_form_product")
        stock_pid = stock_choice.split(" - ")[0]
        new_qty = st.number_input("New on-hand quantity", min_value=0, value=int(get_stock(stock_pid)), step=1,
                                  key="stock_form_qty")
        if st.form_submit_button("Save stock level"):
            update_stock(stock_pid, int(new_qty))
            st.success(f"{stock_pid} stock set to {int(new_qty)}.")

# TAB 2: Rules & ISA Hierarchy Explanations
with tab2:
    st.subheader("Knowledge Representation: First Order Predicate Logic & ISA Hierarchy")
    
    products = get_products()
    selected_prod = st.selectbox("Select Product to Inspect Reasoning:", products["product_id"] + " - " + products["name"])
    prod_id = selected_prod.split(" - ")[0]

    policy = get_category_policy(prod_id)

    st.markdown(f"**Direct Category:** `{policy['category']}`")
    st.markdown(f"**Transitive ISA Closure (`isa_closure`):** `{', '.join(policy['all_categories'])}`")

    st.write("---")
    st.subheader("Rules Fired for this Product:")
    if policy["applied_rules"]:
        for rule in policy["applied_rules"]:
            st.success(rule)
    else:
        st.info("No domain specific category predicate rules triggered for this product.")

# TAB 3: Bayesian Risk & Utility Theory
with tab3:
    st.subheader("Bayesian Risk & Expected Utility Optimization")
    
    if "latest_results" in st.session_state:
        results = st.session_state["latest_results"]
        selected_p = st.selectbox("Select Product for Utility Analysis:", [r["product_id"] + " - " + r["name"] for r in results])
        p_id = selected_p.split(" - ")[0]
        
        res = next(r for r in results if r["product_id"] == p_id)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Posterior Daily Demand Rate by Day Type")
            rates = res["group_rates"]
            if rates:
                groups = list(rates.keys())
                means = [rates[g]["posterior_mean"] for g in groups]
                fig_post = px.bar(
                    x=groups,
                    y=means,
                    error_y=[rates[g]["ci_high"] - rates[g]["posterior_mean"] for g in groups],
                    error_y_minus=[rates[g]["posterior_mean"] - rates[g]["ci_low"] for g in groups],
                    labels={'x': 'Day Type', 'y': 'Units per Day (90% interval)'},
                    title=f"Demand Rates (Stockout Risk: {res['stockout_risk']*100:.1f}%)"
                )
                st.plotly_chart(fig_post, use_container_width=True)

        with col_b:
            st.markdown("#### Expected Profit by Order Quantity")
            util_data = res["profit_by_qty"]
            if util_data:
                fig_util = px.line(
                    x=list(util_data.keys()),
                    y=list(util_data.values()),
                    markers=True,
                    labels={'x': 'Order Quantity', 'y': 'Expected Profit'},
                    title=f"Best Quantity: {res['optimal_qty']} units; order today: {res['recommended_qty']}"
                )
                st.plotly_chart(fig_util, use_container_width=True)
    else:
        st.info("Run the agent cycle to view probability distributions and utility curves.")

# TAB 4: Decision Audit Log
with tab4:
    st.subheader("SQLite Decision Audit Log")
    df_logs = get_logs()
    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No decision logs found. Run an agent cycle to log data.")

# TAB 5: Offline evaluation results (precomputed CSVs from scripts/export_results.py)
RESULTS_DIR = Path(__file__).parent / "results"
SCENARIOS = {
    "Tuned static baseline (main result)": "tuned",
    "Tuned weekend-aware baseline": "weekend",
    "Default static baseline": "default",
    "Misspecified demand (default baseline)": "misspec",
}

with tab5:
    st.subheader("Simulation results on the 10-product sample catalog (data/sample_products.csv)")
    st.caption("Precomputed 90-day simulations over 20 test seeds with synthetic demand; not the live "
               "app's catalog. Profit in Rs. Paired differences use a 95% t confidence interval across seeds.")
    if not (RESULTS_DIR / "tuned_policy_totals.csv").exists():
        st.warning("No results found. Run `python scripts/export_results.py` to create the results/ CSVs.")
    else:
        label = st.selectbox("Comparison:", list(SCENARIOS))
        prefix = SCENARIOS[label]

        st.markdown("#### Per-policy totals (all 10 products, mean over seeds)")
        totals = pd.read_csv(RESULTS_DIR / f"{prefix}_policy_totals.csv")
        st.dataframe(totals.round({"profit": 0, "fill_rate": 3, "stockout_days": 1, "units_spoiled": 1,
                                   "orders_placed": 1, "orders_per_month": 1}), use_container_width=True)

        st.markdown("#### Paired profit differences (95% CI)")
        paired = pd.read_csv(RESULTS_DIR / f"{prefix}_paired.csv")
        fig = px.scatter(paired, x="mean", y="comparison",
                         error_x=paired["ci_high"] - paired["mean"], error_x_minus=paired["mean"] - paired["ci_low"],
                         labels={"mean": "Profit difference (Rs, 90 days)", "comparison": ""})
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(paired.round({"mean": 0, "ci_low": 0, "ci_high": 0, "relative_to_b": 3}), use_container_width=True)

        product_file = RESULTS_DIR / f"{prefix}_per_product.csv"
        if product_file.exists():
            st.markdown("#### Per product: DSS minus baseline")
            st.dataframe(pd.read_csv(product_file).round(3), use_container_width=True)

        cost_file = RESULTS_DIR / "cost_sensitivity.csv"
        if cost_file.exists():
            st.markdown("#### Cost sensitivity: DSS minus default baseline (Rs)")
            st.dataframe(pd.read_csv(cost_file).round({"dss_minus_baseline": 0, "ci_low": 0, "ci_high": 0}),
                         use_container_width=True)


# TAB 6: What-If Explorer (read-only: never writes to the database)
@st.cache_data(ttl=300, show_spinner=False)
def whatif_history() -> tuple[pd.DataFrame, dict, str]:
    """Engine-format sales history. If SalesHistory is still empty, builds the same synthetic history the
    first agent cycle would write, in memory only."""
    products = get_products()
    ids = engine_adapter.engine_id_map(products["product_id"])
    sales = get_sales_history()
    source = "SalesHistory table"
    if sales.empty:
        sales = engine_adapter.synthetic_history_rows(products)
        source = "synthetic history (in memory; SalesHistory is not seeded yet)"
    return engine_adapter.to_engine_history(sales, ids), ids, source


def isa_chain(category: str) -> str:
    relations = get_isa_relations()
    chain, current = [category], category
    while True:
        parents = [parent for child, parent in relations if child == current and parent not in chain]
        if not parents:
            return " → ".join(chain)
        current = parents[0]
        chain.append(current)


def window_day_types(start: date, days: int, promo_dates: set) -> list[str]:
    types = []
    for i in range(days):
        d = start + timedelta(days=i)
        types.append(("weekend" if d.weekday() >= 5 else "weekday") + ("_promo" if d in promo_dates else ""))
    return types


with tab6:
    st.subheader("What-If Explorer")
    st.caption("Change any input to see how the decision engine responds. Read-only: nothing is written to the database.")
    wi_products = get_products()
    wi_choice = st.selectbox("Product", wi_products["product_id"] + " - " + wi_products["name"], key="wi_product")
    wi_pid = wi_choice.split(" - ")[0]
    row = wi_products[wi_products["product_id"] == wi_pid].iloc[0]
    db_on_order = engine_adapter.on_order_quantities(wi_products[wi_products["product_id"] == wi_pid], date.today())[wi_pid]

    c1, c2, c3 = st.columns(3)
    stock = c1.number_input("Current stock (on hand)", min_value=0, value=int(get_stock(wi_pid)), step=1, key="wi_stock")
    on_order = c2.number_input("Units on order", min_value=0, value=int(db_on_order), step=1, key="wi_on_order")
    decision_date = c3.date_input("Decision date", value=date.today(), key="wi_date")
    c3.caption(f"{decision_date:%A}")

    c4, c5 = st.columns(2)
    promo_on = c4.checkbox("Promotion running", value=False, key="wi_promo")
    promo_days = c5.number_input("Promotion days (from the decision date)", min_value=1, max_value=60, value=7,
                                 step=1, key="wi_promo_days", disabled=not promo_on)

    c6, c7, c8, c9 = st.columns(4)
    unit_cost = c6.number_input("Unit cost", min_value=0.0, value=float(row["unit_cost"]), step=0.1, key="wi_unit_cost")
    price = c7.number_input("Price", min_value=0.0, value=float(row["selling_price"]), step=0.1, key="wi_price")
    holding = c8.number_input("Holding cost / day", min_value=0.0, value=float(row["holding_cost_per_day"]),
                              step=0.01, format="%.2f", key="wi_holding")
    penalty = c9.number_input("Stockout penalty", min_value=0.0, value=float(row["stockout_penalty"]), step=0.5,
                              key="wi_penalty")
    c10, c11, c12 = st.columns(3)
    default_shelf = 0 if pd.isna(row["shelf_life_days"]) else int(row["shelf_life_days"])
    shelf = c10.number_input("Shelf life (days, 0 = not perishable)", min_value=0, value=default_shelf, step=1, key="wi_shelf")
    order_cost = c11.number_input("Order cost", min_value=0.0, value=float(engine_adapter.APP_ORDER_COST), step=0.5,
                                  key="wi_order_cost")
    risk_choice = c12.selectbox("Risk tolerance", ["None (risk-neutral)", "200", "50"], key="wi_risk")
    risk_tolerance = None if risk_choice.startswith("None") else float(risk_choice)

    history, ids, history_source = whatif_history()
    product = row.copy()
    product["unit_cost"], product["selling_price"] = unit_cost, price
    product["holding_cost_per_day"], product["stockout_penalty"] = holding, penalty
    product["shelf_life_days"] = shelf if shelf > 0 else None
    promo_dates = [decision_date + timedelta(days=i) for i in range(int(promo_days))] if promo_on else []
    available = int(stock + on_order)
    rec = engine_adapter.recommend(product, ids, available, decision_date, history, promo_dates,
                                   order_cost=order_cost, risk_tolerance=risk_tolerance)
    risk, order = rec["risk"], rec["order"]
    lead = int(row["lead_time_days"])

    decision = f"ORDER {rec['recommended_qty']} units" if rec["should_order"] else "WAIT"
    m1, m2, m3, m4 = st.columns([2, 1, 1, 1])
    m1.metric("Decision", decision)
    m2.metric("Stockout risk", f"{risk['stockout_risk'] * 100:.1f}%")
    m3.metric("Expected lead-time demand", f"{risk['expected_lead_time_demand']:.1f}")
    m4.metric("Review period", f"{rec['review_period_days']} days")

    if order["optimal_qty"] == 0:
        why = "Waiting won: no order quantity above zero is worth its cost over the order horizon."
    elif rec["should_order"]:
        why = (f"Ordering won: waiting one more day would cost an expected {rec['wait_cost']:.2f} in lost margin and "
               f"penalties, more than the {rec['early_holding_cost']:.2f} it costs to hold {order['optimal_qty']} units one extra day.")
    else:
        why = (f"Waiting won: one more day costs an expected {rec['wait_cost']:.2f} in shortages, less than the "
               f"{rec['early_holding_cost']:.2f} it would cost to hold {order['optimal_qty']} units one extra day.")
    st.markdown(f"**Wait cost {rec['wait_cost']:.2f} vs early holding cost {rec['early_holding_cost']:.2f}.** {why}")

    horizon_types = set(window_day_types(decision_date, risk["order_horizon_days"], set(promo_dates)))
    g1, g2 = st.columns(2)
    with g1:
        rates = risk["group_rates"]
        groups = list(rates)
        fig_rates = px.bar(
            x=groups, y=[rates[g]["posterior_mean"] for g in groups],
            error_y=[rates[g]["ci_high"] - rates[g]["posterior_mean"] for g in groups],
            error_y_minus=[rates[g]["posterior_mean"] - rates[g]["ci_low"] for g in groups],
            color=["in this order window" if g in horizon_types else "not in window" for g in groups],
            labels={"x": "Day type", "y": "Units per day (90% interval)", "color": ""},
            title="Demand rate by day type",
        )
        st.plotly_chart(fig_rates, use_container_width=True)
    with g2:
        dist = risk["lead_time_distribution"]
        fig_dist = go.Figure(go.Bar(
            x=list(dist), y=list(dist.values()),
            marker_color=["#d62728" if d > available else "#1f77b4" for d in dist],
        ))
        if dist and max(dist) > available:
            fig_dist.add_vrect(x0=available + 0.5, x1=max(dist) + 0.5, fillcolor="#d62728", opacity=0.08, line_width=0)
        fig_dist.add_vline(x=available + 0.5, line_dash="dash", annotation_text=f"available {available}")
        fig_dist.update_layout(title=f"Demand over the {lead}-day lead time (red = stockout)",
                               xaxis_title="Units demanded", yaxis_title="Probability")
        st.plotly_chart(fig_dist, use_container_width=True)

    candidates = order["candidates"]
    fig_profit = px.line(x=list(candidates), y=[c["expected_profit"] for c in candidates.values()],
                         labels={"x": "Order quantity", "y": "Expected profit"},
                         title=f"Expected profit by order quantity (best: {order['optimal_qty']})")
    fig_profit.add_vline(x=order["optimal_qty"], line_dash="dash",
                         annotation_text=f"Q* = {order['optimal_qty']}" + ("" if rec["should_order"] else " (waiting)"))
    st.plotly_chart(fig_profit, use_container_width=True)

    policy = get_category_policy(wi_pid)
    st.markdown("#### Rule engine")
    st.markdown(f"**ISA chain:** {isa_chain(policy['category'])}")
    st.markdown(f"**Perishable policy:** spoilage applies = `{policy['spoilage_applies']}`, "
                f"promo sensitive = `{policy['promo_sensitive']}`")
    for rule in policy["applied_rules"]:
        st.success(rule)

    lead_types = ", ".join(f"{k} {g}" for g, k in risk["lead_time_day_groups"].items() if k) or "no days"
    with st.expander("How it decided"):
        st.markdown(f"""
1. **Learn demand.** From the {history_source}, Bayes' theorem estimates a daily demand rate for each day type
   (weekday, weekend, with or without promotion), shown with 90% intervals in the first chart.
2. **Look at the calendar.** The decision is made on {decision_date:%A %d %b}. An order placed today arrives in
   {lead} days, so the lead-time window has {lead_types}.
3. **Measure the risk.** With {available} units available (on hand + on order), the chance that demand during the
   lead time exceeds them is {risk['stockout_risk'] * 100:.1f}% (the red bars in the second chart).
4. **Pick the order cycle.** Balancing the order cost against holding cost gives a review period of
   {rec['review_period_days']} days, so an order should cover {risk['order_horizon_days']} days
   ({lead} lead + {rec['review_period_days']} review).
5. **Choose how much.** Over that horizon, {order['optimal_qty']} units gives the highest expected
   {'profit' if risk_tolerance is None else 'utility (risk-averse)'}, trading off lost sales and penalties
   against holding and spoilage (third chart).
6. **Decide when.** Waiting one day costs {rec['wait_cost']:.2f}; ordering now costs {rec['early_holding_cost']:.2f}
   in extra holding. Decision: **{decision}**.
""")
