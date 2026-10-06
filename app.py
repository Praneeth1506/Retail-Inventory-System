import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.agent import InventoryAgent
from src.db import get_products, get_logs, get_isa_relations
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

# Main Layout Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Inventory Dashboard", "🔍 Rules & ISA Reasoning", "📈 AI Risk & Utility", "📜 Decision Logs", "🧪 Evaluation Results"])

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
