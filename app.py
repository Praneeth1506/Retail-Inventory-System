import streamlit as st
import pandas as pd
import plotly.express as px
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Inventory Dashboard", "🔍 Rules & ISA Reasoning", "📈 AI Risk & Utility", "📜 Decision Logs"])

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
            st.markdown("#### Demand Posterior Distribution")
            post_data = res["posterior"]
            if post_data:
                fig_post = px.bar(
                    x=list(post_data.keys()),
                    y=list(post_data.values()),
                    labels={'x': 'Demand Level', 'y': 'Posterior Probability'},
                    title=f"Posterior Probability (Stockout Risk: {res['stockout_risk']*100:.1f}%)"
                )
                st.plotly_chart(fig_post, use_container_width=True)

        with col_b:
            st.markdown("#### Expected Utility by Reorder Quantity")
            util_data = res["utility_candidates"]
            if util_data:
                fig_util = px.line(
                    x=list(util_data.keys()),
                    y=list(util_data.values()),
                    markers=True,
                    labels={'x': 'Order Quantity', 'y': 'Expected Utility ($)'},
                    title=f"Optimal Quantity: {res['recommended_qty']} units (Max Utility)"
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