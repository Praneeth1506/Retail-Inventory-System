# Retail Inventory Decision Support System

A decision support system for a small retail store that recommends, every day and for every product,
whether to reorder and how much. A SQLite database holds the product catalog, stock levels and sales
history. A first-order-logic rule engine classifies products (for example, Milk → Dairy → Perishable)
and raises alerts. An agent state machine runs the daily decision cycle, and a Streamlit dashboard shows
the reasoning. The decisions come from a pure-Python decision engine:
- **Bayes' Theorem** estimates daily demand rates for weekdays, weekends and promotion days, and turns
  them into a stockout risk.
- **Utility Theory** chooses the order quantity, and decides whether ordering today beats waiting a day.

The engine is evaluated in simulation against static reorder rules, with an ablation and robustness checks.

## Who built what

- **Person 1:** SQLite database (`src/db.py`), rule engine with ISA hierarchy (`src/rules.py`), agent state
  machine (`src/agent.py`), Streamlit dashboard (`app.py`).
- **Person 2:** decision engine (`decision_engine/`): Bayesian stockout risk, utility-based order
  optimization, economic review period, daily `recommend_order`. Also the synthetic data generator, the
  evaluation simulation and reports (`scripts/`), and the adapter connecting the agent to the engine
  (`src/engine_adapter.py`).

## Install

Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running the tests

```bash
.venv/bin/python -m pytest                       # decision engine, evaluation and integration tests
.venv/bin/python scripts/smoke_integration.py    # agent on the real engine for 3 simulated days (uses a DB copy)
```

Person 1's `test_agent.py` and `test_rules.py` are scripts: `.venv/bin/python test_agent.py`. Note that
`test_agent.py` runs a real agent cycle against `data/inventory.db`, so it writes decision logs there.

## Running the app

```bash
.venv/bin/streamlit run app.py
```

Click **Run Agent Decision Cycle** in the sidebar.
- **First cycle.** If the `SalesHistory` table is empty, the agent seeds it with 180 days of synthetic
  history ending yesterday. Each product gets a realistic base demand (see `DEMAND_PROXY` in
  `src/engine_adapter.py`).
- **Empty decision log.** The shipped `data/inventory.db` starts with an empty decision log
  (`ReorderLogs`); mock-era rows were removed, and the original is kept in
  `data/inventory_backup_premerge.db`.
- **Stock on order.** The app has no purchase-order table. "On order" is the sum of recommended
  quantities logged in `ReorderLogs` within each product's lead time, and the agent decides using
  on hand + on order. **This assumes every logged recommendation was actually placed as an order.**
- **"Is Weekend?"** makes the agent decide as of the next Saturday.
- **"Active Promotion?"** means a promotion on every product for the next 7 days.
- **Evaluation Results tab.** It shows the simulation results from `results/`; nothing is run live.

## Exporting the evaluation results

```bash
.venv/bin/python scripts/export_results.py
```

This runs every analysis once and writes the CSVs in `results/` (described in `results/README.md`).
Seeds run in parallel. A full run took 15 min 24 s on a 10-core machine. The report scripts
`scripts/evaluation_report.py` and `scripts/robustness_report.py` print the same analyses; all three share
`scripts/reporting.py`.

## Folder layout

```
app.py                     Streamlit dashboard (Person 1; Evaluation Results tab added)
src/
  db.py                    SQLite schema and helpers (Person 1)
  rules.py                 FOL rules, ISA closure, alerts (Person 1)
  agent.py                 agent state machine (Person 1; switched to the real engine)
  engine_adapter.py        id mapping, history seeding, on-order stock, recommend_order call
  mock_person2.py          Person 1's original mock (no longer used)
decision_engine/           Person 2's decision engine (pure functions; contract in interface.md)
  data_generator.py        synthetic sales history
  bayes_risk.py            Bayesian demand rates and stockout risk
  optimizer.py             utility-based order quantity, economic review period
  recommend.py             recommend_order: how much, and whether to order today
  inventory.py             effective inventory position for perishable batches
  evaluation.py            simulation: baselines vs. DSS vs. ablation
  mock.py                  mock with identical signatures
scripts/                   reports, export_results.py, smoke test, previews
results/                   exported evaluation CSVs + README
tests/                     pytest suite
data/
  inventory.db             app database (empty decision log, empty sales history)
  sample_products.csv      10-product catalog used by the evaluation
```

## Results

All results come from simulated 90-day periods on the 10-product sample catalog
(`data/sample_products.csv`, not the live app's catalog), over 20 test seeds of synthetic demand.
Costs (stockout penalty, order cost ₹30, holding) are stated assumptions. Every number below is taken
from the CSVs in `results/`.

- **Main result.** The comparison is against a static reorder rule tuned per product on separate seeds.
  The DSS earned ₹13.4k more profit (95% CI ₹12.4k to ₹14.3k), about
  13% more. Its fill rate was 0.998, against 0.925 for the tuned rule.
- **Where the gain comes from.** An ablation without weekend and promotion evidence shows the Bayes model
  contributes ₹9.0k of this, and the utility-based ordering logic ₹4.4k.
- **Which products.** Milk and bread account for 87% of the gain. On long-shelf-life
  products the DSS's lead is statistically significant but practically negligible: ₹71 to
  ₹254 per product over 90 days.
- **Robustness.** The DSS stayed ahead in all 9 combinations of stockout penalty (×0.5, ×1,
  ×2) and order cost (₹0, 30, 100). It also stayed ahead (by ₹25.2k, 95% CI ₹23.3k to
  ₹27.1k) under demand the model was not built for: extra variability plus a month-start
  spike. These two checks used the untuned baseline, so they show the direction of the result is
  robust, not its size.
- **Calibration.** The Bayes model's 90% credible intervals covered the true demand rate 88.8%
  of the time across 2000 intervals. That's consistent with the nominal 90% within sampling error.
- **Weekend-aware baseline.** Against a baseline that also plans for weekends (but not promotions), tuned
  the same way, the DSS still earned ₹12.5k more (95% CI ₹11.6k to ₹13.3k).

**Limitations**
- The data is synthetic, generated from a model family close to the one the DSS assumes.
- Single store.
- Flat order cost per product.
- Ending inventory valued at cost.
- The live app treats logged recommendations as placed orders.
