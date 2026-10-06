# Results

CSV outputs of `python scripts/export_results.py`. Every number in the root README's Results section and
on the dashboard's **Evaluation Results** tab comes from these files.

**Setup.** Simulated 90-day periods on the 10-product sample catalog (`data/sample_products.csv`, not the
live app's catalog), with synthetic demand from `decision_engine.data_generator`. Test seeds 0–19 are used
unless stated otherwise. The baseline is tuned on separate seeds, 100–119. Profit is in rupees and includes
an ending-minus-starting inventory adjustment at unit cost. Costs (stockout penalty, ₹30 order cost,
holding) are stated assumptions. Simulation details are in `decision_engine/interface.md`, section 8.

**Policies.**
- `baseline`: a static reorder rule. It reorders when stock falls to a reorder point and orders up to a fixed target level (an (s, S) rule).
- `dss`: the decision engine (`recommend_order` daily).
- `dss_no_evidence`: the DSS with a demand model that ignores weekends and promotions (ablation).

## Files

### Main comparisons

There are three scenarios. Each writes `<scenario>_policy_totals.csv` and `<scenario>_paired.csv`; the
three tuned or default scenarios also write `<scenario>_per_product.csv`.

| Prefix | Baseline used | Demand |
|---|---|---|
| `default_` | Static baseline, default parameters (safety factor 1.2, cover 7 days) | Standard |
| `tuned_` | Static baseline, tuned per product on seeds 100–119 | Standard |
| `weekend_` | Weekend-aware baseline (separate weekday and weekend demand, still no promotions), tuned the same way | Standard |
| `misspec_` | Static baseline, default parameters | Misspecified: negative binomial, k = 10, plus a ×1.3 month-start spike. The DSS isn't told. Totals and paired files only. |

- **`*_policy_totals.csv`**: one row per policy. Each value is the total over all 10 products, averaged
  over seeds: `profit`, `fill_rate`, `stockout_days`, `units_spoiled`, `orders_placed`, `orders_per_month`.
- **`*_paired.csv`**: per-seed paired profit differences for `dss − baseline`, `dss_no_evidence − baseline`
  and `dss − dss_no_evidence`. Columns:
  - `mean`, `ci_low`, `ci_high`: 95% t interval over 20 seeds.
  - `excludes_zero`: whether that interval excludes 0.
  - `relative_to_b`: the mean difference divided by the second policy's mean profit.
- **`*_per_product.csv`**: one row per product. Columns:
  - `dss_minus_baseline` with its CI.
  - `fill_rate_<policy>`, `units_spoiled_<policy>` and `orders_per_month_<policy>` for each policy.

### Tuning and robustness

- **`tuned_baseline_params.csv`, `weekend_baseline_params.csv`**: the safety factor and cover days chosen
  per product from {1.0, 1.2, 1.5, 2.0, 2.5} × {3, 7, 14, 21, 28}, by highest mean profit on seeds 100–119.
  Perishables cap cover at shelf life − 1 (`effective_cover_days`). Each file also has the tuning profit
  of the chosen and default parameters.
- **`cost_sensitivity.csv`**: `dss − baseline` (default baseline, seeds 0–9) for each combination of
  stockout penalty × {0.5, 1, 2} and order cost ₹{0, 30, 100}, with its 95% CI.
- **`coverage.csv`**: how often the Bayes model's 90% credible intervals contain the true daily demand
  rate. It covers 50 history seeds (180 days each) × 10 products × 4 day types (weekday, weekend,
  weekday + promotion, weekend + promotion), with one row per day type plus an overall row.
- **`runtime.csv`**: seconds per section of the export run.
