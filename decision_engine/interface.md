# Decision Engine Interface

This is the contract between the `decision_engine` package (Person 2) and its caller (Person 1).
It lists every public function, its exact signature, what each argument means, what comes back,
and the assumptions behind it.

**How to import.** Until the real implementation is ready, use the mock module. It has the same
signatures and return structures, with hardcoded but realistic values:

```python
from decision_engine import mock as engine

rec = engine.recommend_order(product_row, current_stock, today, history_df, promo_dates=promos)
order_qty = rec["recommended_qty"]  # not rec["order"]["optimal_qty"]
```

The agent's daily call is `recommend_order` (section 6). The other functions are public for charts and testing.

When the real modules are ready, only the import changes. Every call stays the same.

---

## Key definitions

| Term | Meaning |
|---|---|
| **`current_stock` (inventory position)** | Units on hand **plus** units already ordered but not yet delivered. **The agent must pass this, not just on-hand stock.** If it passes on-hand stock only, the engine doesn't see orders already in transit and recommends reordering the same item every day until the delivery arrives. |
| **`review_period_days`** | The planned order cycle: days from when this order arrives until the next order can arrive. `recommend_order` sets it to the economic (EOQ) cycle by default. The agent still **runs daily**; `recommend_order`'s when-to-order rule makes it wait on most days. The lower-level functions default to `1`. |
| **Order horizon** | `lead_time_days + review_period_days`. An order placed today must cover demand until the *next* order can arrive. |
| **Timing** | The decision is made at the **start** of `current_date`, before that day's sales. Lead-time window = `current_date` through `current_date + lead_time_days - 1`. Order-horizon window = `current_date` through `current_date + lead_time_days + review_period_days - 1`. |
| **`stockout_penalty`** | Goodwill cost (₹) per unit of unmet demand, **on top of** the lost margin. The lost margin is already counted, because revenue only includes units actually sold. |
| **`spoilage_cost`** | Extra disposal cost (₹) per spoiled unit, **on top of** losing the unit's purchase cost. Default `0`. |
| **`order_cost`** | Fixed cost (₹) per order placed for one product (delivery, handling, admin), regardless of quantity. |

All arguments with default values are keyword-only and must be passed by name.

## General rules

- All functions are pure. They do no database access and no file I/O, and they don't change their inputs.
- Dict results contain only built-in Python types (`int`, `float`, `str`, `bool`, `None`, `dict`, `list`).
  No numpy scalars.
- **JSON note:** several results use **integer dict keys** (`lead_time_distribution`,
  `order_horizon_distribution`, `candidates`). If you serialize a result to JSON (for example, to store it in
  SQLite as text), these keys become **strings** (`5` becomes `"5"`). Convert them back with
  `{int(k): v for k, v in d.items()}` after `json.loads`.
- Functions that use randomness take a `seed`. The same inputs and the same seed always give the same output.
- Invalid arguments raise `ValueError` with a message saying what is wrong.
- Money values are in rupees (₹).

## Input: products DataFrame (owned by Person 1)

| Column | Type | Notes |
|---|---|---|
| `product_id` | int | Unique. |
| `name` | str | |
| `category` | str | |
| `unit_cost` | float | ₹ per unit. |
| `selling_price` | float | ₹ per unit. |
| `holding_cost_per_day` | float | ₹ per unit per day. |
| `stockout_penalty` | float | ₹ per unit short. See Key definitions. |
| `shelf_life_days` | int or null | Null (`None`/`NaN`) means the product is not perishable. |
| `lead_time_days` | int | Days from placing an order to delivery. |

`data/sample_products.csv` is a valid example.

## SalesHistory DataFrame (produced by this package, stored by Person 1)

| Column | Type | Notes |
|---|---|---|
| `date` | `datetime.date` | |
| `product_id` | int | |
| `units_sold` | int | ≥ 0. Equal to **true demand**. See below. |
| `is_weekend` | bool | Saturday or Sunday. |
| `has_promo` | bool | |

---

## 1. `generate_sales_history`

```python
generate_sales_history(
    products_df: pd.DataFrame,
    days: int,
    seed: int,
    *,
    start_date: date | None = None,
    overdispersion: float | None = None,
    month_start_mult: float = 1.0,
) -> pd.DataFrame
```

Generates synthetic daily sales history for every product.

| Argument | Type | Meaning |
|---|---|---|
| `products_df` | DataFrame | Products in the schema above. Uses `product_id`, `name` and `category`. |
| `days` | int | Number of consecutive days to generate. Must be ≥ 1. |
| `seed` | int | Random seed. |
| `start_date` | `date` or None | First day. If `None`, defaults to `date(2025, 1, 1)` so output is reproducible. |
| `overdispersion` | float or None | **Robustness option.** `None` (default) means Poisson demand. A value `k > 0` gives negative binomial demand with the same mean and variance `mean + mean² / k`. |
| `month_start_mult` | float | **Robustness option.** Multiplies demand on days 1–5 of each month. Default `1.0` (no effect). Must be > 0. |

The two robustness options make the true demand differ from what the DSS's model assumes. The model
is **not** told about them. Their defaults leave the output unchanged.

**Returns** a DataFrame in the SalesHistory schema with `days × len(products_df)` rows, sorted by
`date` then `product_id`.

```text
         date  product_id  units_sold  is_weekend  has_promo
0  2025-01-01           1          19       False      False
1  2025-01-01           2          17       False      False
...
```

**Assumptions**
- `units_sold` is **true demand**. Stock limits are **not** applied, so the history never shows
  sales capped by a stockout. If you simulate stock yourself, take `min(units_sold, stock)` as actual sales.
- Demand is Poisson around a base rate per product (looked up by `name`, falling back to `category`), multiplied on weekends and promotion days.
- Promotions run for 3 to 7 consecutive days, on roughly 15% of days per product.
- Each product has its own random stream, so the first N days of a longer run equal a run of N days. Any date slice of the output is a valid SalesHistory frame.

**Usage note:** pass `start_date` so the generated history **ends the day before your simulation
starts**. For example, if the live simulation starts on `sim_start`, use
`start_date = sim_start - timedelta(days=days)`. Otherwise the history and the simulation overlap,
or leave a gap.

## 2. `simulate_one_day`

```python
simulate_one_day(
    products_df: pd.DataFrame,
    current_date: date,
    has_promo_map: dict[int, bool],
    seed: int,
    *,
    overdispersion: float | None = None,
    month_start_mult: float = 1.0,
) -> pd.DataFrame
```

Generates one day of sales rows for every product. Use it to advance a live simulation one day at a time.

| Argument | Type | Meaning |
|---|---|---|
| `products_df` | DataFrame | Products in the schema above. |
| `current_date` | `date` | The day being simulated. Sets `is_weekend`. |
| `has_promo_map` | `dict[int, bool]` | `product_id` → whether that product is on promotion today. Products not in the map are treated as `False`. |
| `seed` | int | Random seed. **Use a different seed each day** (for example, a base seed plus the day index), or every day gets the same random draws. |
| `overdispersion`, `month_start_mult` | | Same robustness options as in `generate_sales_history`. |

**Returns** a DataFrame in the SalesHistory schema with one row per product, all with `date == current_date`.
`units_sold` is true demand, as in `generate_sales_history`.

## 3. `calculate_stockout_risk`

```python
calculate_stockout_risk(
    product_id: int,
    current_stock: int,
    lead_time_days: int,
    current_date: date,
    history_df: pd.DataFrame,
    *,
    review_period_days: int = 1,
    promo_dates: Collection[date] = (),
) -> dict
```

Estimates the demand distribution and the probability of a stockout before a new order can arrive.
It uses Bayes' Theorem to estimate a daily demand rate for each of four evidence groups (`weekday`,
`weekend`, `weekday_promo`, `weekend_promo`) from the product's history. Demand over a window is a
Poisson mixture over those posteriors, with each day of the window placed in its group from its own
date. The full model is in the `bayes_risk.py` module docstring.

| Argument | Type | Meaning |
|---|---|---|
| `product_id` | int | Product to assess. |
| `current_stock` | int | **Inventory position** (on hand + on order). Must be ≥ 0. |
| `lead_time_days` | int | The product's lead time. Must be ≥ 0. |
| `current_date` | `date` | Decision day. Both windows start on this day (see **Timing** in Key definitions). |
| `history_df` | DataFrame | Sales history in the SalesHistory schema. It may contain all products; only rows for `product_id` are used. |
| `review_period_days` | int | Order cycle in days (see Key definitions). Default `1`. Must be ≥ 1. |
| `promo_dates` | collection of `date` | Dates on which **this product** has a promotion. Dates outside the windows are ignored. Default: none. |

Weekend days (Saturday and Sunday) are derived from each window date. `datetime` and `pd.Timestamp`
values are accepted and converted to dates.

**Returns**

| Key | Type | Meaning |
|---|---|---|
| `product_id` | int | Echo of the input. |
| `stockout_risk` | float in [0, 1] | P(demand over the lead-time window > `current_stock`). |
| `group_rates` | `dict[str, dict]` | For each of the four groups: `{"posterior_mean": float, "ci_low": float, "ci_high": float, "n_days": int}`. Rates are units per day. `ci_low`/`ci_high` bound the 90% credible interval. `n_days` is the number of history days in the group (0 means the estimate comes from the prior alone). |
| `lead_time_day_groups` | `dict[str, int]` | Number of days of each group in the lead-time window, from the actual dates. All four keys are always present. The values sum to `lead_time_days`. |
| `lead_time_distribution` | `dict[int, float]` | Units demanded over the lead-time window → probability. |
| `order_horizon_days` | int | `lead_time_days + review_period_days`. |
| `order_horizon_distribution` | `dict[int, float]` | Units demanded over the order-horizon window → probability. **Pass this to `calculate_optimal_order_quantity`.** |
| `expected_lead_time_demand` | float | Mean of `lead_time_distribution`. |
| `expected_order_horizon_demand` | float | Mean of `order_horizon_distribution`. |

Both distributions contain non-negative integer keys and sum to 1 within 1e-6. Demand values with
negligible probability are left out and the rest renormalized, so keys don't always start at 0.

```python
# current_date = Friday 2025-03-07, lead_time_days = 3, promo_dates = {date(2025, 3, 8)}
{
    "product_id": 1,
    "stockout_risk": 0.27,
    "group_rates": {
        "weekday":       {"posterior_mean": 29.9, "ci_low": 29.1, "ci_high": 30.7, "n_days": 113},
        "weekend":       {"posterior_mean": 40.8, "ci_low": 39.3, "ci_high": 42.2, "n_days": 46},
        "weekday_promo": {"posterior_mean": 55.1, "ci_low": 52.0, "ci_high": 58.3, "n_days": 15},
        "weekend_promo": {"posterior_mean": 76.8, "ci_low": 71.1, "ci_high": 82.6, "n_days": 6},
    },
    "lead_time_day_groups": {"weekday": 1, "weekend": 1, "weekday_promo": 0, "weekend_promo": 1},
    "lead_time_distribution": {95: 0.0001, 96: 0.0002, ..., 205: 0.0001},
    "order_horizon_days": 4,
    "order_horizon_distribution": {120: 0.0001, ..., 245: 0.0001},
    "expected_lead_time_demand": 147.5,
    "expected_order_horizon_demand": 177.4,
}
```

**Assumptions**
- Daily demand is Poisson given the group's rate. Days are independent given the rate, and one unknown rate is shared by all days of a group.
- `units_sold` in `history_df` is treated as true demand, not demand censored by stockouts.
- If `history_df` has no rows for the product, the function uses a prior based only on default demand levels. That prior is centered on 8 units per day for every group.
- `stockout_risk` covers the lead-time window only. It is the chance of running out *before an order placed now arrives*.
- Raises `ValueError` if `current_date` or a promo date is not a date, or if `history_df` is missing a SalesHistory column.

## 4. `calculate_optimal_order_quantity`

```python
calculate_optimal_order_quantity(
    current_stock: int,
    demand_distribution: dict[int, float],
    unit_cost: float,
    price: float,
    holding_cost: float,
    stockout_penalty: float,
    horizon_days: int,
    *,
    spoilage_cost: float = 0.0,
    order_cost: float = 0.0,
    shelf_life_days: int | None = None,
    review_period_days: int = 1,
    risk_tolerance: float | None = None,
    max_qty: int | None = None,
    step: int = 1,
) -> dict
```

Chooses the order quantity `Q` that maximizes expected utility of profit over the order horizon.

| Argument | Type | Meaning |
|---|---|---|
| `current_stock` | int | **Inventory position** (on hand + on order). Must be ≥ 0. |
| `demand_distribution` | `dict[int, float]` | **Use `order_horizon_distribution` from `calculate_stockout_risk`.** Must be non-empty, with non-negative probabilities summing to 1 within 1e-6. Keys need not start at 0. String keys of integers (as produced by a JSON round trip) are accepted. |
| `unit_cost` | float | ₹ per unit purchased. Must be ≥ 0. |
| `price` | float | ₹ selling price per unit. Must be ≥ 0. |
| `holding_cost` | float | ₹ per unit per day (`holding_cost_per_day`). Must be ≥ 0. |
| `stockout_penalty` | float | ₹ goodwill cost per unit short, on top of the lost margin. Must be ≥ 0. |
| `horizon_days` | int | **Use `order_horizon_days` from `calculate_stockout_risk`.** Must be ≥ 1. |
| `spoilage_cost` | float | ₹ extra disposal cost per spoiled unit. Default `0.0`. Must be ≥ 0. |
| `order_cost` | float | ₹ fixed cost of placing this order. Charged only when `Q > 0`. Default `0.0`. Must be ≥ 0. |
| `shelf_life_days` | int or None | `None` means not perishable. |
| `review_period_days` | int | **Use the same value given to `calculate_stockout_risk`.** It sets how long the new order has been on the shelf by the end of the horizon. Default `1`. Must be between 1 and `horizon_days`. |
| `risk_tolerance` | float or None | `None` means risk-neutral (linear utility). Otherwise, a positive ₹ amount for exponential utility. Smaller values are more risk-averse. |
| `max_qty` | int or None | Largest order considered. `None` (default) means an automatic bound (see below). Must be ≥ 0 if given. |
| `step` | int | Candidate spacing. Candidates are `0, step, 2*step, …` up to the upper bound. Default `1`. Must be ≥ 1. |

**Automatic upper bound (`max_qty=None`).** Candidates run from 0 to `max(0, ceil(q - current_stock))`,
where `q` is the smallest demand value with cumulative probability ≥ 1 − 1e-4. Beyond this point,
extra units can't measurably add sales, because demand almost never reaches them. They only add
holding or spoilage cost, so the optimum can't lie beyond it. The one exception is
`holding_cost = 0` on a product that doesn't spoil, where extra units cost nothing; `at_upper_bound`
then reports that the bound was reached.

### Profit model

For order size `Q` and demand `d` over the horizon:

```text
available = current_stock + Q
sold      = min(available, d)
leftover  = max(0, available - d)
shortage  = max(0, d - available)
spoiled   = see "Spoilage" below (0 if shelf_life_days is None)

profit = (price - unit_cost) * sold
         - holding_cost * horizon_days * (available + leftover) / 2
         - stockout_penalty * shortage
         - (unit_cost + spoilage_cost) * spoiled
         - (order_cost if Q > 0 else 0)
```

- Leftover units that don't spoil keep their purchase value, because they can be sold later.
- The fixed `order_cost` makes small orders less attractive. When stock is nearly enough, ordering nothing can be optimal.
- Holding cost uses the average inventory over the horizon, approximated as `(start + end) / 2`.

### Spoilage

The new order arrives at the end of the lead time. By the end of the horizon it has been on the shelf
for `review_period_days`, not `horizon_days`. For a perishable product:

```text
remaining_life = shelf_life_days - review_period_days
if remaining_life <= 0:  spoiled = leftover
else:                    spoiled = max(0, leftover - ceil(mean_daily * remaining_life))
mean_daily = expected horizon demand / horizon_days
```

Leftover stock keeps selling for its remaining life at the average daily rate, and whatever is still
unsold then spoils. Leftover that doesn't spoil keeps its value.

Assumptions:
- Stock is sold first in, first out.
- Units already on hand are treated as fresh, because their age isn't tracked.
- Demand after the horizon is approximated by its average.

### Utility

- `risk_tolerance is None`: linear, `U(profit) = profit`. `utility_type = "linear"`.
- Otherwise: exponential, `U(profit) = 1 - exp(-profit / risk_tolerance)`. `utility_type = "exponential"`.

Expected utility is `Σ_d P(d) · U(profit(Q, d))`. `optimal_qty` is the candidate with the highest
expected utility. If two candidates tie, the smaller one is chosen.

The **certainty equivalent** is the guaranteed profit the owner would value exactly as much as the
risky outcome of ordering `Q`. Under linear utility it equals the expected profit. Under exponential
utility it is `-R · log(E[exp(-profit / R)])`, which is never above the expected profit. The difference
is the risk premium. Candidates are ranked by certainty equivalent, which gives the same choice as
expected utility but can't overflow.

A risk-averse owner may order **more or less** than a risk-neutral one. If large shortage penalties
drive the bad outcomes, ordering more reduces risk. If leftover costs (spoilage, holding) drive
them, ordering less does.

**Returns**

| Key | Type | Meaning |
|---|---|---|
| `optimal_qty` | int | Recommended order quantity. One of the candidates. `0` means don't order. |
| `expected_utility` | float | Expected utility at `optimal_qty`. Under linear utility, this equals `expected_profit`. Under exponential utility with extreme losses (more than about 700 × `risk_tolerance`), it can be `-inf`. Use `certainty_equivalent` for display. |
| `expected_profit` | float | Expected profit (₹) at `optimal_qty`. |
| `certainty_equivalent` | float | Guaranteed profit (₹) with the same utility as the optimal order. Equals `expected_profit` under linear utility. |
| `no_stockout_probability` | float | P(demand ≤ `current_stock + optimal_qty`) over the horizon. |
| `at_upper_bound` | bool | `True` if `optimal_qty` is the largest candidate, so the bound may be limiting the order. With the automatic bound, this only happens in the zero-holding-cost case above. |
| `utility_type` | str | `"linear"` or `"exponential"`. |
| `candidates` | `dict[int, dict]` | For every candidate `Q`: `{"expected_utility": float, "expected_profit": float, "certainty_equivalent": float}`. |

```python
{
    "optimal_qty": 27,
    "expected_utility": 61.8,
    "expected_profit": 61.8,
    "certainty_equivalent": 61.8,
    "no_stockout_probability": 0.71,
    "at_upper_bound": False,
    "utility_type": "linear",
    "candidates": {
        0:  {"expected_utility": -40.2, "expected_profit": -40.2, "certainty_equivalent": -40.2},
        1:  {"expected_utility": -35.9, "expected_profit": -35.9, "certainty_equivalent": -35.9},
        ...
        48: {"expected_utility": 2.4,   "expected_profit": 2.4,   "certainty_equivalent": 2.4},
    },
}
```

**Assumptions**
- At most one order is placed per decision. It arrives within the horizon and covers the whole horizon.
- This is a single-decision model applied daily. With `order_cost > 0` it is a sensible heuristic,
  not a proven optimal multi-period policy, because it doesn't look ahead to future order costs.
- Spoilage: first in, first out. Units on hand are treated as fresh. Demand after the horizon is approximated by its average (see Spoilage).
- Units are ordered in multiples of `step`. Demand above the distribution's largest key is treated as impossible.

## 5. `economic_review_period`

```python
economic_review_period(
    mean_daily_demand: float,
    holding_cost: float,
    order_cost: float,
    *,
    shelf_life_days: int | None = None,
    min_days: int = 1,
    max_days: int = 30,
) -> int
```

Returns the order cycle length of the **Economic Order Quantity (EOQ)** model, in days. If you order
every `R` days, the fixed order cost per day is `order_cost / R`. The average holding cost per day is
`holding_cost × mean_daily_demand × R / 2`. Their sum is smallest at

```text
R = round(sqrt(2 * order_cost / (holding_cost * mean_daily_demand)))
```

The result is clipped to `[min_days, max_days]`.

| Argument | Type | Meaning |
|---|---|---|
| `mean_daily_demand` | float | Average units per day. Must be ≥ 0. |
| `holding_cost` | float | ₹ per unit per day. Must be ≥ 0. |
| `order_cost` | float | ₹ fixed cost per order. Must be ≥ 0. |
| `shelf_life_days` | int or None | For perishables, the result is also capped at `max(1, shelf_life_days - 1)`, so new stock has at least one day left to sell after the horizon. This cap wins over `min_days`. |
| `min_days`, `max_days` | int | Bounds on the result. `1 ≤ min_days ≤ max_days`. |

**Edge cases:** `order_cost == 0` gives `min_days`, because ordering is free. `holding_cost == 0` or
`mean_daily_demand == 0` gives `max_days`.

**Assumptions**
- Demand is steady at its mean. The EOQ trade-off ignores demand variability and stockouts;
  `calculate_optimal_order_quantity` handles those when sizing each order.

## 6. `recommend_order` (the agent's daily call)

```python
recommend_order(
    product: Mapping,
    current_stock: int,
    current_date: date,
    history_df: pd.DataFrame,
    *,
    promo_dates: Collection[date] = (),
    order_cost: float = 30.0,
    spoilage_cost: float = 0.0,
    risk_tolerance: float | None = None,
    review_period_days: int | None = None,
) -> dict
```

**Person 1's agent should call `recommend_order` once per product per day and order
`recommended_qty`, not `order["optimal_qty"]`.** It makes two Utility Theory decisions with
consistent arguments. `calculate_stockout_risk` and `calculate_optimal_order_quantity` remain public,
for charts and testing.

**Decision 1: how much (`Q*`).**
1. If `review_period_days` is `None`, it is set by `economic_review_period`. That uses the product's
   mean daily `units_sold` in `history_df` (8 units per day if there is no history), its
   `holding_cost_per_day`, `order_cost`, and `shelf_life_days`.
2. `calculate_stockout_risk(..., review_period_days=..., promo_dates=...)` runs starting on `current_date`.
3. `calculate_optimal_order_quantity` runs on `order_horizon_distribution`, with
   `horizon_days = order_horizon_days` and the same `review_period_days`. Its `optimal_qty` is `Q*`.

**Decision 2: whether to order today.** Re-optimizing every day with a rolling horizon would top
the order up almost daily, paying `order_cost` each time. So it compares two options: the expected
cost of **waiting one day** against the cost of **ordering early**.
- `calculate_stockout_risk` is called with `review_period_days = 1`; this is `wait_risk`. It gives
  `D_L`, demand over days `t … t+L−1` (its `lead_time_distribution`), and `D_(L+1)`, demand over days
  `t … t+L` (its `order_horizon_distribution`). With `pos = current_stock`:
  ```text
  extra_shortage     = E[max(0, D_(L+1) − pos)] − E[max(0, D_L − pos)]
  wait_cost          = (selling_price − unit_cost + stockout_penalty) × extra_shortage
  early_holding_cost = holding_cost_per_day × Q*
  ```
- If you wait, an order placed tomorrow arrives on day `t+L+1`, so day `t+L` must be covered from the
  current position. `wait_cost` is the expected lost margin plus penalty from that day.
- If you order today, `Q*` is held one extra day; that is `early_holding_cost`.
- **Order `Q*` today only if `Q* > 0` and `wait_cost > early_holding_cost`; otherwise order 0.**
  With `lead_time_days = 0`, `D_L` is zero demand.
- This is a one-day look-ahead. It compares today with tomorrow, not with every later day.

| Argument | Type | Meaning |
|---|---|---|
| `product` | dict or `pd.Series` | One row of the products table. Needs `product_id`, `unit_cost`, `selling_price`, `holding_cost_per_day`, `stockout_penalty`, `shelf_life_days` (None/NaN if not perishable), `lead_time_days`. |
| `current_stock` | int | **Inventory position**: on hand + on order, or `effective_inventory_position` if batch ages are known. |
| `current_date` | `date` | Decision day (start of day, before that day's sales). |
| `history_df` | DataFrame | Sales history in the SalesHistory schema. |
| `promo_dates` | collection of `date` | Dates on which this product has a promotion. |
| `order_cost` | float | ₹ fixed cost per order. Default `30.0`. |
| `spoilage_cost` | float | ₹ extra disposal cost per spoiled unit. Default `0.0`. Pass the value from `get_category_policy`. |
| `risk_tolerance` | float or None | `None` for risk-neutral, otherwise a ₹ amount R > 0 for exponential utility. |
| `review_period_days` | int or None | `None` (default) for the economic review period, or a fixed number of days. |

**Returns**

```python
{
    "product_id": 5,
    "review_period_days": 8,
    "recommended_qty": 0,          # what to order today: Q* if should_order, else 0
    "should_order": False,
    "wait_cost": 1.7,              # ₹ expected cost of waiting one day
    "early_holding_cost": 10.2,    # ₹ cost of holding Q* one extra day
    "risk": { ... },               # calculate_stockout_risk over lead time + review period
    "order": { ... },              # calculate_optimal_order_quantity result; order["optimal_qty"] is Q*
    "wait_risk": { ... },          # calculate_stockout_risk with review_period_days = 1
}
```

**Assumptions**
- The agent still runs **daily**. On most days the when-to-order rule says wait, so orders happen
  roughly once per review period.
- Both risk calls share one posterior fit, so the extra call is cheap.
- Raises `ValueError` if `product` is missing a field, or if an underlying function rejects its inputs.

## 7. `effective_inventory_position`

```python
effective_inventory_position(
    batches: list[tuple[date, int]],
    in_transit_qty: int,
    current_date: date,
    lead_time_days: int,
    shelf_life_days: int | None,
    mean_daily_demand: float,
) -> int
```

An inventory position that counts perishable stock only to the extent it can sell before it expires.
Plain on hand + on order overstates a perishable product's stock, because units that will expire
before a new order arrives can't cover demand after that point. A reorder trigger based on them fires
too late.

| Argument | Type | Meaning |
|---|---|---|
| `batches` | list of `(arrival_date, qty)` | On-hand stock. It is sold oldest first; the list may be in any order. |
| `in_transit_qty` | int | Units ordered but not yet received. Always counted in full. |
| `current_date` | `date` | Today, after today's receipts and expiries. |
| `lead_time_days` | int | Days until an order placed today would arrive. |
| `shelf_life_days` | int or None | `None` means not perishable. A batch arriving on day `a` is discarded at the start of day `a + shelf_life_days`. |
| `mean_daily_demand` | float | Expected units demanded per day. |

**Rule**
- **Non-perishable:** sum of the batches + `in_transit_qty`.
- **Perishable:** walk the batches oldest first, with a running count of expected demand already
  absorbed by older batches.
  - A batch that expires on or before `current_date + lead_time_days` (gone by the time a new order
    would arrive) counts only for
    `min(qty, floor(mean_daily_demand × days until its expiry) − demand already absorbed)`.
  - A batch still sellable on `current_date + lead_time_days` counts in full.
  - Then `in_transit_qty` is added.

**Assumptions**
- This is a **deterministic approximation**. Demand is taken to be exactly its mean, and partial
  credit is rounded down.
- All batches of a product share one shelf life.
- **Person 1's live app, which doesn't track batches, can pass on-hand + on-order as `current_stock`
  instead.** This function is for callers that know batch ages, such as the evaluation simulation.

## 8. `run_evaluation`

```python
run_evaluation(
    products_df: pd.DataFrame,
    days: int,
    seed: int,
    *,
    history_days: int = 180,
    policies: dict[int, dict] | None = None,
    order_cost: float = 30.0,
    risk_tolerance: float | None = None,
    review_period_overrides: dict[int, int] | None = None,
    include_ablation: bool = True,
    baseline_safety_factor: float | dict[int, float] = 1.2,
    baseline_cover_days: int | dict[int, int] = 7,
    include_dss: bool = True,
    generator_options: dict | None = None,
    baseline_weekend_aware: bool = False,
) -> pd.DataFrame
```

Simulates `days` days of store operation on the **same** demand under each policy:
- `"baseline"`: the static reorder policy described below.
- `"dss"`: this decision engine, calling `recommend_order` daily.
- `"dss_no_evidence"`: an ablation, included when `include_ablation` is `True`. It is identical to
  `"dss"`, except the demand model pools all history days into one rate, so weekends and promotions
  carry no information. The `dss` − `dss_no_evidence` difference measures what that evidence is worth.

| Argument | Type | Meaning |
|---|---|---|
| `products_df` | DataFrame | Products in the schema above. |
| `days` | int | Days to simulate. Must be a positive int. |
| `seed` | int | Random seed for the history and the simulated demand. All policies see identical demand. |
| `history_days` | int | Days of sales history generated before the simulated period. Default `180`. Must be a positive int. |
| `policies` | `dict[int, dict]` or None | `product_id` → the dict from Person 1's `get_category_policy`, e.g. `{"is_perishable": True, "spoilage_cost": 0.0, "shelf_life_days": 2}`. If `None`, perishability comes from `shelf_life_days` (null means not perishable) and `spoilage_cost = 0`. |
| `order_cost` | float | ₹ fixed cost per order placed for one product. Default `30.0`. Must be ≥ 0. |
| `risk_tolerance` | float or None | Passed to `recommend_order`. `None` means risk-neutral. Must be > 0 if given. |
| `review_period_overrides` | `dict[int, int]` or None | `product_id` → review period passed to `recommend_order`. Products not listed use the economic review period. Values must be positive ints. |
| `include_ablation` | bool | Also run `"dss_no_evidence"`. Default `True`. Ignored if `include_dss` is `False`. |
| `baseline_safety_factor` | float or `dict[int, float]` | Baseline reorder-point safety factor. One value for every product, or `{product_id: value}` (missing products use `1.2`). Must be > 0. |
| `baseline_cover_days` | int or `dict[int, int]` | Baseline cycle cover `D` in days. One value, or `{product_id: value}` (missing products use `7`). Perishables are still capped at `shelf_life_days − 1`. Positive ints. |
| `include_dss` | bool | If `False`, only `"baseline"` rows are produced (for tuning the baseline). Default `True`. |
| `baseline_weekend_aware` | bool | Use the weekend-aware baseline (see below) instead of the static one. Default `False`. A robustness check. |
| `generator_options` | dict or None | Passed to `generate_sales_history`. Keys: `"overdispersion"` and/or `"month_start_mult"`. The DSS's model is **not** told about these effects. Default `None` (standard demand). |

**All of these defaults reproduce the standard evaluation exactly.** A test compares against a saved
snapshot from before these options existed.

**Returns** a DataFrame with one row per (policy, product), plus one total row per policy, in the
policy order above.

| Column | Type | Meaning |
|---|---|---|
| `policy` | str | `"baseline"`, `"dss"` or `"dss_no_evidence"`. |
| `product_id` | int or None | **`None` on total rows.** The column has `object` dtype. |
| `total_profit` | float | ₹ profit, using the accounting below. |
| `stockout_days` | int | Days with unmet demand. Summed on total rows. |
| `units_lost` | int | Units of unmet demand. |
| `units_spoiled` | int | Units discarded after expiry. Always 0 for non-perishable products. |
| `avg_stock_held` | float | Average on-hand units at the end of each day. Summed across products on total rows. |
| `orders_placed` | int | Number of orders placed. Each costs `order_cost`. |
| `fill_rate` | float | Units sold / units demanded. On total rows: total sold / total demanded. |
| `revenue` | float | ₹ `selling_price × units sold`. |
| `units_demanded` | int | Total true demand. This is identical across policies. |

```text
             policy product_id  total_profit  stockout_days  units_lost  units_spoiled  avg_stock_held  orders_placed  fill_rate   revenue  units_demanded
0          baseline          1       ...
10         baseline       None      87626.00    ...   (seed 0, 90 days; columns abbreviated)   fill_rate 0.891   orders 163
21              dss       None     114274.10    ...                                           fill_rate 0.997   orders 208
32  dss_no_evidence       None     104579.57    ...                                           fill_rate 0.966   orders 206
```

### Simulation

1. **Demand.** It generates `history_days + days` of demand from `seed`, starting 2025-01-01, and splits
   it by date. The first `history_days` are the **fixed history** both policies learn from. The rest are
   the true daily demand and promotion flags of the simulated period. Promotion dates are known in
   advance; only the DSS uses them.
2. **State.** Per policy and product, it tracks on-hand stock as first-in-first-out batches (quantity
   and arrival day) and in-transit orders (quantity and arrival day).
3. **Each day `t`**, in this order:
   1. **Receive** orders arriving on day `t`, as a new batch of age 0.
   2. **Expire.** For perishables, discard every batch whose age (`t − arrival day`) has reached
      `shelf_life_days`. Count those units as spoiled and charge `spoilage_cost` per unit.
   3. **Decide.** The inventory position is `effective_inventory_position` (section 7): on hand + in
      transit, except that perishable batches expiring before a new order could arrive count only for
      the demand they're expected to meet first. The baseline passes its `mu` as `mean_daily_demand`;
      the DSS passes the history's mean daily demand, which is the same number. The policy chooses `Q`. If `Q > 0`, pay
      `unit_cost × Q + order_cost`; the order arrives at the start of day `t + lead_time_days`. A lead
      time of 0 means it arrives immediately, before today's sales.
   4. **Sell** first in, first out, up to the stock on hand. Revenue = `selling_price × sold`. Each
      lost unit is charged `stockout_penalty`. A stockout day is any day with lost units.
   5. **Hold.** Charge `holding_cost_per_day` per unit on hand at the end of the day.
4. **Starting state.** Every policy starts each product with on-hand stock equal to the baseline's
   order-up-to level `S` (one fresh batch arriving on day 0) and nothing in transit.
5. **Accounting.**
   ```text
   profit = revenue − purchases − order costs − holding − stockout penalties − spoilage costs
            + value of ending inventory (on hand + in transit, at unit_cost)
            − value of starting inventory (at unit_cost)
   ```
   The inventory adjustment means a policy isn't rewarded or punished for the stock it happens to hold
   on the last day.
6. **DSS decisions.** Each day, for each product, it calls `recommend_order` with
   `current_stock` = the effective inventory position, the actual date, the fixed history, the
   product's promotion dates, `order_cost`, `risk_tolerance`, `spoilage_cost` from `policies`
   (default 0), and the review period override if one is given. It orders `recommended_qty`.
7. **Caching.** The history is fixed, so each product's posterior is computed once per run. Results are
   identical to calling `recommend_order` afresh each day.

**Why the history is fixed.** It is **not** updated during the simulation. On a stockout day the store
only observes sales, not demand. Appending sales would bias demand estimates downward (censored
demand), and appending true demand would use information a real store cannot observe.

**Assumptions**
- Unmet demand is lost, not backordered. Orders arrive exactly `lead_time_days` later, in full.
- All policies use the same accounting. Only the ordering decision differs.
- The simulation tracks real batch ages. The DSS's optimizer approximates spoilage by treating stock on
  hand as fresh. The simulation is the ground truth for every policy.

### Baseline policy (static reorder point, order-up-to level)

For each product:

- `mu` = mean daily `units_sold` for the product over the `history_days` history, using all days. Weekends and promotions are ignored. If the product has no history, `mu` = 8.
- `D` = `cover_days` for non-perishables, and `max(1, min(cover_days, shelf_life_days − 1))` for perishables so a batch can sell before it expires. `cover_days = baseline_cover_days`, default `7`.
- Reorder point `s = ceil(mu * lead_time_days * safety_factor)`, with `safety_factor = baseline_safety_factor` (default `1.2`).
- Order-up-to level `S = s + ceil(mu * D)`.
- Each day, if the effective inventory position `<= s`, order `S - position`.
- It never adjusts for weekends or promotions.

A competent static rule of thumb: per-product reorder point and order-up-to level from average demand, sized so perishable batches can sell before expiry, never adjusted for weekends or promotions.

### Weekend-aware baseline (`baseline_weekend_aware=True`, robustness check)

This is the same rule, except the demand rates depend on the calendar:

- Separate mean daily `units_sold` for weekdays and for weekends, from the history. Promotion days are
  included in these means, not treated separately. If a day type is missing, its mean falls back to
  the overall mean.
- On day `t`: `s_t = ceil(safety_factor × Σ mean(day) over the lead-time days t … t+L−1)`.
- `S_t = s_t + ceil(Σ mean(day) over the D cover days t+L … t+L+D−1)`, with `D` and the perishable
  cap as above. Each `mean(day)` is the weekday or weekend mean.
- Each day, if the effective inventory position `<= s_t`, order `S_t − position`. The effective
  position still uses the overall mean.
- It never adjusts for promotions.
- Every policy starts with `S` as computed for the first simulated day.
