"""Synthetic daily sales history generation (seeded, reproducible)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

SALES_COLUMNS = ["date", "product_id", "units_sold", "is_weekend", "has_promo"]
REQUIRED_PRODUCT_COLUMNS = ["product_id", "name", "category"]
DEFAULT_START_DATE = date(2025, 1, 1)

# Base daily demand (units per day on a weekday without promotion) for a small Indian store.
BASE_DEMAND: dict[str, float] = {
    "Toned Milk 500ml": 30.0,
    "Curd 400g": 12.0,
    "White Bread 400g": 15.0,
    "Eggs (pack of 6)": 10.0,
    "Sona Masoori Rice 5kg": 4.0,
    "Whole Wheat Atta 5kg": 5.0,
    "Sugar 1kg": 9.0,
    "Bath Soap 100g": 8.0,
    "Glucose Biscuits 250g": 20.0,
    "Sunflower Oil 1L": 6.0,
}
# Used when a product's name is not in BASE_DEMAND.
CATEGORY_BASE_DEMAND: dict[str, float] = {
    "Dairy": 15.0,
    "Bakery": 12.0,
    "Staples": 5.0,
    "Personal Care": 6.0,
    "Snacks": 15.0,
}
# Used when neither the name nor the category is known.
GLOBAL_BASE_DEMAND = 8.0

WEEKEND_MULT = 1.4
PROMO_MULT = 1.8
# Per-category overrides, e.g. {"Staples": {"weekend": 1.2, "promo": 1.5}}. Missing keys use the defaults.
CATEGORY_MULT_OVERRIDES: dict[str, dict[str, float]] = {}

# Misspecification options (off by default). The DSS's demand model does not know about them.
MONTH_START_DAYS = 5  # month_start_mult applies to days 1..5 of each month

# Promotions run for PROMO_LENGTH_RANGE days, separated by PROMO_GAP_RANGE days without promotion
# (both inclusive). Mean length 5 and mean gap 28.5 give roughly 15% of days on promotion.
PROMO_LENGTH_RANGE = (3, 7)
PROMO_GAP_RANGE = (15, 42)

# Each product draws from its own streams, keyed by (seed, product_id, stream), so a product's
# output does not depend on which other products are in products_df or their order.
_PROMO_STREAM = 0
_DEMAND_STREAM = 1


def _base_demand(name: str, category: str) -> float:
    if name in BASE_DEMAND:
        return BASE_DEMAND[name]
    return CATEGORY_BASE_DEMAND.get(category, GLOBAL_BASE_DEMAND)


def _demand_rate(name: str, category: str, is_weekend: bool, has_promo: bool) -> float:
    """Expected daily demand (Poisson mean). The single demand model used by every generator."""
    overrides = CATEGORY_MULT_OVERRIDES.get(category, {})
    rate = _base_demand(name, category)
    if is_weekend:
        rate *= overrides.get("weekend", WEEKEND_MULT)
    if has_promo:
        rate *= overrides.get("promo", PROMO_MULT)
    return rate


def _validate_options(overdispersion: float | None, month_start_mult: float) -> None:
    if overdispersion is not None and not overdispersion > 0:
        raise ValueError(f"overdispersion must be > 0 or None, got {overdispersion}")
    if not month_start_mult > 0:
        raise ValueError(f"month_start_mult must be > 0, got {month_start_mult}")


def _month_start_factor(day: date, month_start_mult: float) -> float:
    return month_start_mult if day.day <= MONTH_START_DAYS else 1.0


def _draw_units(rng: np.random.Generator, rates, overdispersion: float | None):
    """Poisson(rate), or with overdispersion k a negative binomial with the same mean and
    variance rate + rate^2 / k (a gamma-Poisson mixture: n = k, p = k / (k + rate))."""
    if overdispersion is None:
        return rng.poisson(rates)
    rates = np.asarray(rates, dtype=float)
    return rng.negative_binomial(overdispersion, overdispersion / (overdispersion + rates))


def _rng(seed: int, product_id: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, product_id, stream])


def _promo_schedule(rng: np.random.Generator, days: int) -> np.ndarray:
    """Boolean promo flag per day. Days are drawn in order, so a shorter run is a prefix of a longer one."""
    flags = np.zeros(days, dtype=bool)
    gap_low, gap_high = PROMO_GAP_RANGE
    length_low, length_high = PROMO_LENGTH_RANGE
    # Random phase so products do not all start their first promotion on the same day.
    day = int(rng.integers(0, gap_high + 1))
    while day < days:
        length = int(rng.integers(length_low, length_high + 1))
        flags[day : day + length] = True
        day += length + int(rng.integers(gap_low, gap_high + 1))
    return flags


def _validate_products(products_df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PRODUCT_COLUMNS if c not in products_df.columns]
    if missing:
        raise ValueError(f"products_df is missing columns: {missing}")
    if products_df["product_id"].duplicated().any():
        raise ValueError("products_df has duplicate product_id values")


def _sales_frame(columns: dict[str, list | np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(columns, columns=SALES_COLUMNS)
    return df.astype(
        {"product_id": "int64", "units_sold": "int64", "is_weekend": "bool", "has_promo": "bool"}
    )


def generate_sales_history(
    products_df: pd.DataFrame,
    days: int,
    seed: int,
    *,
    start_date: date | None = None,
    overdispersion: float | None = None,
    month_start_mult: float = 1.0,
) -> pd.DataFrame:
    """Generate synthetic daily sales history for every product.

    Demand model: units_sold ~ Poisson(base * weekend_mult * promo_mult), where base comes from
    BASE_DEMAND by product name (falling back to CATEGORY_BASE_DEMAND, then GLOBAL_BASE_DEMAND),
    weekend_mult = WEEKEND_MULT on Saturday and Sunday, and promo_mult = PROMO_MULT on promotion
    days. Both multipliers can be overridden per category in CATEGORY_MULT_OVERRIDES.

    Assumptions:
        - units_sold is TRUE DEMAND. Stock limits are not applied, so the history never shows
          sales capped by a stockout.
        - Promotions run for 3 to 7 consecutive days with gaps of 15 to 42 days, so roughly 15%
          of days per product are on promotion. They are drawn independently per product.
        - Daily demands are independent given the day's weekend and promo status.
        - Each product draws from its own random streams keyed by (seed, product_id), and draws
          day by day. So the first N days of a longer run equal a run of N days, and a product's
          rows do not depend on the other products in products_df.
        - Every row stands alone, so any slice of the output (e.g. splitting history from a
          simulated period) is a valid SalesHistory frame. Promotions may cross the split point.
        - start_date defaults to 2025-01-01 so output is reproducible. Pass start_date so the
          history ends the day before your simulation starts.
        - Misspecification options, for robustness checks (defaults leave the output unchanged):
          overdispersion = k draws negative binomial demand with the same mean and variance
          mean + mean^2 / k instead of Poisson; month_start_mult multiplies the mean on days 1-5
          of every month. The DSS's demand model is not told about either effect.

    Args:
        products_df: Products table. Uses product_id, name and category.
        days: Number of consecutive days to generate. Must be >= 1.
        seed: Non-negative random seed.
        start_date: First day of the history.
        overdispersion: None for Poisson demand, or k > 0 for negative binomial demand.
        month_start_mult: Demand multiplier on days 1-5 of each month (> 0). Default 1.0.

    Returns:
        DataFrame in the SalesHistory schema (date, product_id, units_sold, is_weekend,
        has_promo) with days * len(products_df) rows, sorted by date then product_id.

    Raises:
        ValueError: If days < 1, products_df is missing columns or has duplicate product_ids,
            or an option is out of range.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    _validate_products(products_df)
    _validate_options(overdispersion, month_start_mult)
    start = start_date if start_date is not None else DEFAULT_START_DATE

    dates = [start + timedelta(days=offset) for offset in range(days)]
    weekend = np.array([d.weekday() >= 5 for d in dates], dtype=bool)

    parts = []
    for product in products_df.sort_values("product_id").itertuples(index=False):
        product_id = int(product.product_id)
        promo = _promo_schedule(_rng(seed, product_id, _PROMO_STREAM), days)
        rates = np.array(
            [
                _demand_rate(product.name, product.category, bool(w), bool(p))
                * _month_start_factor(d, month_start_mult)
                for d, w, p in zip(dates, weekend, promo)
            ]
        )
        units = _draw_units(_rng(seed, product_id, _DEMAND_STREAM), rates, overdispersion)
        parts.append(
            _sales_frame(
                {
                    "date": dates,
                    "product_id": np.full(days, product_id),
                    "units_sold": units,
                    "is_weekend": weekend,
                    "has_promo": promo,
                }
            )
        )

    if not parts:
        return _sales_frame({c: [] for c in SALES_COLUMNS})
    history = pd.concat(parts, ignore_index=True)
    return history.sort_values(["date", "product_id"], kind="stable", ignore_index=True)


def simulate_one_day(
    products_df: pd.DataFrame,
    current_date: date,
    has_promo_map: dict[int, bool],
    seed: int,
    *,
    overdispersion: float | None = None,
    month_start_mult: float = 1.0,
) -> pd.DataFrame:
    """Generate one day of sales rows for every product, using the same demand model as
    generate_sales_history.

    Assumptions:
        - units_sold is TRUE DEMAND. Stock limits are not applied.
        - Products missing from has_promo_map are not on promotion.
        - The caller varies seed from day to day. The same seed gives the same draws.
        - overdispersion and month_start_mult behave as in generate_sales_history.

    Args:
        products_df: Products table. Uses product_id, name and category.
        current_date: The day being simulated. Sets is_weekend.
        has_promo_map: product_id -> whether that product is on promotion today.
        seed: Non-negative random seed.
        overdispersion: None for Poisson demand, or k > 0 for negative binomial demand.
        month_start_mult: Demand multiplier on days 1-5 of each month (> 0). Default 1.0.

    Returns:
        DataFrame in the SalesHistory schema with one row per product, sorted by product_id.

    Raises:
        ValueError: If products_df is missing columns or has duplicate product_ids, or an option
            is out of range.
    """
    _validate_products(products_df)
    _validate_options(overdispersion, month_start_mult)
    is_weekend = current_date.weekday() >= 5

    rows: dict[str, list] = {c: [] for c in SALES_COLUMNS}
    for product in products_df.sort_values("product_id").itertuples(index=False):
        product_id = int(product.product_id)
        has_promo = bool(has_promo_map.get(product_id, False))
        rate = _demand_rate(product.name, product.category, is_weekend, has_promo)
        rate *= _month_start_factor(current_date, month_start_mult)
        rows["date"].append(current_date)
        rows["product_id"].append(product_id)
        rows["units_sold"].append(int(_draw_units(_rng(seed, product_id, _DEMAND_STREAM), rate, overdispersion)))
        rows["is_weekend"].append(is_weekend)
        rows["has_promo"].append(has_promo)
    return _sales_frame(rows)
