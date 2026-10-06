"""Stockout risk estimation using Bayes' Theorem.

Model
-----
Demand for one product on one day is Poisson with an unknown daily rate lambda that depends on
the day's evidence group:

    weekday, weekend, weekday_promo, weekend_promo

Each group's rate is estimated separately with Bayes' theorem over a discrete grid of candidate
rates, and the posteriors are turned into a predictive distribution of demand over a window of days.

Timing. The decision is made at the start of current_date, before that day's sales. The lead-time
window is current_date through current_date + lead_time_days - 1, and the order-horizon window is
current_date through current_date + lead_time_days + review_period_days - 1. Each day in a window
is assigned to its group from its own date (Saturday/Sunday = weekend) and promo_dates.

1. Sufficient statistics. history_df is filtered to the product, and each day is assigned to its
   evidence group. For each group g we keep n_g (days observed) and T_g (total units sold). For a
   Poisson likelihood these two numbers carry all the information in the group's data.

2. Hypotheses. A grid of 4000 candidate daily rates, linearly spaced from 0.01 to
   max(5, 3 * the largest observed group mean). All four groups of a product share the grid.

3. Prior. For each group, a lognormal prior over the grid: log(lambda) ~ Normal(log(c_g), 0.5^2),
   evaluated on the grid and normalized. The centers c_g come from a structural estimate:
       mu  = mean daily units on weekday (no promo) days; fallback: mean over all days
       r_w = weekend mean / mu          (fallback 1.0 if there are no weekend days)
       r_p = weekday_promo mean / mu    (fallback 1.0 if there are no weekday promo days)
       c_weekday = mu, c_weekend = mu * r_w, c_weekday_promo = mu * r_p,
       c_weekend_promo = mu * r_w * r_p
   This is an EMPIRICAL BAYES prior: it reuses the same data the likelihood uses. It is kept wide
   (sigma = 0.5 on the log scale, i.e. roughly a factor of 2.7 for a 95% range) so that any group
   with a reasonable number of days is dominated by its own data. Its real job is the sparse case:
   a group with few or no days (typically weekend_promo) borrows strength from the multiplicative
   structure, e.g. weekend_promo ~ weekday * weekend effect * promo effect.

4. Likelihood and posterior. For a Poisson rate lambda and the group's data,
       log L(lambda) = T_g * log(lambda) - n_g * lambda        (terms without lambda dropped)
   By Bayes' theorem, posterior(lambda) is proportional to L(lambda) * prior(lambda). This is
   computed in log space and normalized with logsumexp for numerical stability.

5. Group rates. Each group's posterior mean and 90% credible interval (5th and 95th percentiles)
   summarize what the data say about that group's daily rate. The percentiles interpolate the
   posterior CDF between grid points (each grid point's mass is centered on its cell), so they
   are not limited to the grid spacing.

6. Predictive window demand. For a window with k_g days in group g (counted from the window's
   actual dates), the demand from that group is a Poisson mixture:
       P(S_g = s) = sum over lambda of posterior_g(lambda) * Poisson(s; k_g * lambda)
   A single lambda is shared by all k_g days of the group: the rate is unknown but fixed, so
   uncertainty about it does not average out across days. (Drawing an independent lambda per day
   would understate the risk.) The groups' rates are estimated from disjoint data and are treated
   as independent, so the group distributions are convolved to give the total window demand.

7. Truncation. The support is cut where the cumulative probability exceeds 1 - 1e-9 (and, at the
   bottom, below 1e-9), then renormalized, so each distribution sums to 1 within 1e-6.

8. Stockout risk. stockout_risk = P(demand over the lead-time window > current_stock).
   The order-horizon distribution is computed the same way over the order-horizon window.

With no history for the product, there is no data to estimate mu: mu falls back to
DEFAULT_DAILY_DEMAND, both ratios to 1.0, and every group's posterior equals its prior.

Pooled variant (private, used only by the evaluation's "dss_no_evidence" ablation). All history days
form ONE group with a lognormal prior centered on the overall mean, giving a single posterior rate.
Every window day uses that one rate, shared across the whole window (the window is treated as one
group of k = window length days), so weekends and promotions carry no information.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import lognorm, poisson

GROUPS = ("weekday", "weekend", "weekday_promo", "weekend_promo")

GRID_SIZE = 4000
GRID_MIN = 0.01
GRID_MIN_UPPER = 5.0
GRID_UPPER_FACTOR = 3.0
PRIOR_SIGMA = 0.5
CREDIBLE_MASS = 0.90
TAIL_PROBABILITY = 1e-9
DEFAULT_DAILY_DEMAND = 8.0

_REQUIRED_HISTORY_COLUMNS = ["product_id", "units_sold", "is_weekend", "has_promo"]


@dataclass(frozen=True)
class _ProductFit:
    """Posterior over daily rates for each evidence group of one product."""

    grid: np.ndarray
    posteriors: dict[str, np.ndarray]
    n_days: dict[str, int]
    totals: dict[str, int]
    centers: dict[str, float]
    mu: float
    pooled: bool = False


def _as_date(value: date) -> date:
    """Normalize datetime / pd.Timestamp to date (a datetime never equals a date in Python)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise ValueError(f"expected a date, got {value!r}")


def _group_name(is_weekend: bool, has_promo: bool) -> str:
    return ("weekend" if is_weekend else "weekday") + ("_promo" if has_promo else "")


def _window_day_groups(start: date, days: int, promo_dates: set[date]) -> dict[str, int]:
    """Number of days of each group in the window start .. start + days - 1."""
    counts = {g: 0 for g in GROUPS}
    for offset in range(days):
        day = start + timedelta(days=offset)
        counts[_group_name(day.weekday() >= 5, day in promo_dates)] += 1
    return counts


def _group_stats(history_df: pd.DataFrame, product_id: int) -> tuple[dict[str, int], dict[str, int]]:
    rows = history_df[history_df["product_id"] == product_id]
    n_days = {g: 0 for g in GROUPS}
    totals = {g: 0 for g in GROUPS}
    if len(rows) == 0:
        return n_days, totals
    groups = [
        _group_name(bool(w), bool(p)) for w, p in zip(rows["is_weekend"], rows["has_promo"])
    ]
    summary = rows.assign(group=groups).groupby("group")["units_sold"].agg(["count", "sum"])
    for group, row in summary.iterrows():
        n_days[group] = int(row["count"])
        totals[group] = int(row["sum"])
    return n_days, totals


def _prior_centers(n_days: dict[str, int], totals: dict[str, int]) -> tuple[float, dict[str, float]]:
    means = {g: totals[g] / n_days[g] for g in GROUPS if n_days[g] > 0}
    if "weekday" in means:
        mu = means["weekday"]
    elif means:
        mu = sum(totals.values()) / sum(n_days.values())
    else:
        mu = DEFAULT_DAILY_DEMAND
    mu = max(mu, GRID_MIN)
    r_w = means["weekend"] / mu if "weekend" in means else 1.0
    r_p = means["weekday_promo"] / mu if "weekday_promo" in means else 1.0
    centers = {
        "weekday": mu,
        "weekend": mu * r_w,
        "weekday_promo": mu * r_p,
        "weekend_promo": mu * r_w * r_p,
    }
    return mu, {g: max(c, GRID_MIN) for g, c in centers.items()}


def _fit_product(history_df: pd.DataFrame, product_id: int, pooled: bool = False) -> _ProductFit:
    """Steps 1-4 of the model: group statistics, grid, empirical Bayes prior, posterior.

    With pooled=True, all days are one group (see "Pooled variant" in the module docstring); every
    group name then maps to the same pooled statistics and posterior.
    """
    n_days, totals = _group_stats(history_df, product_id)
    if pooled:
        n_all, total_all = sum(n_days.values()), sum(totals.values())
        n_days = {g: n_all for g in GROUPS}
        totals = {g: total_all for g in GROUPS}
        mu = max(total_all / n_all if n_all else DEFAULT_DAILY_DEMAND, GRID_MIN)
        centers = {g: mu for g in GROUPS}
    else:
        mu, centers = _prior_centers(n_days, totals)

    observed_means = [totals[g] / n_days[g] for g in GROUPS if n_days[g] > 0]
    largest_mean = max(observed_means) if observed_means else mu
    grid = np.linspace(GRID_MIN, max(GRID_MIN_UPPER, GRID_UPPER_FACTOR * largest_mean), GRID_SIZE)
    log_grid = np.log(grid)

    posteriors = {}
    for g in GROUPS:
        log_prior = lognorm.logpdf(grid, s=PRIOR_SIGMA, scale=centers[g])
        log_prior -= logsumexp(log_prior)
        log_likelihood = totals[g] * log_grid - n_days[g] * grid
        log_post = log_prior + log_likelihood
        posteriors[g] = np.exp(log_post - logsumexp(log_post))

    return _ProductFit(grid, posteriors, n_days, totals, centers, mu, pooled)


def _group_rates(fit: _ProductFit) -> dict[str, dict]:
    """Step 5: posterior mean and 90% credible interval of each group's daily rate."""
    tail = (1 - CREDIBLE_MASS) / 2
    rates = {}
    for g in GROUPS:
        post = fit.posteriors[g]
        mid_cdf = np.cumsum(post) - post / 2
        low, high = np.interp([tail, 1 - tail], mid_cdf, fit.grid)
        rates[g] = {
            "posterior_mean": float(np.dot(fit.grid, post)),
            "ci_low": float(low),
            "ci_high": float(high),
            "n_days": int(fit.n_days[g]),
        }
    return rates


def _group_window_pmf(grid: np.ndarray, posterior: np.ndarray, k: int) -> np.ndarray:
    """Demand over k days of one group: a Poisson mixture with one lambda shared by all k days."""
    if k == 0:
        return np.array([1.0])
    support = posterior > TAIL_PROBABILITY * 1e-6
    lam, weights = grid[support], posterior[support]
    s_max = int(poisson.isf(TAIL_PROBABILITY * 1e-3, k * lam.max())) + 1
    s = np.arange(s_max + 1)
    return poisson.pmf(s[:, None], k * lam[None, :]) @ weights


def _truncate(pmf: np.ndarray) -> dict[int, float]:
    """Step 7: drop both tails below TAIL_PROBABILITY and renormalize."""
    cdf = np.cumsum(pmf)
    low = int(np.searchsorted(cdf, TAIL_PROBABILITY))
    high = min(int(np.searchsorted(cdf, 1 - TAIL_PROBABILITY)), len(pmf) - 1)
    kept = pmf[low : high + 1]
    kept = kept / kept.sum()
    return {int(low + i): float(p) for i, p in enumerate(kept)}


def _window_distribution(fit: _ProductFit, group_days: dict[str, int]) -> dict[int, float]:
    """Step 6: convolve the per-group predictive distributions over a window of days."""
    if fit.pooled:
        # One rate shared by every day of the window: a single group of k = window length.
        group_days = {"weekday": sum(group_days.values())}
    pmf = np.array([1.0])
    for g, k in group_days.items():
        if k > 0:
            pmf = np.convolve(pmf, _group_window_pmf(fit.grid, fit.posteriors[g], k))
    return _truncate(pmf)


def calculate_stockout_risk(
    product_id: int,
    current_stock: int,
    lead_time_days: int,
    current_date: date,
    history_df: pd.DataFrame,
    *,
    review_period_days: int = 1,
    promo_dates: Collection[date] = (),
) -> dict:
    """Estimate the demand distribution and the probability of a stockout before an order arrives.

    See the module docstring for the full model. In short: each evidence group's daily Poisson rate
    gets a posterior from Bayes' theorem (empirical Bayes lognormal prior x Poisson likelihood), and
    window demand is a Poisson mixture over that posterior, using each day's own group.

    Assumptions:
        - The decision is made at the start of current_date, before that day's sales.
        - Lead-time window: current_date .. current_date + lead_time_days - 1.
          Order-horizon window: current_date .. current_date + lead_time_days + review_period_days - 1.
        - current_stock is the inventory position (on hand + on order).
        - Weekend days are Saturday and Sunday. promo_dates lists the dates on which THIS product
          is on promotion; dates outside the windows are ignored. datetime / pd.Timestamp values
          are converted to dates.
        - history_df units_sold is true demand (not censored by stockouts).
        - Daily demands are independent Poisson given the group's rate; one unknown rate per group,
          shared by every day of that group.
        - stockout_risk covers the lead-time window only; the order-horizon distribution is
          returned for the optimizer.
        - With no history for the product, every group's posterior is a prior centered on
          DEFAULT_DAILY_DEMAND (8 units per day).

    Args:
        product_id: Product to assess.
        current_stock: Inventory position. Must be >= 0.
        lead_time_days: Lead time in days. Must be >= 0.
        current_date: Decision day; the first day of both windows.
        history_df: Sales history in the SalesHistory schema. Only rows for product_id are used.
        review_period_days: Days between decisions. Must be >= 1.
        promo_dates: Dates on which this product has a promotion.

    Returns:
        Dict with product_id (int), stockout_risk (float), group_rates ({group: {"posterior_mean",
        "ci_low", "ci_high", "n_days"}}), lead_time_day_groups ({group: int}),
        lead_time_distribution (int -> float), order_horizon_days (int),
        order_horizon_distribution (int -> float), expected_lead_time_demand (float) and
        expected_order_horizon_demand (float).

    Raises:
        ValueError: If current_stock < 0, lead_time_days < 0, review_period_days < 1,
            current_date or a promo date is not a date, or history_df is missing a
            SalesHistory column.
    """
    return _stockout_risk(
        product_id, current_stock, lead_time_days, current_date, history_df,
        review_period_days=review_period_days, promo_dates=promo_dates,
    )


def _stockout_risk(
    product_id: int,
    current_stock: int,
    lead_time_days: int,
    current_date: date,
    history_df: pd.DataFrame,
    *,
    review_period_days: int = 1,
    promo_dates: Collection[date] = (),
    fit: _ProductFit | None = None,
    pooled: bool = False,
) -> dict:
    """calculate_stockout_risk with two private options for the evaluation.

    fit: a cached _fit_product(history_df, product_id, pooled) result. The history is fixed during
        an evaluation, so reusing it gives results identical to refitting on every call.
    pooled: use the pooled (no weekend/promo evidence) model; ignored when fit is given.
    """
    if current_stock < 0:
        raise ValueError(f"current_stock must be >= 0, got {current_stock}")
    if lead_time_days < 0:
        raise ValueError(f"lead_time_days must be >= 0, got {lead_time_days}")
    if review_period_days < 1:
        raise ValueError(f"review_period_days must be >= 1, got {review_period_days}")
    missing = [c for c in _REQUIRED_HISTORY_COLUMNS if c not in history_df.columns]
    if missing:
        raise ValueError(f"history_df is missing columns: {missing}")
    start = _as_date(current_date)
    promos = {_as_date(d) for d in promo_dates}

    if fit is None:
        fit = _fit_product(history_df, product_id, pooled=pooled)
    order_horizon_days = lead_time_days + review_period_days
    lead_time_day_groups = _window_day_groups(start, lead_time_days, promos)
    horizon_day_groups = _window_day_groups(start, order_horizon_days, promos)

    lead_time_distribution = _window_distribution(fit, lead_time_day_groups)
    order_horizon_distribution = _window_distribution(fit, horizon_day_groups)

    stockout_risk = sum(p for d, p in lead_time_distribution.items() if d > current_stock)

    return {
        "product_id": int(product_id),
        "stockout_risk": float(min(1.0, max(0.0, stockout_risk))),
        "group_rates": _group_rates(fit),
        "lead_time_day_groups": lead_time_day_groups,
        "lead_time_distribution": lead_time_distribution,
        "order_horizon_days": int(order_horizon_days),
        "order_horizon_distribution": order_horizon_distribution,
        "expected_lead_time_demand": float(sum(d * p for d, p in lead_time_distribution.items())),
        "expected_order_horizon_demand": float(
            sum(d * p for d, p in order_horizon_distribution.items())
        ),
    }
