"""Effective inventory position: stock that can still serve demand once a new order would arrive."""

from __future__ import annotations

import math
from datetime import date, timedelta


def effective_inventory_position(
    batches: list[tuple[date, int]],
    in_transit_qty: int,
    current_date: date,
    lead_time_days: int,
    shelf_life_days: int | None,
    mean_daily_demand: float,
) -> int:
    """Inventory position that counts perishable stock only to the extent it can sell before expiring.

    The ordinary inventory position (on hand + in transit) overstates the stock of a perishable
    product: units that will expire before a new order could arrive cannot cover demand after that
    point, so a reorder trigger based on them fires too late.

    Rule:
        - Non-perishable (shelf_life_days is None): sum of batches + in_transit_qty.
        - Perishable: a batch arriving on day a is discarded at the start of day a + shelf_life_days.
          Walk the batches oldest first (they are sold first in, first out) with a running counter
          of expected demand already absorbed by older batches.
            * A batch that expires on or before current_date + lead_time_days (gone by the time a
              new order would arrive) counts only for
                  min(qty, floor(expected demand from current_date until its expiry - demand
                                 already absorbed by older batches)),
              at mean_daily_demand per day; that credit is added to the counter.
            * A batch still sellable on current_date + lead_time_days counts in full.
          Then in_transit_qty is added in full (it arrives fresh).

    Assumptions:
        - This is a DETERMINISTIC approximation: demand is taken to be exactly its mean, so random
          variation in how fast old batches sell is ignored. Partial credit is rounded down.
        - All batches of a product share one shelf life, so newer batches never expire before
          older ones.
        - Person 1's live app, which does not track batches, can pass on-hand + on-order as the
          inventory position instead; this function is for callers that know batch ages (such as
          the evaluation simulation).

    Args:
        batches: On-hand stock as (arrival_date, qty), in any order; quantities >= 0.
        in_transit_qty: Units ordered but not yet received. Must be >= 0.
        current_date: Today (start of day, after today's receipts and expiries).
        lead_time_days: Days until an order placed today would arrive. Must be >= 0.
        shelf_life_days: Shelf life in days, or None if not perishable. Must be >= 1 if given.
        mean_daily_demand: Expected units demanded per day. Must be >= 0.

    Returns:
        Effective inventory position (int).

    Raises:
        ValueError: On negative quantities, lead time or demand, or shelf_life_days < 1.
    """
    if in_transit_qty < 0:
        raise ValueError(f"in_transit_qty must be >= 0, got {in_transit_qty}")
    if lead_time_days < 0:
        raise ValueError(f"lead_time_days must be >= 0, got {lead_time_days}")
    if shelf_life_days is not None and shelf_life_days < 1:
        raise ValueError(f"shelf_life_days must be >= 1, got {shelf_life_days}")
    if not mean_daily_demand >= 0:
        raise ValueError(f"mean_daily_demand must be >= 0, got {mean_daily_demand}")
    if any(qty < 0 for _, qty in batches):
        raise ValueError("batch quantities must be >= 0")

    if shelf_life_days is None:
        return int(sum(qty for _, qty in batches) + in_transit_qty)

    arrival_of_new_order = current_date + timedelta(days=lead_time_days)
    absorbed = 0
    position = 0
    for arrival, qty in sorted(batches, key=lambda b: b[0]):
        expiry = arrival + timedelta(days=shelf_life_days)
        if expiry <= arrival_of_new_order:
            days_left = max(0, (expiry - current_date).days)
            demand_left = max(0, math.floor(mean_daily_demand * days_left) - absorbed)
            credit = min(qty, demand_left)
            absorbed += credit
        else:
            credit = qty
        position += credit
    return int(position + in_transit_qty)
