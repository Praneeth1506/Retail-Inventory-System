"""Tests for decision_engine.inventory.effective_inventory_position."""

import inspect
from datetime import date

import pytest

from decision_engine import inventory, mock
from decision_engine.inventory import effective_inventory_position

TODAY = date(2025, 3, 10)


def position(batches, in_transit=0, lead=2, shelf=3, mean=10.0, today=TODAY):
    return effective_inventory_position(batches, in_transit, today, lead, shelf, mean)


def test_non_perishable_counts_everything():
    batches = [(date(2024, 1, 1), 10), (date(2025, 3, 9), 5)]
    assert position(batches, in_transit=7, shelf=None) == 22


def test_perishable_batch_expiring_before_lead_time_gets_partial_credit():
    # Arrived 9 Mar, shelf 3 -> discarded at the start of 12 Mar = today + lead 2: gone when a new
    # order arrives. It can sell on 10 and 11 Mar: 2 days x 10 = 20 of its 50 units.
    assert position([(date(2025, 3, 9), 50)], in_transit=5) == 20 + 5


def test_fresh_batch_gets_full_credit():
    # Arrived today: expires 13 Mar, still sellable when an order placed today arrives on 12 Mar.
    assert position([(TODAY, 50)]) == 50


def test_partial_credit_capped_at_batch_quantity():
    assert position([(date(2025, 3, 9), 8)]) == 8


def test_multiple_batches_sold_oldest_first():
    batches = [
        (date(2025, 3, 10), 40),  # fresh: full 40
        (date(2025, 3, 8), 15),   # expires 11 Mar: 1 day x 10 -> 10 of 15
        (date(2025, 3, 9), 30),   # expires 12 Mar: 20 by then, minus 10 absorbed -> 10 of 30
    ]
    assert position(batches) == 10 + 10 + 40


def test_older_batch_absorbs_demand_first():
    # The older batch alone covers both days of expected demand, so the next expiring batch gets nothing.
    batches = [(date(2025, 3, 8), 100), (date(2025, 3, 9), 30)]
    # 8 Mar batch expires 11 Mar: 1 day -> 10. 9 Mar batch: floor(20) - 10 = 10.
    assert position(batches) == 20
    assert position([(date(2025, 3, 9), 100), (date(2025, 3, 9), 30)]) == 20


def test_zero_lead_time_counts_unexpired_batches_in_full():
    assert position([(date(2025, 3, 8), 25)], lead=0) == 25


def test_without_expiring_stock_equals_on_hand_plus_on_order():
    batches = [(TODAY, 12), (date(2025, 3, 9), 0)]
    assert position(batches, in_transit=30, lead=1, shelf=5) == 42


@pytest.mark.parametrize(
    "kwargs",
    [{"in_transit": -1}, {"lead": -1}, {"shelf": 0}, {"mean": -0.5}, {"batches": [(TODAY, -3)]}],
)
def test_rejects_invalid_arguments(kwargs):
    args = {"batches": [(TODAY, 5)], **kwargs}
    with pytest.raises(ValueError):
        position(**args)


def test_mock_exports_same_function():
    assert mock.effective_inventory_position is inventory.effective_inventory_position
    assert inspect.signature(mock.effective_inventory_position) == inspect.signature(effective_inventory_position)
