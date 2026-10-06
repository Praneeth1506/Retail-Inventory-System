"""Streamlit AppTest checks for the What-If Explorer tab (run on a copy of the database)."""

import hashlib
import shutil
import time
from datetime import date
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.db as db

ROOT = Path(__file__).resolve().parent.parent
MONDAY = date(2026, 10, 5)


@pytest.fixture
def app(tmp_path, monkeypatch):
    copy = tmp_path / "inventory.db"
    shutil.copy(ROOT / "data" / "inventory.db", copy)
    monkeypatch.setattr(db, "DB_PATH", copy)
    st.cache_data.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key="wi_product").select("P001 - Whole Milk 1L").run()
    at.date_input(key="wi_date").set_value(MONDAY).run()
    # Pin the inputs that default from the database, so the tests don't depend on its current contents.
    at.number_input(key="wi_on_order").set_value(0).run()
    yield at, copy
    st.cache_data.clear()


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def risk(at: AppTest) -> float:
    return float(metric(at, "Stockout risk").rstrip("%"))


def test_stock_change_switches_order_to_wait(app):
    at, _ = app
    at.number_input(key="wi_stock").set_value(18).run()
    assert metric(at, "Decision").startswith("ORDER ")
    at.number_input(key="wi_stock").set_value(300).run()
    assert not at.exception
    assert metric(at, "Decision") == "WAIT"


def test_promotion_raises_stockout_risk(app):
    at, _ = app
    at.number_input(key="wi_stock").set_value(80).run()  # about the lead-time demand: risk well inside (0, 1)
    without = risk(at)
    at.checkbox(key="wi_promo").check().run()
    assert risk(at) > without
    assert 0 < without < 100


def test_explorer_never_writes_to_the_database(app):
    at, copy = app
    before = hashlib.sha256(copy.read_bytes()).hexdigest()
    at.number_input(key="wi_stock").set_value(5).run()
    at.checkbox(key="wi_promo").check().run()
    at.selectbox(key="wi_risk").select("50").run()
    at.number_input(key="wi_shelf").set_value(0).run()
    assert not at.exception
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == before


def test_each_change_recomputes_quickly(app):
    at, _ = app
    at.number_input(key="wi_stock").set_value(40).run()  # warm the history cache
    started = time.perf_counter()
    at.number_input(key="wi_stock").set_value(41).run()
    assert time.perf_counter() - started < 1.0


def test_empty_history_is_built_in_memory_without_writing(tmp_path, monkeypatch):
    """With SalesHistory empty (as shipped), the explorer must not seed it: it builds history in memory."""
    import sqlite3

    copy = tmp_path / "inventory.db"
    shutil.copy(ROOT / "data" / "inventory.db", copy)
    with sqlite3.connect(copy) as conn:
        conn.execute("DELETE FROM SalesHistory")
        conn.execute("DELETE FROM ReorderLogs")
    monkeypatch.setattr(db, "DB_PATH", copy)
    st.cache_data.clear()
    before = hashlib.sha256(copy.read_bytes()).hexdigest()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.number_input(key="wi_stock").set_value(18).run()
    assert not at.exception, [e.value for e in at.exception]
    assert metric(at, "Decision").startswith("ORDER ")
    assert any("in memory" in m.value for m in at.markdown)
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == before
    st.cache_data.clear()
