import sqlite3
from pathlib import Path
import pandas as pd

# Define path to SQLite database file inside the data/ directory
DB_PATH = Path(__file__).parent.parent / "data" / "inventory.db"


def get_connection():
    """Returns a connection object to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Creates the 5 core tables and seeds initial products if database is empty."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Products Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Products (
        product_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_cost REAL NOT NULL,
        selling_price REAL NOT NULL,
        holding_cost_per_day REAL NOT NULL,
        stockout_penalty REAL NOT NULL,
        shelf_life_days INTEGER,
        lead_time_days INTEGER NOT NULL
    );
    """)

    # 2. IsA Category Hierarchy Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS IsA (
        child TEXT NOT NULL,
        parent TEXT NOT NULL,
        PRIMARY KEY (child, parent)
    );
    """)

    # 3. StockLevels Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS StockLevels (
        product_id TEXT PRIMARY KEY,
        quantity INTEGER NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(product_id) REFERENCES Products(product_id)
    );
    """)

    # 4. SalesHistory Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS SalesHistory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        product_id TEXT NOT NULL,
        units_sold INTEGER NOT NULL,
        is_weekend INTEGER NOT NULL,
        has_promo INTEGER NOT NULL,
        FOREIGN KEY(product_id) REFERENCES Products(product_id)
    );
    """)

    # 5. ReorderLogs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ReorderLogs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        product_id TEXT NOT NULL,
        agent_state TEXT NOT NULL,
        stockout_risk REAL NOT NULL,
        recommended_qty INTEGER NOT NULL,
        expected_utility REAL NOT NULL,
        alert_reason TEXT,
        FOREIGN KEY(product_id) REFERENCES Products(product_id)
    );
    """)

    conn.commit()

    # Check if database is newly created to seed initial data
    cursor.execute("SELECT COUNT(*) FROM Products;")
    if cursor.fetchone()[0] == 0:
        _seed_initial_data(conn)

    conn.close()


def _seed_initial_data(conn):
    """Populates initial sample inventory, relationships, and stock levels."""
    cursor = conn.cursor()

    # Seed 10 sample products across Perishable, Seasonal, and Standard categories
    products = [
        ("P001", "Whole Milk 1L", "Milk", 1.20, 2.50, 0.05, 3.00, 7, 2),
        ("P002", "Greek Yogurt 500g", "Yogurt", 1.80, 3.80, 0.08, 4.00, 10, 2),
        ("P003", "Cheddar Cheese 200g", "Cheese", 2.50, 5.00, 0.10, 5.00, 30, 3),
        ("P004", "Fresh Strawberries 250g", "Berries", 1.50, 3.50, 0.15, 4.50, 4, 1),
        ("P005", "Organic Bananas 1kg", "Fruit", 0.80, 1.90, 0.05, 2.00, 5, 1),
        ("P006", "Sunscreen SPF50", "SummerCare", 6.00, 14.00, 0.20, 10.00, None, 4),
        ("P007", "Beach Towel Premium", "BeachGear", 8.00, 22.00, 0.25, 12.00, None, 5),
        ("P008", "Basmati Rice 5kg", "Grains", 7.00, 15.00, 0.10, 10.00, None, 3),
        ("P009", "Olive Oil 1L", "Pantry", 5.50, 12.00, 0.12, 8.00, None, 4),
        ("P010", "Dark Chocolate 100g", "Snacks", 1.00, 2.80, 0.04, 2.50, None, 2),
    ]

    cursor.executemany("""
    INSERT INTO Products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, products)

    # Seed Category Hierarchies (IsA)
    isa_relations = [
        ("Milk", "Dairy"),
        ("Yogurt", "Dairy"),
        ("Cheese", "Dairy"),
        ("Dairy", "Perishable"),
        ("Berries", "Produce"),
        ("Fruit", "Produce"),
        ("Produce", "Perishable"),
        ("SummerCare", "Seasonal"),
        ("BeachGear", "Seasonal"),
        ("Grains", "Pantry"),
        ("Snacks", "NonPerishable"),
        ("Pantry", "NonPerishable"),
    ]

    cursor.executemany("""
    INSERT INTO IsA VALUES (?, ?);
    """, isa_relations)

    # Seed Initial Stock Levels
    stock_levels = [
        ("P001", 18),
        ("P002", 25),
        ("P003", 40),
        ("P004", 8),
        ("P005", 12),
        ("P006", 30),
        ("P007", 15),
        ("P008", 50),
        ("P009", 35),
        ("P010", 60),
    ]

    cursor.executemany("""
    INSERT INTO StockLevels (product_id, quantity) VALUES (?, ?);
    """, stock_levels)

    conn.commit()


# =========================================================
# Database Helper Functions (No raw SQL outside this file)
# =========================================================

def get_products() -> pd.DataFrame:
    """Retrieves all products as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM Products;", conn)
    conn.close()
    return df


def get_stock(product_id: str) -> int:
    """Gets current stock quantity for a given product."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT quantity FROM StockLevels WHERE product_id = ?;", (product_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0


def update_stock(product_id: str, qty: int) -> None:
    """Updates stock quantity for a product."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE StockLevels 
    SET quantity = ?, last_updated = CURRENT_TIMESTAMP 
    WHERE product_id = ?;
    """, (qty, product_id))
    conn.commit()
    conn.close()


def get_sales_history(product_id: str = None) -> pd.DataFrame:
    """Retrieves sales history, optionally filtered by product_id."""
    conn = get_connection()
    if product_id:
        query = "SELECT * FROM SalesHistory WHERE product_id = ? ORDER BY date ASC;"
        df = pd.read_sql_query(query, conn, params=(product_id,))
    else:
        query = "SELECT * FROM SalesHistory ORDER BY date ASC;"
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def insert_sales_rows(df: pd.DataFrame) -> None:
    """Bulk inserts sales history records from a DataFrame."""
    conn = get_connection()
    df.to_sql("SalesHistory", conn, if_exists="append", index=False)
    conn.close()


def log_decision(product_id: str, agent_state: str, stockout_risk: float, 
                 recommended_qty: int, expected_utility: float, alert_reason: str = None) -> None:
    """Logs an agent decision into the ReorderLogs table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO ReorderLogs (product_id, agent_state, stockout_risk, recommended_qty, expected_utility, alert_reason)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (product_id, agent_state, stockout_risk, recommended_qty, expected_utility, alert_reason))
    conn.commit()
    conn.close()


def get_logs() -> pd.DataFrame:
    """Retrieves all reorder decision logs."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM ReorderLogs ORDER BY timestamp DESC;", conn)
    conn.close()
    return df


def get_isa_relations() -> list[tuple[str, str]]:
    """Fetches category hierarchy tuples (child, parent) for rule evaluation."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT child, parent FROM IsA;")
    rows = cursor.fetchall()
    conn.close()
    return rows