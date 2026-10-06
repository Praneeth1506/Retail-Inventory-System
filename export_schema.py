import pandas as pd
from src.db import init_db, get_products

if __name__ == "__main__":
    print("1. Initializing Database and Tables...")
    init_db()

    print("2. Exporting Products table for Person 2...")
    df_products = get_products()
    export_path = "data/products_export.csv"
    df_products.to_csv(export_path, index=False)

    print(f"\nSuccess! Exported {len(df_products)} products to '{export_path}'.")
    print("\n--- Handoff Information for Person 2 ---")
    print("• SQLite Database created at 'data/inventory.db'")
    print("• Seeded Products CSV created at 'data/products_export.csv'")
    print("• Database helper 'insert_sales_rows(df)' is ready for loading historical sales.")