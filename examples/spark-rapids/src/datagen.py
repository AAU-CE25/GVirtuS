# datagen.py

"""
Generates synthetic e-commerce data at configurable scale.
Uses pyarrow for Spark-compatible parquet output.
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
try:
    from .config import (
        SCALE_FACTOR, NUM_CUSTOMERS, NUM_PRODUCTS,
        NUM_ORDERS, NUM_ORDER_ITEMS, NUM_CLICKSTREAM, DATA_DIR
    )
except ImportError:
    from config import (
        SCALE_FACTOR, NUM_CUSTOMERS, NUM_PRODUCTS,
        NUM_ORDERS, NUM_ORDER_ITEMS, NUM_CLICKSTREAM, DATA_DIR
    )


def save_parquet(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to parquet with Spark-compatible timestamps.
    Forces all timestamps to microsecond precision.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert any datetime columns to microsecond precision
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.floor("us")

    # Use pyarrow to write with explicit timestamp type
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Replace any nanosecond timestamp columns with microsecond
    new_fields = []
    for i, field in enumerate(table.schema):
        if pa.types.is_timestamp(field.type):
            new_fields.append(pa.field(field.name, pa.timestamp("us")))
        else:
            new_fields.append(field)

    new_schema = pa.schema(new_fields)
    table = table.cast(new_schema)

    pq.write_table(table, output_path)
    print(f"  → Saved to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def generate_customers(output_path: str, num_customers: int):
    """Customer dimension table."""
    print(f"Generating {num_customers:,} customers...")
    np.random.seed(42)

    regions = ["North", "South", "East", "West", "Central"]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    channels = ["Web", "Mobile", "Store", "Marketplace"]

    df = pd.DataFrame({
        "customer_id": np.arange(1, num_customers + 1),
        "region": np.random.choice(regions, num_customers),
        "customer_tier": np.random.choice(tiers, num_customers, p=[0.4, 0.3, 0.2, 0.1]),
        "signup_channel": np.random.choice(channels, num_customers),
        "account_age_days": np.random.randint(1, 3650, num_customers),
        "is_active": np.random.choice([True, False], num_customers, p=[0.7, 0.3]),
        "credit_score": np.random.normal(650, 100, num_customers).clip(300, 850).astype(int),
        "annual_income": np.random.lognormal(10.5, 0.8, num_customers).clip(15000, 500000).astype(int),
    })

    save_parquet(df, output_path)
    return df


def generate_products(output_path: str, num_products: int):
    """Product dimension table."""
    print(f"Generating {num_products:,} products...")
    np.random.seed(43)

    categories = [
        "Electronics", "Clothing", "Home & Garden", "Sports",
        "Books", "Toys", "Food & Grocery", "Health & Beauty",
        "Automotive", "Office Supplies"
    ]
    subcategories = {
        "Electronics": ["Phones", "Laptops", "Tablets", "Accessories", "Audio"],
        "Clothing": ["Men", "Women", "Kids", "Shoes", "Accessories"],
        "Home & Garden": ["Furniture", "Kitchen", "Decor", "Garden", "Tools"],
        "Sports": ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Cycling"],
        "Books": ["Fiction", "Non-Fiction", "Academic", "Children", "Comics"],
        "Toys": ["Action Figures", "Board Games", "Puzzles", "Dolls", "LEGO"],
        "Food & Grocery": ["Snacks", "Beverages", "Fresh", "Frozen", "Organic"],
        "Health & Beauty": ["Skincare", "Haircare", "Supplements", "Makeup", "Wellness"],
        "Automotive": ["Parts", "Accessories", "Tools", "Care", "Electronics"],
        "Office Supplies": ["Paper", "Pens", "Furniture", "Tech", "Storage"],
    }

    cats = np.random.choice(categories, num_products)
    subcats = [np.random.choice(subcategories[c]) for c in cats]

    df = pd.DataFrame({
        "product_id": np.arange(1, num_products + 1),
        "category": cats,
        "subcategory": subcats,
        "base_price": np.random.lognormal(3.0, 1.0, num_products).clip(1, 5000).round(2),
        "weight_kg": np.random.lognormal(0.5, 1.0, num_products).clip(0.01, 50).round(2),
        "rating": np.random.uniform(1.0, 5.0, num_products).round(1),
        "review_count": np.random.exponential(50, num_products).astype(int),
        "is_prime_eligible": np.random.choice([True, False], num_products, p=[0.6, 0.4]),
    })

    save_parquet(df, output_path)
    return df


def generate_orders(output_path: str, num_orders: int, num_customers: int):
    """Orders fact table."""
    print(f"Generating {num_orders:,} orders...")
    np.random.seed(44)

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    statuses = ["completed", "shipped", "processing", "cancelled", "returned"]
    payment_methods = ["credit_card", "debit_card", "paypal", "apple_pay", "gift_card"]

    chunk_size = min(5_000_000, num_orders)
    chunks = []

    for start in range(0, num_orders, chunk_size):
        end = min(start + chunk_size, num_orders)
        size = end - start

        # Generate dates as strings first, then convert — avoids nanosecond issues
        random_days = np.random.randint(0, date_range_days, size)
        base = np.datetime64('2022-01-01', 'D')
        order_dates = base + random_days.astype('timedelta64[D]')

        chunk = pd.DataFrame({
            "order_id": np.arange(start + 1, end + 1),
            "customer_id": np.random.randint(1, num_customers + 1, size),
            "order_date": pd.to_datetime(order_dates),
            "status": np.random.choice(statuses, size, p=[0.65, 0.15, 0.10, 0.07, 0.03]),
            "payment_method": np.random.choice(payment_methods, size),
            "shipping_cost": np.random.exponential(8.0, size).clip(0, 50).round(2),
            "discount_pct": np.random.choice([0, 5, 10, 15, 20, 25, 30], size,
                                              p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]),
        })
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    save_parquet(df, output_path)
    return df


def generate_order_items(output_path: str, num_items: int, num_orders: int, num_products: int):
    """Order items fact table."""
    print(f"Generating {num_items:,} order items...")
    np.random.seed(45)

    chunk_size = min(5_000_000, num_items)
    chunks = []

    for start in range(0, num_items, chunk_size):
        end = min(start + chunk_size, num_items)
        size = end - start

        quantities = np.random.geometric(0.5, size).clip(1, 20)
        unit_prices = np.random.lognormal(3.0, 1.0, size).clip(1, 5000).round(2)

        chunk = pd.DataFrame({
            "item_id": np.arange(start + 1, end + 1),
            "order_id": np.random.randint(1, num_orders + 1, size),
            "product_id": np.random.randint(1, num_products + 1, size),
            "quantity": quantities,
            "unit_price": unit_prices,
            "total_price": (quantities * unit_prices).round(2),
        })
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    save_parquet(df, output_path)
    return df


def generate_clickstream(output_path: str, num_events: int, num_customers: int, num_products: int):
    """Clickstream / browsing behavior."""
    print(f"Generating {num_events:,} clickstream events...")
    np.random.seed(46)

    actions = ["view", "click", "add_to_cart", "remove_from_cart", "wishlist", "search"]
    devices = ["desktop", "mobile", "tablet"]
    browsers = ["chrome", "safari", "firefox", "edge", "app"]

    chunk_size = min(5_000_000, num_events)
    chunks = []

    date_range_days = 365 * 4

    for start in range(0, num_events, chunk_size):
        end = min(start + chunk_size, num_events)
        size = end - start

        # Generate dates using numpy — avoids nanosecond issues
        random_days = np.random.randint(0, date_range_days, size)
        base = np.datetime64('2022-01-01', 'D')
        event_dates = base + random_days.astype('timedelta64[D]')

        chunk = pd.DataFrame({
            "event_id": np.arange(start + 1, end + 1),
            "customer_id": np.random.randint(1, num_customers + 1, size),
            "product_id": np.random.randint(1, num_products + 1, size),
            "action": np.random.choice(actions, size, p=[0.45, 0.25, 0.12, 0.05, 0.05, 0.08]),
            "device": np.random.choice(devices, size, p=[0.4, 0.45, 0.15]),
            "browser": np.random.choice(browsers, size, p=[0.45, 0.25, 0.1, 0.05, 0.15]),
            "session_duration_sec": np.random.exponential(300, size).clip(1, 7200).astype(int),
            "event_date": pd.to_datetime(event_dates),
        })
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    save_parquet(df, output_path)
    return df


def generate_all():
    """Generate all tables at the configured scale factor."""
    print(f"\n{'='*60}")
    print(f"  E-Commerce Data Generator — Scale Factor {SCALE_FACTOR}")
    print(f"{'='*60}\n")

    generate_customers(f"{DATA_DIR}/customers.parquet", NUM_CUSTOMERS)
    generate_products(f"{DATA_DIR}/products.parquet", NUM_PRODUCTS)
    generate_orders(f"{DATA_DIR}/orders.parquet", NUM_ORDERS, NUM_CUSTOMERS)
    generate_order_items(f"{DATA_DIR}/order_items.parquet", NUM_ORDER_ITEMS, NUM_ORDERS, NUM_PRODUCTS)
    generate_clickstream(f"{DATA_DIR}/clickstream.parquet", NUM_CLICKSTREAM, NUM_CUSTOMERS, NUM_PRODUCTS)

    print(f"\n{'='*60}")
    print(f"  Data generation complete!")
    total_size = sum(
        os.path.getsize(f"{DATA_DIR}/{f}")
        for f in os.listdir(DATA_DIR) if f.endswith('.parquet')
    )
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_all()