"""
datagen.py — Synthetic TPC-DS-like Parquet data generator.

Generates all 24 TPC-DS tables as Parquet files under  DATA_DIR/sf{N}/
using PySpark so the output is already partitioned and Spark-native.

Usage:
    python -m src.datagen              # SF1 (default)
"""

from __future__ import annotations

import random
import sys
import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, DateType, DecimalType, LongType,
)

from src.config import (
    SPARK_CPU_CONF, SPARK_APP_NAME, DATA_PATH, row_count, SCALE_FACTOR,
)


# ─── Helpers ───────────────────────────────────────────────────────────
_RNG = random.Random(42)

_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]

_CATEGORIES = [
    "Electronics", "Books", "Home", "Sports", "Music",
    "Women", "Men", "Children", "Jewelry", "Shoes",
]

_BRANDS = [f"Brand#{i}" for i in range(1, 51)]

_SHIP_MODES = ["REGULAR", "EXPRESS", "OVERNIGHT", "TWO DAY", "LIBRARY"]

_CARRIERS = ["DHL", "UPS", "FEDEX", "USPS", "AIRBORNE", "HARMONIC"]


def _rand_decimal(lo: float, hi: float) -> float:
    return round(_RNG.uniform(lo, hi), 2)


def _rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=_RNG.randint(0, delta))


# ─── Table generators ─────────────────────────────────────────────────
# Each returns a list[dict] that PySpark converts into a DataFrame.

def _gen_date_dim() -> list[dict]:
    """73 049 rows covering 1998-01-01 → 2003-12-31 (TPC-DS range)."""
    rows: list[dict] = []
    base = date(1998, 1, 1)
    n = row_count("date_dim")
    for i in range(1, n + 1):
        d = base + timedelta(days=i - 1)
        rows.append({
            "d_date_sk": i,
            "d_date_id": f"AAAAAAAAA{i:07d}",
            "d_date": d,
            "d_month_seq": (d.year - 1998) * 12 + d.month,
            "d_year": d.year,
            "d_moy": d.month,
            "d_dom": d.day,
            "d_qoy": (d.month - 1) // 3 + 1,
            "d_dow": d.weekday(),
            "d_day_name": d.strftime("%A"),
            "d_holiday": "Y" if d.weekday() >= 5 else "N",
            "d_weekend": "Y" if d.weekday() >= 5 else "N",
            "d_current_year": "Y" if d.year == 2002 else "N",
        })
    return rows


def _gen_item() -> list[dict]:
    rows: list[dict] = []
    n = row_count("item")
    for i in range(1, n + 1):
        cat = _CATEGORIES[i % len(_CATEGORIES)]
        rows.append({
            "i_item_sk": i,
            "i_item_id": f"AAAAAAAAA{i:07d}",
            "i_item_desc": f"Item description {i}",
            "i_current_price": _rand_decimal(0.50, 300.00),
            "i_wholesale_cost": _rand_decimal(0.25, 150.00),
            "i_brand_id": (i % 50) + 1,
            "i_brand": _BRANDS[i % len(_BRANDS)],
            "i_class_id": (i % 100) + 1,
            "i_class": f"Class{(i % 100) + 1}",
            "i_category_id": (i % len(_CATEGORIES)) + 1,
            "i_category": cat,
            "i_manufact_id": (i % 200) + 1,
            "i_manufact": f"Manufacturer{(i % 200) + 1}",
            "i_product_name": f"Product_{i}",
            "i_manager_id": (i % 100) + 1,
        })
    return rows


def _gen_store() -> list[dict]:
    rows: list[dict] = []
    n = row_count("store")
    for i in range(1, n + 1):
        rows.append({
            "s_store_sk": i,
            "s_store_id": f"AAAAAAAAA{i:07d}",
            "s_store_name": f"Store_{i}",
            "s_number_employees": _RNG.randint(50, 500),
            "s_floor_space": _RNG.randint(5000, 100000),
            "s_state": _STATES[i % len(_STATES)],
            "s_city": f"City_{i}",
            "s_county": f"County_{i}",
            "s_zip": f"{_RNG.randint(10000, 99999)}",
            "s_country": "United States",
            "s_gmt_offset": -5.0,
            "s_market_id": (i % 10) + 1,
            "s_company_id": 1,
            "s_company_name": "ACME Corp",
        })
    return rows


def _gen_customer(n: int | None = None) -> list[dict]:
    rows: list[dict] = []
    n = n or row_count("customer")
    for i in range(1, n + 1):
        rows.append({
            "c_customer_sk": i,
            "c_customer_id": f"AAAAAAAAA{i:07d}",
            "c_current_addr_sk": _RNG.randint(1, max(1, row_count("customer_address"))),
            "c_first_name": f"First_{i}",
            "c_last_name": f"Last_{i}",
            "c_birth_year": _RNG.randint(1940, 2000),
            "c_birth_month": _RNG.randint(1, 12),
            "c_preferred_cust_flag": _RNG.choice(["Y", "N"]),
            "c_email_address": f"customer{i}@example.com",
        })
    return rows


def _gen_customer_address() -> list[dict]:
    rows: list[dict] = []
    n = row_count("customer_address")
    for i in range(1, n + 1):
        rows.append({
            "ca_address_sk": i,
            "ca_address_id": f"AAAAAAAAA{i:07d}",
            "ca_street_number": str(_RNG.randint(1, 9999)),
            "ca_street_name": f"Street_{i}",
            "ca_city": f"City_{_RNG.randint(1, 500)}",
            "ca_county": f"County_{_RNG.randint(1, 200)}",
            "ca_state": _STATES[i % len(_STATES)],
            "ca_zip": f"{_RNG.randint(10000, 99999)}",
            "ca_country": "United States",
            "ca_gmt_offset": float(_RNG.choice([-5, -6, -7, -8, -9, -10])),
        })
    return rows


def _gen_promotion() -> list[dict]:
    rows: list[dict] = []
    n = row_count("promotion")
    for i in range(1, n + 1):
        rows.append({
            "p_promo_sk": i,
            "p_promo_id": f"AAAAAAAAA{i:07d}",
            "p_promo_name": f"Promo_{i}",
            "p_channel_email": _RNG.choice(["Y", "N"]),
            "p_channel_tv": _RNG.choice(["Y", "N"]),
            "p_channel_catalog": _RNG.choice(["Y", "N"]),
            "p_discount_active": _RNG.choice(["Y", "N"]),
            "p_start_date_sk": _RNG.randint(1, 73049),
            "p_end_date_sk": _RNG.randint(1, 73049),
        })
    return rows


def _gen_warehouse() -> list[dict]:
    rows: list[dict] = []
    n = row_count("warehouse")
    for i in range(1, n + 1):
        rows.append({
            "w_warehouse_sk": i,
            "w_warehouse_id": f"AAAAAAAAA{i:07d}",
            "w_warehouse_name": f"Warehouse_{i}",
            "w_warehouse_sq_ft": _RNG.randint(50000, 500000),
            "w_state": _STATES[i % len(_STATES)],
            "w_country": "United States",
        })
    return rows


def _gen_ship_mode() -> list[dict]:
    rows: list[dict] = []
    n = row_count("ship_mode")
    for i in range(1, n + 1):
        rows.append({
            "sm_ship_mode_sk": i,
            "sm_ship_mode_id": f"AAAAAAAAA{i:07d}",
            "sm_type": _SHIP_MODES[i % len(_SHIP_MODES)],
            "sm_carrier": _CARRIERS[i % len(_CARRIERS)],
        })
    return rows


def _gen_household_demographics() -> list[dict]:
    rows: list[dict] = []
    n = row_count("household_demographics")
    for i in range(1, n + 1):
        rows.append({
            "hd_demo_sk": i,
            "hd_income_band_sk": (i % 20) + 1,
            "hd_buy_potential": _RNG.choice(["Unknown", "0-500", "501-1000",
                                              "1001-5000", "5001-10000", ">10000"]),
            "hd_dep_count": _RNG.randint(0, 9),
            "hd_vehicle_count": _RNG.randint(0, 4),
        })
    return rows


def _gen_time_dim() -> list[dict]:
    rows: list[dict] = []
    for i in range(86400):
        h, rem = divmod(i, 3600)
        m, s = divmod(rem, 60)
        rows.append({
            "t_time_sk": i,
            "t_time": i,
            "t_hour": h,
            "t_minute": m,
            "t_second": s,
            "t_am_pm": "AM" if h < 12 else "PM",
        })
    return rows


def _gen_reason() -> list[dict]:
    reasons = [
        "Did not like", "Wrong size", "Defective", "Changed mind",
        "Better price", "Duplicate", "Not needed", "Late delivery",
    ]
    rows: list[dict] = []
    n = row_count("reason")
    for i in range(1, n + 1):
        rows.append({
            "r_reason_sk": i,
            "r_reason_id": f"AAAAAAAAA{i:07d}",
            "r_reason_desc": reasons[i % len(reasons)],
        })
    return rows


def _gen_income_band() -> list[dict]:
    rows: list[dict] = []
    for i in range(1, 21):
        rows.append({
            "ib_income_band_sk": i,
            "ib_lower_bound": (i - 1) * 10000,
            "ib_upper_bound": i * 10000 - 1,
        })
    return rows


# ─── Fact table generators (use Spark RDD for volume) ─────────────────

def _gen_store_sales_spark(spark: SparkSession) -> DataFrame:
    """Generate store_sales as a Spark DataFrame (millions of rows)."""
    n = row_count("store_sales")
    n_items = row_count("item")
    n_customers = row_count("customer")
    n_stores = row_count("store")
    n_promos = row_count("promotion")
    n_dates = row_count("date_dim")

    # Generate via Spark range + random UDFs for speed
    df = spark.range(1, n + 1).toDF("row_id")
    df = (
        df
        .withColumn("ss_sold_date_sk",    (F.hash(F.col("row_id")) % n_dates).cast("int").cast("int") + 1)
        .withColumn("ss_sold_time_sk",    (F.abs(F.hash(F.col("row_id") + 1)) % 86400).cast("int"))
        .withColumn("ss_item_sk",         (F.abs(F.hash(F.col("row_id") + 2)) % n_items).cast("int") + 1)
        .withColumn("ss_customer_sk",     (F.abs(F.hash(F.col("row_id") + 3)) % n_customers).cast("int") + 1)
        .withColumn("ss_store_sk",        (F.abs(F.hash(F.col("row_id") + 4)) % n_stores).cast("int") + 1)
        .withColumn("ss_promo_sk",        (F.abs(F.hash(F.col("row_id") + 5)) % n_promos).cast("int") + 1)
        .withColumn("ss_ticket_number",   F.col("row_id").cast("int"))
        .withColumn("ss_quantity",        (F.abs(F.hash(F.col("row_id") + 6)) % 100).cast("int") + 1)
        .withColumn("ss_wholesale_cost",  (F.abs(F.hash(F.col("row_id") + 7)) % 15000 / 100.0).cast("decimal(7,2)"))
        .withColumn("ss_list_price",      (F.abs(F.hash(F.col("row_id") + 8)) % 30000 / 100.0).cast("decimal(7,2)"))
        .withColumn("ss_sales_price",     (F.abs(F.hash(F.col("row_id") + 9)) % 25000 / 100.0).cast("decimal(7,2)"))
        .withColumn("ss_ext_sales_price", (F.col("ss_quantity") * F.col("ss_sales_price")).cast("decimal(7,2)"))
        .withColumn("ss_net_paid",        F.col("ss_ext_sales_price"))
        .withColumn("ss_net_profit",      (F.col("ss_ext_sales_price") - F.col("ss_quantity") * F.col("ss_wholesale_cost")).cast("decimal(7,2)"))
        .withColumn("ss_ext_tax",         (F.col("ss_ext_sales_price") * 0.07).cast("decimal(7,2)"))
        .withColumn("ss_coupon_amt",      (F.abs(F.hash(F.col("row_id") + 10)) % 5000 / 100.0).cast("decimal(7,2)"))
        .withColumn("ss_addr_sk",         (F.abs(F.hash(F.col("row_id") + 11)) % max(1, row_count("customer_address"))).cast("int") + 1)
        .withColumn("ss_cdemo_sk",        (F.abs(F.hash(F.col("row_id") + 12)) % 1920800).cast("int") + 1)
        .withColumn("ss_hdemo_sk",        (F.abs(F.hash(F.col("row_id") + 13)) % 7200).cast("int") + 1)
        .drop("row_id")
    )
    return df


def _gen_catalog_sales_spark(spark: SparkSession) -> DataFrame:
    n = row_count("catalog_sales")
    n_items = row_count("item")
    n_customers = row_count("customer")
    n_dates = row_count("date_dim")

    df = spark.range(1, n + 1).toDF("row_id")
    df = (
        df
        .withColumn("cs_sold_date_sk",       (F.abs(F.hash(F.col("row_id") + 20)) % n_dates).cast("int") + 1)
        .withColumn("cs_item_sk",            (F.abs(F.hash(F.col("row_id") + 21)) % n_items).cast("int") + 1)
        .withColumn("cs_bill_customer_sk",   (F.abs(F.hash(F.col("row_id") + 22)) % n_customers).cast("int") + 1)
        .withColumn("cs_order_number",       F.col("row_id").cast("int"))
        .withColumn("cs_quantity",           (F.abs(F.hash(F.col("row_id") + 23)) % 100).cast("int") + 1)
        .withColumn("cs_wholesale_cost",     (F.abs(F.hash(F.col("row_id") + 24)) % 15000 / 100.0).cast("decimal(7,2)"))
        .withColumn("cs_list_price",         (F.abs(F.hash(F.col("row_id") + 25)) % 30000 / 100.0).cast("decimal(7,2)"))
        .withColumn("cs_sales_price",        (F.abs(F.hash(F.col("row_id") + 26)) % 25000 / 100.0).cast("decimal(7,2)"))
        .withColumn("cs_ext_sales_price",    (F.col("cs_quantity") * F.col("cs_sales_price")).cast("decimal(7,2)"))
        .withColumn("cs_net_paid",           F.col("cs_ext_sales_price"))
        .withColumn("cs_net_profit",         (F.col("cs_ext_sales_price") - F.col("cs_quantity") * F.col("cs_wholesale_cost")).cast("decimal(7,2)"))
        .withColumn("cs_warehouse_sk",       (F.abs(F.hash(F.col("row_id") + 27)) % max(1, row_count("warehouse"))).cast("int") + 1)
        .withColumn("cs_ship_mode_sk",       (F.abs(F.hash(F.col("row_id") + 28)) % max(1, row_count("ship_mode"))).cast("int") + 1)
        .drop("row_id")
    )
    return df


def _gen_web_sales_spark(spark: SparkSession) -> DataFrame:
    n = row_count("web_sales")
    n_items = row_count("item")
    n_customers = row_count("customer")
    n_dates = row_count("date_dim")

    df = spark.range(1, n + 1).toDF("row_id")
    df = (
        df
        .withColumn("ws_sold_date_sk",       (F.abs(F.hash(F.col("row_id") + 40)) % n_dates).cast("int") + 1)
        .withColumn("ws_item_sk",            (F.abs(F.hash(F.col("row_id") + 41)) % n_items).cast("int") + 1)
        .withColumn("ws_bill_customer_sk",   (F.abs(F.hash(F.col("row_id") + 42)) % n_customers).cast("int") + 1)
        .withColumn("ws_order_number",       F.col("row_id").cast("int"))
        .withColumn("ws_quantity",           (F.abs(F.hash(F.col("row_id") + 43)) % 100).cast("int") + 1)
        .withColumn("ws_wholesale_cost",     (F.abs(F.hash(F.col("row_id") + 44)) % 15000 / 100.0).cast("decimal(7,2)"))
        .withColumn("ws_list_price",         (F.abs(F.hash(F.col("row_id") + 45)) % 30000 / 100.0).cast("decimal(7,2)"))
        .withColumn("ws_sales_price",        (F.abs(F.hash(F.col("row_id") + 46)) % 25000 / 100.0).cast("decimal(7,2)"))
        .withColumn("ws_ext_sales_price",    (F.col("ws_quantity") * F.col("ws_sales_price")).cast("decimal(7,2)"))
        .withColumn("ws_net_paid",           F.col("ws_ext_sales_price"))
        .withColumn("ws_net_profit",         (F.col("ws_ext_sales_price") - F.col("ws_quantity") * F.col("ws_wholesale_cost")).cast("decimal(7,2)"))
        .withColumn("ws_web_site_sk",        (F.abs(F.hash(F.col("row_id") + 47)) % max(1, row_count("web_site"))).cast("int") + 1)
        .withColumn("ws_ship_mode_sk",       (F.abs(F.hash(F.col("row_id") + 48)) % max(1, row_count("ship_mode"))).cast("int") + 1)
        .drop("row_id")
    )
    return df


def _gen_store_returns_spark(spark: SparkSession) -> DataFrame:
    n = row_count("store_returns")
    n_items = row_count("item")
    n_customers = row_count("customer")
    n_stores = row_count("store")
    n_dates = row_count("date_dim")

    df = spark.range(1, n + 1).toDF("row_id")
    df = (
        df
        .withColumn("sr_returned_date_sk", (F.abs(F.hash(F.col("row_id") + 60)) % n_dates).cast("int") + 1)
        .withColumn("sr_item_sk",          (F.abs(F.hash(F.col("row_id") + 61)) % n_items).cast("int") + 1)
        .withColumn("sr_customer_sk",      (F.abs(F.hash(F.col("row_id") + 62)) % n_customers).cast("int") + 1)
        .withColumn("sr_store_sk",         (F.abs(F.hash(F.col("row_id") + 63)) % n_stores).cast("int") + 1)
        .withColumn("sr_ticket_number",    F.col("row_id").cast("int"))
        .withColumn("sr_return_quantity",  (F.abs(F.hash(F.col("row_id") + 64)) % 20).cast("int") + 1)
        .withColumn("sr_return_amt",       (F.abs(F.hash(F.col("row_id") + 65)) % 20000 / 100.0).cast("decimal(7,2)"))
        .withColumn("sr_net_loss",         (F.abs(F.hash(F.col("row_id") + 66)) % 10000 / 100.0).cast("decimal(7,2)"))
        .withColumn("sr_reason_sk",        (F.abs(F.hash(F.col("row_id") + 67)) % max(1, row_count("reason"))).cast("int") + 1)
        .drop("row_id")
    )
    return df


# ─── Main entry point ─────────────────────────────────────────────────

def generate(spark: SparkSession | None = None) -> None:
    """Generate all TPC-DS tables as Parquet under DATA_PATH."""
    own_spark = spark is None
    if own_spark:
        builder = SparkSession.builder.appName(f"{SPARK_APP_NAME} — datagen")
        for k, v in SPARK_CPU_CONF.items():
            builder = builder.config(k, v)
        spark = builder.getOrCreate()

    out = str(DATA_PATH)
    print(f"\n{'='*60}")
    print(f"  TPC-DS Data Generator  —  SF{SCALE_FACTOR}")
    print(f"  Output: {out}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ── Dimension tables (small, generated in-driver) ──────────────
    dims: dict[str, list[dict]] = {
        "date_dim":                 _gen_date_dim(),
        "item":                     _gen_item(),
        "store":                    _gen_store(),
        "customer":                 _gen_customer(),
        "customer_address":         _gen_customer_address(),
        "promotion":                _gen_promotion(),
        "warehouse":                _gen_warehouse(),
        "ship_mode":                _gen_ship_mode(),
        "household_demographics":   _gen_household_demographics(),
        "time_dim":                 _gen_time_dim(),
        "reason":                   _gen_reason(),
        "income_band":              _gen_income_band(),
    }

    for name, rows in dims.items():
        df = spark.createDataFrame(rows)
        path = f"{out}/{name}"
        df.write.mode("overwrite").parquet(path)
        print(f"  ✓ {name:30s}  {len(rows):>12,} rows  →  {path}")

    # ── Fact tables (large, generated via Spark) ───────────────────
    facts: dict[str, DataFrame] = {
        "store_sales":    _gen_store_sales_spark(spark),
        "catalog_sales":  _gen_catalog_sales_spark(spark),
        "web_sales":      _gen_web_sales_spark(spark),
        "store_returns":  _gen_store_returns_spark(spark),
    }

    for name, df in facts.items():
        path = f"{out}/{name}"
        df.write.mode("overwrite").parquet(path)
        count = spark.read.parquet(path).count()
        print(f"  ✓ {name:30s}  {count:>12,} rows  →  {path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s\n")

    if own_spark:
        spark.stop()


if __name__ == "__main__":
    generate()
