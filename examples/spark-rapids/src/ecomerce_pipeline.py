"""
E-Commerce Customer Analytics Pipeline — Native Spark (CPU)

Real-world parallel: Daily/weekly analytics pipeline that a retail
data team would run to understand customer behavior, segment customers,
and generate business insights.

Stages:
  1. Data Loading         — Read parquet files
  2. Revenue Analytics    — Joins + aggregations (like a BI dashboard query)
  3. Customer 360         — Complex multi-table join (like a CDP build)
  4. RFM Segmentation     — Window functions + scoring (marketing analytics)
  5. Cohort Analysis      — Time-based grouping (product analytics)
  6. Funnel Analysis      — Clickstream sessionization (web analytics)
  7. Customer Clustering  — ML segmentation (data science)
"""

import time
import json
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
try:
    from .config import DATA_DIR, RESULTS_DIR, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY
except ImportError:
    from config import DATA_DIR, RESULTS_DIR, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY


class EcommercePipeline:
    """
    End-to-end e-commerce analytics pipeline.
    Each stage represents a real business analytics task.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.timings = {}
        self.results = {}

    def _time_stage(self, stage_name: str):
        """Decorator-like context manager for timing stages."""
        class Timer:
            def __init__(self, name, timings):
                self.name = name
                self.timings = timings
            def __enter__(self):
                self.start = time.time()
                print(f"\n{'─'*50}")
                print(f"  Stage: {self.name}")
                print(f"{'─'*50}")
                return self
            def __exit__(self, *args):
                elapsed = time.time() - self.start
                self.timings[self.name] = elapsed
                print(f"  ✓ Completed in {elapsed:.2f}s")
        return Timer(stage_name, self.timings)

    # ──────────────────────────────────────────────
    # Stage 1: Data Loading
    # ──────────────────────────────────────────────
    def load_data(self):
        """
        Load all tables from parquet.
        Real-world: Reading from data lake (S3, HDFS, ADLS).
        """
        with self._time_stage("1_data_loading"):
            self.customers = self.spark.read.parquet(f"{DATA_DIR}/customers.parquet")
            self.products = self.spark.read.parquet(f"{DATA_DIR}/products.parquet")
            self.orders = self.spark.read.parquet(f"{DATA_DIR}/orders.parquet")
            self.order_items = self.spark.read.parquet(f"{DATA_DIR}/order_items.parquet")
            self.clickstream = self.spark.read.parquet(f"{DATA_DIR}/clickstream.parquet")

            # Force materialization (Spark is lazy)
            counts = {
                "customers": self.customers.count(),
                "products": self.products.count(),
                "orders": self.orders.count(),
                "order_items": self.order_items.count(),
                "clickstream": self.clickstream.count(),
            }
            self.results["row_counts"] = counts
            for table, count in counts.items():
                print(f"  {table}: {count:,} rows")

    # ──────────────────────────────────────────────
    # Stage 2: Revenue Analytics
    # Real-world: Daily revenue dashboard, category performance
    # ──────────────────────────────────────────────
    def revenue_analytics(self):
        """
        Multi-table join + aggregation.
        Business question: What is the revenue by category, region, and time?
        SQL equivalent of a complex BI dashboard query.
        """
        with self._time_stage("2_revenue_analytics"):
            # Join orders → order_items → products → customers
            revenue = (
                self.orders
                .filter(F.col("status") == "completed")
                .join(self.order_items, "order_id")
                .join(self.products, "product_id")
                .join(self.customers, "customer_id")
            )

            # Revenue by category and region (monthly)
            monthly_revenue = (
                revenue
                .withColumn("year_month", F.date_format("order_date", "yyyy-MM"))
                .groupBy("year_month", "category", "region")
                .agg(
                    F.sum("total_price").alias("total_revenue"),
                    F.count("order_id").alias("num_orders"),
                    F.avg("total_price").alias("avg_order_value"),
                    F.countDistinct("customer_id").alias("unique_customers"),
                )
                .orderBy("year_month", F.desc("total_revenue"))
            )

            # Top 10 product categories by revenue
            top_categories = (
                revenue
                .groupBy("category", "subcategory")
                .agg(
                    F.sum("total_price").alias("total_revenue"),
                    F.sum("quantity").alias("total_units"),
                    F.avg("rating").alias("avg_rating"),
                )
                .orderBy(F.desc("total_revenue"))
                .limit(10)
            )

            result_count = monthly_revenue.count()
            top_cats = top_categories.collect()
            self.results["monthly_revenue_rows"] = result_count
            self.results["top_category"] = top_cats[0]["category"] if top_cats else None
            print(f"  Monthly revenue rows: {result_count:,}")
            print(f"  Top category: {self.results['top_category']}")

    # ──────────────────────────────────────────────
    # Stage 3: Customer 360 View
    # Real-world: Building a Customer Data Platform (CDP)
    # ──────────────────────────────────────────────
    def customer_360(self):
        """
        Build a unified customer profile by joining all data sources.
        Business question: What does each customer's complete profile look like?
        This is the foundation of personalization and targeting.
        """
        with self._time_stage("3_customer_360"):
            # Order-level metrics per customer
            customer_orders = (
                self.orders
                .filter(F.col("status") == "completed")
                .join(self.order_items, "order_id")
                .groupBy("customer_id")
                .agg(
                    F.count("order_id").alias("total_orders"),
                    F.sum("total_price").alias("lifetime_revenue"),
                    F.avg("total_price").alias("avg_item_value"),
                    F.min("order_date").alias("first_order_date"),
                    F.max("order_date").alias("last_order_date"),
                    F.countDistinct("product_id").alias("unique_products_bought"),
                    F.sum("quantity").alias("total_items_bought"),
                    F.avg("discount_pct").alias("avg_discount_used"),
                )
            )

            # Clickstream metrics per customer
            customer_clicks = (
                self.clickstream
                .groupBy("customer_id")
                .agg(
                    F.count("event_id").alias("total_events"),
                    F.sum(F.when(F.col("action") == "view", 1).otherwise(0)).alias("total_views"),
                    F.sum(F.when(F.col("action") == "add_to_cart", 1).otherwise(0)).alias("total_add_to_cart"),
                    F.avg("session_duration_sec").alias("avg_session_duration"),
                    F.countDistinct("product_id").alias("unique_products_viewed"),
                    F.countDistinct("device").alias("num_devices_used"),
                )
            )

            # Join everything into Customer 360
            self.customer_360_df = (
                self.customers
                .join(customer_orders, "customer_id", "left")
                .join(customer_clicks, "customer_id", "left")
                .fillna(0)
                .withColumn("conversion_rate",
                    F.when(F.col("total_views") > 0,
                           F.col("total_orders") / F.col("total_views"))
                    .otherwise(0)
                )
                .withColumn("cart_abandonment_rate",
                    F.when(F.col("total_add_to_cart") > 0,
                           1 - (F.col("total_orders") / F.col("total_add_to_cart")))
                    .otherwise(0)
                )
            )

            count = self.customer_360_df.count()
            self.results["customer_360_rows"] = count
            print(f"  Customer 360 profiles: {count:,}")

    # ──────────────────────────────────────────────
    # Stage 4: RFM Segmentation
    # Real-world: Marketing team's customer segmentation
    # ──────────────────────────────────────────────
    def rfm_segmentation(self):
        """
        Recency-Frequency-Monetary segmentation with window functions.
        Business question: Who are our best/worst customers?
        Used by marketing for targeted campaigns.
        """
        with self._time_stage("4_rfm_segmentation"):
            reference_date = "2025-12-31"

            rfm = (
                self.orders
                .filter(F.col("status") == "completed")
                .join(self.order_items, "order_id")
                .groupBy("customer_id")
                .agg(
                    F.datediff(F.lit(reference_date), F.max("order_date")).alias("recency_days"),
                    F.countDistinct("order_id").alias("frequency"),
                    F.sum("total_price").alias("monetary"),
                )
            )

            # Score using percentile-based ranking (window functions)
            for col_name in ["recency_days", "frequency", "monetary"]:
                quantiles = rfm.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
                if col_name == "recency_days":
                    # Lower recency = better (more recent)
                    rfm = rfm.withColumn(
                        f"{col_name}_score",
                        F.when(F.col(col_name) <= quantiles[0], 4)
                        .when(F.col(col_name) <= quantiles[1], 3)
                        .when(F.col(col_name) <= quantiles[2], 2)
                        .otherwise(1)
                    )
                else:
                    # Higher frequency/monetary = better
                    rfm = rfm.withColumn(
                        f"{col_name}_score",
                        F.when(F.col(col_name) >= quantiles[2], 4)
                        .when(F.col(col_name) >= quantiles[1], 3)
                        .when(F.col(col_name) >= quantiles[0], 2)
                        .otherwise(1)
                    )

            # Composite RFM segment
            rfm = rfm.withColumn(
                "rfm_segment",
                F.concat(
                    F.col("recency_days_score").cast("string"),
                    F.col("frequency_score").cast("string"),
                    F.col("monetary_score").cast("string"),
                )
            )

            # Label segments
            rfm = rfm.withColumn(
                "customer_segment",
                F.when(
                    (F.col("recency_days_score") >= 3) &
                    (F.col("frequency_score") >= 3) &
                    (F.col("monetary_score") >= 3), "Champions"
                )
                .when(
                    (F.col("recency_days_score") >= 3) &
                    (F.col("frequency_score") >= 2), "Loyal"
                )
                .when(
                    (F.col("recency_days_score") >= 3) &
                    (F.col("monetary_score") >= 2), "Potential Loyalist"
                )
                .when(F.col("recency_days_score") >= 3, "New Customer")
                .when(
                    (F.col("recency_days_score") <= 2) &
                    (F.col("frequency_score") >= 3), "At Risk"
                )
                .when(F.col("recency_days_score") == 1, "Lost")
                .otherwise("Needs Attention")
            )

            segment_counts = rfm.groupBy("customer_segment").count().collect()
            self.results["rfm_segments"] = {row["customer_segment"]: row["count"] for row in segment_counts}
            print(f"  Segments: {self.results['rfm_segments']}")

    # ──────────────────────────────────────────────
    # Stage 5: Cohort Analysis
    # Real-world: Product analytics, retention analysis
    # ──────────────────────────────────────────────
    def cohort_analysis(self):
        """
        Monthly cohort retention analysis.
        Business question: How well do we retain customers over time?
        """
        with self._time_stage("5_cohort_analysis"):
            completed_orders = self.orders.filter(F.col("status") == "completed")

            # Find each customer's first order month (cohort)
            first_order = (
                completed_orders
                .groupBy("customer_id")
                .agg(F.min(F.date_format("order_date", "yyyy-MM")).alias("cohort_month"))
            )

            # Join back and calculate months since cohort
            cohort_data = (
                completed_orders
                .join(first_order, "customer_id")
                .withColumn("order_month", F.date_format("order_date", "yyyy-MM"))
                .withColumn("months_since_first",
                    F.months_between(
                        F.to_date("order_month", "yyyy-MM"),
                        F.to_date("cohort_month", "yyyy-MM")
                    ).cast("int")
                )
            )

            # Cohort retention table
            cohort_retention = (
                cohort_data
                .groupBy("cohort_month", "months_since_first")
                .agg(F.countDistinct("customer_id").alias("active_customers"))
                .orderBy("cohort_month", "months_since_first")
            )

            count = cohort_retention.count()
            self.results["cohort_retention_rows"] = count
            print(f"  Cohort retention data points: {count:,}")

    # ──────────────────────────────────────────────
    # Stage 6: Funnel Analysis
    # Real-world: Web analytics, conversion optimization
    # ──────────────────────────────────────────────
    def funnel_analysis(self):
        """
        Clickstream funnel: view → click → add_to_cart → purchase.
        Business question: Where do customers drop off?
        Heavy aggregation on the largest table (clickstream).
        """
        with self._time_stage("6_funnel_analysis"):
            # Monthly funnel by device
            funnel = (
                self.clickstream
                .withColumn("year_month", F.date_format("event_date", "yyyy-MM"))
                .groupBy("year_month", "device")
                .agg(
                    F.count("event_id").alias("total_events"),
                    F.sum(F.when(F.col("action") == "view", 1).otherwise(0)).alias("views"),
                    F.sum(F.when(F.col("action") == "click", 1).otherwise(0)).alias("clicks"),
                    F.sum(F.when(F.col("action") == "add_to_cart", 1).otherwise(0)).alias("add_to_carts"),
                    F.sum(F.when(F.col("action") == "search", 1).otherwise(0)).alias("searches"),
                    F.avg("session_duration_sec").alias("avg_session_sec"),
                    F.countDistinct("customer_id").alias("unique_visitors"),
                )
                .withColumn("click_through_rate", F.col("clicks") / F.col("views"))
                .withColumn("cart_rate", F.col("add_to_carts") / F.col("clicks"))
                .orderBy("year_month", "device")
            )

            count = funnel.count()
            self.results["funnel_rows"] = count
            print(f"  Funnel data points: {count:,}")

    # ──────────────────────────────────────────────
    # Stage 7: Customer Clustering (ML)
    # Real-world: Data science team's segmentation model
    # ──────────────────────────────────────────────
    def customer_clustering(self):
        """
        K-Means clustering on Customer 360 features.
        Business question: What natural customer segments exist?
        This is the GPU-heavy stage — matrix operations, iterative algorithm.
        """
        with self._time_stage("7_customer_clustering"):
            # Select numeric features for clustering
            feature_cols = [
                "total_orders", "lifetime_revenue", "avg_item_value",
                "unique_products_bought", "total_items_bought",
                "total_events", "total_views", "total_add_to_cart",
                "avg_session_duration", "conversion_rate",
                "credit_score", "annual_income", "account_age_days",
            ]

            # Prepare features
            ml_data = self.customer_360_df.select(["customer_id"] + feature_cols).na.fill(0)

            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
            scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
            kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=5, seed=42, maxIter=20)

            pipeline = Pipeline(stages=[assembler, scaler, kmeans])
            model = pipeline.fit(ml_data)

            # Get predictions
            predictions = model.transform(ml_data)
            cluster_counts = predictions.groupBy("cluster").count().collect()
            self.results["clusters"] = {row["cluster"]: row["count"] for row in cluster_counts}

            # Cluster profiles
            cluster_profiles = (
                predictions
                .groupBy("cluster")
                .agg(*[F.avg(c).alias(f"avg_{c}") for c in feature_cols])
            )
            cluster_profiles.show(truncate=False)
            print(f"  Clusters: {self.results['clusters']}")

