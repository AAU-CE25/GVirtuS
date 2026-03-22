# Spark CPU vs GPU (RAPIDS) Performance Comparison

This document compares the execution times of the PySpark pipeline running on a traditional CPU setup versus an NVIDIA GPU setup using RAPIDS.

## Execution Timings (Seconds)

| Pipeline Phase | CPU Time (s) | GPU Time (s) | Speedup (CPU / GPU) |
| :--- | :--- | :--- | :--- |
| **1_data_loading** | 3.70 | 4.16 | 0.89x |
| **2_revenue_analytics** | 3.94 | 15.03 | 0.26x |
| **3_customer_360** | 0.20 | 0.24 | 0.83x |
| **4_rfm_segmentation** | 6.95 | 5.60 | 1.24x |
| **5_cohort_analysis** | 2.62 | 2.48 | 1.06x |
| **6_funnel_analysis** | 0.75 | 1.13 | 0.66x |
| **7_customer_clustering** | 17.95 | 11.79 | 1.52x |
| **Total Time** | **36.11** | **40.43** | **0.89x** |

## Summary of Results
- The **GPU** outperformed the **CPU** in computationally heavier tasks like **RFM Segmentation** (1.24x speedup) and **Customer Clustering** (1.52x speedup).
- The **CPU** was faster in **Data Loading**, **Revenue Analytics**, and **Funnel Analysis**. For smaller scale scale factors or operations involving primarily I/O, CPU overhead is lower. Revenue Analytics on GPU had a significant initialization or execution overhead.
- Overall, for this specific dataset size (sf1), the CPU execution was slightly faster in total elapsed time.

## Business Logic Differences
While the row counts remain identical between the two setups (e.g., 100,000 customers, 1,000,000 orders), slight non-deterministic differences are observed in data groupings due to internal hashing and clustering algorithms (like KMeans for `customer_clustering` or `rfm_segmentation`).
