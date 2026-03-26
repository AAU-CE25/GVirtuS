# Spark CPU vs GPU (RAPIDS) Performance Comparison

This document compares the execution times of the PySpark pipeline running on a traditional CPU setup versus an NVIDIA GPU setup using RAPIDS at different scale factors.

## Scale Factor 10 (sf10)
At this scale (~10 million orders, 1 million customers), the computational benefits of the GPU are highly apparent.

### Execution Timings (Seconds)

| Pipeline Phase | CPU Time (s) | GPU Time (s) | Speedup (CPU / GPU) |
| :--- | :--- | :--- | :--- |
| **1_data_loading** | 3.69 | 4.20 | 0.88x |
| **2_revenue_analytics** | 36.81 | 13.22 | 2.78x |
| **3_customer_360** | 0.17 | 0.25 | 0.68x |
| **4_rfm_segmentation** | 61.50 | 14.83 | 4.15x |
| **5_cohort_analysis** | 13.22 | 7.37 | 1.79x |
| **6_funnel_analysis** | 10.07 | 7.73 | 1.30x |
| **7_customer_clustering** | 306.23 | 45.03 | 6.80x |
| **Total Time** | **431.69** | **92.63** | **4.66x** |

### Summary of sf10 Results
- **Massive Overall Speedup:** The GPU completed the full analytics pipeline in **92.6 seconds**, while the CPU took **431.7 seconds** – an **overall speedup of 4.66x**.
- **Significant Clustering Gains:** The most compute-intensive part of the pipeline, **Customer Clustering (K-Means)**, received a massive **6.8x speedup**.
- **RFM and Revenue Benefits:** Heavy grouping, aggregations, and window functions significantly benefited the GPU. **RFM Segmentation** saw a 4.15x speedup and **Revenue Analytics** gained a 2.78x speedup.

---

## Scale Factor 1 (sf1)
Scale Factor 1 acts as a development and testing scale factor with ~1 million orders and 100,000 customers.

### Execution Timings (Seconds)

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

### Summary of sf1 Results
- The **GPU** outperformed the **CPU** in computationally heavier tasks like **RFM Segmentation** (1.24x) and **Customer Clustering** (1.52x).
- For smaller scale factors or operations involving primarily I/O, CPU overhead is lower. Revenue Analytics on GPU had a significant initialization or execution overhead here.
- Overall, for this small specific dataset size, the CPU execution was slightly faster in total elapsed time.

---

## Conclusion

- **Dataset Size Matters:** At `sf1`, the initialization overhead of spinning up the GPU processing eclipses the actual computational work, leading to the CPU winning slightly. However, at `sf10`, the massive payload unlocks the computational power of RAPIDS, leading to a huge **4.66x overall speedup**.
- **Business Logic Differences:** Over all setups, row counts remain identical between the two setups. The slight non-deterministic differences in data groupings are due to internal hashing algorithms and ML variations (like KMeans initialization) between Spark and CuML.