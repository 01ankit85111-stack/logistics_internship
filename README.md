# Strategic Planning and Data Exploration in Logistics

**Week 1 Task — Internship Strategic Planning Report**
Author: Ankit ([01ankit85111-stack](https://github.com/01ankit85111-stack))
Track: AI & ML / Data Science — Logistics Analytics

## Overview

This repository contains the Week 1 strategic planning deliverable for a
logistics analytics project. The goal is to demonstrate how data science
techniques — regression, clustering, and optimization — can be applied to a
realistic e-commerce warehousing and last-mile delivery scenario to improve
supply chain efficiency.

The scenario covers a mid-sized e-commerce fulfillment network with three
regional warehouses that faces three recurring problems:

- **Inventory imbalance** across warehouses
- **Inefficient, manually planned delivery routes**
- **Unpredictable demand spikes**

Five KPIs were defined to measure improvement: On-Time Delivery Rate, Order
Fulfillment Cycle Time, Inventory Turnover Ratio, Vehicle/Route Utilization,
and Warehouse Picking Accuracy.

## Repository Contents

| File | Description |
|---|---|
| `Week1_Logistics_Strategic_Planning_Report.docx` | Full strategic planning report — scenario, KPIs, research, roadmap, code illustrations, and conclusion |
| `logistics_pipeline.py` | End-to-end, tested Python pipeline implementing the report's roadmap (data cleaning → EDA → clustering → forecasting → optimization) |
| `orders.csv` | Sample order-level dataset (1,200+ synthetic orders) |
| `vehicle_routes.csv` | Sample route/vehicle dataset (15 routes) used to join with orders on `route_id` |
| `README.md` | This file |

## Dataset Schema

**orders.csv**
`order_id, route_id, sku_id, region, order_date, delivery_date, promised_hrs, qty_ordered, unit_price, order_value, promo_flag`

**vehicle_routes.csv**
`route_id, vehicle_id, warehouse, region, capacity, cost_per_trip, avg_distance_km`

`route_id` is the join key between the two files. The datasets include
intentional data-quality issues (a few missing `delivery_date` values and
duplicate `order_id` rows) to exercise the cleaning steps in the pipeline.

## Pipeline Steps (`logistics_pipeline.py`)

1. **Data Collection & Cleaning** — loads both CSVs, validates the join key,
   merges without column collisions, parses dates, drops invalid/duplicate
   records, computes delivery lead time and on-time flag.
2. **Exploratory Data Analysis** — lead-time distribution and on-time
   delivery rate by region (saved as PNG charts).
3. **Clustering** — K-Means groups SKUs by demand pattern for warehouse
   slotting decisions.
4. **Demand Forecasting** — aggregates orders into a daily demand series,
   engineers calendar/lag features, and trains a Random Forest regressor.
5. **Route Optimization** — PuLP linear program selects the lowest-cost
   combination of real routes (from `vehicle_routes.csv`) that meets the
   day's demand.

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/01ankit85111-stack/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn pulp

# 3. Run the pipeline (orders.csv and vehicle_routes.csv must be in the same folder)
python logistics_pipeline.py
```

Expected output: console logs for each pipeline stage, plus two chart files
(`eda_lead_time_distribution.png`, `eda_otd_by_region.png`) written to the
working directory.

## Results Summary (on the included sample data)

| Metric | Value |
|---|---|
| Overall On-Time Delivery Rate | ~44.6% |
| Demand Forecast MAE | ~17 units/day (avg demand ~51/day) |
| SKU Clusters Found | 4 |
| Optimized Route Selection | Lowest-cost route meeting daily demand, chosen via PuLP |

These numbers are from synthetic sample data and are meant to validate that
the pipeline runs correctly end-to-end, not as real business conclusions.

## Next Steps

This is the Week 1 **planning** deliverable. Subsequent weeks will apply
this same pipeline to a larger/real dataset, tune the forecasting and
clustering models, expand the optimization to a full vehicle routing
problem (VRP), and build a KPI dashboard for stakeholders.