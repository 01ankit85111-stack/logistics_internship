"""
Logistics Analytics Pipeline
Week 1 Strategic Planning Project

This script implements the full roadmap described in the strategic planning
report (Sections 5.1 - 5.5) against the actual orders.csv / vehicle_routes.csv
schema. It fixes three issues found in the report's illustrative snippets
during review:

  1. Both orders.csv and vehicle_routes.csv contain a 'region' column, so a
     plain merge on 'route_id' silently creates 'region_x' / 'region_y'.
     Fixed by dropping the duplicate route-level region before merging.
  2. The forecasting snippet referenced 'day_of_week', 'month', 'promo_flag'
     and 'lag_7day_avg_demand' as if they already existed per order. These
     are DAILY features, so they must be built by aggregating orders to a
     daily demand series first.
  3. The optimization snippet used a hardcoded example route dict instead of
     the real vehicle_routes.csv capacity/cost columns.

Run:  python logistics_pipeline.py
Requires: pandas, numpy, matplotlib, seaborn, scikit-learn, pulp
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 120)

# =========================================================================
# 5.1  DATA COLLECTION & CLEANING
# =========================================================================

def load_and_clean(orders_path="orders.csv", routes_path="vehicle_routes.csv"):
    orders = pd.read_csv(orders_path)
    routes = pd.read_csv(routes_path)

    # --- KEY CHECK: verify the join key is valid before merging ---------
    missing_routes = set(orders["route_id"]) - set(routes["route_id"])
    if missing_routes:
        print(f"WARNING: {len(missing_routes)} orders reference a route_id "
              f"not present in vehicle_routes.csv: {missing_routes}")
    else:
        print(f"Key check OK: all {orders['route_id'].nunique()} route_ids "
              f"in orders.csv exist in vehicle_routes.csv.")

    # --- FIX: both files have a 'region' column -> drop the duplicate ---
    routes_for_join = routes.drop(columns=["region"])
    df = orders.merge(routes_for_join, on="route_id", how="left")

    # --- parse dates, coercing anything unparseable to NaT --------------
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["delivery_date", "order_date"])
    print(f"Dropped {before - len(df)} rows with missing order/delivery dates "
          f"({before} -> {len(df)}).")

    # --- lead time in hours + sanity filter ------------------------------
    df["lead_time_hrs"] = (df["delivery_date"] - df["order_date"]).dt.total_seconds() / 3600
    before = len(df)
    df = df[df["lead_time_hrs"] > 0]
    print(f"Removed {before - len(df)} rows with non-positive lead time.")

    # --- duplicate order_id check ----------------------------------------
    dup_count = df["order_id"].duplicated().sum()
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"])
    print(f"Removed {before - len(df)} duplicate order_id rows "
          f"(found {dup_count} duplicates).")

    df["on_time"] = df["lead_time_hrs"] <= df["promised_hrs"]

    return df, routes


# =========================================================================
# 5.2  EXPLORATORY DATA ANALYSIS
# =========================================================================

def run_eda(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["lead_time_hrs"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Delivery Lead Time (hrs)")
    fig.tight_layout()
    fig.savefig("eda_lead_time_distribution.png", dpi=150)
    plt.close(fig)
    plt.show()

    otd_by_region = df.groupby("region")["on_time"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    otd_by_region.plot(kind="barh", ax=ax)
    ax.set_title("On-Time Delivery Rate by Region")
    ax.set_xlabel("On-time rate")
    fig.tight_layout()
    fig.savefig("eda_otd_by_region.png", dpi=150)
    plt.close(fig)
    plt.show()

    print("\nOverall on-time delivery rate: {:.1%}".format(df["on_time"].mean()))
    print("On-time rate by region:\n", otd_by_region)
    return otd_by_region


# =========================================================================
# 5.3  CLUSTERING FOR WAREHOUSE / SKU SLOTTING
# =========================================================================

def cluster_skus(df, n_clusters=4):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = df.groupby("sku_id").agg(
        avg_daily_demand=("qty_ordered", "mean"),
        demand_std=("qty_ordered", "std"),
        avg_order_value=("order_value", "mean"),
    ).fillna(0)

    X_scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features["cluster"] = kmeans.fit_predict(X_scaled)

    print("\nSKU cluster sizes:\n", features["cluster"].value_counts().sort_index())
    return features


# =========================================================================
# 5.4  DEMAND FORECASTING (daily aggregation -> regression)
# =========================================================================

def build_daily_demand(df):
    """orders.csv is at order-grain; forecasting needs a daily time series."""
    daily = (
        df.set_index("order_date")
        .resample("D")
        .agg(daily_demand=("qty_ordered", "sum"), promo_flag=("promo_flag", "max"))
        .fillna(0)
    )
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["lag_7day_avg_demand"] = (
        daily["daily_demand"].rolling(window=7, min_periods=1).mean().shift(1).fillna(0)
    )
    return daily.reset_index()


def forecast_demand(daily):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    feature_cols = ["day_of_week", "month", "promo_flag", "lag_7day_avg_demand"]
    X = daily[feature_cols]
    y = daily["daily_demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"\nDemand forecast MAE: {mae:.2f} units/day "
          f"(avg daily demand = {y.mean():.1f})")
    return model, mae


# =========================================================================
# 5.5  ROUTE / VEHICLE OPTIMIZATION (uses real vehicle_routes.csv data)
# =========================================================================

def optimize_routes(routes, daily_demand_target):
    import pulp

    prob = pulp.LpProblem("Route_Allocation", pulp.LpMinimize)

    route_ids = routes["route_id"].tolist()
    cost = dict(zip(routes["route_id"], routes["cost_per_trip"]))
    capacity = dict(zip(routes["route_id"], routes["capacity"]))

    x = pulp.LpVariable.dicts("use", route_ids, cat="Binary")

    prob += pulp.lpSum(cost[r] * x[r] for r in route_ids)  # minimize total cost
    prob += pulp.lpSum(capacity[r] * x[r] for r in route_ids) >= daily_demand_target

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = [r for r in route_ids if x[r].value() == 1]
    total_cost = sum(cost[r] for r in selected)
    total_capacity = sum(capacity[r] for r in selected)

    print(f"\nDemand target for the day: {daily_demand_target} packages")
    print(f"Routes selected to dispatch: {selected}")
    print(f"Total capacity covered: {total_capacity} | Total cost: {total_cost}")
    return selected


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: Load & clean data")
    print("=" * 70)
    df, routes = load_and_clean()

    print("\n" + "=" * 70)
    print("STEP 2: Exploratory data analysis")
    print("=" * 70)
    run_eda(df)

    print("\n" + "=" * 70)
    print("STEP 3: SKU clustering")
    print("=" * 70)
    sku_clusters = cluster_skus(df)

    print("\n" + "=" * 70)
    print("STEP 4: Demand forecasting")
    print("=" * 70)
    daily = build_daily_demand(df)
    model, mae = forecast_demand(daily)

    print("\n" + "=" * 70)
    print("STEP 5: Route optimization")
    print("=" * 70)
    typical_daily_demand = int(daily["daily_demand"].mean())
    optimize_routes(routes, daily_demand_target=typical_daily_demand)

    print("\nPipeline completed successfully.")