import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pulp
 
# Load raw order and delivery data
orders = pd.read_csv('orders.csv')
routes = pd.read_csv('vehicle_routes.csv')
 
# Merge datasets on order/route key
df = orders.merge(routes, on='route_id', how='left')
 
# Clean missing and inconsistent values
df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df = df.dropna(subset=['delivery_date', 'order_date'])
df['lead_time_hrs'] = (df['delivery_date'] - df['order_date']).dt.total_seconds() / 3600
df = df[df['lead_time_hrs'] > 0]        # remove impossible values
df = df.drop_duplicates(subset=['order_id'])

 
# Distribution of delivery lead time
sns.histplot(df['lead_time_hrs'], bins=30, kde=True)
plt.title('Distribution of Delivery Lead Time (hrs)')
plt.show()
 
# On-time vs delayed orders by region
df['on_time'] = df['lead_time_hrs'] <= df['promised_hrs']
otd_by_region = df.groupby('region')['on_time'].mean().sort_values()
otd_by_region.plot(kind='barh', title='On-Time Delivery Rate by Region')
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
 
# Cluster warehouses/SKUs by demand pattern for smarter slotting
features = df.groupby('sku_id').agg(
    avg_daily_demand=('qty_ordered', 'mean'),
    demand_std=('qty_ordered', 'std'),
    avg_order_value=('order_value', 'mean')
).fillna(0)
 
X_scaled = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
features['cluster'] = kmeans.fit_predict(X_scaled)
# High-velocity SKUs (one cluster) can be slotted closer to packing stations

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
 
features_cols = ['day_of_week', 'month', 'promo_flag', 'lag_7day_avg_demand']
X = df[features_cols]
y = df['daily_demand']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
 
preds = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, preds))
# Forecasted demand feeds directly into inventory replenishment planning

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
 
features_cols = ['day_of_week', 'month', 'promo_flag', 'lag_7day_avg_demand']
X = df[features_cols]
y = df['daily_demand']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
 
preds = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, preds))
# Forecasted demand feeds directly into inventory replenishment planning
