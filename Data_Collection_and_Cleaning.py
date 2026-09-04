import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
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
