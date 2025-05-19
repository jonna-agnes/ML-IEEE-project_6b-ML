# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the SmartBin IoT dataset
data = pd.read_csv('smartbin_data.csv')

# Data Preprocessing
# Fill missing values only in numeric columns
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

# Feature extraction: extract hour and day of week from timestamp
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['dayofweek'] = data['timestamp'].dt.dayofweek

# Normalize features for ANN
scaler = StandardScaler()
data[['fill_level', 'hour', 'dayofweek']] = scaler.fit_transform(data[['fill_level', 'hour', 'dayofweek']])

# KMeans Clustering: group bins based on location and fill level
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(data[['latitude', 'longitude', 'fill_level']])

# Define features and target
X = data[['hour', 'dayofweek', 'latitude', 'longitude']]
y = data['fill_level']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)
print(f'Linear Regression MAE: {lr_mae:.2f}')

# Artificial Neural Network (ANN)
ann_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_model.compile(optimizer='adam', loss='mean_squared_error')
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
ann_predictions = ann_model.predict(X_test)
ann_mae = mean_absolute_error(y_test, ann_predictions)
print(f'ANN MAE: {ann_mae:.2f}')

# Define distance calculation
def calculate_distance(bin1, bin2):
    return np.sqrt((bin1['latitude'] - bin2['latitude'])**2 + (bin1['longitude'] - bin2['longitude'])**2)

# Greedy route optimization
def greedy_route_optimization(data):
    start_bin = data.iloc[0]
    route = [start_bin]
    remaining_bins = data.iloc[1:].copy()
    
    while not remaining_bins.empty:
        last_bin = route[-1]
        distances = remaining_bins.apply(lambda row: calculate_distance(last_bin, row), axis=1)
        next_bin_idx = distances.idxmin()
        route.append(remaining_bins.loc[next_bin_idx])
        remaining_bins = remaining_bins.drop(next_bin_idx)
        
    return route

# Optimize route
optimized_route = greedy_route_optimization(data)

# Print route
print("\nOptimized Route (Bin Order):")
for bin in optimized_route:
    print(f'Bin ID: {bin["bin_id"]}, Location: ({bin["latitude"]}, {bin["longitude"]})')

# Visualization
optimized_route_df = pd.DataFrame(optimized_route)
plt.plot(optimized_route_df['latitude'], optimized_route_df['longitude'], marker='o', color='b')
plt.title('Optimized Waste Collection Route')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()

# Energy savings estimation
baseline_distance = sum(data.apply(lambda row: calculate_distance(row, data.iloc[0]), axis=1))
optimized_distance = sum(optimized_route_df.apply(lambda row: calculate_distance(row, optimized_route_df.iloc[0]), axis=1))
energy_saved = (baseline_distance - optimized_distance) / baseline_distance * 100
print(f'\nEnergy Saved: {energy_saved:.2f}%')

# Summary
print("\nExperiment Summary:")
print(f"Linear Regression MAE: {lr_mae:.2f}")
print(f"ANN MAE: {ann_mae:.2f}")
print(f"Energy Saved: {energy_saved:.2f}%")
