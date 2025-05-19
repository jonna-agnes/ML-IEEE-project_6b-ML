# SmartBin ML Model

## Overview

This project implements a machine learning pipeline for optimizing waste collection using SmartBin IoT data. The goal is to predict bin fill levels and optimize collection routes to save energy and resources.

## Features

- **Data Preprocessing:** Handles missing values, extracts time-based features, and normalizes data.
- **Clustering:** Uses KMeans to group bins based on location and fill level.
- **Prediction Models:**
  - **Linear Regression:** Predicts bin fill levels using time and location features.
  - **Artificial Neural Network (ANN):** Deep learning model for more accurate fill level prediction.
- **Route Optimization:** Implements a greedy algorithm to optimize the waste collection route.
- **Visualization:** Plots the optimized route using Matplotlib.
- **Energy Savings Estimation:** Calculates the percentage of energy saved by route optimization.

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `pandas` and `numpy` for data manipulation
  - `scikit-learn` for machine learning models and preprocessing
  - `tensorflow.keras` for building the ANN
  - `matplotlib` for visualization

## Implementation Details

1. **Data Loading:** Reads IoT data from `smartbin_data.csv`.
2. **Preprocessing:** Fills missing numeric values, extracts hour and day of week from timestamps, and normalizes features.
3. **Clustering:** Groups bins using KMeans based on latitude, longitude, and fill level.
4. **Model Training:**
   - **Linear Regression:** Trained to predict fill levels.
   - **ANN:** Trained with two hidden layers for improved accuracy.
5. **Route Optimization:** Uses a greedy algorithm to determine the most efficient collection order.
6. **Visualization:** Displays the optimized route on a 2D plot.
7. **Energy Calculation:** Compares baseline and optimized routes to estimate energy savings.

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib
   ```
2. Place your `smartbin_data.csv` file in the project directory.
3. Run the script:
   ```bash
   python smartbin_model.py
   ```

## Output

- Prints model performance (MAE for Linear Regression and ANN).
- Displays the optimized collection route.
- Shows estimated energy savings.

---

**Author:** [jonna-agnes](https://github.com/jonna-agnes)
