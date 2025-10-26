import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("car_dataset.csv")

# 2. Check for missing values
print("Missing values:\n", df.isnull().sum())

# 3. Extract feature and target
X = df['year_bought'].values.reshape(-1, 1)
y = df['selling_price'].values.reshape(-1, 1)

# 4. Compute means
x_mean = np.mean(X)
y_mean = np.mean(y)

# 5. Compute slope and intercept
b1 = np.sum((X - x_mean)*(y - y_mean)) / np.sum((X - x_mean)**2)
b0 = y_mean - b1*x_mean

# 6. Make predictions
y_pred = b0 + b1*X

# 7. Calculate RMSE
rmse = np.sqrt(np.mean((y - y_pred)**2))

# 8. Print results
print(f"Intercept (b0): {b0}")
print(f"Slope (b1): {b1}")
print(f"RMSE: {rmse}")