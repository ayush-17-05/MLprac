import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("car_dataset.csv")

# 3. Encode categorical feature
df['transmission_encoded'] = df['transmission'].map({'Manual':0, 'Automatic':1})

# 4. Extract feature and target
X = df['transmission_encoded'].values.reshape(-1, 1)
y = df['selling_price'].values.reshape(-1, 1)

# 5. Compute means
x_mean = np.mean(X)
y_mean = np.mean(y)

# 6. Compute slope and intercept
b1 = np.sum((X - x_mean)*(y - y_mean)) / np.sum((X - x_mean)**2)
b0 = y_mean - b1*x_mean

# 7. Make predictions
y_pred = b0 + b1*X

# 8. Calculate RMSE
rmse = np.sqrt(np.mean((y - y_pred)**2))

# 9. Print results
print(f"Intercept (b0): {b0}")
print(f"Slope (b1): {b1}")
print(f"RMSE: {rmse}")
