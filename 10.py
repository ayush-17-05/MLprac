import pandas as pd
import numpy as np

df = pd.read_csv("car_dataset.csv")

# wrt owner
df['owner_encoded'] = df['owner'].map({'First':1, 'Second':2, 'Third':3, 'Fourth & Above':4})
X = df['owner_encoded'].values.reshape(-1, 1)
Y = df['selling_price'].values.reshape(-1, 1)

x_mean = np.mean(X)
y_mean = np.mean(Y)

b1 = np.sum((X - x_mean)*(Y - y_mean)) / np.sum((X - x_mean)**2)
b0 = y_mean - b1*x_mean
Y_pred = b0 + b1*X

rmse = np.sqrt(np.mean((Y - Y_pred)**2))
print(f"Intercept (b0): {b0}")
print(f"Slope (b1): {b1}")
print(f"RMSE: {rmse}")