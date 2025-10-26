import pandas as pd
import numpy as np

df = pd.read_csv("advertising.csv")
X = df['TV'].values.reshape(-1, 1)
y = df['Sales'].values.reshape(-1, 1)

x_mean = np.mean(X)
y_mean = np.mean(y)

b1 = np.sum((X - x_mean)*(y - y_mean)) / np.sum((X - x_mean)**2)
b0 = y_mean - b1*x_mean

y_pred = b0 + b1*X
rmse = np.sqrt(np.mean((y - y_pred)**2))

print(b0, b1, rmse)
