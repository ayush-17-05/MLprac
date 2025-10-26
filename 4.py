import pandas as pd
import numpy as np

df = pd.read_csv("advertising.csv")
X_radio = df['Radio'].values
X_tv = df['TV'].values
y = df['Sales'].values
n = len(y)

mean_radio = np.mean(X_radio)
mean_tv = np.mean(X_tv)
mean_y = np.mean(y)

S_x1x1 = np.sum((X_radio - mean_radio)**2)
S_x2x2 = np.sum((X_tv - mean_tv)**2)
S_x1x2 = np.sum((X_radio - mean_radio)*(X_tv - mean_tv))
S_x1y = np.sum((X_radio - mean_radio)*(y - mean_y))
S_x2y = np.sum((X_tv - mean_tv)*(y - mean_y))

denominator = S_x1x1*S_x2x2 - S_x1x2**2
b1 = (S_x2x2*S_x1y - S_x1x2*S_x2y)/denominator
b2 = (S_x1x1*S_x2y - S_x1x2*S_x1y)/denominator
b0 = mean_y - b1*mean_radio - b2*mean_tv

y_pred = b0 + b1*X_radio + b2*X_tv
rmse = np.sqrt(np.mean((y - y_pred)**2))

print(b0, b1, b2, rmse)