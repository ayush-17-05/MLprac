import pandas as pd
import numpy as np

df = pd.read_csv("advertising.csv")
X_news = df['Newspaper'].values
X_tv = df['TV'].values
y = df['Sales'].values
n = len(y)

mean_news = np.mean(X_news)
mean_tv = np.mean(X_tv)
mean_y = np.mean(y)

S_x1x1 = np.sum((X_news - mean_news)**2)
S_x2x2 = np.sum((X_tv - mean_tv)**2)
S_x1x2 = np.sum((X_news - mean_news)*(X_tv - mean_tv))
S_x1y = np.sum((X_news - mean_news)*(y - mean_y))
S_x2y = np.sum((X_tv - mean_tv)*(y - mean_y))

denominator = S_x1x1*S_x2x2 - S_x1x2**2
b1 = (S_x2x2*S_x1y - S_x1x2*S_x2y)/denominator
b2 = (S_x1x1*S_x2y - S_x1x2*S_x1y)/denominator
b0 = mean_y - b1*mean_news - b2*mean_tv

y_pred = b0 + b1*X_news + b2*X_tv
rmse = np.sqrt(np.mean((y - y_pred)**2))

print(b0, b1, b2, rmse)
