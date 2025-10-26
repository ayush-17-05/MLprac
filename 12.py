import pandas as pd
import numpy as np

df = pd.read_csv("car_dataset.csv")
X_year = df['year_bought'].values
df['transmission_encoded']=df['transmission'].map({
    'Manual':0,
    'Automatic':1
})
X_transmission = df['transmission_encoded'].values.reshape(-1,1)
y = df['selling_price'].values.reshape(-1,1)
n = len(y)

mean_year = np.mean(X_year)
mean_transmission = np.mean(X_transmission)
mean_y = np.mean(y)

S_x1x1 = np.sum((X_year - mean_year)**2)
S_x2x2 = np.sum((X_transmission - mean_transmission)**2)
S_x1x2 = np.sum((X_year - mean_year)*(X_transmission - mean_transmission))
S_x1y = np.sum((X_year - mean_year)*(y - mean_y))
S_x2y = np.sum((X_transmission - mean_transmission)*(y - mean_y))

denominator = S_x1x1*S_x2x2 - S_x1x2**2
b1 = (S_x2x2*S_x1y - S_x1x2*S_x2y)/denominator
b2 = (S_x1x1*S_x2y - S_x1x2*S_x1y)/denominator
b0 = mean_y - b1*mean_year - b2*mean_transmission

y_pred = b0 + b1*X_year + b2*X_transmission
rmse = np.sqrt(np.mean((y - y_pred)**2))

print(b0, b1, b2, rmse)