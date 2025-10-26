# 1. Import Libraries
import pandas as pd       # For handling CSV data
import numpy as np        # For math operations like arrays
# 2. Load the dataset
df = pd.read_csv("advertising.csv")

# Quick look at first 5 rows
print("First 5 rows of dataset:")
print(df.head())
print("\nCheck for missing values:")
print(df.isnull().sum())

# 3. Define a function for Linear Regression from scratch
def linear_regression(X, y, learning_rate=0.00001, iterations=2000):
    """
    X: feature(s) as numpy array (1D or 2D)
    y: target variable as numpy array
    Returns: learned coefficients (theta) and RMSE
    """
    # If X is a single feature, reshape to 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Add a column of 1s for bias (intercept)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize weights to zeros (theta0 = bias, theta1..n = slopes)
    theta = np.zeros((X_b.shape[1], 1))
    
    # Convert y to column vector
    y = y.reshape(-1, 1)
    m = len(y)  # number of data points
    
    # Gradient Descent loop
    for i in range(iterations):
        predictions = X_b.dot(theta)          # Predicted values
        error = predictions - y               # Difference from actual
        gradients = (2/m) * X_b.T.dot(error) # Compute gradients
        theta = theta - learning_rate * gradients  # Update weights
    
    # Final predictions
    final_pred = X_b.dot(theta)
    rmse = np.sqrt(np.mean((y - final_pred)**2))  # Compute RMSE
    return theta, rmse

# 4. Define function to print equation nicely
def print_equation(theta, feature_names):
    eq = f"Sales = {theta[0][0]:.4f}"
    for i, name in enumerate(feature_names):
        eq += f" + {theta[i+1][0]:.4f}*{name}"
    return eq

# 5. Solve all 6 Questions
y = df['Sales'].values  # Target variable (same for all)

# Q1: Sales vs Radio
X1 = df['Radio'].values
theta1, rmse1 = linear_regression(X1, y)
print("\nQ1: Sales vs Radio")
print(print_equation(theta1, ['Radio']))
print(f"RMSE: {rmse1:.4f}")

# Q2: Sales vs TV
X2 = df['TV'].values
theta2, rmse2 = linear_regression(X2, y, learning_rate=0.00001)
print("\nQ2: Sales vs TV")
print(print_equation(theta2, ['TV']))
print(f"RMSE: {rmse2:.4f}")

# Q3: Sales vs Newspaper
X3 = df['Newspaper'].values
theta3, rmse3 = linear_regression(X3, y, learning_rate=0.00001)
print("\nQ3: Sales vs Newspaper")
print(print_equation(theta3, ['Newspaper']))
print(f"RMSE: {rmse3:.4f}")

# Q4: Sales vs Radio + TV
X4 = df[['Radio','TV']].values
theta4, rmse4 = linear_regression(X4, y, learning_rate=0.0000005, iterations=5000)
print("\nQ4: Sales vs Radio + TV")
print(print_equation(theta4, ['Radio','TV']))
print(f"RMSE: {rmse4:.4f}")

# Q5: Sales vs Newspaper + TV
X5 = df[['Newspaper','TV']].values
theta5, rmse5 = linear_regression(X5, y, learning_rate=0.0000005, iterations=5000)
print("\nQ5: Sales vs Newspaper + TV")
print(print_equation(theta5, ['Newspaper','TV']))
print(f"RMSE: {rmse5:.4f}")

# Q6: Sales vs Newspaper + Radio
X6 = df[['Newspaper','Radio']].values
theta6, rmse6 = linear_regression(X6, y, learning_rate=0.0000005, iterations=5000)
print("\nQ6: Sales vs Newspaper + Radio")
print(print_equation(theta6, ['Newspaper','Radio']))
print(f"RMSE: {rmse6:.4f}")