from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
print("Shape:", X.shape)
print("First 5 rows:\n", X[:5])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Prediction & accuracy
y_pred = model.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, y_pred))