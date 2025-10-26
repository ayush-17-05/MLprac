from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model - GINI
model = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_split=2)
model.fit(X_train, y_train)

# Predict & accuracy
y_pred = model.predict(X_test)
print("Accuracy (GINI):", accuracy_score(y_test, y_pred))