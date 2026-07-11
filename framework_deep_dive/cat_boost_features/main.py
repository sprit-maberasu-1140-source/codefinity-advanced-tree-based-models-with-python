# Importing the libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Generating the dataset
X, y = make_classification(
    n_samples=8000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initializing the CatBoost model
model = CatBoostClassifier(
    iterations=150,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=False
)

# Fitting the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Computing the accuracy
accuracy_value = accuracy_score(y_test, y_pred)

# Printing the results
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Accuracy:", accuracy_value)