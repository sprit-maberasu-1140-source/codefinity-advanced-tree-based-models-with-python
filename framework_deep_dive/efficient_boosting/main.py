import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# 1) Generate dataset
X, y = make_classification(
    n_samples=15000,
    n_features=40,
    n_informative=20,
    n_redundant=10,
    random_state=42
)

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Initialize LightGBM model
model = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 4) Fit model
model.fit(X_train, y_train)

# 5) Predict
y_pred = model.predict(X_test)

# 6) Compute accuracy
accuracy_value = accuracy_score(y_test, y_pred)

# Print results
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Model depth:", model.max_depth)
print("Accuracy:", accuracy_value)