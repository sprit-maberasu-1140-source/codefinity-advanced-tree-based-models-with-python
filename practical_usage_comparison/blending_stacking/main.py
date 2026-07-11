import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1) Generate dataset
X, y = make_classification(
    n_samples=6000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) Initialize models
model_cat = CatBoostClassifier(
    iterations=150,
    learning_rate=0.1,
    depth=6,
    verbose=False,
    random_state=42
)
model_xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model_lgb = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train all models
model_cat.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
model_lgb.fit(X_train, y_train)

# 3) Predict probabilities (positive class)
p_cat = model_cat.predict_proba(X_test)[:, 1]
p_xgb = model_xgb.predict_proba(X_test)[:, 1]
p_lgb = model_lgb.predict_proba(X_test)[:, 1]

# 4) Blend (average)
p_blend = (p_cat + p_xgb + p_lgb) / 3.0
y_pred = (p_blend > 0.5).astype(int)

# 5) Compute accuracy
accuracy_value = accuracy_score(y_test, y_pred)

print("Train/Test shapes:", X_train.shape, X_test.shape)
print("Models:", type(model_cat), type(model_xgb), type(model_lgb))
print("Blended accuracy:", accuracy_value)