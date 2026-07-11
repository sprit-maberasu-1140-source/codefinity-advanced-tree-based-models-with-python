import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 1) Load dataset
df = pd.read_csv('https://codefinity-content-media-v2.s3.eu-west-1.amazonaws.com/courses/4e92d172-a423-47c0-a243-a38dd496de94/california_housing.csv')
X = df.drop(columns=['median_house_value']).values
y = df['median_house_value'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) Initialize model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
    verbosity=0
)

# 3) Fit the model
model.fit(X_train, y_train)

# 4) Predict on test set
y_pred = model.predict(X_test)

# 5) Compute MSE
mse_value = mean_squared_error(y_test, y_pred)

# Print results
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Model depth:", model.max_depth)
print("MSE:", mse_value)