import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# ------------------ LOAD DATA ------------------
df = pd.read_csv("data/processed/processed.csv")

# ------------------ FEATURES & TARGET ------------------
X = df.drop(["Weekly_Sales", "Date"], axis=1)
y = df["Weekly_Sales"]

# ------------------ TRAIN TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ MODEL ------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# ------------------ TRAIN ------------------
model.fit(X_train, y_train)

# ------------------ PREDICT ------------------
y_pred = model.predict(X_test)

# ------------------ EVALUATION ------------------
mse = mean_squared_error(y_test, y_pred)
print(f"Model Trained ✔ | MSE: {mse:.2f}")

# ------------------ SAVE MODEL ------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")
print("Model saved at models/rf_model.pkl")

# ------------------ SAVE ACTUAL vs PREDICTED ------------------
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results_df.to_csv("data/processed/predictions.csv", index=False)
print("Predictions saved at data/processed/predictions.csv")
