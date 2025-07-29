import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

# Load the dataset
df = pd.read_csv("ci_data.csv")

# Drop missing durations (these are likely failed or incomplete jobs)
df = df.dropna(subset=["duration_seconds"])

# Convert categorical features
df["job_name"] = df["job_name"].astype("category").cat.codes
df["status_bin"] = df["status"].apply(lambda x: 1 if x != "success" else 0)

# Features to use
features = ["job_name"]
target_risk = "status_bin"
target_time = "duration_seconds"

# Split dataset
X = df[features]
y_risk = df[target_risk]
y_time = df[target_time]

X_train, X_test, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
_, _, y_time_train, y_time_test = train_test_split(X, y_time, test_size=0.2, random_state=42)

# === Train Risk Classifier ===
risk_model = RandomForestClassifier()
risk_model.fit(X_train, y_risk_train)
risk_preds = risk_model.predict(X_test)
risk_acc = accuracy_score(y_risk_test, risk_preds)
print(f"✅ Risk model accuracy: {risk_acc:.2f}")

# === Train Time Predictor ===
time_model = RandomForestRegressor()
time_model.fit(X_train, y_time_train)
time_preds = time_model.predict(X_test)
time_mae = mean_absolute_error(y_time_test, time_preds)
print(f"⏱️ Duration model MAE: {time_mae:.2f} seconds")

# === Save models ===
joblib.dump(risk_model, "risk_model.pkl")
joblib.dump(time_model, "duration_model.pkl")
joblib.dump(df["job_name"].astype("category").cat.categories.tolist(), "job_categories.pkl")

print("✅ Models saved: risk_model.pkl, duration_model.pkl")
