import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

# Load the dataset
df = pd.read_csv("ci_data.csv")

# Drop missing durations
df = df.dropna(subset=["duration_seconds"])

# Convert job_name to categories and save original names
df["job_name"] = df["job_name"].astype("category")
job_categories = df["job_name"].cat.categories.tolist()  # Save original names
df["job_name"] = df["job_name"].cat.codes  # Encode as numbers

# Binary risk target
df["status_bin"] = df["status"].apply(lambda x: 1 if x != "success" else 0)

# Split features and targets
X = df[["job_name"]]
y_risk = df["status_bin"]
y_time = df["duration_seconds"]

X_train, X_test, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
_, _, y_time_train, y_time_test = train_test_split(X, y_time, test_size=0.2, random_state=42)

# === Train Models ===
risk_model = RandomForestClassifier()
risk_model.fit(X_train, y_risk_train)
risk_acc = accuracy_score(y_risk_test, risk_model.predict(X_test))
print(f"Risk model accuracy: {risk_acc:.2f}")

time_model = RandomForestRegressor()
time_model.fit(X_train, y_time_train)
time_mae = mean_absolute_error(y_time_test, time_model.predict(X_test))
print(f"Duration model MAE: {time_mae:.2f} seconds")

# Save all artifacts
joblib.dump(risk_model, "risk_model.pkl")
joblib.dump(time_model, "duration_model.pkl")
joblib.dump(job_categories, "job_categories.pkl")  # Fixed line

print("Models saved: risk_model.pkl, duration_model.pkl, job_categories.pkl")
