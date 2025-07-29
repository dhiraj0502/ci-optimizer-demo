import os
import sys
import joblib
import json

# Load models
risk_model = joblib.load("risk_model.pkl")
time_model = joblib.load("duration_model.pkl")
job_categories = joblib.load("job_categories.pkl")

# Read input job name from argument
job_name = sys.argv[1]

# Convert job name to category code
try:
    job_code = job_categories.index(job_name)
except ValueError:
    print("❌ Unknown job name. Will run by default.")
    print("::set-output name=skip::false")
    exit(0)

X = [[job_code]]
pred_risk = risk_model.predict(X)[0]
pred_time = time_model.predict(X)[0]

# === Simple policy ===
# If job is low risk AND takes > 8 seconds → skip
if pred_risk == 0 and pred_time > 8:
    decision = "true"
else:
    decision = "false"

# Output for GitHub
print(f"Predicted risk: {pred_risk} | Predicted time: {pred_time:.2f}")
print(f"::set-output name=skip::{decision}")

