import os
import sys
import joblib
import csv
from datetime import datetime, timezone

# Load models
risk_model = joblib.load("risk_model.pkl")
time_model = joblib.load("duration_model.pkl")
job_categories = joblib.load("job_categories.pkl")

# Get job name
job_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
print(f"[DEBUG] Input job name: {job_name}")

try:
    job_code = job_categories.index(job_name)
except ValueError:
    print(f"::warning:: Unknown job '{job_name}' not in model. Logging skipped.")
    print("❌ Unknown job. Will run.")
    print("::set-output name=skip::false")
    exit(0)

# Predict
X = [[job_code]]
pred_risk = risk_model.predict(X)[0]
pred_time = time_model.predict(X)[0]

# Decision logic
skip = pred_risk == 0 and pred_time > 8
decision = "true" if skip else "false"

# Output to GitHub Actions
print(f"::set-output name=skip::{decision}")
print(f"Predicted risk: {pred_risk}, Predicted time: {pred_time:.2f} sec, Skip: {decision}")

# Log results
log_file = "decision_log.csv"
log_exists = os.path.exists(log_file)

with open(log_file, mode="a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "job_name", "predicted_risk", "predicted_time", "decision"])
    writer.writerow([
        datetime.now(timezone.utc).isoformat(),
        job_name,
        pred_risk,
        round(pred_time, 2),
        decision
    ])
