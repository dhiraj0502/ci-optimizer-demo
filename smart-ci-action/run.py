import os
import sys
import joblib
from datetime import datetime

# Load models
risk_model = joblib.load("risk_model.pkl")
time_model = joblib.load("duration_model.pkl")
job_categories = joblib.load("job_categories.pkl")

# Job name from CLI
job_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
print(f"[DEBUG] Input job name: {job_name}")

try:
    job_code = job_categories.index(job_name)
except ValueError:
    print(f"::warning:: Unknown job '{job_name}' not in model. Logging skipped.")
    print("❌ Unknown job. Will run.")
    print("::set-output name=skip::false")
    exit(0)

X = [[job_code]]
pred_risk = risk_model.predict(X)[0]
pred_time = time_model.predict(X)[0]

# Decision rule
skip = pred_risk == 0 and pred_time > 8
decision = "true" if skip else "false"

# Output to GitHub Actions
print(f"::set-output name=skip::{decision}")
print(f"Predicted risk: {pred_risk}, Predicted time: {pred_time:.2f} sec, Skip: {decision}")

# Log to CSV
log_file = "decision_log.csv"
log_exists = os.path.exists(log_file)

with open(log_file, "a") as f:
    if not log_exists:
        f.write("timestamp,job_name,predicted_risk,predicted_time,decision\n")
    f.write(f"{datetime.utcnow().isoformat()},{job_name},{pred_risk},{round(pred_time, 2)},{decision}\n")
