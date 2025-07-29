import requests
import csv
import os

# === CONFIG ===
TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "dhiraj0502/ci-optimizer-demo"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# === OUTPUT FILE ===
CSV_FILE = "ci_data.csv"

def fetch_workflow_runs():
    url = f"https://api.github.com/repos/{REPO}/actions/runs"
    response = requests.get(url, headers=HEADERS)
    return response.json().get("workflow_runs", [])

def fetch_jobs_for_run(run_id):
    url = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/jobs"
    response = requests.get(url, headers=HEADERS)
    return response.json().get("jobs", [])

def save_to_csv(data):
    keys = data[0].keys()
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def main():
    all_data = []
    runs = fetch_workflow_runs()
    print(f"Found {len(runs)} runs")
    for run in runs:
        run_id = run["id"]
        commit_sha = run["head_sha"]
        created_at = run["created_at"]

        jobs = fetch_jobs_for_run(run_id)
        for job in jobs:
            all_data.append({
                "commit_sha": commit_sha,
                "run_id": run_id,
                "job_name": job["name"],
                "status": job["conclusion"],
                "duration_seconds": (job["completed_at"] and job["started_at"]) and
                    (parse_time(job["completed_at"]) - parse_time(job["started_at"])).total_seconds(),
                "created_at": created_at
            })
    if all_data:
        save_to_csv(all_data)
        print(f"✅ Saved {len(all_data)} job records to {CSV_FILE}")
    else:
        print("⚠️ No data found.")

# Helper to parse ISO date
from datetime import datetime
def parse_time(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")

if __name__ == "__main__":
    main()
