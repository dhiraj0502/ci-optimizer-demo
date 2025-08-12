import os, sys, csv, requests
import joblib
from datetime import datetime, timezone

# --- Load models ---
risk_model = joblib.load("risk_model.pkl")
time_model = joblib.load("duration_model.pkl")
vectorizer = joblib.load("feature_vectorizer.pkl")

# --- Inputs & env from GitHub ---
job_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
repo = os.getenv("GITHUB_REPOSITORY")       # owner/repo
sha  = os.getenv("GITHUB_SHA")
token = os.getenv("GITHUB_TOKEN")

print(f"[DEBUG] job={job_name} repo={repo} sha={sha}")

# --- Fetch commit context ---
headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
def get_json(url, params=None):
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json()

additions=deletions=files_changed=0
exts = {"py":0,"js":0,"ts":0,"tsx":0,"java":0,"md":0,"yml":0,"json":0}
author_login="unknown"
msg=""

try:
    commit = get_json(f"https://api.github.com/repos/{repo}/commits/{sha}")
    msg = (commit.get("commit") or {}).get("message","")
    stats = commit.get("stats") or {}
    additions = stats.get("additions",0) or 0
    deletions = stats.get("deletions",0) or 0
    files = commit.get("files") or []
    files_changed = len(files)
    if commit.get("author") and commit["author"].get("login"):
        author_login = commit["author"]["login"]
    # count extensions
    for f in files:
        fn = f.get("filename","")
        if "." in fn:
            ext = fn.split(".")[-1].lower()
            if ext in exts:
                exts[ext]+=1
            if ext in ("yaml","yml"):
                exts["yml"] += 1
except Exception as e:
    print(f"::warning:: Could not fetch commit details: {e}")

# --- author failure rate from last ~20 runs (lightweight heuristic) ---
# If you want a stronger signal, precompute in training set. Here we just set 0.1 default.
author_fail_rate = 0.1

# --- keywords ---
m = msg.lower()
kw_feat    = int("feat" in m)
kw_fix     = int(any(k in m for k in ["fix","bug","bugfix","hotfix"]))
kw_refactor= int("refactor" in m)
kw_docs    = int(("docs" in m) or ("readme" in m))
kw_test    = int(("test" in m) or ("ci" in m))
kw_chore   = int("chore" in m)

# --- Build feature dict & vectorize ---
feat = {
    "job_name": job_name,
    "files_changed": float(files_changed),
    "additions": float(additions),
    "deletions": float(deletions),
    "author_fail_rate": float(author_fail_rate),
    "ext_py": float(exts["py"]),
    "ext_js": float(exts["js"]),
    "ext_ts": float(exts["ts"]),
    "ext_tsx": float(exts["tsx"]),
    "ext_java": float(exts["java"]),
    "ext_md": float(exts["md"]),
    "ext_yml": float(exts["yml"]),
    "ext_json": float(exts["json"]),
    "kw_feat": kw_feat,
    "kw_fix": kw_fix,
    "kw_refactor": kw_refactor,
    "kw_docs": kw_docs,
    "kw_test": kw_test,
    "kw_chore": kw_chore,
}
X = vectorizer.transform([feat])

# --- Predict ---
risk_prob = float(risk_model.predict_proba(X)[0][1])  # prob of failure
pred_time = float(time_model.predict(X)[0])

# --- Policy (tune thresholds here) ---
TIME_HEAVY = 8.0
TIME_MED   = 5.0
LOW_RISK   = 0.05
MID_RISK   = 0.20

if (risk_prob <= LOW_RISK) and (pred_time > TIME_HEAVY):
    mode = "skip"
elif (risk_prob <= MID_RISK) and (pred_time > TIME_MED):
    mode = "partial"
else:
    mode = "full"

skip = "true" if mode == "skip" else "false"

print(f"[INFO] risk_prob={risk_prob:.4f} pred_time={pred_time:.2f}s mode={mode}")

# --- Outputs to GitHub ---
out_path = os.getenv("GITHUB_OUTPUT")
if out_path:
    with open(out_path,"a") as f:
        f.write(f"skip={skip}\n")
        f.write(f"mode={mode}\n")
        f.write(f"risk_prob={risk_prob:.6f}\n")
        f.write(f"predicted_time={pred_time:.2f}\n")

# --- Append decision log ---
log_file = "decision_log.csv"
write_header = not os.path.exists(log_file)
with open(log_file, "a", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["timestamp","commit_sha","job_name","risk_prob","predicted_time","mode"])
    w.writerow([datetime.now(timezone.utc).isoformat(), sha, job_name, f"{risk_prob:.6f}", f"{pred_time:.2f}", mode])
