import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

df = pd.read_csv("ci_data.csv")

# drop rows without duration
df = df.dropna(subset=["duration_s"])

# binary target for risk: non-success == 1 (risky), success == 0 (safe)
df["risk"] = (df["status"].fillna("").str.lower() != "success").astype(int)

# author failure rate (historical)
author_rate = (
    df.groupby("author_login")["risk"]
      .mean()
      .rename("author_fail_rate")
      .reset_index()
)
df = df.merge(author_rate, on="author_login", how="left")
df["author_fail_rate"] = df["author_fail_rate"].fillna(0.0)

# commit message keywords (simple presence flags)
msg = df["msg"].fillna("").str.lower()
df["kw_feat"]    = msg.str.contains(r"\bfeat\b").astype(int)
df["kw_fix"]     = msg.str.contains(r"\bfix|bug|bugfix|hotfix\b").astype(int)
df["kw_refactor"]= msg.str.contains(r"\brefactor\b").astype(int)
df["kw_docs"]    = msg.str.contains(r"\bdocs|readme\b").astype(int)
df["kw_test"]    = msg.str.contains(r"\btest|ci\b").astype(int)
df["kw_chore"]   = msg.str.contains(r"\bchore\b").astype(int)

# minimal numeric/categorical feature set
feature_dicts = []
for _, r in df.iterrows():
    feat = {
        # categorical
        "job_name": str(r["job_name"]),
        # numeric
        "files_changed": float(r["files_changed"] or 0),
        "additions": float(r["additions"] or 0),
        "deletions": float(r["deletions"] or 0),
        "author_fail_rate": float(r["author_fail_rate"] or 0),
        # file ext counts
        "ext_py": float(r["ext_py"] or 0),
        "ext_js": float(r["ext_js"] or 0),
        "ext_ts": float(r["ext_ts"] or 0),
        "ext_tsx": float(r["ext_tsx"] or 0),
        "ext_java": float(r["ext_java"] or 0),
        "ext_md": float(r["ext_md"] or 0),
        "ext_yml": float(r["ext_yml"] or 0),
        "ext_json": float(r["ext_json"] or 0),
        # keywords
        "kw_feat": int(r["kw_feat"]),
        "kw_fix": int(r["kw_fix"]),
        "kw_refactor": int(r["kw_refactor"]),
        "kw_docs": int(r["kw_docs"]),
        "kw_test": int(r["kw_test"]),
        "kw_chore": int(r["kw_chore"]),
    }
    feature_dicts.append(feat)

vec = DictVectorizer(sparse=False)
X = vec.fit_transform(feature_dicts)
y_risk = df["risk"].values
y_time = df["duration_s"].values

X_train, X_test, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
X_train_t, X_test_t, y_time_train, y_time_test = train_test_split(X, y_time, test_size=0.2, random_state=42)

risk_model = RandomForestClassifier(n_estimators=200, random_state=42)
risk_model.fit(X_train, y_risk_train)
risk_acc = accuracy_score(y_risk_test, risk_model.predict(X_test))

time_model = RandomForestRegressor(n_estimators=250, random_state=42)
time_model.fit(X_train_t, y_time_train)
time_mae = mean_absolute_error(y_time_test, time_model.predict(X_test_t))

joblib.dump(risk_model, "risk_model.pkl")
joblib.dump(time_model, "duration_model.pkl")
joblib.dump(vec, "feature_vectorizer.pkl")

print(f"Risk accuracy: {risk_acc:.3f}")
print(f"Duration MAE: {time_mae:.2f}s")
print("Saved: risk_model.pkl, duration_model.pkl, feature_vectorizer.pkl")
