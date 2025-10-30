# Full updated script (preprocessing + fast approximate-RBF + IP lookup + predict function)
import pandas as pd
import numpy as np
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib, time

# --------- CONFIG ----------
CSV_PATH = "new_indian_training_data.csv"
RBF_N_COMPONENTS = 300
RBF_GAMMA = 0.1
SGD_MAX_ITER = 2000
SGD_TOL = 1e-4
MIN_CITY_COUNT = 5  # group cities with < MIN_CITY_COUNT samples into "Other"
# --------------------------

# ---------- LOAD ----------
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Loaded rows:", len(df))

# ---------- CLEAN / DTYPE FIXES ----------
# Normalize common sentinel values
df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)

# Ensure columns exist
for col in ["ip_from", "ip_to", "ip_numeric", "asn", "lat", "lon", "local_rtt_ms", "reachable", "rdns_hostname", "asn_description", "state", "isp", "city"]:
    if col not in df.columns:
        df[col] = np.nan

# Convert ip numeric columns (they are numeric in your sample)
df["ip_from"] = pd.to_numeric(df["ip_from"], errors="coerce")
df["ip_to"]   = pd.to_numeric(df["ip_to"], errors="coerce")
df["ip_numeric"] = pd.to_numeric(df["ip_numeric"], errors="coerce")

# ASN → numeric (in sample it's numeric like 9583)
df["asn"] = pd.to_numeric(df["asn"], errors="coerce")

# lat/lon numeric
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

# local_rtt numeric and missing-flag
df["local_rtt_ms"] = pd.to_numeric(df["local_rtt_ms"], errors="coerce")
df["is_rtt_missing"] = df["local_rtt_ms"].isna().astype(int)  # important signal

# reachable -> 0/1
df["reachable_flag"] = df["reachable"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
df["reachable_flag"] = pd.to_numeric(df["reachable_flag"], errors="coerce").fillna(0).astype(int)

# rdns and asn_description: keep as strings (NaN okay)
df["rdns_hostname"] = df["rdns_hostname"].fillna(np.nan).astype(object)
df["asn_description"] = df["asn_description"].fillna(np.nan).astype(object)
df["state"] = df["state"].fillna(np.nan).astype(object)
df["isp"] = df["isp"].fillna(np.nan).astype(object)

# Drop rows without city label (we need labels)
df["city"] = df["city"].astype(str).str.strip().replace({"nan": np.nan})
df = df.dropna(subset=["city"]).reset_index(drop=True)
print("Rows after dropping empty-city:", len(df))

# ---------- GROUP RARE CITIES ----------
city_counts = df["city"].value_counts()
rare = city_counts[city_counts < MIN_CITY_COUNT].index
df["city_grouped"] = df["city"].where(~df["city"].isin(rare), other="Other")
print("Unique cities before:", len(city_counts), "-> after grouping:", df["city_grouped"].nunique())

# ---------- SELECT FEATURES ----------
# Use only features that exist / are meaningful in sample
feature_cols = [
    "state",             # categorical
    "asn_description",   # text categorical
    "rdns_hostname",     # text categorical
    "lat", "lon",        # numeric coords (may be duplicated from city but useful)
    "asn",               # numeric ASN
    "reachable_flag",    # numeric/binary
    "is_rtt_missing"     # binary flag: RTT missing
]
# keep only columns that are present
feature_cols = [c for c in feature_cols if c in df.columns]

df_model = df[feature_cols + ["city_grouped"]].copy()
X = df_model.drop(columns=["city_grouped"])
y = df_model["city_grouped"]

# ---------- TRAIN/TEST SPLIT (stratified on grouped city) ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
print("Train size:", len(X_train), "Test size:", len(X_test))

# ---------- PREPROCESSORS ----------
# Categorical columns (text-like): we'll impute missing with "__MISSING__" then one-hot
categorical_cols = [c for c in ["state", "asn_description", "rdns_hostname"] if c in X_train.columns]
# Numeric columns: ensure they are numeric
numeric_cols = [c for c in ["lat", "lon", "asn", "reachable_flag", "is_rtt_missing"] if c in X_train.columns]

# categorical pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# numeric pipeline -> mean impute then standardize
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_cols),
    ("num", numeric_transformer, numeric_cols)
], remainder="drop", verbose_feature_names_out=False)

# ---------- FAST APPROXIMATE RBF PIPELINE ----------
rbf_approx = RBFSampler(gamma=RBF_GAMMA, n_components=RBF_N_COMPONENTS, random_state=42)

fast_pipeline = Pipeline([
    ("pre", preprocessor),
    ("rbf", rbf_approx),
    ("clf", SGDClassifier(
        loss="log_loss",   # probabilistic linear classifier
        max_iter=SGD_MAX_ITER,
        tol=SGD_TOL,
        random_state=42
    ))
])

# ---------- TRAIN ----------
print("Training approximate-RBF pipeline (this should be quick)...")
t0 = time.time()
fast_pipeline.fit(X_train, y_train)
t1 = time.time()
print(f"Training done in {t1 - t0:.1f}s")

# ---------- EVALUATE ----------
y_pred = fast_pipeline.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred, zero_division=0, digits=3))

# Save pipeline
joblib.dump(fast_pipeline, "city_rbf_approx_sgd_pipeline.joblib")
print("Saved pipeline -> city_rbf_approx_sgd_pipeline.joblib")

# ---------- LOAD LOOKUP DF (for IP -> record) ----------
# keep ip numeric ranges for lookup
df_lookup = pd.read_csv(CSV_PATH, low_memory=False)
# normalize sentinel values
df_lookup.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)
# coerce ip numeric ranges (if present)
for col in ["ip_from", "ip_to", "ip_numeric"]:
    if col in df_lookup.columns:
        df_lookup[col] = pd.to_numeric(df_lookup[col], errors="coerce")

# drop rows missing ip range
if "ip_from" in df_lookup.columns and "ip_to" in df_lookup.columns:
    df_lookup = df_lookup.dropna(subset=["ip_from", "ip_to"]).reset_index(drop=True)

# ---------- IP lookup + predict function ----------
def find_ip_record(ip_str, lookup_df):
    """Return first matching row (Series) for an IPv4 string, or None"""
    try:
        ipn = int(ipaddress.ip_address(ip_str))
    except Exception as e:
        print("Invalid IP:", e)
        return None
    # make sure lookup_df has numeric ip_from/ip_to
    if not {"ip_from", "ip_to"}.issubset(set(lookup_df.columns)):
        print("Lookup table missing ip_from/ip_to")
        return None
    mask = (lookup_df["ip_from"] <= ipn) & (lookup_df["ip_to"] >= ipn)
    matches = lookup_df.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]

def predict_city_from_ip(ip_str, pipeline=fast_pipeline, lookup_df=df_lookup):
    rec = find_ip_record(ip_str, lookup_df)
    if rec is None:
        print(f"No matching IP range found for {ip_str}")
        return None
    # build an input row with same feature columns used in training
    input_row = {}
    for c in feature_cols:
        if c in ["lat", "lon", "asn", "reachable_flag", "is_rtt_missing"]:
            # numeric fields: coerce to numeric
            val = rec.get(c, np.nan)
            try:
                input_row[c] = float(val) if (val is not None and not pd.isna(val)) else np.nan
            except:
                input_row[c] = np.nan
        else:
            # categorical/text fields (state, asn_description, rdns_hostname)
            v = rec.get(c, "")
            if pd.isna(v):
                input_row[c] = "__MISSING__"
            else:
                input_row[c] = str(v)
    # if is_rtt_missing not present in lookup, compute from local_rtt_ms
    if "is_rtt_missing" not in input_row:
        input_row["is_rtt_missing"] = 1 if pd.isna(rec.get("local_rtt_ms")) else 0

    X_in = pd.DataFrame([input_row])[feature_cols]  # ensure column order
    pred = pipeline.predict(X_in)[0]
    print(f"\nPrediction for {ip_str}: {pred}")
    print("Record info (best effort):")
    for k in ("ip_start", "ip_end", "country", "state", "city", "asn", "asn_description", "rdns_hostname", "local_rtt_ms", "reachable"):
        if k in rec.index:
            print(f"  {k}: {rec[k]}")
    return pred

# ---------- USAGE ----------
# Example: call predict_city_from_ip("1.6.68.129")
# To prompt user:
# user_ip = input("Enter IP: ").strip()
# predict_city_from_ip(user_ip)

