# Decision Tree training + inference for IP -> city, lat, lon, reachable
# Paste this entire cell into Colab and run.

import os, time, ipaddress, numpy as np, pandas as pd, joblib
from math import sqrt
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

# ---------------- CONFIG ----------------
CSV_PATH = "new_indian_training_data.csv"
OUT_DIR = "dt_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_CITY_COUNT = 5             # group rare cities into 'Other'
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Decision tree hyperparams (tune as needed)
DT_CITY_PARAMS = dict(max_depth=20, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE)
DT_REG_PARAMS  = dict(max_depth=20, min_samples_leaf=5, random_state=RANDOM_STATE)

# TF-IDF sizes
TFIDF_RDNS_DIM = 300
TFIDF_ASND_DIM = 300

# ---------------- LOAD & CLEAN ----------------
print("Loading CSV...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Rows loaded:", len(df))

# normalize sentinel values
df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)

# ensure columns exist
for c in ["ip_from","ip_to","ip_start","ip_end","ip_numeric","asn","lat","lon","local_rtt_ms","reachable","rdns_hostname","asn_description","state","city"]:
    if c not in df.columns:
        df[c] = np.nan

# coerce numeric ip columns if present
df["ip_from"] = pd.to_numeric(df["ip_from"], errors="coerce")
df["ip_to"]   = pd.to_numeric(df["ip_to"], errors="coerce")
df["ip_numeric"] = pd.to_numeric(df["ip_numeric"], errors="coerce")

# if ip_from empty but ip_start/ip_end present, convert dotted to ints
def dotted_to_int_safe(x):
    try:
        return int(ipaddress.ip_address(str(x)))
    except:
        return np.nan

if df["ip_from"].isna().all() and "ip_start" in df.columns:
    print("Converting dotted ip_start/ip_end to numeric ip_from/ip_to ...")
    df["ip_from"] = df["ip_start"].apply(dotted_to_int_safe)
    df["ip_to"]   = df["ip_end"].apply(dotted_to_int_safe)

# numeric conversions
df["asn"] = pd.to_numeric(df["asn"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["local_rtt_ms"] = pd.to_numeric(df["local_rtt_ms"], errors="coerce")

# flags
df["is_rtt_missing"] = df["local_rtt_ms"].isna().astype(int)
df["reachable_flag"] = df["reachable"].astype(str).str.upper().map({"TRUE":1,"FALSE":0})
df["reachable_flag"] = pd.to_numeric(df["reachable_flag"], errors="coerce").fillna(0).astype(int)

# text cleanup
df["rdns_hostname"] = df["rdns_hostname"].fillna("").astype(str)
df["asn_description"] = df["asn_description"].fillna("").astype(str)
df["state"] = df["state"].fillna("__MISSING__").astype(str)

# drop rows without city label (we need labels)
df["city"] = df["city"].astype(str).str.strip().replace({"nan": np.nan})
df = df.dropna(subset=["city"]).reset_index(drop=True)
print("Rows after dropping empty city:", len(df))

# compute ip_mid feature (midpoint) for generalization
df["ip_from_f"] = df["ip_from"].fillna(df["ip_numeric"])
df["ip_to_f"]   = df["ip_to"].fillna(df["ip_numeric"])
df["ip_mid"] = (df["ip_from_f"] + df["ip_to_f"]) / 2.0
df["ip_mid"] = pd.to_numeric(df["ip_mid"], errors="coerce")

# ---------------- TARGET PREP: group rare cities ----------------
city_counts = df["city"].value_counts()
rare = city_counts[city_counts < MIN_CITY_COUNT].index
df["city_grouped"] = df["city"].where(~df["city"].isin(rare), other="Other")
print("Unique cities before:", len(city_counts), "after grouping:", df["city_grouped"].nunique())

# label encode city
le_city = LabelEncoder()
df["city_label"] = le_city.fit_transform(df["city_grouped"])

# ---------------- FEATURES ----------------
TEXT_RDNS = "rdns_hostname"
TEXT_ASND = "asn_description"
CAT_STATE = "state"
NUMERIC = ["ip_mid", "asn", "lat", "lon", "local_rtt_ms", "is_rtt_missing", "reachable_flag"]

# keep only columns that exist
NUMERIC = [c for c in NUMERIC if c in df.columns]

# build X, y
X_df = df[[TEXT_RDNS, TEXT_ASND] + NUMERIC + [CAT_STATE]].copy()
y_city = df["city_label"].values
y_lat = df["lat"].fillna(df["lat"].mean()).values
y_lon = df["lon"].fillna(df["lon"].mean()).values
y_reach = df["reachable_flag"].values

# train/test split (stratify by grouped city)
X_train_df, X_test_df, y_city_train, y_city_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test, y_reach_train, y_reach_test = train_test_split(
    X_df, y_city, y_lat, y_lon, y_reach, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["city_grouped"]
)

print("Train size:", len(X_train_df), "Test size:", len(X_test_df))

# ---------------- PREPROCESSING (fit) ----------------
print("Fitting TF-IDF and encoders...")
tfidf_rdns = TfidfVectorizer(max_features=TFIDF_RDNS_DIM, ngram_range=(1,2))
tfidf_asnd = TfidfVectorizer(max_features=TFIDF_ASND_DIM, ngram_range=(1,2))
tfidf_rdns.fit(X_train_df[TEXT_RDNS])
tfidf_asnd.fit(X_train_df[TEXT_ASND])

# OneHot for state (sparse)
ohe_state = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
ohe_state.fit(X_train_df[[CAT_STATE]])

# Numeric imputer (decision trees don't need scaling but impute missing)
num_imputer = SimpleImputer(strategy="mean")
num_imputer.fit(X_train_df[NUMERIC])

# Build features for train / test: text -> tfidf (sparse), numeric -> impute -> dense, cat -> ohe (sparse)
print("Transforming training data...")
X_train_rdns = tfidf_rdns.transform(X_train_df[TEXT_RDNS])
X_train_asnd = tfidf_asnd.transform(X_train_df[TEXT_ASND])
X_train_cat  = ohe_state.transform(X_train_df[[CAT_STATE]])
X_train_num  = num_imputer.transform(X_train_df[NUMERIC])
# Convert numeric to sparse to stack, then we will convert full stack to dense for Decision Tree
from scipy import sparse
X_train_stack = hstack([csr_matrix(X_train_rdns), csr_matrix(X_train_asnd), csr_matrix(X_train_num), csr_matrix(X_train_cat)]).tocsr()

print("Transforming test data...")
X_test_rdns = tfidf_rdns.transform(X_test_df[TEXT_RDNS])
X_test_asnd = tfidf_asnd.transform(X_test_df[TEXT_ASND])
X_test_cat  = ohe_state.transform(X_test_df[[CAT_STATE]])
X_test_num  = num_imputer.transform(X_test_df[NUMERIC])
X_test_stack = hstack([csr_matrix(X_test_rdns), csr_matrix(X_test_asnd), csr_matrix(X_test_num), csr_matrix(X_test_cat)]).tocsr()

print("Converting stacked sparse matrices to dense arrays for Decision Tree (may use memory).")
X_train_full = X_train_stack.toarray()
X_test_full  = X_test_stack.toarray()
print("Final shapes:", X_train_full.shape, X_test_full.shape)

# ---------------- TRAIN DECISION TREE MODELS ----------------
print("Training DecisionTreeClassifier for city (multi-class)...")
dt_city = DecisionTreeClassifier(**DT_CITY_PARAMS)
t0 = time.time()
dt_city.fit(X_train_full, y_city_train)
t1 = time.time()
print(f"Trained city DT in {t1-t0:.1f}s")

print("Training DecisionTreeRegressor for lat...")
dt_lat = DecisionTreeRegressor(**DT_REG_PARAMS)
dt_lat.fit(X_train_full, y_lat_train)

print("Training DecisionTreeRegressor for lon...")
dt_lon = DecisionTreeRegressor(**DT_REG_PARAMS)
dt_lon.fit(X_train_full, y_lon_train)

print("Training DecisionTreeClassifier for reachable...")
dt_reach = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE)
dt_reach.fit(X_train_full, y_reach_train)

# ---------------- EVALUATE ----------------
print("\nEvaluating on test set...")
y_city_pred = dt_city.predict(X_test_full)
acc_city = accuracy_score(y_city_test, y_city_pred)
print("City accuracy (test):", round(acc_city*100,2), "%")
print(classification_report(y_city_test, y_city_pred, zero_division=0, digits=3))

lat_pred = dt_lat.predict(X_test_full)
lon_pred = dt_lon.predict(X_test_full)
lat_rmse = sqrt(mean_squared_error(y_lat_test, lat_pred))
lon_rmse = sqrt(mean_squared_error(y_lon_test, lon_pred))
print(f"Lat RMSE: {lat_rmse:.4f}, Lon RMSE: {lon_rmse:.4f}")

reach_pred = dt_reach.predict(X_test_full)
print("Reachable accuracy:", round(accuracy_score(y_reach_test, reach_pred)*100,2), "%")

# ---------------- SAVE ARTIFACTS ----------------
print("Saving artifacts to", OUT_DIR)
joblib.dump(dt_city, os.path.join(OUT_DIR, "dt_city.joblib"))
joblib.dump(dt_lat,  os.path.join(OUT_DIR, "dt_lat.joblib"))
joblib.dump(dt_lon,  os.path.join(OUT_DIR, "dt_lon.joblib"))
joblib.dump(dt_reach,os.path.join(OUT_DIR, "dt_reach.joblib"))
joblib.dump(le_city, os.path.join(OUT_DIR, "label_encoder_city.joblib"))
preproc = {
    "tfidf_rdns": tfidf_rdns,
    "tfidf_asnd": tfidf_asnd,
    "ohe_state": ohe_state,
    "num_imputer": num_imputer,
    "numeric_cols": NUMERIC,
    "categorical_cols": [CAT_STATE],
    "text_cols": [TEXT_RDNS, TEXT_ASND]
}
joblib.dump(preproc, os.path.join(OUT_DIR, "preproc_objects.joblib"))
print("Saved artifacts.")

# ---------------- LOOKUP DF for inference ----------------
lookup_df = pd.read_csv(CSV_PATH, low_memory=False)
lookup_df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)
for c in ("ip_from","ip_to","ip_numeric"):
    if c in lookup_df.columns:
        lookup_df[c] = pd.to_numeric(lookup_df[c], errors="coerce")
if (("ip_from" not in lookup_df.columns or lookup_df["ip_from"].isna().all()) and "ip_start" in lookup_df.columns):
    lookup_df["ip_from"] = lookup_df["ip_start"].apply(dotted_to_int_safe)
    lookup_df["ip_to"]   = lookup_df["ip_end"].apply(dotted_to_int_safe)
if {"ip_from","ip_to"}.issubset(lookup_df.columns):
    lookup_df = lookup_df.dropna(subset=["ip_from","ip_to"]).reset_index(drop=True)

# ---------------- INFERENCE HELPERS ----------------
preproc_loaded = joblib.load(os.path.join(OUT_DIR, "preproc_objects.joblib"))
dt_city_loaded = joblib.load(os.path.join(OUT_DIR, "dt_city.joblib"))
dt_lat_loaded  = joblib.load(os.path.join(OUT_DIR, "dt_lat.joblib"))
dt_lon_loaded  = joblib.load(os.path.join(OUT_DIR, "dt_lon.joblib"))
dt_reach_loaded= joblib.load(os.path.join(OUT_DIR, "dt_reach.joblib"))
le_loaded = joblib.load(os.path.join(OUT_DIR, "label_encoder_city.joblib"))

def ip_to_int(ip_str):
    try:
        return int(ipaddress.ip_address(ip_str))
    except:
        return None

def find_exact_record(ip_str, lookup_df):
    ipn = ip_to_int(ip_str)
    if ipn is None:
        return None
    mask = (lookup_df["ip_from"] <= ipn) & (lookup_df["ip_to"] >= ipn)
    hits = lookup_df.loc[mask]
    if hits.empty:
        return None
    hits = hits.assign(range_size=(hits["ip_to"] - hits["ip_from"]))
    return hits.sort_values("range_size").iloc[0]

def find_nearest_record(ip_str, lookup_df):
    ipn = ip_to_int(ip_str)
    if ipn is None:
        return None
    d_from = (lookup_df["ip_from"] - ipn).abs()
    d_to   = (lookup_df["ip_to"] - ipn).abs()
    dist = np.minimum(d_from.fillna(np.inf).to_numpy(), d_to.fillna(np.inf).to_numpy())
    idx = int(np.nanargmin(dist))
    return lookup_df.iloc[idx]

def build_feature_row_from_record(rec):
    # build a one-row DataFrame with text, numeric, categorical matching preproc keys
    feat = {}
    # text
    for t in preproc_loaded["text_cols"]:
        v = rec.get(t, "") if t in rec.index else ""
        feat[t] = v if not pd.isna(v) else ""
    # numeric
    for n in preproc_loaded["numeric_cols"]:
        if n in rec.index:
            try:
                feat[n] = float(rec[n]) if not pd.isna(rec[n]) else np.nan
            except:
                feat[n] = np.nan
        else:
            feat[n] = np.nan
    # categorical
    for c in preproc_loaded["categorical_cols"]:
        v = rec.get(c, "__MISSING__") if c in rec.index else "__MISSING__"
        feat[c] = v if not pd.isna(v) else "__MISSING__"
    # ensure ip_mid computed if missing
    if pd.isna(feat.get("ip_mid", None)):
        ipf = rec.get("ip_from", np.nan) if "ip_from" in rec.index else np.nan
        ipt = rec.get("ip_to", np.nan) if "ip_to" in rec.index else np.nan
        ipn = rec.get("ip_numeric", np.nan) if "ip_numeric" in rec.index else np.nan
        ipf_val = ipf if not pd.isna(ipf) else ipn
        ipt_val = ipt if not pd.isna(ipt) else ipn
        if not pd.isna(ipf_val) or not pd.isna(ipt_val):
            try:
                feat["ip_mid"] = float(((ipf_val if not pd.isna(ipf_val) else ipn) + (ipt_val if not pd.isna(ipt_val) else ipn)) / 2.0)
            except:
                feat["ip_mid"] = np.nan
    # return one-row DF with column order used earlier
    cols = preproc_loaded["text_cols"] + preproc_loaded["numeric_cols"] + preproc_loaded["categorical_cols"]
    return pd.DataFrame([feat])[cols]

def transform_row_to_model_input(row_df):
    # transform one-row df to stacked dense array for DT
    tr = preproc_loaded
    tf_rdns = tr["tfidf_rdns"].transform(row_df[tr["text_cols"][0]])
    tf_asnd = tr["tfidf_asnd"].transform(row_df[tr["text_cols"][1]])
    num_arr = tr["num_imputer"].transform(row_df[tr["numeric_cols"]])
    cat_arr = tr["ohe_state"].transform(row_df[tr["categorical_cols"]])
    stacked = hstack([csr_matrix(tf_rdns), csr_matrix(tf_asnd), csr_matrix(num_arr), csr_matrix(cat_arr)]).tocsr()
    return stacked.toarray()

def predict_from_ip(ip_str, use_nearest_fallback=True, topk=5):
    rec = find_exact_record(ip_str, lookup_df)
    used_fallback = False
    if rec is None and use_nearest_fallback:
        rec = find_nearest_record(ip_str, lookup_df)
        used_fallback = True
    if rec is None:
        # create synthetic record from ip numeric only
        ipn = ip_to_int(ip_str)
        if ipn is None:
            return {"error": "invalid ip"}
        rec = pd.Series({
            "ip_numeric": ipn,
            "ip_from": np.nan, "ip_to": np.nan,
            "asn": np.nan, "lat": np.nan, "lon": np.nan, "local_rtt_ms": np.nan,
            "is_rtt_missing": 1, "reachable": False, "reachable_flag": 0,
            "rdns_hostname": "", "asn_description": "", "state": "__MISSING__"
        })

    row_df = build_feature_row_from_record(rec)
    X_in = transform_row_to_model_input(row_df)

    # city
    city_pred_idx = dt_city_loaded.predict(X_in)[0]
    city_pred = le_loaded.inverse_transform([int(city_pred_idx)])[0]
    # top-k probabilities if available
    topk_list = None
    try:
        probs = dt_city_loaded.predict_proba(X_in)[0]
        idx = np.argsort(probs)[::-1][:topk]
        topk_list = [(le_loaded.inverse_transform([int(i)])[0], float(probs[i])) for i in idx]
    except Exception:
        topk_list = None

    # lat/lon
    lat_pred = float(dt_lat_loaded.predict(X_in)[0])
    lon_pred = float(dt_lon_loaded.predict(X_in)[0])
    # reachable
    reach_pred = bool(int(dt_reach_loaded.predict(X_in)[0]))

    return {
        "ip": ip_str,
        "predicted_city": city_pred,
        "topk": topk_list,
        "predicted_lat": lat_pred,
        "predicted_lon": lon_pred,
        "predicted_reachable": reach_pred,
        "used_fallback_nearest_range": bool(used_fallback),
        "matched_record_sample": rec.to_dict() if isinstance(rec, pd.Series) else None
    }

# ---------------- Example usage ----------------
print("\nExample inference (you can call predict_from_ip('1.6.68.129'))")
res = predict_from_ip("1.6.68.129")
print(res)
