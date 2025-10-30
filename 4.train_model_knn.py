# Full fixed KNN training + inference cell
# paste into Colab and run

import os, time, ipaddress, numpy as np, pandas as pd, joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from math import sqrt

# -------------------- Config --------------------
CSV_PATH = "new_indian_training_data.csv"
OUT_DIR = "knn_artifacts_fixed"
os.makedirs(OUT_DIR, exist_ok=True)

K_NEIGHBORS_CITY = 7
K_NEIGHBORS_REG = 7
KNN_METRIC = "cosine"         # good with TF-IDF + numeric stacking
N_TFIDF_RDNS = 200
N_TFIDF_ASND = 200
TEST_SIZE = 0.20
RANDOM_STATE = 42
MIN_CITY_COUNT = 5

# -------------------- Load & clean --------------------
print("Loading CSV...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Rows:", len(df))

# Normalize sentinel values
df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)

# Ensure expected columns exist
expected_cols = ["ip_from","ip_to","ip_start","ip_end","ip_numeric","asn","lat","lon","local_rtt_ms","reachable","rdns_hostname","asn_description","state","city"]
for c in expected_cols:
    if c not in df.columns:
        df[c] = np.nan

# Coerce numeric ip ranges if present; else convert dotted ip_start/ip_end
df["ip_from"] = pd.to_numeric(df["ip_from"], errors="coerce")
df["ip_to"]   = pd.to_numeric(df["ip_to"], errors="coerce")
df["ip_numeric"] = pd.to_numeric(df["ip_numeric"], errors="coerce")

def dotted_to_int(x):
    try:
        return int(ipaddress.ip_address(str(x)))
    except Exception:
        return np.nan

if df["ip_from"].isna().all() and "ip_start" in df.columns:
    print("Converting dotted ip_start/ip_end to numeric ip_from/ip_to ...")
    df["ip_from"] = df["ip_start"].apply(dotted_to_int)
    df["ip_to"]   = df["ip_end"].apply(dotted_to_int)

# Numeric conversions for other features
df["asn"] = pd.to_numeric(df["asn"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["local_rtt_ms"] = pd.to_numeric(df["local_rtt_ms"], errors="coerce")

# Flags
df["is_rtt_missing"] = df["local_rtt_ms"].isna().astype(int)
df["reachable_flag"] = df["reachable"].astype(str).str.upper().map({"TRUE":1,"FALSE":0})
df["reachable_flag"] = pd.to_numeric(df["reachable_flag"], errors="coerce").fillna(0).astype(int)

# Text cleanup
df["rdns_hostname"] = df["rdns_hostname"].fillna("").astype(str)
df["asn_description"] = df["asn_description"].fillna("").astype(str)
df["state"] = df["state"].fillna("__MISSING__").astype(str)

# Drop rows without city label
df["city"] = df["city"].fillna("").astype(str).str.strip()
df = df[df["city"] != ""].reset_index(drop=True)
print("Rows after keeping labeled cities:", len(df))

# ip_mid (midpoint) for generalization
df["ip_from_f"] = df["ip_from"].fillna(df["ip_numeric"])
df["ip_to_f"]   = df["ip_to"].fillna(df["ip_numeric"])
df["ip_mid"] = (df["ip_from_f"] + df["ip_to_f"]) / 2.0
df["ip_mid"] = pd.to_numeric(df["ip_mid"], errors="coerce")

# -------------------- Prepare target and group rare cities --------------------
city_counts = df["city"].value_counts()
rare_cities = city_counts[city_counts < MIN_CITY_COUNT].index
df["city_grouped"] = df["city"].where(~df["city"].isin(rare_cities), other="Other")
print("Unique cities before:", len(city_counts), "after grouping:", df["city_grouped"].nunique())

le_city = LabelEncoder()
df["city_label"] = le_city.fit_transform(df["city_grouped"])

# -------------------- Feature columns --------------------
NUMERIC_COLS = [c for c in ["ip_mid","asn","local_rtt_ms","is_rtt_missing","reachable_flag"] if c in df.columns]
CATEGORICAL_COLS = ["state"]
TEXT_RDNS = "rdns_hostname"
TEXT_ASND = "asn_description"

# Build X and y
X_df = df[[TEXT_RDNS, TEXT_ASND] + NUMERIC_COLS + CATEGORICAL_COLS].copy()
y = df["city_label"].values

# Train/test split stratified on grouped city
X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["city_grouped"])
print("Train rows:", len(X_train_df), "Test rows:", len(X_test_df))

# -------------------- Preprocessing fit --------------------
print("Fitting TF-IDF for rdns and asn_description ...")
tfidf_rdns = TfidfVectorizer(max_features=N_TFIDF_RDNS, ngram_range=(1,1))
tfidf_asnd = TfidfVectorizer(max_features=N_TFIDF_ASND, ngram_range=(1,1))
tfidf_rdns.fit(X_train_df[TEXT_RDNS])
tfidf_asnd.fit(X_train_df[TEXT_ASND])

# OneHotEncoder: use sparse_output for sklearn >=1.4
ohe_state = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
ohe_state.fit(X_train_df[CATEGORICAL_COLS])

# Numeric imputer + scaler
num_imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X_train_num = num_imputer.fit_transform(X_train_df[NUMERIC_COLS])
X_train_num = scaler.fit_transform(X_train_num)
X_test_num  = num_imputer.transform(X_test_df[NUMERIC_COLS])
X_test_num  = scaler.transform(X_test_num)

# Transform text and categorical to sparse matrices
X_train_rdns = tfidf_rdns.transform(X_train_df[TEXT_RDNS])
X_train_asnd = tfidf_asnd.transform(X_train_df[TEXT_ASND])
X_train_cat  = ohe_state.transform(X_train_df[CATEGORICAL_COLS])

X_test_rdns = tfidf_rdns.transform(X_test_df[TEXT_RDNS])
X_test_asnd = tfidf_asnd.transform(X_test_df[TEXT_ASND])
X_test_cat  = ohe_state.transform(X_test_df[CATEGORICAL_COLS])

# Stack into final feature matrices (sparse)
print("Stacking feature matrices ...")
X_train_full = hstack([csr_matrix(X_train_rdns), csr_matrix(X_train_asnd), csr_matrix(X_train_num), csr_matrix(X_train_cat)]).tocsr()
X_test_full  = hstack([csr_matrix(X_test_rdns), csr_matrix(X_test_asnd), csr_matrix(X_test_num), csr_matrix(X_test_cat)]).tocsr()
print("X_train shape:", X_train_full.shape, "X_test shape:", X_test_full.shape)

# -------------------- Train KNN models --------------------
print("Training KNN city classifier ...")
knn_city = KNeighborsClassifier(n_neighbors=K_NEIGHBORS_CITY, metric=KNN_METRIC, n_jobs=-1)
knn_city.fit(X_train_full, y_train)

# prepare lat/lon training targets aligned with X_train_df index
train_idx = X_train_df.index.to_numpy()
lat_train = df.loc[train_idx, "lat"].fillna(df["lat"].mean()).to_numpy()
lon_train = df.loc[train_idx, "lon"].fillna(df["lon"].mean()).to_numpy()

print("Training KNN regressors for lat & lon ...")
knn_lat = KNeighborsRegressor(n_neighbors=K_NEIGHBORS_REG, metric=KNN_METRIC, n_jobs=-1)
knn_lon = KNeighborsRegressor(n_neighbors=K_NEIGHBORS_REG, metric=KNN_METRIC, n_jobs=-1)
knn_lat.fit(X_train_full, lat_train)
knn_lon.fit(X_train_full, lon_train)

print("Training KNN reachable classifier ...")
reach_train = df.loc[train_idx, "reachable_flag"].to_numpy()
knn_reach = KNeighborsClassifier(n_neighbors=K_NEIGHBORS_CITY, metric=KNN_METRIC, n_jobs=-1)
knn_reach.fit(X_train_full, reach_train)

# -------------------- Evaluate --------------------
print("\nEvaluating on test set...")
y_pred_city = knn_city.predict(X_test_full)
acc_city = accuracy_score(y_test, y_pred_city)
print("City accuracy (test):", round(acc_city*100,2), "%")
print(classification_report(y_test, y_pred_city, zero_division=0, digits=3))

# lat/lon test arrays aligned with X_test_df
test_idx = X_test_df.index.to_numpy()
lat_test = df.loc[test_idx, "lat"].fillna(df["lat"].mean()).to_numpy()
lon_test = df.loc[test_idx, "lon"].fillna(df["lon"].mean()).to_numpy()

lat_pred = knn_lat.predict(X_test_full)
lon_pred = knn_lon.predict(X_test_full)
lat_rmse = sqrt(mean_squared_error(lat_test, lat_pred))
lon_rmse = sqrt(mean_squared_error(lon_test, lon_pred))
print(f"Lat RMSE: {lat_rmse:.4f}, Lon RMSE: {lon_rmse:.4f}")

reach_test = df.loc[test_idx, "reachable_flag"].to_numpy()
reach_pred = knn_reach.predict(X_test_full)
print("Reachable accuracy:", round(accuracy_score(reach_test, reach_pred)*100,2), "%")

# -------------------- Save artifacts --------------------
print("Saving artifacts...")
joblib.dump(knn_city, os.path.join(OUT_DIR, "knn_city.joblib"))
joblib.dump(knn_lat, os.path.join(OUT_DIR, "knn_lat.joblib"))
joblib.dump(knn_lon, os.path.join(OUT_DIR, "knn_lon.joblib"))
joblib.dump(knn_reach, os.path.join(OUT_DIR, "knn_reach.joblib"))
joblib.dump(le_city, os.path.join(OUT_DIR, "label_encoder.joblib"))
preproc = {
    "tfidf_rdns": tfidf_rdns,
    "tfidf_asnd": tfidf_asnd,
    "ohe_state": ohe_state,
    "num_imputer": num_imputer,
    "scaler": scaler,
    "numeric_cols": NUMERIC_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "text_cols": [TEXT_RDNS, TEXT_ASND]
}
joblib.dump(preproc, os.path.join(OUT_DIR, "preproc.joblib"))
print("Saved in", OUT_DIR)

# -------------------- Inference helpers --------------------
# load lookup df for IP->record queries
lookup_df = pd.read_csv(CSV_PATH, low_memory=False)
lookup_df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)
for col in ("ip_from","ip_to","ip_numeric"):
    if col in lookup_df.columns:
        lookup_df[col] = pd.to_numeric(lookup_df[col], errors="coerce")
if (("ip_from" not in lookup_df.columns or lookup_df["ip_from"].isna().all()) and "ip_start" in lookup_df.columns):
    lookup_df["ip_from"] = lookup_df["ip_start"].apply(dotted_to_int)
    lookup_df["ip_to"]   = lookup_df["ip_end"].apply(dotted_to_int)
if {"ip_from","ip_to"}.issubset(lookup_df.columns):
    lookup_df = lookup_df.dropna(subset=["ip_from","ip_to"]).reset_index(drop=True)

preproc_loaded = joblib.load(os.path.join(OUT_DIR, "preproc.joblib"))
le_loaded = joblib.load(os.path.join(OUT_DIR, "label_encoder.joblib"))
knn_city_loaded = joblib.load(os.path.join(OUT_DIR, "knn_city.joblib"))
knn_lat_loaded = joblib.load(os.path.join(OUT_DIR, "knn_lat.joblib"))
knn_lon_loaded = joblib.load(os.path.join(OUT_DIR, "knn_lon.joblib"))
knn_reach_loaded = joblib.load(os.path.join(OUT_DIR, "knn_reach.joblib"))

def find_exact_record(ip_str, lookup_df):
    try:
        ipn = int(ipaddress.ip_address(ip_str))
    except:
        return None
    mask = (lookup_df["ip_from"] <= ipn) & (lookup_df["ip_to"] >= ipn)
    hits = lookup_df.loc[mask]
    if hits.empty:
        return None
    hits = hits.assign(range_size=(hits["ip_to"] - hits["ip_from"]))
    return hits.sort_values("range_size").iloc[0]

def find_nearest_record(ip_str, lookup_df):
    try:
        ipn = int(ipaddress.ip_address(ip_str))
    except:
        return None
    d_from = (lookup_df["ip_from"] - ipn).abs()
    d_to   = (lookup_df["ip_to"] - ipn).abs()
    dist = np.minimum(d_from.fillna(np.inf).to_numpy(), d_to.fillna(np.inf).to_numpy())
    idx = int(np.nanargmin(dist))
    return lookup_df.iloc[idx]

def build_feature_vector_from_record(rec, preproc_dict):
    feat = {}
    # numeric
    for n in preproc_dict["numeric_cols"]:
        if n in rec.index:
            try:
                feat[n] = float(rec[n]) if not pd.isna(rec[n]) else np.nan
            except:
                feat[n] = np.nan
        else:
            feat[n] = np.nan
    # categorical
    for c in preproc_dict["categorical_cols"]:
        val = rec.get(c, "__MISSING__") if c in rec.index else "__MISSING__"
        feat[c] = val if not pd.isna(val) else "__MISSING__"
    # text
    for t in preproc_dict["text_cols"]:
        val = rec.get(t, "") if t in rec.index else ""
        feat[t] = val if not pd.isna(val) else ""
    # ip_mid fallback
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
    cols = preproc_dict["text_cols"] + preproc_dict["numeric_cols"] + preproc_dict["categorical_cols"]
    return pd.DataFrame([feat])[cols]

def transform_row_to_feature_matrix(row_df, preproc_dict):
    tr = preproc_dict
    tf1 = tr["tfidf_rdns"].transform(row_df[tr["text_cols"][0]])
    tf2 = tr["tfidf_asnd"].transform(row_df[tr["text_cols"][1]])
    num_arr = tr["num_imputer"].transform(row_df[tr["numeric_cols"]])
    num_arr = tr["scaler"].transform(num_arr)
    cat_arr = tr["ohe_state"].transform(row_df[tr["categorical_cols"]])
    return hstack([csr_matrix(tf1), csr_matrix(tf2), csr_matrix(num_arr), csr_matrix(cat_arr)]).tocsr()

def predict_from_ip(ip_str, use_nearest_fallback=True, topk=5):
    rec = find_exact_record(ip_str, lookup_df)
    used_fallback = False
    if rec is None and use_nearest_fallback:
        rec = find_nearest_record(ip_str, lookup_df)
        used_fallback = True
    if rec is None:
        # synthetic record from IP numeric only
        try:
            ipn = int(ipaddress.ip_address(ip_str))
        except:
            return {"error": "invalid ip"}
        synthetic_rec = {
            "ip_numeric": ipn, "ip_from": np.nan, "ip_to": np.nan,
            "asn": np.nan, "lat": np.nan, "lon": np.nan, "local_rtt_ms": np.nan,
            "is_rtt_missing": 1, "reachable_flag": 0, "rdns_hostname": "", "asn_description":"", "state":"__MISSING__"
        }
        rec = pd.Series(synthetic_rec)

    row_df = build_feature_vector_from_record(rec, preproc_loaded)
    X_feat = transform_row_to_feature_matrix(row_df, preproc_loaded)

    city_label = knn_city_loaded.predict(X_feat)[0]
    city_name = le_loaded.inverse_transform([int(city_label)])[0]
    topk_list = None
    try:
        probs = knn_city_loaded.predict_proba(X_feat)[0]
        idx = np.argsort(probs)[::-1][:topk]
        topk_list = [(le_loaded.inverse_transform([int(i)])[0], float(probs[i])) for i in idx]
    except Exception:
        topk_list = None

    lat_pred = float(knn_lat_loaded.predict(X_feat)[0])
    lon_pred = float(knn_lon_loaded.predict(X_feat)[0])
    reach_pred = int(knn_reach_loaded.predict(X_feat)[0])

    return {
        "ip": ip_str,
        "predicted_city": city_name,
        "topk": topk_list,
        "predicted_lat": lat_pred,
        "predicted_lon": lon_pred,
        "predicted_reachable": bool(reach_pred),
        "used_fallback_nearest_range": bool(used_fallback),
        "matched_record_sample": rec.to_dict() if isinstance(rec, pd.Series) else None
    }

# -------------------- Example usage --------------------
print("\nExample prediction for an IP (it will fallback to nearest range if not found):")
example_ip = "1.6.68.129"
result = predict_from_ip(example_ip)
print(result)

print("\nDone. Artifacts in:", OUT_DIR)
