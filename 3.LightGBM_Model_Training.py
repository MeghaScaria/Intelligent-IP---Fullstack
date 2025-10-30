# Single Colab cell: install LightGBM (if needed), then train LightGBM multiclass on your CSV
# Paste and run in a Colab cell.

# 1) Install LightGBM (quiet). This requires internet to pip-install.
import sys, subprocess, pkgutil, time
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    import lightgbm as lgb
except Exception:
    print("Installing lightgbm...")
    install("lightgbm")
    import lightgbm as lgb

print("LightGBM version:", lgb.__version__)

# 2) Imports
import pandas as pd, numpy as np, ipaddress, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
from sklearn.utils import shuffle

# 3) Config
CSV_PATH = "new_indian_training_data.csv"
OUTPUT_DIR = "artifacts_lgbm"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MIN_CITY_COUNT = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42

# 4) Load CSV (local file)
print("Loading CSV...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Rows loaded:", len(df))

# 5) Basic cleaning & sentinel normalization
df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)

# Ensure key columns exist
for c in ["ip_from","ip_to","ip_numeric","ip_start","ip_end","asn","lat","lon","local_rtt_ms","reachable","rdns_hostname","asn_description","state","city"]:
    if c not in df.columns:
        df[c] = np.nan

# Convert ip_from/ip_to or dotted ip_start/ip_end to numeric integers
def dotted_to_int_safe(x):
    try:
        return int(ipaddress.ip_address(str(x)))
    except Exception:
        return np.nan

# If numeric ip_from/ip_to exist, coerce; otherwise convert dotted ip_start/ip_end
df["ip_from"] = pd.to_numeric(df["ip_from"], errors="coerce")
df["ip_to"]   = pd.to_numeric(df["ip_to"], errors="coerce")

if df["ip_from"].isna().all() and "ip_start" in df.columns:
    print("Converting dotted ip_start/ip_end to numeric ip_from/ip_to ...")
    df["ip_from"] = df["ip_start"].apply(dotted_to_int_safe)
    df["ip_to"]   = df["ip_end"].apply(dotted_to_int_safe)

# ip_mid feature: midpoint of range or ip_numeric fallback
df["ip_numeric"] = pd.to_numeric(df["ip_numeric"], errors="coerce")
df["ip_from_f"] = df["ip_from"].fillna(df["ip_numeric"])
df["ip_to_f"]   = df["ip_to"].fillna(df["ip_numeric"])
df["ip_mid"] = (df["ip_from_f"] + df["ip_to_f"]) / 2.0
df["ip_mid"] = pd.to_numeric(df["ip_mid"], errors="coerce")

# Numeric conversions
df["asn"] = pd.to_numeric(df["asn"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["local_rtt_ms"] = pd.to_numeric(df["local_rtt_ms"], errors="coerce")
df["is_rtt_missing"] = df["local_rtt_ms"].isna().astype(int)

# reachable -> 0/1
df["reachable_flag"] = df["reachable"].astype(str).str.upper().map({"TRUE":1,"FALSE":0})
df["reachable_flag"] = pd.to_numeric(df["reachable_flag"], errors="coerce").fillna(0).astype(int)

# Clean text columns
df["rdns_hostname"] = df["rdns_hostname"].fillna("").astype(str)
df["asn_description"] = df["asn_description"].fillna("").astype(str)
df["state"] = df["state"].fillna("__MISSING__").astype(str)
df["city"] = df["city"].fillna("").astype(str).str.strip()

# Drop rows without city label (we need labels)
df = df[df["city"] != ""].reset_index(drop=True)
print("Rows after dropping empty-city:", len(df))

# Group rare cities into 'Other' so stratified split works
city_counts = df["city"].value_counts()
rare = city_counts[city_counts < MIN_CITY_COUNT].index
df["city_grouped"] = df["city"].where(~df["city"].isin(rare), other="Other")
print("Unique cities before:", len(city_counts), "after grouping:", df["city_grouped"].nunique())

# Prepare features for modeling
# Text features
TEXT_RDNS = "rdns_hostname"
TEXT_ASND = "asn_description"
# Numeric features
NUMERIC = ["ip_mid","asn","lat","lon","local_rtt_ms","is_rtt_missing","reachable_flag"]
# Categorical (low-card)
CATEGORICAL = ["state"]

# Fill text and cat columns
df[TEXT_RDNS] = df[TEXT_RDNS].fillna("")
df[TEXT_ASND] = df[TEXT_ASND].fillna("")
df[CATEGORICAL] = df[CATEGORICAL].fillna("__MISSING__")

# Label encode target
le = LabelEncoder()
df["city_label"] = le.fit_transform(df["city_grouped"])
n_classes = len(le.classes_)
print("Training classes:", n_classes)

# Subset relevant columns and shuffle to ensure balanced batches
cols_keep = [TEXT_RDNS, TEXT_ASND] + NUMERIC + CATEGORICAL + ["city_label"]
df = df[cols_keep].copy()
df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)

# Train/test split stratified
X = df.drop(columns=["city_label"])
y = df["city_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print("Train rows:", len(X_train), "Test rows:", len(X_test))

# 6) Fit TF-IDF on text; numeric preprocessing; categorical one-hot
print("Fitting TF-IDF + numeric pipelines...")
tfidf_rdns = TfidfVectorizer(max_features=400, ngram_range=(1,2))
tfidf_asnd = TfidfVectorizer(max_features=400, ngram_range=(1,2))

tfidf_rdns.fit(X_train[TEXT_RDNS])
tfidf_asnd.fit(X_train[TEXT_ASND])

# Numeric pipeline: impute mean + scale
num_imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X_train_num = num_imputer.fit_transform(X_train[NUMERIC])
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test[NUMERIC])
X_test_num = scaler.transform(X_test_num)

# Categorical: one-hot (low-cardinality)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[CATEGORICAL])
X_test_cat  = ohe.transform(X_test[CATEGORICAL])

# Text -> sparse
X_train_rdns = tfidf_rdns.transform(X_train[TEXT_RDNS])
X_test_rdns  = tfidf_rdns.transform(X_test[TEXT_RDNS])
X_train_asnd = tfidf_asnd.transform(X_train[TEXT_ASND])
X_test_asnd  = tfidf_asnd.transform(X_test[TEXT_ASND])

# Stack into sparse matrix for LightGBM efficiency
print("Stacking feature matrices (sparse)...")
X_train_comb = hstack([csr_matrix(X_train_rdns), csr_matrix(X_train_asnd),
                       csr_matrix(X_train_num), csr_matrix(X_train_cat)]).tocsr()
X_test_comb  = hstack([csr_matrix(X_test_rdns), csr_matrix(X_test_asnd),
                       csr_matrix(X_test_num), csr_matrix(X_test_cat)]).tocsr()

print("Shape train:", X_train_comb.shape, "shape test:", X_test_comb.shape)

# 7) Train LightGBM multiclass
print("Preparing LightGBM Dataset...")
train_data = lgb.Dataset(X_train_comb, label=y_train)
valid_data = lgb.Dataset(X_test_comb, label=y_test, reference=train_data)

params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 64,
    "verbosity": -1,
    "num_threads": 4
}

print("Training LightGBM (this may take a couple minutes)...")
start = time.time()
bst = lgb.train(
    params,
    train_data,
    num_boost_round=400,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

end = time.time()
print(f"LightGBM training completed in {end - start:.1f} seconds")

# 8) Evaluate
y_prob = bst.predict(X_test_comb, num_iteration=bst.best_iteration)
y_pred = np.argmax(y_prob, axis=1)
acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", round(acc*100,2), "%")
print(classification_report(y_test, y_pred, zero_division=0))

# 9) Save artifacts
print("Saving artifacts...")
bst.save_model(os.path.join(OUTPUT_DIR, "3.city_lgbm.txt"))
joblib.dump(le, os.path.join(OUTPUT_DIR, "3.city_label_encoder.joblib"))
# Save TF-IDF, numeric imputer/scaler, and OHE as tuple
preproc = {
    "tfidf_rdns": tfidf_rdns,
    "tfidf_asnd": tfidf_asnd,
    "num_imputer": num_imputer,
    "scaler": scaler,
    "onehot": ohe,
    "numeric_cols": NUMERIC,
    "categorical_cols": CATEGORICAL,
    "text_cols": [TEXT_RDNS, TEXT_ASND]
}
joblib.dump(preproc, os.path.join(OUTPUT_DIR, "preproc_objects.joblib"))
print("Artifacts saved to", OUTPUT_DIR)

# 10) Inference helper: predict from IP (uses numeric ip_mid and DB lookup if available)
print("\nExample inference helper (uses lookup if available, else uses ip_mid numeric):")

# load lookup DF (coerce ip ranges)
lookup = pd.read_csv(CSV_PATH, low_memory=False)
lookup.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)
for col in ("ip_from","ip_to","ip_numeric"):
    if col in lookup.columns:
        lookup[col] = pd.to_numeric(lookup[col], errors="coerce")
if ("ip_from" not in lookup.columns or lookup["ip_from"].isna().all()) and ("ip_start" in lookup.columns):
    lookup["ip_from"] = lookup["ip_start"].apply(dotted_to_int_safe)
    lookup["ip_to"]   = lookup["ip_end"].apply(dotted_to_int_safe)
lookup = lookup.dropna(subset=["ip_from","ip_to"]) if {"ip_from","ip_to"}.issubset(lookup.columns) else lookup

# load preproc & label encoder & booster (for convenience in this cell)
le_loaded = joblib.load(os.path.join(OUTPUT_DIR, "3.city_label_encoder.joblib"))
preproc_loaded = joblib.load(os.path.join(OUTPUT_DIR, "3.preproc_objects.joblib"))
bst_loaded = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, "3.city_lgbm.txt"))

def ip_to_int_safe(ip_str):
    try:
        return int(ipaddress.ip_address(ip_str))
    except:
        return None

def find_record(ip_str, lookup_df):
    ipn = ip_to_int_safe(ip_str)
    if ipn is None:
        return None
    mask = (lookup_df["ip_from"] <= ipn) & (lookup_df["ip_to"] >= ipn)
    hits = lookup_df.loc[mask]
    if hits.empty:
        return None
    hits = hits.assign(range_size=(hits["ip_to"] - hits["ip_from"]))
    return hits.sort_values("range_size").iloc[0]

def build_features_for_inference(rec_or_ip, use_lookup=True):
    """
    rec_or_ip: either a Series record from lookup DF (preferred) or an IP string.
    If record is provided, build full features from that record. Otherwise build features from the IP (ip_mid numeric)
    """
    if isinstance(rec_or_ip, str):
        # convert ip to ip_mid - generalization only
        ipn = ip_to_int_safe(rec_or_ip)
        if ipn is None:
            return None
        # build minimal feature dict
        feat = {}
        for c in NUMERIC + CATEGORICAL + [TEXT_RDNS, TEXT_ASND]:
            if c in NUMERIC:
                if c == "ip_mid":
                    feat[c] = float(ipn)
                else:
                    feat[c] = np.nan
            elif c in CATEGORICAL:
                feat[c] = "__MISSING__"
            else:
                feat[c] = ""
        return pd.DataFrame([feat])[NUMERIC + CATEGORICAL + [TEXT_RDNS, TEXT_ASND]]
    else:
        # rec_or_ip is a Series record from lookup df
        rec = rec_or_ip
        feat = {}
        # numeric
        feat["ip_mid"] = float(((rec.get("ip_from") if not pd.isna(rec.get("ip_from")) else rec.get("ip_numeric")) + (rec.get("ip_to") if not pd.isna(rec.get("ip_to")) else rec.get("ip_numeric"))) / 2.0) if (not pd.isna(rec.get("ip_from")) or not pd.isna(rec.get("ip_numeric"))) else np.nan
        feat["asn"] = float(rec.get("asn")) if not pd.isna(rec.get("asn")) else np.nan
        feat["lat"] = float(rec.get("lat")) if not pd.isna(rec.get("lat")) else np.nan
        feat["lon"] = float(rec.get("lon")) if not pd.isna(rec.get("lon")) else np.nan
        feat["local_rtt_ms"] = float(rec.get("local_rtt_ms")) if not pd.isna(rec.get("local_rtt_ms")) else np.nan
        feat["is_rtt_missing"] = 1 if pd.isna(rec.get("local_rtt_ms")) else 0
        feat["reachable_flag"] = 1 if str(rec.get("reachable","")).upper() == "TRUE" else 0
        # categorical/text
        feat["state"] = rec.get("state", "__MISSING__") if not pd.isna(rec.get("state")) else "__MISSING__"
        feat[TEXT_RDNS] = rec.get("rdns_hostname","") if not pd.isna(rec.get("rdns_hostname")) else ""
        feat[TEXT_ASND] = rec.get("asn_description","") if not pd.isna(rec.get("asn_description")) else ""
        return pd.DataFrame([feat])[NUMERIC + CATEGORICAL + [TEXT_RDNS, TEXT_ASND]]

def predict_city_from_ip(ip_str, topk=5, use_nearest_when_missing=True):
    # try exact lookup record
    rec = find_record(ip_str, lookup) if isinstance(lookup, pd.DataFrame) and not lookup.empty else None
    used_fallback = False
    if rec is None and use_nearest_when_missing and isinstance(lookup, pd.DataFrame) and not lookup.empty:
        # fallback: nearest by distance to boundaries
        ipn = ip_to_int_safe(ip_str)
        if ipn is None:
            return {"error": "invalid ip"}
        d_from = (lookup["ip_from"] - ipn).abs()
        d_to = (lookup["ip_to"] - ipn).abs()
        dist = np.minimum(d_from.fillna(np.inf).to_numpy(), d_to.fillna(np.inf).to_numpy())
        idx = int(np.nanargmin(dist))
        rec = lookup.iloc[idx]
        used_fallback = True

    if rec is not None:
        Xrow = build_features_for_inference(rec)
    else:
        Xrow = build_features_for_inference(ip_str)

    # transform features: text -> tfidf, numeric -> impute+scale, cat -> onehot
    tr = preproc_loaded
    tf_rdns = tr["tfidf_rdns"].transform(Xrow[TEXT_RDNS])
    tf_asnd = tr["tfidf_asnd"].transform(Xrow[TEXT_ASND])
    Xnum = tr["num_imputer"].transform(Xrow[NUMERIC])
    Xnum = tr["scaler"].transform(Xnum)
    Xcat = tr["onehot"].transform(Xrow[CATEGORICAL])

    Xfull = hstack([csr_matrix(tf_rdns), csr_matrix(tf_asnd), csr_matrix(Xnum), csr_matrix(Xcat)]).tocsr()
    probs = bst_loaded.predict(Xfull, num_iteration=bst_loaded.best_iteration)
    top_idx = np.argsort(probs[0])[::-1][:topk]
    topk_list = [(le_loaded.inverse_transform([i])[0], float(probs[0][i])) for i in top_idx]
    top1_label = le_loaded.inverse_transform([int(np.argmax(probs[0]))])[0]

    return {"ip": ip_str, "predicted_city": top1_label, "topk": topk_list, "used_fallback_nearest_range": used_fallback, "matched_record": rec.to_dict() if rec is not None else None}

# Example usage:
print("\nExample predictions (you can call predict_city_from_ip('1.6.68.129'))")
example_ip = "1.6.68.129"
res = predict_city_from_ip(example_ip)
print("Example:", res)
print("\nNow you can call predict_city_from_ip(<ip>) to get predictions.")
