# ip_inference.py - use saved pipeline to predict city from an IP address
#TAKES INPUT AND GIVES OUTPUT IN THE FORM OF A CITY
import joblib
import pandas as pd
import numpy as np
import ipaddress
import os

MODEL_PATH = "2.RBF_city_rbf_approx_sgd_pipeline.joblib"
CSV_PATH = "new_indian_training_data.csv"

# --------- Helpers ----------
def dotted_ip_to_int(ip_str):
    try:
        return int(ipaddress.ip_address(ip_str))
    except Exception:
        return None

def load_lookup_df(csv_path):
    # Load CSV and coerce numeric ip ranges if present; also try to convert ip_start/ip_end dotted to integers
    df = pd.read_csv(csv_path, low_memory=False)
    # Normalize sentinel values
    df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)

    # Attempt to coerce ip_from/ip_to columns to numeric if they exist
    for col in ("ip_from", "ip_to", "ip_numeric"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If ip_from/ip_to are missing but ip_start/ip_end (dotted strings) exist, convert them
    if (("ip_from" not in df.columns or df["ip_from"].isna().all())
        and "ip_start" in df.columns and "ip_end" in df.columns):
        def conv(s):
            try:
                return int(ipaddress.ip_address(str(s)))
            except Exception:
                return np.nan
        df["ip_from"] = df["ip_start"].apply(conv)
        df["ip_to"] = df["ip_end"].apply(conv)

    # Drop rows without numeric ranges (can't lookup those)
    if "ip_from" in df.columns and "ip_to" in df.columns:
        df = df.dropna(subset=["ip_from", "ip_to"]).reset_index(drop=True)
    else:
        raise ValueError("No numeric ip_from/ip_to columns found or convertible in CSV. Provide ip_from/ip_to or ip_start/ip_end dotted columns.")
    return df

def find_ip_record(ip_str, lookup_df):
    ipn = dotted_ip_to_int(ip_str)
    if ipn is None:
        raise ValueError(f"Invalid IP address: {ip_str}")
    mask = (lookup_df["ip_from"] <= ipn) & (lookup_df["ip_to"] >= ipn)
    matches = lookup_df.loc[mask]
    if matches.empty:
        return None
    # If multiple matches, pick the most specific range (smallest ip_to - ip_from) if present
    matches = matches.assign(range_size=(matches["ip_to"] - matches["ip_from"]))
    matches = matches.sort_values("range_size")
    return matches.iloc[0]

# --------- Load model & lookup dataframe ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Make sure you trained and saved it.")

pipeline = joblib.load(MODEL_PATH)
print("Loaded pipeline:", MODEL_PATH)

lookup_df = load_lookup_df(CSV_PATH)
print("Loaded lookup table rows:", len(lookup_df))

# Inspect what feature columns the pipeline expects.
# Most pipelines accept DataFrame rows with the same column names used in training.
# We will attempt to infer the expected column set from the ColumnTransformer inside pipeline.
def infer_feature_columns_from_pipeline(pl):
    # Best-effort: look for 'pre' or 'preprocessor' step that is a ColumnTransformer
    if hasattr(pl, "named_steps"):
        for name, step in pl.named_steps.items():
            from sklearn.compose import ColumnTransformer
            if isinstance(step, ColumnTransformer):
                # collect columns from transformers
                cols = []
                for t in step.transformers_:
                    # transformer tuple: (name, transformer, columns)
                    if len(t) >= 3:
                        cols_spec = t[2]
                        # cols_spec may be list, slice, or single string
                        if isinstance(cols_spec, (list, tuple)):
                            cols.extend(list(cols_spec))
                        else:
                            cols.append(cols_spec)
                return [c for c in cols if isinstance(c, str)]
    # fallback: use common features we used for training (update if you trained with different list)
    fallback = ["state", "asn_description", "rdns_hostname", "lat", "lon", "asn", "reachable_flag", "is_rtt_missing"]
    return fallback

# If pipeline is a simple Pipeline wrapping ColumnTransformer under 'pre', adjust accordingly:
feature_cols = None
try:
    # try to find column transformer at pipeline.named_steps["pre"] or ["preprocessor"]
    if hasattr(pipeline, "named_steps"):
        for candidate in ("pre", "preprocessor", "preprocess"):
            if candidate in pipeline.named_steps:
                ct = pipeline.named_steps[candidate]
                if isinstance(ct, type(pipeline.named_steps.get(candidate))):
                    # try to read transformers_ if it's ColumnTransformer
                    from sklearn.compose import ColumnTransformer
                    if isinstance(ct, ColumnTransformer):
                        cols = []
                        for t in ct.transformers_:
                            if len(t) >= 3:
                                cs = t[2]
                                if isinstance(cs, (list, tuple)):
                                    cols.extend(list(cs))
                                else:
                                    cols.append(cs)
                        feature_cols = [c for c in cols if isinstance(c, str)]
                        break
except Exception:
    feature_cols = None

if feature_cols is None:
    # best-effort fallback:
    feature_cols = ["state", "asn_description", "rdns_hostname", "lat", "lon", "asn", "reachable_flag", "is_rtt_missing"]

print("Using feature columns:", feature_cols)

# --------- Build prediction function that takes an IP ----------
def build_feature_row_from_record(rec, feature_cols):
    """Given a matched lookup record (Series) build a DataFrame row matching feature_cols."""
    row = {}
    for c in feature_cols:
        if c in rec.index:
            val = rec[c]
            # numeric columns: coerce where appropriate
            if c in ("lat", "lon", "asn", "reachable_flag", "is_rtt_missing", "ip_numeric", "ip_from", "ip_to"):
                try:
                    row[c] = float(val) if (val is not None and not pd.isna(val)) else np.nan
                except:
                    row[c] = np.nan
            else:
                # categorical/text columns
                if pd.isna(val):
                    row[c] = "__MISSING__"
                else:
                    row[c] = str(val)
        else:
            # column not present in CSV, supply sensible default
            if c in ("lat", "lon", "asn", "reachable_flag", "is_rtt_missing"):
                row[c] = np.nan
            else:
                row[c] = "__MISSING__"
    # Ensure is_rtt_missing computed from local_rtt_ms if not provided
    if "is_rtt_missing" in feature_cols and pd.isna(row.get("is_rtt_missing")):
        row["is_rtt_missing"] = 1 if pd.isna(rec.get("local_rtt_ms")) else 0
    return pd.DataFrame([row], columns=feature_cols)

def predict_from_ip(ip_str, pipeline, lookup_df, topk=5):
    rec = find_ip_record(ip_str, lookup_df)
    if rec is None:
        print(f"No matching IP range found for {ip_str}")
        return None

    X_row = build_feature_row_from_record(rec, feature_cols)

    # Predict city/class
    pred = pipeline.predict(X_row)[0]

    # Try probabilities if available
    proba_list = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(X_row)[0]
            # pipeline.classes_ holds class labels
            classes = pipeline.classes_
            # sort top-k
            idx = np.argsort(probs)[::-1][:topk]
            proba_list = [(classes[i], float(probs[i])) for i in idx]
        except Exception:
            proba_list = None

    # Build output
    out = {
        "ip": ip_str,
        "predicted_city": pred,
        "topk": proba_list,
        "matched_record": {}
    }
    # add some matched record info for user
    for k in ("ip_start", "ip_end", "ip_from", "ip_to", "country", "state", "city", "asn", "asn_description", "rdns_hostname", "local_rtt_ms", "reachable"):
        if k in rec.index:
            out["matched_record"][k] = rec[k]
    return out

# --------- Example usage ----------
# prompt user (or call function directly)
user_ip = input("Enter an IPv4 address (e.g. 1.6.68.129): ").strip()
try:
    result = predict_from_ip(user_ip, pipeline, lookup_df, topk=5)
    if result is not None:
        print("\nPredicted city:", result["predicted_city"])
        if result["topk"] is not None:
            print("Top predictions (city, prob):")
            for c, p in result["topk"]:
                print(f"  {c}: {p:.3f}")
        print("\nMatched DB record (some fields):")
        for k, v in result["matched_record"].items():
            print(f"  {k}: {v}")
except Exception as e:
    print("Error while predicting:", e)

# --------- Notes ----------
# - If you want predicted lat/lon as model outputs, we need to train separate regressors (SVR or LightGBM)
#   and save them similarly (joblib). Right now this returns the lat/lon present in the matched DB row.
# - If your CSV does not contain numeric ip_from/ip_to, the script converts ip_start/ip_end dotted to integers.
# - Ensure the feature_cols above match the exact columns used in training; update the list if you trained with a different set.
