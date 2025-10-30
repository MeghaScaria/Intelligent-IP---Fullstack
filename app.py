from flask import Flask, render_template_string, request
import os
import json
import pandas as pd
import numpy as np
import joblib
import importlib.util
from types import ModuleType


app = Flask(__name__)


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>IP City Prediction - Comparison</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      .row { display: flex; gap: 24px; }
      .card { flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 16px; }
      .muted { color: #666; font-size: 12px; }
      input[type=text] { width: 360px; padding: 8px; }
      button { padding: 8px 16px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { text-align: left; border-bottom: 1px solid #eee; padding: 6px 8px; }
      .topk { font-family: monospace; }
      .err { color: #b00020; }
      .ok { color: #0b7; }
      .hdr { margin-bottom: 12px; }
      .subtitle { color: #333; font-weight: bold; }
      .features { white-space: pre-wrap; font-family: monospace; font-size: 12px; color: #333; background: #fafafa; padding: 8px; border: 1px dashed #ddd; border-radius: 4px; }
    </style>
  </head>
  <body>
    <h2>IP City Prediction - LightGBM vs RBF Approx SGD vs KNN vs Decision Tree</h2>
    <form method="post">
      <label>IP address:</label>
      <input type="text" name="ip" placeholder="e.g. 1.6.68.129" value="{{ ip or '' }}" />
      <button type="submit">Predict</button>
    </form>
    <div class="muted hdr">Models load once on server start. Update accuracy in config if needed.</div>

    {% if error %}
      <div class="err">{{ error }}</div>
    {% endif %}

    {% if lgbm or rbf %}
    <div class="row">
      <div class="card">
        <div class="subtitle">LightGBM</div>
        {% if lgbm.error %}
          <div class="err">{{ lgbm.error }}</div>
        {% else %}
          <div><b>Predicted City:</b> {{ lgbm.predicted_city }}</div>
          <div><b>Confidence (Top-1):</b> {{ '%.3f'|format(lgbm.topk[0].prob if lgbm.topk and lgbm.topk|length>0 else 0.0) }}</div>
          <div><b>Top-k:</b></div>
          <div class="topk">
            {% for item in lgbm.topk %}
              {{ item.city }}: {{ '%.3f'|format(item.prob) }}<br/>
            {% endfor %}
          </div>
          <div><b>Used Fallback:</b> {{ lgbm.used_fallback_no_exact_range }}</div>
          <div><b>Accuracy (config):</b> {{ config.lgbm_accuracy }}</div>
          {% if lgbm_distance_commercial is not none %}
            <div><b>Distance to Commercial Ref:</b> {{ '%.1f'|format(lgbm_distance_commercial) }} km</div>
          {% endif %}
          {% if lgbm_distance_whois is not none %}
            <div><b>Distance to WHOIS Ref:</b> {{ '%.1f'|format(lgbm_distance_whois) }} km</div>
          {% endif %}
          <div style="margin-top:8px;"><b>Features</b></div>
          <div class="features">{{ lgbm.used_features|tojson(indent=2) }}</div>
        {% endif %}
      </div>

      <div class="card">
        <div class="subtitle">RBF Approx SGD</div>
        {% if rbf_error %}
          <div class="err">{{ rbf_error }}</div>
        {% else %}
          <div><b>Predicted City:</b> {{ rbf.predicted_city }}</div>
          <div><b>Confidence (Top-1):</b> {{ '%.3f'|format(rbf.top1_prob if rbf.top1_prob is not none else 0.0) }}</div>
          {% if rbf.topk %}
          <div><b>Top-k:</b></div>
          <div class="topk">
            {% for city, prob in rbf.topk %}
              {{ city }}: {{ '%.3f'|format(prob) }}<br/>
            {% endfor %}
          </div>
          {% endif %}
          <div><b>Accuracy (config):</b> {{ config.rbf_accuracy }}</div>
          {% if rbf_distance_commercial is not none %}
            <div><b>Distance to Commercial Ref:</b> {{ '%.1f'|format(rbf_distance_commercial) }} km</div>
          {% endif %}
          {% if rbf_distance_whois is not none %}
            <div><b>Distance to WHOIS Ref:</b> {{ '%.1f'|format(rbf_distance_whois) }} km</div>
          {% endif %}
          <div style="margin-top:8px;"><b>Matched Record (subset)</b></div>
          <div class="features">{{ rbf.matched_record|tojson(indent=2) }}</div>
        {% endif %}
      </div>
      
      <div class="card">
        <div class="subtitle">K-Nearest Neighbor (KNN) Algorithm</div>
        {% if knn_error %}
          <div class="err">{{ knn_error }}</div>
        {% else %}
          <div><b>Predicted City:</b> {{ knn.predicted_city }}</div>
          <div><b>Confidence (Top-1):</b> {{ '%.3f'|format(knn.top1_prob if knn.top1_prob is not none else 0.0) }}</div>
          {% if knn.topk %}
          <div><b>Top-k:</b></div>
          <div class="topk">
            {% for city, prob in knn.topk %}
              {{ city }}: {{ '%.3f'|format(prob) }}<br/>
            {% endfor %}
          </div>
          {% endif %}
          {% if knn.predicted_lat is not none %}
            <div><b>Predicted Lat (KNN):</b> {{ '%.5f'|format(knn.predicted_lat) }}</div>
          {% endif %}
          {% if knn.predicted_reachable is not none %}
            <div><b>Predicted Reachable (KNN):</b> {{ knn.predicted_reachable }}</div>
          {% endif %}
          {% if knn_distance_commercial is not none %}
            <div><b>Distance to Commercial Ref:</b> {{ '%.1f'|format(knn_distance_commercial) }} km</div>
          {% endif %}
          {% if knn_distance_whois is not none %}
            <div><b>Distance to WHOIS Ref:</b> {{ '%.1f'|format(knn_distance_whois) }} km</div>
          {% endif %}
          <div style="margin-top:8px;"><b>Used Features</b></div>
          <div class="features">{{ knn.used_features|tojson(indent=2) }}</div>
        {% endif %}
      </div>
      
      <div class="card">
        <div class="subtitle">Decision Tree</div>
        {% if dt_error %}
          <div class="err">{{ dt_error }}</div>
        {% else %}
          <div><b>Predicted City:</b> {{ dt.predicted_city }}</div>
          {% if dt.topk %}
          <div><b>Top-k:</b></div>
          <div class="topk">
            {% for city, prob in dt.topk %}
              {{ city }}: {{ '%.3f'|format(prob) }}<br/>
            {% endfor %}
          </div>
          {% endif %}
          {% if dt.predicted_lat is not none %}
            <div><b>Predicted Lat (DT):</b> {{ '%.5f'|format(dt.predicted_lat) }}</div>
          {% endif %}
          {% if dt.predicted_lon is not none %}
            <div><b>Predicted Lon (DT):</b> {{ '%.5f'|format(dt.predicted_lon) }}</div>
          {% endif %}
          {% if dt.predicted_reachable is not none %}
            <div><b>Predicted Reachable (DT):</b> {{ dt.predicted_reachable }}</div>
          {% endif %}
          {% if dt_distance_commercial is not none %}
            <div><b>Distance to Commercial Ref:</b> {{ '%.1f'|format(dt_distance_commercial) }} km</div>
          {% endif %}
          {% if dt_distance_whois is not none %}
            <div><b>Distance to WHOIS Ref:</b> {{ '%.1f'|format(dt_distance_whois) }} km</div>
          {% endif %}
          <div style="margin-top:8px;"><b>Used Features</b></div>
          <div class="features">{{ dt.used_features|tojson(indent=2) }}</div>
        {% endif %}
      </div>
    </div>

    
    {% endif %}
  </body>
  </html>
"""


# --- Configurable paths and (optional) accuracy display ---
MODEL_LGBM = os.environ.get("MODEL_LGBM", "3.city_lgbm.txt")
LABEL_LGBM = os.environ.get("LABEL_LGBM", "3.city_label_encoder.joblib")
PREPROC_LGBM = os.environ.get("PREPROC_LGBM", "3..joblib")

MODEL_RBF = os.environ.get("MODEL_RBF", "2.RBF_city_rbf_approx_sgd_pipeline.joblib")
CSV_LOOKUP = os.environ.get("CSV_LOOKUP", "new_indian_training_data.csv")

# KNN artifacts
KNN_CITY = os.environ.get("KNN_CITY", "4.knn_city.joblib")
KNN_LABEL = os.environ.get("KNN_LABEL", "4.label_encoder.joblib")
KNN_PREPROC = os.environ.get("KNN_PREPROC", "4.preproc.joblib")
KNN_LAT = os.environ.get("KNN_LAT", "4.knn_lat.joblib")
KNN_REACH = os.environ.get("KNN_REACH", "4.knn_reach.joblib")

# Decision Tree artifacts
DT_CITY = os.environ.get("DT_CITY", "5.dt_city.joblib")
DT_LAT  = os.environ.get("DT_LAT",  "5.dt_lat.joblib")
DT_LON  = os.environ.get("DT_LON",  "5.dt_lon.joblib")
DT_REACH= os.environ.get("DT_REACH","5.dt_reach.joblib")
DT_LABEL= os.environ.get("DT_LABEL","5.label_encoder_city.joblib")
DT_PRE  = os.environ.get("DT_PRE",  "5.preproc_objects.joblib")

CONFIG = {
    "lgbm_accuracy": os.environ.get("LGBM_ACCURACY", "N/A"),
    "rbf_accuracy": os.environ.get("RBF_ACCURACY", "N/A"),
}


# --- Utility: make objects JSON-serializable (convert numpy/pandas types) ---
def _to_serializable(obj):
    import numpy as _np
    import pandas as _pd

    if obj is None:
        return None
    # numpy scalars
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    # pandas scalars
    if isinstance(obj, (_pd.Timestamp,)):
        return obj.isoformat()
    # sequences
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [ _to_serializable(x) for x in obj ]
    # mappings
    if isinstance(obj, dict):
        return { str(k): _to_serializable(v) for k, v in obj.items() }
    # pandas Series/DataFrame
    if isinstance(obj, _pd.Series):
        return { str(k): _to_serializable(v) for k, v in obj.items() }
    if isinstance(obj, _pd.DataFrame):
        return [ _to_serializable(rec) for rec in obj.to_dict(orient="records") ]
    # fallback
    return obj


# --- Dynamically load LightGBM inference from file (filename starts with digit) ---
def _load_lgbm_predict_func():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "3.run_lgbm_ip_inference.py")
    if not os.path.exists(script_path):
        return None
    spec = importlib.util.spec_from_file_location("lgbm_infer", script_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)  # type: ModuleType
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        return None
    return getattr(module, "predict_for_ip", None)

lgbm_predict = _load_lgbm_predict_func()


# --- Load RBF pipeline and lookup once ---
try:
    rbf_pipeline = joblib.load(MODEL_RBF) if os.path.exists(MODEL_RBF) else None
except Exception:
    rbf_pipeline = None

def load_lookup_df(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)
        for col in ("ip_from", "ip_to", "ip_numeric"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if (("ip_from" not in df.columns or df["ip_from"].isna().all()) and
            ("ip_start" in df.columns and "ip_end" in df.columns)):
            import ipaddress
            def conv(s):
                try:
                    return int(ipaddress.ip_address(str(s)))
                except Exception:
                    return np.nan
            df["ip_from"] = df["ip_start"].apply(conv)
            df["ip_to"] = df["ip_end"].apply(conv)
        if {"ip_from", "ip_to"}.issubset(df.columns):
            df = df.dropna(subset=["ip_from", "ip_to"]).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

LOOKUP_DF = load_lookup_df(CSV_LOOKUP)

# Precompute simple state centroids from lookup as a WHOIS proxy
STATE_CENTROIDS = {}
try:
    if not LOOKUP_DF.empty and {"state","lat","lon"}.issubset(LOOKUP_DF.columns):
        grp = LOOKUP_DF.dropna(subset=["lat","lon"]).groupby("state")
        cent = grp[["lat","lon"]].mean().reset_index()
        STATE_CENTROIDS = { str(r["state"]): (float(r["lat"]), float(r["lon"])) for _, r in cent.iterrows() }
except Exception:
    STATE_CENTROIDS = {}


def rbf_predict_from_ip(ip_str: str, topk: int = 5):
    if rbf_pipeline is None or LOOKUP_DF.empty:
        return None, "RBF pipeline or lookup not available"

    # Find matching record
    import ipaddress
    try:
        ipn = int(ipaddress.ip_address(ip_str))
    except Exception:
        return None, "Invalid IP"
    mask = (LOOKUP_DF["ip_from"] <= ipn) & (LOOKUP_DF["ip_to"] >= ipn)
    matches = LOOKUP_DF.loc[mask]
    if matches.empty:
        return None, "No matching IP range"
    rec = matches.assign(range_size=(matches["ip_to"] - matches["ip_from"]))\
                 .sort_values("range_size").iloc[0]

    # Infer feature columns from the pipeline if possible
    feature_cols = None
    try:
        if hasattr(rbf_pipeline, "named_steps"):
            from sklearn.compose import ColumnTransformer
            for name, step in rbf_pipeline.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    cols = []
                    for t in step.transformers_:
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
        feature_cols = ["state", "asn_description", "rdns_hostname", "lat", "lon", "asn", "reachable_flag", "is_rtt_missing"]

    # Build a feature row
    row = {}
    for c in feature_cols:
        if c in rec.index:
            val = rec[c]
            if c in ("lat", "lon", "asn", "reachable_flag", "is_rtt_missing", "ip_numeric", "ip_from", "ip_to"):
                try:
                    row[c] = float(val) if (val is not None and not pd.isna(val)) else np.nan
                except Exception:
                    row[c] = np.nan
            else:
                row[c] = "__MISSING__" if pd.isna(val) else str(val)
        else:
            row[c] = np.nan if c in ("lat", "lon", "asn", "reachable_flag", "is_rtt_missing") else "__MISSING__"
    if "is_rtt_missing" in feature_cols and pd.isna(row.get("is_rtt_missing")):
        row["is_rtt_missing"] = 1 if pd.isna(rec.get("local_rtt_ms")) else 0
    X = pd.DataFrame([row], columns=feature_cols)

    pred_city = rbf_pipeline.predict(X)[0]
    topk_list = None
    top1_prob = None
    if hasattr(rbf_pipeline, "predict_proba"):
        try:
            probs = rbf_pipeline.predict_proba(X)[0]
            classes = getattr(rbf_pipeline, "classes_", None)
            if classes is not None:
                idx = np.argsort(probs)[::-1][:topk]
                topk_list = [(classes[i], float(probs[i])) for i in idx]
                top1_prob = float(probs[idx[0]]) if len(idx) else None
        except Exception:
            pass

    matched_rec_subset = {}
    for k in ("ip_start", "ip_end", "ip_from", "ip_to", "country", "state", "city", "asn", "asn_description", "rdns_hostname", "local_rtt_ms", "reachable"):
        if k in rec.index:
            matched_rec_subset[k] = rec[k]

    return {
        "ip": ip_str,
        "predicted_city": pred_city,
        "topk": topk_list,
        "top1_prob": top1_prob,
        "matched_record": matched_rec_subset,
    }, None


def _build_base_features_for_ip_record(rec: pd.Series, ip_str: str) -> pd.DataFrame:
    # Use the same scheme as LightGBM features
    TEXT_RDNS = "rdns_hostname"
    TEXT_ASND = "asn_description"
    NUMERIC = ["ip_mid", "asn", "lat", "lon", "local_rtt_ms", "is_rtt_missing", "reachable_flag"]
    CATEGORICAL = ["state"]

    if rec is None:
        # fallback minimal features using ip_mid only
        try:
            import ipaddress as _ip
            ipn = int(_ip.ip_address(ip_str))
        except Exception:
            return pd.DataFrame()
        feat = {}
        for c in NUMERIC + CATEGORICAL + [TEXT_RDNS, TEXT_ASND]:
            if c in NUMERIC:
                feat[c] = float(ipn) if c == "ip_mid" else np.nan
            elif c in CATEGORICAL:
                feat[c] = "__MISSING__"
            else:
                feat[c] = ""
        return pd.DataFrame([feat])[NUMERIC + CATEGORICAL + [TEXT_RDNS, TEXT_ASND]]

    # construct from record
    ip_from = rec.get("ip_from") if "ip_from" in rec else np.nan
    ip_to = rec.get("ip_to") if "ip_to" in rec else np.nan
    ip_numeric = rec.get("ip_numeric") if "ip_numeric" in rec else np.nan
    if pd.isna(ip_from) and not pd.isna(ip_numeric):
        ip_from = ip_numeric
    if pd.isna(ip_to) and not pd.isna(ip_numeric):
        ip_to = ip_numeric
    feat = {
        "ip_mid": float((pd.to_numeric(ip_from, errors="coerce") + pd.to_numeric(ip_to, errors="coerce")) / 2.0) if not (pd.isna(ip_from) and pd.isna(ip_to)) else np.nan,
        "asn": float(rec.get("asn")) if not pd.isna(rec.get("asn")) else np.nan,
        "lat": float(rec.get("lat")) if not pd.isna(rec.get("lat")) else np.nan,
        "lon": float(rec.get("lon")) if not pd.isna(rec.get("lon")) else np.nan,
        "local_rtt_ms": float(rec.get("local_rtt_ms")) if not pd.isna(rec.get("local_rtt_ms")) else np.nan,
        "is_rtt_missing": 1 if pd.isna(rec.get("local_rtt_ms")) else 0,
        "reachable_flag": 1 if str(rec.get("reachable", "")).upper() == "TRUE" else 0,
        "state": rec.get("state", "__MISSING__") if not pd.isna(rec.get("state")) else "__MISSING__",
        "rdns_hostname": rec.get("rdns_hostname", "") if not pd.isna(rec.get("rdns_hostname")) else "",
        "asn_description": rec.get("asn_description", "") if not pd.isna(rec.get("asn_description")) else "",
    }
    cols_order = ["ip_mid", "asn", "lat", "lon", "local_rtt_ms", "is_rtt_missing", "reachable_flag", "state", "rdns_hostname", "asn_description"]
    return pd.DataFrame([feat])[cols_order]


def _transform_with_preproc(preproc_obj, Xrow: pd.DataFrame):
    # If this is a full sklearn Pipeline/Transformer, use it directly
    if hasattr(preproc_obj, "transform"):
        try:
            return preproc_obj.transform(Xrow)
        except Exception:
            pass
    # Else expect dict parts like the LightGBM preproc
    try:
        # Read dynamic column lists from preproc dict
        text_cols = preproc_obj.get("text_cols", ["rdns_hostname","asn_description"])
        numeric_cols = preproc_obj.get("numeric_cols", ["ip_mid","asn","local_rtt_ms","is_rtt_missing","reachable_flag"])
        categorical_cols = preproc_obj.get("categorical_cols", ["state"])

        tf_rdns = preproc_obj["tfidf_rdns"].transform(Xrow[text_cols[0]])
        tf_asnd = preproc_obj["tfidf_asnd"].transform(Xrow[text_cols[1]])
        Xnum = preproc_obj["num_imputer"].transform(Xrow[numeric_cols])
        Xnum = preproc_obj["scaler"].transform(Xnum)
        # ohe_state is key used by KNN training artifacts
        ohe = preproc_obj.get("ohe_state") or preproc_obj.get("onehot")
        Xcat = ohe.transform(Xrow[categorical_cols]) if ohe is not None else None
        from scipy.sparse import hstack, csr_matrix
        parts = [csr_matrix(tf_rdns), csr_matrix(tf_asnd), csr_matrix(Xnum)]
        if Xcat is not None:
            parts.append(csr_matrix(Xcat))
        return hstack(parts).tocsr()
    except Exception:
        return None


def knn_predict_from_ip(ip_str: str, topk: int = 5):
    # Ensure artifacts exist
    if not (os.path.exists(KNN_CITY) and os.path.exists(KNN_LABEL) and os.path.exists(KNN_PREPROC)):
        return None, "KNN artifacts not available"

    # Load artifacts lazily
    try:
        knn_city = joblib.load(KNN_CITY)
        knn_label = joblib.load(KNN_LABEL)
        knn_pre = joblib.load(KNN_PREPROC)
        knn_lat = joblib.load(KNN_LAT) if os.path.exists(KNN_LAT) else None
        knn_reach = joblib.load(KNN_REACH) if os.path.exists(KNN_REACH) else None
    except Exception as e:
        return None, f"Failed to load KNN artifacts: {e}"

    # Lookup matching record
    import ipaddress
    try:
        ipn = int(ipaddress.ip_address(ip_str))
    except Exception:
        return None, "Invalid IP"
    rec = None
    if not LOOKUP_DF.empty and {"ip_from","ip_to"}.issubset(LOOKUP_DF.columns):
        mask = (LOOKUP_DF["ip_from"] <= ipn) & (LOOKUP_DF["ip_to"] >= ipn)
        matches = LOOKUP_DF.loc[mask]
        if not matches.empty:
            rec = matches.assign(range_size=(matches["ip_to"] - matches["ip_from"]))\
                       .sort_values("range_size").iloc[0]

    Xrow = _build_base_features_for_ip_record(rec, ip_str)
    if Xrow.empty:
        return None, "Failed to build features for IP"

    Xfull = _transform_with_preproc(knn_pre, Xrow)
    if Xfull is None:
        return None, "KNN preprocessor could not transform features"

    # City prediction
    try:
        pred_idx = int(knn_city.predict(Xfull)[0])
        pred_city = knn_label.inverse_transform([pred_idx])[0] if hasattr(knn_label, "inverse_transform") else str(pred_idx)
    except Exception as e:
        return None, f"KNN city prediction failed: {e}"

    # Probabilities/topk if available
    topk_list = None
    top1_prob = None
    if hasattr(knn_city, "predict_proba"):
        try:
            probs = knn_city.predict_proba(Xfull)[0]
            classes = getattr(knn_city, "classes_", None)
            if classes is not None:
                # Map class indices back to label encoder indices if needed
                idx_sorted = np.argsort(probs)[::-1][:topk]
                # Convert class labels (which should be encoded ints) to city names via label encoder
                cities = knn_label.inverse_transform(classes[idx_sorted].astype(int)) if hasattr(knn_label, "inverse_transform") else classes[idx_sorted]
                topk_list = [(str(cities[i]), float(probs[idx_sorted][i])) for i in range(len(idx_sorted))]
                top1_prob = float(probs[idx_sorted][0]) if len(idx_sorted) else None
        except Exception:
            pass

    # Optional regress/classify aux targets
    pred_lat = None
    if knn_lat is not None:
        try:
            pred_lat = float(knn_lat.predict(Xfull)[0])
        except Exception:
            pred_lat = None
    pred_reach = None
    if knn_reach is not None:
        try:
            pred_reach = int(knn_reach.predict(Xfull)[0])
        except Exception:
            pred_reach = None

    used_features = Xrow.iloc[0].to_dict()
    used_features = _to_serializable(used_features)

    return {
        "ip": ip_str,
        "predicted_city": pred_city,
        "topk": topk_list,
        "top1_prob": top1_prob,
        "predicted_lat": pred_lat,
        "predicted_reachable": pred_reach,
        "used_features": used_features,
    }, None


def dt_predict_from_ip(ip_str: str, topk: int = 5):
    # Ensure artifacts exist
    needed = [DT_CITY, DT_LAT, DT_LON, DT_REACH, DT_LABEL, DT_PRE]
    if not all(os.path.exists(p) for p in needed):
        return None, "Decision Tree artifacts not available"

    # Load artifacts lazily
    try:
        dt_city = joblib.load(DT_CITY)
        dt_lat  = joblib.load(DT_LAT)
        dt_lon  = joblib.load(DT_LON)
        dt_reach= joblib.load(DT_REACH)
        dt_label= joblib.load(DT_LABEL)
        dt_pre  = joblib.load(DT_PRE)
    except Exception as e:
        return None, f"Failed to load Decision Tree artifacts: {e}"

    # Lookup matching record
    import ipaddress
    try:
        ipn = int(ipaddress.ip_address(ip_str))
    except Exception:
        return None, "Invalid IP"
    rec = None
    if not LOOKUP_DF.empty and {"ip_from","ip_to"}.issubset(LOOKUP_DF.columns):
        mask = (LOOKUP_DF["ip_from"] <= ipn) & (LOOKUP_DF["ip_to"] >= ipn)
        matches = LOOKUP_DF.loc[mask]
        if not matches.empty:
            rec = matches.assign(range_size=(matches["ip_to"] - matches["ip_from"]))\
                   .sort_values("range_size").iloc[0]

    # Build row with dt_pre column lists
    feat = {}
    # text
    for t in dt_pre.get("text_cols", ["rdns_hostname","asn_description"]):
        v = rec.get(t, "") if (rec is not None and t in rec.index) else ""
        feat[t] = v if not pd.isna(v) else ""
    # numeric
    for n in dt_pre.get("numeric_cols", ["ip_mid","asn","lat","lon","local_rtt_ms","is_rtt_missing","reachable_flag"]):
        if rec is not None and n in rec.index:
            try:
                feat[n] = float(rec[n]) if not pd.isna(rec[n]) else np.nan
            except Exception:
                feat[n] = np.nan
        else:
            feat[n] = np.nan
    # categorical
    for c in dt_pre.get("categorical_cols", ["state"]):
        v = rec.get(c, "__MISSING__") if (rec is not None and c in rec.index) else "__MISSING__"
        feat[c] = v if not pd.isna(v) else "__MISSING__"
    # ensure ip_mid
    if pd.isna(feat.get("ip_mid", None)):
        ipf = rec.get("ip_from", np.nan) if (rec is not None and "ip_from" in rec.index) else np.nan
        ipt = rec.get("ip_to", np.nan) if (rec is not None and "ip_to" in rec.index) else np.nan
        ipn2 = rec.get("ip_numeric", np.nan) if (rec is not None and "ip_numeric" in rec.index) else np.nan
        ipf_val = ipf if not pd.isna(ipf) else ipn2
        ipt_val = ipt if not pd.isna(ipt) else ipn2
        if not pd.isna(ipf_val) or not pd.isna(ipt_val):
            try:
                feat["ip_mid"] = float(((ipf_val if not pd.isna(ipf_val) else ipn2) + (ipt_val if not pd.isna(ipt_val) else ipn2)) / 2.0)
            except Exception:
                feat["ip_mid"] = np.nan
    cols = dt_pre.get("text_cols", []) + dt_pre.get("numeric_cols", []) + dt_pre.get("categorical_cols", [])
    row_df = pd.DataFrame([feat])[cols]

    # Transform to model input (dense)
    try:
        tf_rdns = dt_pre["tfidf_rdns"].transform(row_df[dt_pre["text_cols"][0]])
        tf_asnd = dt_pre["tfidf_asnd"].transform(row_df[dt_pre["text_cols"][1]])
        num_arr = dt_pre["num_imputer"].transform(row_df[dt_pre["numeric_cols"]])
        ohe = dt_pre.get("ohe_state")
        cat_arr = ohe.transform(row_df[dt_pre["categorical_cols"]]) if ohe is not None else None
        from scipy.sparse import hstack, csr_matrix
        parts = [csr_matrix(tf_rdns), csr_matrix(tf_asnd), csr_matrix(num_arr)]
        if cat_arr is not None:
            parts.append(csr_matrix(cat_arr))
        Xstack = hstack(parts).tocsr()
        Xin = Xstack.toarray()
    except Exception as e:
        return None, f"DT preprocessing failed: {e}"

    # Predictions
    try:
        city_idx = int(dt_city.predict(Xin)[0])
        city_name = dt_label.inverse_transform([city_idx])[0] if hasattr(dt_label, "inverse_transform") else str(city_idx)
    except Exception as e:
        return None, f"DT city prediction failed: {e}"

    topk_list = None
    try:
        probs = dt_city.predict_proba(Xin)[0]
        idx = np.argsort(probs)[::-1][:topk]
        topk_list = [(dt_label.inverse_transform([int(i)])[0], float(probs[i])) for i in idx]
    except Exception:
        pass

    pred_lat = None
    pred_lon = None
    pred_reach = None
    try:
        pred_lat = float(dt_lat.predict(Xin)[0])
    except Exception:
        pass
    try:
        pred_lon = float(dt_lon.predict(Xin)[0])
    except Exception:
        pass
    try:
        pred_reach = bool(int(dt_reach.predict(Xin)[0]))
    except Exception:
        pass

    used_features = row_df.iloc[0].to_dict()
    used_features = _to_serializable(used_features)

    return {
        "ip": ip_str,
        "predicted_city": city_name,
        "topk": topk_list,
        "predicted_lat": pred_lat,
        "predicted_lon": pred_lon,
        "predicted_reachable": pred_reach,
        "used_features": used_features,
    }, None
@app.route("/", methods=["GET", "POST"])
def index():
    ip = None
    error = None
    lgbm = None
    rbf = None
    rbf_error = None
    knn = None
    knn_error = None
    dt = None
    dt_error = None
    # per-model distances
    lgbm_distance_commercial = None
    lgbm_distance_whois = None
    rbf_distance_commercial = None
    rbf_distance_whois = None
    knn_distance_commercial = None
    knn_distance_whois = None
    dt_distance_commercial = None
    dt_distance_whois = None

    if request.method == "POST":
        ip = (request.form.get("ip") or "").strip()
        if not ip:
            error = "Please enter an IP address"
        else:
            if lgbm_predict is None:
                lgbm = {"error": "LightGBM inference script not available"}
            else:
                try:
                    lgbm = lgbm_predict(
                        ip=ip,
                        model_path=MODEL_LGBM,
                        label_encoder_path=LABEL_LGBM,
                        preproc_path=PREPROC_LGBM,
                        csv_lookup_path=CSV_LOOKUP,
                        topk=5,
                    )
                except Exception as e:
                    lgbm = {"error": str(e)}

            rbf, rbf_error = rbf_predict_from_ip(ip, topk=5)
            knn, knn_error = knn_predict_from_ip(ip, topk=5)
            dt, dt_error = dt_predict_from_ip(ip, topk=5)

            # Sanitize objects for Jinja tojson (convert numpy/pandas types)
            if isinstance(lgbm, dict):
                lgbm = _to_serializable(lgbm)
            if isinstance(rbf, dict):
                rbf = _to_serializable(rbf)
            if isinstance(knn, dict):
                knn = _to_serializable(knn)
            if isinstance(dt, dict):
                dt = _to_serializable(dt)

            # Helper: haversine in KM
            def _haversine_km(lat1, lon1, lat2, lon2):
                from math import radians, sin, cos, asin, sqrt
                try:
                    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
                except Exception:
                    return None
                R = 6371.0
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                return R * c

            # Find reference record for this IP (commercial proxy)
            ref_rec = None
            try:
                import ipaddress as _ip
                ipn = int(_ip.ip_address(ip))
                if not LOOKUP_DF.empty and {"ip_from","ip_to"}.issubset(LOOKUP_DF.columns):
                    m = (LOOKUP_DF["ip_from"] <= ipn) & (LOOKUP_DF["ip_to"] >= ipn)
                    hits = LOOKUP_DF.loc[m]
                    if not hits.empty:
                        ref_rec = hits.assign(range_size=(hits["ip_to"] - hits["ip_from"]))\
                                  .sort_values("range_size").iloc[0]
            except Exception:
                ref_rec = None

            # Prepare references
            ref_lat = float(ref_rec.get("lat")) if (ref_rec is not None and not pd.isna(ref_rec.get("lat"))) else None
            ref_lon = float(ref_rec.get("lon")) if (ref_rec is not None and not pd.isna(ref_rec.get("lon"))) else None
            whois_lat = None
            whois_lon = None
            st = None
            if ref_rec is not None and not pd.isna(ref_rec.get("state")):
                st = str(ref_rec.get("state"))
            elif isinstance(rbf, dict) and rbf.get("matched_record") and rbf["matched_record"].get("state"):
                st = str(rbf["matched_record"]["state"])
            if st and st in STATE_CENTROIDS:
                whois_lat, whois_lon = STATE_CENTROIDS[st]

            # Estimate model lat/lon where possible
            # KNN has explicit lat
            knn_lat = knn.get("predicted_lat") if isinstance(knn, dict) else None
            knn_lon = knn.get("predicted_lon") if isinstance(knn, dict) else None
            # Decision Tree has explicit lat/lon
            dt_lat = dt.get("predicted_lat") if isinstance(dt, dict) else None
            dt_lon = dt.get("predicted_lon") if isinstance(dt, dict) else None
            # LightGBM/RBF: no lat predictor; approximate with ref_rec lat/lon if matched
            # This still gives distance 0 to commercial reference; for WHOIS, we use centroid distance
            lgbm_lat = ref_lat if isinstance(lgbm, dict) else None
            lgbm_lon = ref_lon if isinstance(lgbm, dict) else None
            rbf_lat = ref_lat if isinstance(rbf, dict) else None
            rbf_lon = ref_lon if isinstance(rbf, dict) else None

            # Compute distances
            if knn_lat is not None and ref_lat is not None and (knn_lon is not None or ref_lon is not None):
                use_lon = knn_lon if knn_lon is not None else ref_lon
                knn_distance_commercial = _haversine_km(knn_lat, use_lon, ref_lat, ref_lon)
            if knn_lat is not None and whois_lat is not None and (knn_lon is not None or whois_lon is not None):
                use_lon = knn_lon if knn_lon is not None else whois_lon
                knn_distance_whois = _haversine_km(knn_lat, use_lon, whois_lat, whois_lon)

            if lgbm_lat is not None and ref_lat is not None and lgbm_lon is not None and ref_lon is not None:
                lgbm_distance_commercial = _haversine_km(lgbm_lat, lgbm_lon, ref_lat, ref_lon)
            if lgbm_lat is not None and whois_lat is not None and lgbm_lon is not None and whois_lon is not None:
                lgbm_distance_whois = _haversine_km(lgbm_lat, lgbm_lon, whois_lat, whois_lon)

            if rbf_lat is not None and ref_lat is not None and rbf_lon is not None and ref_lon is not None:
                rbf_distance_commercial = _haversine_km(rbf_lat, rbf_lon, ref_lat, ref_lon)
            if rbf_lat is not None and whois_lat is not None and rbf_lon is not None and whois_lon is not None:
                rbf_distance_whois = _haversine_km(rbf_lat, rbf_lon, whois_lat, whois_lon)

            # DT distances
            dt_distance_commercial = None
            dt_distance_whois = None
            if dt_lat is not None and ref_lat is not None and (dt_lon is not None or ref_lon is not None):
                use_lon = dt_lon if dt_lon is not None else ref_lon
                dt_distance_commercial = _haversine_km(dt_lat, use_lon, ref_lat, ref_lon)
            if dt_lat is not None and whois_lat is not None and (dt_lon is not None or whois_lon is not None):
                use_lon = dt_lon if dt_lon is not None else whois_lon
                dt_distance_whois = _haversine_km(dt_lat, use_lon, whois_lat, whois_lon)

    return render_template_string(TEMPLATE, ip=ip, lgbm=lgbm, rbf=rbf, rbf_error=rbf_error, knn=knn, knn_error=knn_error, dt=dt, dt_error=dt_error, lgbm_distance_commercial=lgbm_distance_commercial, lgbm_distance_whois=lgbm_distance_whois, rbf_distance_commercial=rbf_distance_commercial, rbf_distance_whois=rbf_distance_whois, knn_distance_commercial=knn_distance_commercial, knn_distance_whois=knn_distance_whois, dt_distance_commercial=dt_distance_commercial, dt_distance_whois=dt_distance_whois, error=error, config=CONFIG)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)


