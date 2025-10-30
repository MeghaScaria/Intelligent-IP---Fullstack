import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import ipaddress
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix


def ip_to_int_safe(ip_str: str) -> Optional[int]:
    try:
        return int(ipaddress.ip_address(ip_str))
    except Exception:
        return None


def dotted_to_int_safe(x: Any) -> Optional[int]:
    try:
        return int(ipaddress.ip_address(str(x)))
    except Exception:
        return None


def load_lookup_dataframe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path, low_memory=False)
    df.replace({"-": np.nan, "no_rdns": np.nan, "": np.nan}, inplace=True)

    # Coerce numeric ip range columns if present; otherwise try dotted fallbacks
    for col in ("ip_from", "ip_to", "ip_numeric"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if ("ip_from" not in df.columns or df["ip_from"].isna().all()) and ("ip_start" in df.columns):
        df["ip_from"] = df["ip_start"].apply(dotted_to_int_safe)
        df["ip_to"] = df.get("ip_end", np.nan)
        if "ip_to" in df.columns:
            df["ip_to"] = df["ip_to"].apply(dotted_to_int_safe)

    # Keep only rows with valid ranges if we have both bounds
    if {"ip_from", "ip_to"}.issubset(df.columns):
        df = df.dropna(subset=["ip_from", "ip_to"]).reset_index(drop=True)

    return df


def find_record_for_ip(ip_str: str, lookup_df: pd.DataFrame) -> Optional[pd.Series]:
    if lookup_df is None or lookup_df.empty:
        return None
    ipn = ip_to_int_safe(ip_str)
    if ipn is None:
        return None
    if not {"ip_from", "ip_to"}.issubset(lookup_df.columns):
        return None
    mask = (lookup_df["ip_from"] <= ipn) & (lookup_df["ip_to"] >= ipn)
    hits = lookup_df.loc[mask]
    if hits.empty:
        return None
    hits = hits.assign(range_size=(hits["ip_to"] - hits["ip_from"]))
    return hits.sort_values("range_size").iloc[0]


def build_features(rec_or_ip: Any) -> pd.DataFrame:
    TEXT_RDNS = "rdns_hostname"
    TEXT_ASND = "asn_description"
    NUMERIC = [
        "ip_mid",
        "asn",
        "lat",
        "lon",
        "local_rtt_ms",
        "is_rtt_missing",
        "reachable_flag",
    ]
    CATEGORICAL = ["state"]

    if isinstance(rec_or_ip, str):
        ipn = ip_to_int_safe(rec_or_ip)
        if ipn is None:
            return pd.DataFrame()
        feat: Dict[str, Any] = {}
        for c in NUMERIC + CATEGORICAL + [TEXT_RDNS, TEXT_ASND]:
            if c in NUMERIC:
                feat[c] = float(ipn) if c == "ip_mid" else np.nan
            elif c in CATEGORICAL:
                feat[c] = "__MISSING__"
            else:
                feat[c] = ""
        return pd.DataFrame([feat])[NUMERIC + CATEGORICAL + [TEXT_RDNS, TEXT_ASND]]

    rec = rec_or_ip
    feat = {}
    # numeric
    ip_from = rec.get("ip_from") if "ip_from" in rec else np.nan
    ip_to = rec.get("ip_to") if "ip_to" in rec else np.nan
    ip_numeric = rec.get("ip_numeric") if "ip_numeric" in rec else np.nan
    if pd.isna(ip_from) and not pd.isna(ip_numeric):
        ip_from = ip_numeric
    if pd.isna(ip_to) and not pd.isna(ip_numeric):
        ip_to = ip_numeric
    feat["ip_mid"] = float((pd.to_numeric(ip_from, errors="coerce") + pd.to_numeric(ip_to, errors="coerce")) / 2.0) if not (pd.isna(ip_from) and pd.isna(ip_to)) else np.nan
    feat["asn"] = float(rec.get("asn")) if not pd.isna(rec.get("asn")) else np.nan
    feat["lat"] = float(rec.get("lat")) if not pd.isna(rec.get("lat")) else np.nan
    feat["lon"] = float(rec.get("lon")) if not pd.isna(rec.get("lon")) else np.nan
    feat["local_rtt_ms"] = float(rec.get("local_rtt_ms")) if not pd.isna(rec.get("local_rtt_ms")) else np.nan
    feat["is_rtt_missing"] = 1 if pd.isna(rec.get("local_rtt_ms")) else 0
    feat["reachable_flag"] = 1 if str(rec.get("reachable", "")).upper() == "TRUE" else 0
    # categorical/text
    feat["state"] = rec.get("state", "__MISSING__") if not pd.isna(rec.get("state")) else "__MISSING__"
    feat["rdns_hostname"] = rec.get("rdns_hostname", "") if not pd.isna(rec.get("rdns_hostname")) else ""
    feat["asn_description"] = rec.get("asn_description", "") if not pd.isna(rec.get("asn_description")) else ""

    cols_order = [
        "ip_mid",
        "asn",
        "lat",
        "lon",
        "local_rtt_ms",
        "is_rtt_missing",
        "reachable_flag",
        "state",
        "rdns_hostname",
        "asn_description",
    ]
    return pd.DataFrame([feat])[cols_order]


def predict_for_ip(
    ip: str,
    model_path: str,
    label_encoder_path: str,
    preproc_path: str,
    csv_lookup_path: Optional[str] = None,
    topk: int = 5,
) -> Dict[str, Any]:
    # Lazy import lightgbm to avoid dependency issues until needed
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:
        raise RuntimeError("LightGBM is required to run inference. Please install it.") from e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessing file not found: {preproc_path}")

    le = joblib.load(label_encoder_path)
    preproc = joblib.load(preproc_path)
    booster = lgb.Booster(model_file=model_path)

    lookup_df = load_lookup_dataframe(csv_lookup_path) if csv_lookup_path else pd.DataFrame()
    rec = find_record_for_ip(ip, lookup_df)
    used_fallback = rec is None

    Xrow = build_features(rec if rec is not None else ip)
    if Xrow.empty:
        return {"error": "invalid or unsupported IP", "ip": ip}

    # Names used in training
    TEXT_RDNS = "rdns_hostname"
    TEXT_ASND = "asn_description"
    NUMERIC = ["ip_mid", "asn", "lat", "lon", "local_rtt_ms", "is_rtt_missing", "reachable_flag"]
    CATEGORICAL = ["state"]

    # Transform using saved preprocessors
    tf_rdns = preproc["tfidf_rdns"].transform(Xrow[TEXT_RDNS])
    tf_asnd = preproc["tfidf_asnd"].transform(Xrow[TEXT_ASND])
    Xnum = preproc["num_imputer"].transform(Xrow[NUMERIC])
    Xnum = preproc["scaler"].transform(Xnum)
    Xcat = preproc["onehot"].transform(Xrow[CATEGORICAL])
    Xfull = hstack([csr_matrix(tf_rdns), csr_matrix(tf_asnd), csr_matrix(Xnum), csr_matrix(Xcat)]).tocsr()

    probs = booster.predict(Xfull, num_iteration=booster.best_iteration)
    probs_row = probs[0]
    top_idx = np.argsort(probs_row)[::-1][: max(1, topk)]
    topk_list = [
        {"city": le.inverse_transform([int(i)])[0], "prob": float(probs_row[i])}
        for i in top_idx
    ]
    top1_city = le.inverse_transform([int(np.argmax(probs_row))])[0]

    used_features = Xrow.iloc[0].to_dict()
    matched_record = rec.to_dict() if isinstance(rec, pd.Series) else None

    return {
        "ip": ip,
        "predicted_city": top1_city,
        "topk": topk_list,
        "used_fallback_no_exact_range": used_fallback,
        "used_features": used_features,
        "matched_record": matched_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightGBM city prediction from IP")
    parser.add_argument("--ip", required=True, help="IP address to predict (IPv4 or IPv6)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument(
        "--model",
        default="3.city_lgbm.txt",
        help="Path to LightGBM model txt file",
    )
    parser.add_argument(
        "--label_encoder",
        default="3.city_label_encoder.joblib",
        help="Path to label encoder joblib",
    )
    parser.add_argument(
        "--preproc",
        default="3..joblib",
        help="Path to preprocessing objects joblib (TFIDF/Imputer/Scaler/OHE)",
    )
    parser.add_argument(
        "--csv",
        default="new_indian_training_data.csv",
        help="Optional CSV for IP range lookup (speeds and enriches features)",
    )
    args = parser.parse_args()

    try:
        result = predict_for_ip(
            ip=args.ip,
            model_path=args.model,
            label_encoder_path=args.label_encoder,
            preproc_path=args.preproc,
            csv_lookup_path=args.csv,
            topk=args.topk,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


