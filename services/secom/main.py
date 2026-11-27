import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "secom_model.joblib"
META_PATH = BASE_DIR / "artifacts" / "secom_pipeline_metadata.json"
THRESH_PATH = BASE_DIR / "data" / "thresholds.json"
RANGES_PATH = BASE_DIR / "data" / "reference_ranges.json"

# Register shell class so joblib deserializes without executing secom_main.py
# Register shell class so joblib deserializes without executing secom_main.py
dummy_mod = types.ModuleType("secom_main")


class SECOMTelemetryPipeline:
    drop_cols_: List[str]
    imputer_: Any
    scaler_: Any
    collinear_cols_: List[str]
    selected_features_: List[str]
    models_: Dict[str, Any]

    def preprocess_transform(self, X_raw: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        nan_cnt = np.asarray(X_raw.isnull().sum(axis=1), dtype=float).reshape(-1, 1)
        X_c = X_raw.drop(columns=self.drop_cols_)
        X_scaled = self.scaler_.transform(self.imputer_.transform(X_c))
        df_out = pd.DataFrame(X_scaled, columns=X_c.columns, index=X_raw.index)
        df_out["missing_sensor_count"] = nan_cnt
        return df_out

    def transform_features(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X_proc = self.preprocess_transform(X_raw, fit=False)
        X_uncorr = X_proc.drop(columns=self.collinear_cols_)
        return X_uncorr[self.selected_features_]


setattr(dummy_mod, "SECOMTelemetryPipeline", SECOMTelemetryPipeline)
sys.modules["secom_main"] = dummy_mod

# Load artifacts once at startup
bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipe"]
explainer = bundle["explainer"]
RAW_COLUMNS: List[str] = bundle["raw_columns"]
SELECTED_FEATURES: List[str] = bundle["selected_features"]

with open(META_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)
with open(THRESH_PATH, "r", encoding="utf-8") as f:
    THRESHOLDS = json.load(f)
with open(RANGES_PATH, "r", encoding="utf-8") as f:
    REF_RANGES = json.load(f)

MISSING_CAP = 0.40

app = FastAPI(title="SECOM Telemetry Screening Service", version="1.0.0")


class SensorRequest(BaseModel):
    values: List[Optional[float]]
    columns: Optional[List[str]] = None
    explain: bool = True


def validate_sensor_row(values: List[Optional[float]], columns: Optional[List[str]] = None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if len(values) != len(RAW_COLUMNS):
        errors.append(f"Expected exactly {len(RAW_COLUMNS)} sensor values, got {len(values)}.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if columns is not None and list(columns) != RAW_COLUMNS:
        errors.append("Sensor column names or order do not match training schema ('0'..'589').")
        return {"valid": False, "errors": errors, "warnings": warnings}

    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    if np.isinf(arr).any():
        errors.append("Infinite values (inf/-inf) are not allowed.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    missing_frac = float(np.isnan(arr).mean())
    if missing_frac > MISSING_CAP:
        errors.append(f"Missing sensor fraction ({missing_frac:.1%}) exceeds {MISSING_CAP:.0%} limit.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if missing_frac > 0.10:
        warnings.append(f"High missingness ({missing_frac:.1%} of sensors missing); imputed via KNN(k=5).")

    out_of_range = 0
    for col, val in zip(RAW_COLUMNS, arr):
        if not np.isnan(val) and col in REF_RANGES:
            lo, hi = REF_RANGES[col]["min"], REF_RANGES[col]["max"]
            span = max(abs(hi - lo), 1e-6)
            if val < lo - 0.5 * span or val > hi + 0.5 * span:
                out_of_range += 1
    if out_of_range > 0:
        warnings.append(f"{out_of_range} sensor reading(s) lie outside reference test bounds.")

    return {"valid": True, "errors": errors, "warnings": warnings}


@app.get("/health")
def health():
    return {"status": "ok", "service": "secom", "model_loaded": True, "num_features": len(RAW_COLUMNS)}


@app.get("/models")
def list_models():
    return [
        {
            "model_id": "secom_ensemble",
            "domain": "sensor_vector",
            "name": "SECOM Balanced RF + SPE-DT Risk Ranker",
            "classes": ["Pass", "Fail"],
            "input_shape": [len(RAW_COLUMNS)],
            "protocol": METADATA["protocol_B_deployment"],
            "limitations": [
                "Sensors are anonymized ('0'..'589'); SHAP drivers show statistical association, not physical root cause.",
                "Trained on first 80% chronological split of UCI SECOM (1,567 runs, 6.6% fail rate); rankings degrade under process drift.",
                "Unpaired with WM-811K wafer maps; never fused with spatial predictions."
            ],
        }
    ]


@app.post("/validate")
def validate(req: SensorRequest):
    return validate_sensor_row(req.values, req.columns)


@app.post("/predict")
def predict(req: SensorRequest):
    t0 = time.perf_counter()
    check = validate_sensor_row(req.values, req.columns)
    if not check["valid"]:
        raise HTTPException(status_code=400, detail={"errors": check["errors"]})

    row = [np.nan if v is None else float(v) for v in req.values]
    X_raw = pd.DataFrame([row], columns=RAW_COLUMNS)

    X_sel = pipe.transform_features(X_raw)
    prob_brf = float(pipe.models_["Balanced Random Forest"].predict_proba(X_sel)[:, 1][0])
    prob_spe = float(pipe.models_["SPE-DT (Optuna Tuned)"].predict_proba(X_sel)[:, 1][0])
    fused_risk = 0.5 * prob_brf + 0.5 * prob_spe

    top10_thresh = THRESHOLDS["top10_cutoff"]
    top20_thresh = THRESHOLDS["top20_cutoff"]
    if fused_risk >= top10_thresh:
        risk_tier = "High Risk (Top 10% Inspection Tier)"
        label = "Fail"
    elif fused_risk >= top20_thresh:
        risk_tier = "Elevated Risk (Top 20% Inspection Tier)"
        label = "Pass"
    else:
        risk_tier = "Normal Screening Tier"
        label = "Pass"

    explanation = None
    if req.explain:
        sv = explainer.shap_values(X_sel)
        sv_fail = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        row_sv = sv_fail[0]
        top_idx = np.argsort(np.abs(row_sv))[::-1][:8]
        explanation = {
            "type": "shap_top_sensors",
            "model_explained": "Balanced Random Forest",
            "note": "Anonymized sensor indices; indicates model association, not physical root cause.",
            "drivers": [
                {
                    "sensor": SELECTED_FEATURES[idx],
                    "shap_value": round(float(row_sv[idx]), 4),
                    "is_stable_top10": SELECTED_FEATURES[idx] in METADATA["top_stable_associated_sensors"],
                }
                for idx in top_idx
            ],
        }

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "model_id": "secom_ensemble",
        "domain": "sensor_vector",
        "label": label,
        "risk_tier": risk_tier,
        "scores": {
            "fused_telemetry_risk": round(fused_risk, 4),
            "prob_balanced_rf": round(prob_brf, 4),
            "prob_spe_dt": round(prob_spe, 4),
        },
        "threshold": round(top10_thresh, 4),
        "warnings": check["warnings"],
        "explanation": explanation,
        "metadata": {
            "version": METADATA["pipeline_version"],
            "threshold_source": THRESHOLDS["source"],
            "latency_ms": latency_ms,
        },
    }