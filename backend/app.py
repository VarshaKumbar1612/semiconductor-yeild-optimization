import io
import sys
import types
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. Register a lightweight shell class so joblib loads the saved model
#    WITHOUT re-running your 20-minute secom_main.py script
dummy_mod = types.ModuleType("secom_main")

class SECOMTelemetryPipeline:
    def preprocess_transform(self, X_raw: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        nan_cnt = X_raw.isnull().sum(axis=1).values.reshape(-1, 1)
        X_c = X_raw.drop(columns=self.drop_cols_)
        X_scaled = self.scaler_.transform(self.imputer_.transform(X_c))
        df_out = pd.DataFrame(X_scaled, columns=X_c.columns, index=X_raw.index)
        df_out["missing_sensor_count"] = nan_cnt
        return df_out

    def transform_features(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X_proc = self.preprocess_transform(X_raw, fit=False)
        X_uncorr = X_proc.drop(columns=self.collinear_cols_)
        return X_uncorr[self.selected_features_]

dummy_mod.SECOMTelemetryPipeline = SECOMTelemetryPipeline
sys.modules["secom_main"] = dummy_mod

# 2. Load the saved model ONCE when the backend starts (< 1 second)
bundle = joblib.load("models/secom_model.joblib")
pipe = bundle["pipe"]
explainer = bundle["explainer"]
raw_columns = bundle["raw_columns"]
selected_features = bundle["selected_features"]

# 3. Start FastAPI app
app = FastAPI(title="SECOM Defect Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    contents = await file.read()
    df_in = pd.read_csv(io.BytesIO(contents))

    # Align columns with training data
    X_raw = df_in.drop(columns=["Time", "Pass/Fail", "time_rank"], errors="ignore")
    missing = set(raw_columns) - set(X_raw.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing sensor columns: {list(missing)[:5]}")
    X_raw = X_raw[raw_columns]

    # Instant preprocessing & risk prediction
    X_sel = pipe.transform_features(X_raw)
    prob_brf = pipe.models_["Balanced Random Forest"].predict_proba(X_sel)[:, 1]
    prob_spe = pipe.models_["SPE-DT (Optuna Tuned)"].predict_proba(X_sel)[:, 1]
    fused_risk = 0.5 * prob_brf + 0.5 * prob_spe

    # Instant TreeSHAP sensor attribution
    sv = explainer.shap_values(X_sel)
    sv_fail = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)

    n = len(X_sel)
    top10_idx = set(np.argsort(fused_risk)[::-1][: max(1, int(np.ceil(0.10 * n)))])

    wafers = []
    for i in range(n):
        top3 = np.argsort(np.abs(sv_fail[i]))[::-1][:3]
        wafers.append({
            "wafer_id": i + 1,
            "prob_brf": round(float(prob_brf[i]), 4),
            "prob_spe": round(float(prob_spe[i]), 4),
            "fused_risk": round(float(fused_risk[i]), 4),
            "high_risk_alarm": i in top10_idx,
            "top_sensors": [
                {"sensor": selected_features[idx], "shap": round(float(sv_fail[i][idx]), 4)}
                for idx in top3
            ],
        })

    return {"total_wafers": n, "results": wafers}