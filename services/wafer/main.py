import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.wafer.libs.b1_lib import build_b1_model, preprocess_b1
from services.wafer.libs.bar_features import all_feats
from services.wafer.libs.spp_lib import build_spp, make_pred, onehot

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

CLASSES = [
    "center",
    "donut",
    "edge-loc",
    "edge-ring",
    "loc",
    "near-full",
    "none",
    "random",
    "scratch",
]

with open(DATA_DIR / "validator_limits.json", "r", encoding="utf-8") as f:
    LIMITS = json.load(f)

CAUSE_DF = pd.read_csv(DATA_DIR / "cause_table_v2.csv", keep_default_na=False).rename(
    columns={
        "class": "pattern",
        "literature_hypothesis": "possible_causes",
        "sources": "references",
    }
)
CAUSE_LOOKUP = CAUSE_DF.set_index("pattern").to_dict(orient="index")

# Load all 3 models once at startup
bar_model = joblib.load(ARTIFACTS_DIR / "wafer_bar" / "bar_model.joblib")

b1_model = build_b1_model(num_classes=len(CLASSES))
b1_model.load_weights(ARTIFACTS_DIR / "wafer_b1" / "b1_s0_last.weights.h5")

spp_model = build_spp()
spp_model.load_weights(ARTIFACTS_DIR / "wafer_spp" / "spp_s0_last.weights.h5")
spp_pred_fn = make_pred(spp_model)

app = FastAPI(title="WM-811K Wafer Spatial Classification Service", version="1.0.0")


class WaferRequest(BaseModel):
    matrix: List[List[Any]]
    models: Optional[List[str]] = None
    explain: bool = True


def validate_wafer_matrix(matrix: List[List[Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not matrix or not isinstance(matrix, list) or not all(isinstance(r, list) for r in matrix):
        return {"valid": False, "errors": ["Input must be a non-empty 2D matrix."], "warnings": warnings}

    row_lens = {len(r) for r in matrix}
    if len(row_lens) != 1 or 0 in row_lens:
        return {"valid": False, "errors": ["Ragged or empty rows detected; matrix must be rectangular."], "warnings": warnings}

    try:
        arr = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError):
        return {"valid": False, "errors": ["Matrix contains non-numeric values."], "warnings": warnings}

    if arr.ndim != 2:
        return {"valid": False, "errors": ["Input must be strictly 2-dimensional."], "warnings": warnings}

    if np.isnan(arr).any() or np.isinf(arr).any():
        return {"valid": False, "errors": ["Matrix contains NaN or infinite values."], "warnings": warnings}

    unique_vals = set(np.unique(arr).tolist())
    invalid_vals = sorted(unique_vals - {0.0, 1.0, 2.0})
    if invalid_vals:
        return {
            "valid": False,
            "errors": [f"Invalid cell values {invalid_vals}; only {{0, 1, 2}} are allowed."],
            "warnings": warnings,
        }

    arr_int = arr.astype(np.int32)
    H, W = arr_int.shape
    min_s, max_s = LIMITS["hard_min_side"], LIMITS["hard_max_side"]
    if H < min_s or W < min_s:
        return {"valid": False, "errors": [f"Dimensions ({H}x{W}) below minimum side {min_s}."], "warnings": warnings}
    if H > max_s or W > max_s:
        return {"valid": False, "errors": [f"Dimensions ({H}x{W}) exceed maximum side {max_s}."], "warnings": warnings}

    die_size = int((arr_int > 0).sum())
    if die_size == 0:
        return {"valid": False, "errors": ["All-zero matrix (no active wafer dies)."], "warnings": warnings}

    if not (LIMITS["min_height"] <= H <= LIMITS["max_height"] and LIMITS["min_width"] <= W <= LIMITS["max_width"]):
        warnings.append(
            f"Dimensions ({H}x{W}) fall outside training range "
            f"({LIMITS['min_height']}-{LIMITS['max_height']} x {LIMITS['min_width']}-{LIMITS['max_width']})."
        )

    if not (LIMITS["min_die_size"] <= die_size <= LIMITS["max_die_size"]):
        warnings.append(
            f"Active die count ({die_size}) falls outside training range "
            f"({LIMITS['min_die_size']}-{LIMITS['max_die_size']})."
        )

    corners = [arr_int[0, 0], arr_int[0, -1], arr_int[-1, 0], arr_int[-1, -1]]
    if any(c != 0 for c in corners):
        warnings.append("Non-zero corner cells detected; verify wafer background is encoded as 0.")

    if int((arr_int == 2).sum()) == 0:
        warnings.append("No failing dies (value 2) present; prediction will almost certainly be 'none'.")

    return {"valid": True, "errors": errors, "warnings": warnings}


def compute_ig_b1(arr_int: np.ndarray, class_idx: int, steps: int = 32) -> List[List[float]]:
    x = tf.constant(preprocess_b1(arr_int)[None, ...], dtype=tf.float32)
    baseline_map = np.where(arr_int == 2, 1, arr_int)
    b = tf.constant(preprocess_b1(baseline_map)[None, ...], dtype=tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)
    alphas = 0.5 * (alphas[:-1] + alphas[1:])
    interp = b + alphas[:, None, None, None] * (x - b)

    with tf.GradientTape() as tape:
        tape.watch(interp)
        preds = b1_model(interp, training=False)[:, class_idx]
    grads = tape.gradient(preds, interp)
    ig = (x - b)[0] * tf.reduce_mean(grads, axis=0)
    sal = tf.reduce_sum(tf.abs(ig), axis=-1).numpy()
    max_val = float(sal.max())
    if max_val > 0:
        sal = sal / max_val
    return np.round(sal, 4).tolist()


@app.get("/health")
def health():
    return {"status": "ok", "service": "wafer", "models_loaded": ["wafer_bar", "wafer_b1", "wafer_spp"]}


@app.get("/models")
def list_models():
    return [
        {
            "model_id": "wafer_bar",
            "domain": "wafer_map",
            "name": "Classical GBDT (137 Hand-Crafted Features)",
            "classes": CLASSES,
            "headline_macro_f1": 0.869,
            "limitations": [
                "Pattern classification only; no fab process telemetry is used.",
                "Cannot detect transposed matrices or swapped {0,1,2} encodings.",
            ],
        },
        {
            "model_id": "wafer_b1",
            "domain": "wafer_map",
            "name": "Resize CNN (B1, 64x64 Area-Resized)",
            "classes": CLASSES,
            "headline_macro_f1": 0.820,
            "limitations": [
                "Higher false-alarm rate on normal ('none') wafers than classical features.",
                "Resizes inputs to 64x64.",
            ],
        },
        {
            "model_id": "wafer_spp",
            "domain": "wafer_map",
            "name": "Spatial Pyramid Pooling CNN (Native Size)",
            "classes": CLASSES,
            "headline_macro_f1": 0.800,
            "limitations": [
                "Preserves native dimensions via SPP but scores ~0.020 macro-F1 below Resize CNN.",
                "Grad-CAM is disabled because it failed pixel-deletion checks.",
            ],
        },
    ]


@app.post("/validate")
def validate(req: WaferRequest):
    return validate_wafer_matrix(req.matrix)


@app.post("/predict")
def predict(req: WaferRequest):
    check = validate_wafer_matrix(req.matrix)
    if not check["valid"]:
        raise HTTPException(status_code=400, detail={"errors": check["errors"]})

    arr_int = np.asarray(req.matrix, dtype=np.int32)
    selected = set(req.models) if req.models else {"wafer_bar", "wafer_b1", "wafer_spp"}
    results = []

    if "wafer_bar" in selected:
        t0 = time.perf_counter()
        feats = all_feats(arr_int)[None, :]
        probs = bar_model.predict_proba(feats)[0]
        top_idx = int(np.argmax(probs))
        label = CLASSES[top_idx]
        results.append({
            "model_id": "wafer_bar",
            "domain": "wafer_map",
            "label": label,
            "scores": {c: round(float(p), 6) for c, p in zip(CLASSES, probs)},
            "threshold": None,
            "warnings": check["warnings"],
            "explanation": None,
            "cause_lookup": CAUSE_LOOKUP.get(label, {}),
            "metadata": {"version": "bar_v1", "latency_ms": round((time.perf_counter() - t0) * 1000, 2)},
        })

    if "wafer_b1" in selected:
        t0 = time.perf_counter()
        x_b1 = preprocess_b1(arr_int)[None, ...]
        probs = b1_model(x_b1, training=False).numpy()[0]
        top_idx = int(np.argmax(probs))
        label = CLASSES[top_idx]
        expl = None
        if req.explain:
            expl = {
                "type": "integrated_gradients_64x64",
                "steps": 32,
                "heatmap": compute_ig_b1(arr_int, top_idx, steps=32),
            }
        results.append({
            "model_id": "wafer_b1",
            "domain": "wafer_map",
            "label": label,
            "scores": {c: round(float(p), 6) for c, p in zip(CLASSES, probs)},
            "threshold": None,
            "warnings": check["warnings"],
            "explanation": expl,
            "cause_lookup": CAUSE_LOOKUP.get(label, {}),
            "metadata": {"version": "b1_s0_last", "latency_ms": round((time.perf_counter() - t0) * 1000, 2)},
        })

    if "wafer_spp" in selected:
        t0 = time.perf_counter()
        x_spp = tf.constant(onehot([arr_int]))
        probs = spp_pred_fn(x_spp).numpy()[0]
        top_idx = int(np.argmax(probs))
        label = CLASSES[top_idx]
        results.append({
            "model_id": "wafer_spp",
            "domain": "wafer_map",
            "label": label,
            "scores": {c: round(float(p), 6) for c, p in zip(CLASSES, probs)},
            "threshold": None,
            "warnings": check["warnings"],
            "explanation": None,
            "cause_lookup": CAUSE_LOOKUP.get(label, {}),
            "metadata": {"version": "spp_s0_last", "latency_ms": round((time.perf_counter() - t0) * 1000, 2)},
        })

    return {"results": results, "warnings": check["warnings"]}
