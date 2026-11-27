import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
GATEWAY_URL = "http://127.0.0.1:8000"

WAFER_DEMO_PATH = ROOT_DIR / "services" / "wafer" / "data" / "demo_samples.npz"
SECOM_DEMO_PATH = ROOT_DIR / "services" / "secom" / "data" / "demo_rows.csv"
SECOM_COLS_PATH = ROOT_DIR / "services" / "secom" / "data" / "feature_order.json"
SECOM_DATA_DIR = ROOT_DIR / "services" / "secom" / "data"
ASSETS_DIR = ROOT_DIR / "frontend" / "assets"

st.set_page_config(
    page_title="Semiconductor Yield Diagnostics Platform",
    page_icon="🔬",
    layout="wide",
)


@st.cache_data
def load_wafer_demos() -> Dict[str, Any]:
    d = np.load(WAFER_DEMO_PATH, allow_pickle=True)
    return {
        "maps": d["maps"],
        "labels": [str(x) for x in d["labels"]],
        "lot_names": [str(x) for x in d["lot_names"]],
        "wafer_indices": [int(x) for x in d["wafer_indices"]],
        "classes": [str(x) for x in d["classes"]],
    }


@st.cache_data
def load_secom_demos() -> pd.DataFrame:
    return pd.read_csv(SECOM_DEMO_PATH)


@st.cache_data
def load_secom_cols() -> List[str]:
    with open(SECOM_COLS_PATH, "r", encoding="utf-8") as f:
        return [str(c) for c in json.load(f)]


def fetch_health() -> Dict[str, Any]:
    try:
        r = httpx.get(f"{GATEWAY_URL}/health", timeout=3.0)
        return r.json()
    except Exception as exc:
        return {"status": "offline", "error": str(exc), "services": {}}


def render_wafer_heatmap(matrix: np.ndarray, title: str):
    colorscale = [
        [0.0, "#1e222a"],   # 0: outside wafer
        [0.33, "#1e222a"],
        [0.33, "#3b82f6"],  # 1: passing die
        [0.66, "#3b82f6"],
        [0.66, "#ef4444"],  # 2: failing die
        [1.0, "#ef4444"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            zmin=0,
            zmax=2,
            colorscale=colorscale,
            showscale=False,
        )
    )
    fig.update_layout(
        title=title,
        height=290,
        margin=dict(l=10, r=10, t=35, b=10),
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1, showticklabels=False),
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")


# --- HEADER & LIVE HEALTH BAR ---
st.title("🔬 Semiconductor Yield Diagnostics Platform")
st.caption(
    "Independent multi-modal diagnostic workbench: **WM-811K Spatial Wafer Pattern Classification** "
    "& **UCI SECOM Process Telemetry Risk Screening** (unpaired datasets — never fused)."
)

health = fetch_health()
c_h1, c_h2, c_h3, c_h4 = st.columns([2, 2, 2, 1])
gw_ok = health.get("status") in ("ok", "degraded")
wf_st = health.get("services", {}).get("wafer", {}).get("status", "offline")
sc_st = health.get("services", {}).get("secom", {}).get("status", "offline")

c_h1.markdown(f"**Gateway (`:8000`):** {'🟢 Online' if gw_ok else '🔴 Offline'}")
c_h2.markdown(f"**WM-811K Service (`:8001`):** {'🟢 Ready' if wf_st == 'ok' else '🔴 ' + wf_st}")
c_h3.markdown(f"**SECOM Service (`:8002`):** {'🟢 Ready' if sc_st == 'ok' else '🔴 ' + sc_st}")
if c_h4.button("Refresh Status"):
    st.rerun()

st.divider()

tab_diag, tab_about = st.tabs(["🧪 Diagnostic Workbench", "📋 Model Cards & Benchmarks"])

# ==============================================================================
# TAB 1: DIAGNOSTIC WORKBENCH (INPUT + RESULTS)
# ==============================================================================
with tab_diag:
    col_w, col_s = st.columns(2, gap="large")

    wafer_matrix: Optional[List[List[int]]] = None
    wafer_gt: Optional[str] = None
    sensor_values: Optional[List[Optional[float]]] = None
    sensor_gt: Optional[str] = None

    # --- LEFT CARD: WM-811K WAFER MAP ---
    with col_w:
        st.subheader("1. WM-811K Wafer Map (Spatial)")
        w_mode = st.radio(
            "Wafer Input Source",
            ["Demo Validation Wafer", "Upload Custom Matrix (.npy / .csv)", "Skip Wafer Panel"],
            horizontal=True,
        )
        if w_mode == "Demo Validation Wafer":
            demos = load_wafer_demos()
            options = [
                f"Sample #{i} | True: {demos['labels'][i]} | Lot: {demos['lot_names'][i]} ({demos['maps'][i].shape[0]}x{demos['maps'][i].shape[1]})"
                for i in range(len(demos["maps"]))
            ]
            idx = st.selectbox("Select Demo Wafer", range(len(options)), format_func=lambda i: options[i])
            arr = np.asarray(demos["maps"][idx], dtype=int)
            wafer_matrix = arr.tolist()
            wafer_gt = demos["labels"][idx]
            render_wafer_heatmap(arr, f"Wafer Preview ({arr.shape[0]}×{arr.shape[1]} | Blue=Pass, Red=Fail)")

        elif w_mode == "Upload Custom Matrix (.npy / .csv)":
            up_w = st.file_uploader("Upload 2D integer matrix with values {0, 1, 2}", type=["npy", "csv"])
            if up_w is not None:
                try:
                    if up_w.name.endswith(".npy"):
                        arr = np.load(io.BytesIO(up_w.read())).astype(int)
                    else:
                        arr = pd.read_csv(up_w, header=None).values.astype(int)
                    wafer_matrix = arr.tolist()
                    render_wafer_heatmap(arr, f"Uploaded Wafer ({arr.shape[0]}×{arr.shape[1]})")
                except Exception as e:
                    st.error(f"Could not parse matrix file: {e}")

    # --- RIGHT CARD: UCI SECOM TELEMETRY ---
    with col_s:
        st.subheader("2. UCI SECOM Telemetry (590 Sensors)")
        s_mode = st.radio(
            "Telemetry Input Source",
            ["Demo Chronological Holdout Row", "Upload Custom Sensor CSV", "Skip Telemetry Panel"],
            horizontal=True,
        )
        raw_cols = load_secom_cols()

        if s_mode == "Demo Chronological Holdout Row":
            df_sec = load_secom_demos()
            s_options = []
            for i, row in df_sec.iterrows():
                miss_pct = row[raw_cols].isna().mean() * 100
                gt_lbl = "Fail (1)" if row.get("Pass/Fail", -1) == 1 else "Pass (-1)"
                s_options.append(f"Holdout Row #{i} | Actual: {gt_lbl} | Missing: {miss_pct:.1f}% | {row.get('Time', '')}")
            s_idx = st.selectbox("Select SECOM Test Run", range(len(s_options)), format_func=lambda i: s_options[i])
            sel_row = df_sec.iloc[s_idx]
            sensor_values = [None if pd.isna(sel_row[c]) else float(sel_row[c]) for c in raw_cols]
            sensor_gt = "Fail (1)" if sel_row.get("Pass/Fail", -1) == 1 else "Pass (-1)"

            non_null_preview = pd.DataFrame({
                "Sensor": ["33", "130", "103", "21", "205", "510"],
                "Reading": [sel_row.get(k, np.nan) for k in ["33", "130", "103", "21", "205", "510"]],
            })
            st.caption("Preview of Top Stable Associated Sensors (out of 590):")
            st.dataframe(non_null_preview.T, width="stretch")

        elif s_mode == "Upload Custom Sensor CSV":
            up_s = st.file_uploader("Upload 1-row CSV with sensor columns '0'..'589'", type=["csv"])
            if up_s is not None:
                try:
                    df_up = pd.read_csv(up_s)
                    missing_cols = set(raw_cols) - set(df_up.columns)
                    if missing_cols:
                        st.error(f"Missing required sensor columns: {list(missing_cols)[:5]}")
                    else:
                        row0 = df_up[raw_cols].iloc[0]
                        sensor_values = [None if pd.isna(row0[c]) else float(row0[c]) for c in raw_cols]
                        st.success(f"Loaded row 0 ({len(raw_cols)} sensors, {row0.isna().mean():.1%} missing).")
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")

    st.divider()
    run_btn = st.button("🚀 Run Independent Diagnostics", type="primary", width="stretch")

    if run_btn:
        if wafer_matrix is None and sensor_values is None:
            st.warning("Select or upload at least one input (Wafer Map or SECOM Sensor Row).")
        else:
            payload: Dict[str, Any] = {}
            if wafer_matrix is not None:
                payload["wafer"] = {"matrix": wafer_matrix, "explain": True}
            if sensor_values is not None:
                payload["sensor"] = {"values": sensor_values, "columns": raw_cols, "explain": True}

            with st.spinner("Running parallel inference via Gateway (:8000)..."):
                try:
                    resp = httpx.post(f"{GATEWAY_URL}/analyze", json=payload, timeout=25.0)
                    out = resp.json()
                except Exception as exc:
                    st.error(f"Gateway request failed: {exc}. Ensure `run_all.ps1` is running.")
                    st.stop()

            st.info(out.get("disclaimer", ""))
            res_w, res_s = st.columns(2, gap="large")

            # --- WAFER RESULTS PANEL ---
            with res_w:
                st.subheader("WM-811K Spatial Diagnosis")
                w_panel = out["panels"]["wafer"]
                if w_panel["status"] == "skipped":
                    st.info("Wafer panel skipped.")
                elif w_panel["status"] == "error":
                    st.error(f"Wafer service error: {w_panel.get('detail')}")
                else:
                    w_data = w_panel["data"]
                    if wafer_gt:
                        st.caption(f"Ground-Truth Dataset Label: **{wafer_gt}**")
                    for w_warn in w_data.get("warnings", []):
                        st.warning(w_warn)

                    rows_cmp = []
                    b1_ig = None
                    top_cause = {}
                    for item in w_data["results"]:
                        top_prob = max(item["scores"].values())
                        rows_cmp.append({
                            "Model": item["model_id"],
                            "Predicted Pattern": item["label"],
                            "Confidence": f"{top_prob:.1%}",
                            "Latency (ms)": item["metadata"]["latency_ms"],
                        })
                        if item["model_id"] == "wafer_bar":
                            top_cause = item.get("cause_lookup", {})
                        if item["model_id"] == "wafer_b1" and item.get("explanation"):
                            b1_ig = np.array(item["explanation"]["heatmap"])

                    st.dataframe(pd.DataFrame(rows_cmp), width="stretch", hide_index=True)

                    if b1_ig is not None:
                        fig_ig = px.imshow(
                            b1_ig,
                            color_continuous_scale="Inferno",
                            title="Resize CNN (B1) — Integrated Gradients Attribution (64×64)",
                        )
                        fig_ig.update_layout(height=270, margin=dict(l=10, r=10, t=35, b=10))
                        st.plotly_chart(fig_ig, width="stretch")

                    if top_cause:
                        with st.expander("📚 Literature Process Hypothesis & Caveats (Classical Model)", expanded=True):
                            st.markdown(f"**Literature Hypothesis:** {top_cause.get('possible_causes', 'N/A')}")
                            st.markdown(f"**Suggested Check (Author-Inferred):** {top_cause.get('suggested_check_AUTHOR_INFERRED', 'N/A')}")
                            st.markdown(f"**Evidence Level:** `{top_cause.get('evidence_level', 'N/A')}` | **Sources:** {top_cause.get('references', 'N/A')}")
                            if top_cause.get("caveat"):
                                st.warning(f"**Caveat:** {top_cause['caveat']}")
                            if top_cause.get("disclaimer"):
                                st.caption(top_cause["disclaimer"])

            # --- SECOM RESULTS PANEL ---
            with res_s:
                st.subheader("UCI SECOM Telemetry Screening")
                s_panel = out["panels"]["secom"]
                if s_panel["status"] == "skipped":
                    st.info("SECOM telemetry panel skipped.")
                elif s_panel["status"] == "error":
                    st.error(f"SECOM service error: {s_panel.get('detail')}")
                else:
                    s_data = s_panel["data"]
                    if sensor_gt:
                        st.caption(f"Ground-Truth Holdout Outcome: **{sensor_gt}**")
                    for s_warn in s_data.get("warnings", []):
                        st.warning(s_warn)

                    sc = s_data["scores"]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Fused Risk Score", f"{sc['fused_telemetry_risk']:.4f}", f"Top-10% Cutoff: {s_data['threshold']}")
                    m2.metric("Balanced RF", f"{sc['prob_balanced_rf']:.4f}")
                    m3.metric("SPE-DT (Optuna)", f"{sc['prob_spe_dt']:.4f}")

                    tier = s_data["risk_tier"]
                    if "High Risk" in tier:
                        st.error(f"🚨 **{tier}**")
                    elif "Elevated" in tier:
                        st.warning(f"⚠️ **{tier}**")
                    else:
                        st.success(f"✅ **{tier}")

                    expl = s_data.get("explanation")
                    if expl and expl.get("drivers"):
                        df_drv = pd.DataFrame(expl["drivers"])
                        df_drv["Sensor Label"] = df_drv.apply(
                            lambda r: f"Sensor {r['sensor']} ★" if r["is_stable_top10"] else f"Sensor {r['sensor']}",
                            axis=1,
                        )
                        fig_shap = px.bar(
                            df_drv.iloc[::-1],
                            x="shap_value",
                            y="Sensor Label",
                            orientation="h",
                            title="Local TreeSHAP Sensor Drivers (★ = Top-10 Stable Sensor)",
                            color="shap_value",
                            color_continuous_scale="RdBu_r",
                            color_continuous_midpoint=0.0,
                        )
                        fig_shap.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
                        st.plotly_chart(fig_shap, width="stretch")
                        st.caption(expl.get("note", ""))

# ==============================================================================
# TAB 2: MODEL CARDS & BENCHMARKS
# ==============================================================================
with tab_about:
    st.subheader("Model Cards, Evaluation Protocols & Scientific Limitations")
    try:
        cards_resp = httpx.get(f"{GATEWAY_URL}/models", timeout=5.0).json()
        st.info(cards_resp.get("separation_policy", ""))
        for mid, card in cards_resp.get("model_cards", {}).items():
            with st.expander(f"** (`{mid}`) — {card['domain']}", expanded=True):
                st.markdown(f"Architecture:")
                st.markdown(f"Evaluation Protocol:")
                st.json(card["metrics"])
                st.markdown("**Known Limitations:**")
                for lim in card["limitations"]:
                    st.markdown(f"- {lim}")
    except Exception as exc:
        st.warning(f"Could not fetch live model cards from Gateway: {exc}")

    st.divider()
    st.subheader("SECOM Cross-Protocol Benchmark Tables & Global SHAP Figures")
    t1_path = SECOM_DATA_DIR / "table1_protocol_a_summary.csv"
    t2_path = SECOM_DATA_DIR / "table2_protocol_b_summary.csv"
    t3_path = SECOM_DATA_DIR / "table3_shap_sensor_stability.csv"

    c_t1, c_t2 = st.columns(2)
    if t1_path.exists():
        c_t1.markdown("**Table 1 — Protocol A (10 Repeated Stratified 80/20 Splits)**")
        c_t1.dataframe(pd.read_csv(t1_path), width="stretch", hide_index=True)
    if t2_path.exists():
        c_t2.markdown("**Table 2 — Protocol B (3 Expanding-Window Chronological Splits)**")
        c_t2.dataframe(pd.read_csv(t2_path), width="stretch", hide_index=True)

    if t3_path.exists():
        st.markdown("**Table 3 — Cross-Split Top SHAP Sensor Stability**")
        st.dataframe(pd.read_csv(t3_path), width="stretch", hide_index=True)

    f1_img = ASSETS_DIR / "fig1_final_shap_global_summary.png"
    f2_img = ASSETS_DIR / "fig2_final_shap_local_waterfall.png"
    c_f1, c_f2 = st.columns(2)
    if f1_img.exists():
        c_f1.image(str(f1_img), caption="Global SHAP Summary (Protocol B Holdout)")
    if f2_img.exists():
        c_f2.image(str(f2_img), caption="Example Local SHAP Waterfall (Reference Holdout Run)")