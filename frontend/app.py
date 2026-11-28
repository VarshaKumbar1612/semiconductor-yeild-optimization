import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gateway.main import app as gateway_app

WAFER_DEMO_PATH = ROOT_DIR / "services" / "wafer" / "data" / "demo_samples.npz"
SECOM_DEMO_PATH = ROOT_DIR / "services" / "secom" / "data" / "demo_rows.csv"
SECOM_COLS_PATH = ROOT_DIR / "services" / "secom" / "data" / "feature_order.json"
SECOM_DATA_DIR = ROOT_DIR / "services" / "secom" / "data"
ASSETS_DIR = ROOT_DIR / "frontend" / "assets"


def _port_ready(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_resource
def get_gateway_client() -> TestClient:
    """Start :8001 and :8002 sequentially with 1-thread limits, then wrap Gateway in-process."""
    low_cpu_env = os.environ.copy()
    low_cpu_env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    })

    services = [
        (8001, [sys.executable, "-m", "uvicorn", "services.wafer.main:app", "--host", "127.0.0.1", "--port", "8001"]),
        (8002, [sys.executable, "-m", "uvicorn", "services.secom.main:app", "--host", "127.0.0.1", "--port", "8002"]),
    ]
    for port, cmd in services:
        if not _port_ready(port):
            subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=low_cpu_env)
            for _ in range(45):
                time.sleep(2.0)
                if _port_ready(port):
                    break

    return TestClient(gateway_app)


@st.cache_resource
def ensure_backends_running() -> bool:
    """Start :8001, :8002, and :8000 automatically when hosted on a single-container cloud."""
    try:
        r = get_gateway_client().get("/health")
        if r.status_code == 200 and r.json().get("status") == "ok":
            return True
    except Exception:
        pass

    services = [
        [sys.executable, "-m", "uvicorn", "services.wafer.main:app", "--host", "127.0.0.1", "--port", "8001"],
        [sys.executable, "-m", "uvicorn", "services.secom.main:app", "--host", "127.0.0.1", "--port", "8002"],
        [sys.executable, "-m", "uvicorn", "gateway.main:app", "--host", "127.0.0.1", "--port", "8000"],
    ]
    for cmd in services:
        subprocess.Popen(cmd, cwd=str(ROOT_DIR))

    for _ in range(25):
        time.sleep(1.5)
        try:
            r = get_gateway_client().get("/health")
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
    return False


ensure_backends_running()
st.set_page_config(
    page_title="Semiconductor Yield Diagnostics Platform",
    page_icon="🔬",
    layout="wide",
)

# ==============================================================================
# CUSTOM THEME & DASHBOARD STYLING (Matches Lavender/Purple Design)
# ==============================================================================
st.markdown(
    """
    <style>
    /* Page Canvas */
    .stApp {
        background: linear-gradient(180deg, #F4F1FC 0%, #F9F8FD 22%, #F8F7FC 100%);
        color: #18181B;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px;
    }

    /* Hero Header */
    .hero-wrap {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 0 0.9rem 0;
        overflow: hidden;
    }
    .hero-left {
        display: flex;
        align-items: center;
        gap: 16px;
        z-index: 2;
    }
    .hero-icon-box {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: #EFEAFC;
        border: 1px solid #DDD6FE;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #111827;
        margin: 0;
        line-height: 1.15;
    }
    .hero-title span {
        color: #6D28D9;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #4B5563;
        margin: 4px 0 0 0;
    }
    .hero-watermark {
        opacity: 0.22;
        pointer-events: none;
    }

    /* Status Pill Bar */
    .status-bar {
        background: #FFFFFF;
        border: 1px solid #E5E1F5;
        border-radius: 16px;
        padding: 12px 22px;
        box-shadow: 0 2px 10px rgba(109, 40, 217, 0.04);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 1.1rem;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.95rem;
        font-weight: 700;
        color: #1F2937;
    }
    .status-divider {
        width: 1px;
        height: 24px;
        background: #E5E7EB;
    }
    .badge-ok {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .badge-err {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .dot-ok {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        display: inline-block;
    }
    .dot-err {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #EF4444;
        display: inline-block;
    }

    /* Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid #E5E1F5;
        padding-bottom: 10px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E1F5;
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        color: #4B5563;
        height: auto;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6D28D9 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border-color: #6D28D9 !important;
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.22);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* Bordered Card Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #E4E0F5 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 18px rgba(30, 27, 75, 0.04);
        padding: 0 !important;
        overflow: hidden;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 1.15rem 1.35rem 1.35rem 1.35rem;
    }

    /* Lavender Card Header Strip */
    .card-header-strip {
        background: #EFEAFC;
        margin: -1.15rem -1.35rem 1rem -1.35rem;
        padding: 13px 20px;
        border-bottom: 1px solid #E2D9F9;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.08rem;
        font-weight: 700;
        color: #1E1B4B;
    }

    /* Soft Pill Label */
    .lavender-pill {
        display: inline-block;
        background: #EFEAFC;
        color: #5B21B6;
        border: 1px solid #DDD6FE;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 8px;
        margin: 6px 0 10px 0;
    }

    /* Alerts & Banners */
    .banner-info {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #1E3A8A;
        padding: 11px 16px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .banner-warn {
        background: #FEF9C3;
        border: 1px solid #FDE047;
        color: #854D0E;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.85rem;
    }
    .banner-danger {
        background: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0.6rem 0;
    }
    .banner-success {
        background: #DCFCE7;
        border: 1px solid #86EFAC;
        color: #166534;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.6rem 0;
    }

    /* SECOM Metric Sub-Cards */
    .metric-subcard {
        background: #F8F7FC;
        border: 1px solid #E5E1F5;
        border-radius: 12px;
        padding: 12px 14px;
        height: 100%;
    }
    .metric-subcard-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 4px;
    }
    .metric-subcard-val {
        font-size: 1.55rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.2;
    }
    .metric-cutoff-pill {
        display: inline-block;
        margin-top: 6px;
        background: #DCFCE7;
        color: #15803D;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 2px 9px;
        border-radius: 999px;
    }

    /* Primary Gradient CTA Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6D28D9 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.72rem 1.5rem !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 18px rgba(109, 40, 217, 0.25) !important;
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.06);
        box-shadow: 0 8px 22px rgba(109, 40, 217, 0.34) !important;
    }

    /* Secondary Refresh Button */
    div.stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #6D28D9 !important;
        border: 1px solid #DDD6FE !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        padding: 0.4rem 1.1rem !important;
    }

    /* Model Card Inner Panels */
    .mc-box {
        background: #F9F8FD;
        border: 1px solid #E5E1F5;
        border-radius: 12px;
        padding: 14px 16px;
        height: 100%;
    }
    .mc-label {
        font-size: 0.88rem;
        font-weight: 700;
        color: #1E1B4B;
        margin-bottom: 6px;
    }
    .mc-arch-pill {
        background: #FFFFFF;
        border: 1px solid #E2D9F9;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.88rem;
        color: #374151;
        margin-bottom: 12px;
    }
    .lim-list {
        margin: 0;
        padding-left: 1.15rem;
        color: #374151;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .lim-list li::marker {
        color: #7C3AED;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# DATA LOADERS & HELPERS
# ==============================================================================
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
        r = get_gateway_client().get("/health")
        return r.json()
    except Exception as exc:
        return {"status": "offline", "error": str(exc), "services": {}}


def render_wafer_heatmap(matrix: np.ndarray, title: str):
    st.markdown(f'<div class="lavender-pill">{title}</div>', unsafe_allow_html=True)
    colorscale = [
        [0.0, "#1E222A"],   # 0: outside wafer
        [0.33, "#1E222A"],
        [0.33, "#3B82F6"],  # 1: passing die
        [0.66, "#3B82F6"],
        [0.66, "#EF4444"],  # 2: failing die
        [1.0, "#EF4444"],
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
        height=265,
        margin=dict(l=6, r=6, t=6, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1, showticklabels=False),
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")


# ==============================================================================
# HERO BANNER & LIVE STATUS BAR
# ==============================================================================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-left">
            <div class="hero-icon-box">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="4" y="4" width="16" height="16" rx="2"></rect>
                    <rect x="9" y="9" width="6" height="6"></rect>
                    <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"></path>
                </svg>
            </div>
            <div>
                <h1 class="hero-title">Semiconductor <span>Yield Diagnostics</span> Platform</h1>
                <p class="hero-subtitle">AI-powered semiconductor manufacturing analytics for wafer pattern classification, SECOM telemetry, and defect diagnostics.</p>
            </div>
        </div>
        <svg class="hero-watermark" width="130" height="72" viewBox="0 0 120 70" fill="none">
            <ellipse cx="65" cy="35" rx="50" ry="28" stroke="#7C3AED" stroke-width="2"/>
            <path d="M25 25h80M20 35h90M25 45h80M45 10v50M65 8v54M85 10v50" stroke="#7C3AED" stroke-width="1.4"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

health = fetch_health()
gw_ok = health.get("status") in ("ok", "degraded")
wf_st = health.get("services", {}).get("wafer", {}).get("status", "offline")
sc_st = health.get("services", {}).get("secom", {}).get("status", "offline")

gw_badge = '<span class="badge-ok"><span class="dot-ok"></span>Online</span>' if gw_ok else '<span class="badge-err"><span class="dot-err"></span>Offline</span>'
wf_badge = '<span class="badge-ok"><span class="dot-ok"></span>Ready</span>' if wf_st == "ok" else f'<span class="badge-err"><span class="dot-err"></span>{wf_st}</span>'
sc_badge = '<span class="badge-ok"><span class="dot-ok"></span>Ready</span>' if sc_st == "ok" else f'<span class="badge-err"><span class="dot-err"></span>{sc_st}</span>'

col_sb, col_ref = st.columns([6.2, 1.1])
with col_sb:
    st.markdown(
        f"""
        <div class="status-bar">
            <div class="status-item">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                Gateway (:8000) {gw_badge}
            </div>
            <div class="status-divider"></div>
            <div class="status-item">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
                WM-811K Service (:8001) {wf_badge}
            </div>
            <div class="status-divider"></div>
            <div class="status-item">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                SECOM Service (:8002) {sc_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_ref:
    st.write("")
    if st.button("↻ Refresh Status", width="stretch"):
        st.rerun()

tab_diag, tab_about = st.tabs(["📈 Diagnostic Workbench", "📊 Model Cards & Benchmarks"])


# ==============================================================================
# TAB 1: DIAGNOSTIC WORKBENCH
# ==============================================================================
with tab_diag:
    col_w, col_s = st.columns(2, gap="large")

    wafer_matrix: Optional[List[List[int]]] = None
    wafer_gt: Optional[str] = None
    sensor_values: Optional[List[Optional[float]]] = None
    sensor_gt: Optional[str] = None

    # --- LEFT CARD: WM-811K WAFER MAP ---
    with col_w:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-header-strip">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2.2"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>
                    1. WM-811K Wafer Map (Spatial)
                </div>
                """,
                unsafe_allow_html=True,
            )
            w_mode = st.radio(
                "Wafer Input Source",
                ["Demo Validation Wafer", "Upload Custom Wafer Map (.npy / .csv)", "Skip Wafer Panel"],
                horizontal=True,
            )
            if w_mode == "Demo Validation Wafer":
                demos = load_wafer_demos()
                options = [
                    f"Sample #{i} | True: {demos['labels'][i]} | Lot: {demos['lot_names'][i]} ({demos['maps'][i].shape[0]}×{demos['maps'][i].shape[1]})"
                    for i in range(len(demos["maps"]))
                ]
                idx = st.selectbox("Select Demo Wafer", range(len(options)), format_func=lambda i: options[i])
                arr = np.asarray(demos["maps"][idx], dtype=int)
                wafer_matrix = arr.tolist()
                wafer_gt = demos["labels"][idx]
                render_wafer_heatmap(arr, f"Demo Wafer ({arr.shape[0]} × {arr.shape[1]} | Blue=Pass, Red=Fail)")

            elif w_mode == "Upload Custom Wafer Map (.npy / .csv)":
                up_w = st.file_uploader("Upload 2D integer matrix (values: 0, 1, 2)", type=["npy", "csv"])
                if up_w is not None:
                    try:
                        if up_w.name.endswith(".npy"):
                            arr = np.load(io.BytesIO(up_w.read())).astype(int)
                        else:
                            arr = pd.read_csv(up_w, header=None).values.astype(int)
                        wafer_matrix = arr.tolist()
                        render_wafer_heatmap(arr, f"Uploaded Wafer ({arr.shape[0]} × {arr.shape[1]})")
                    except Exception as e:
                        st.error(f"Could not parse matrix file: {e}")
            else:
                st.info("Wafer spatial panel will be skipped during diagnostics.")

    # --- RIGHT CARD: UCI SECOM TELEMETRY ---
    with col_s:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-header-strip">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2.2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    2. UCI SECOM Telemetry (590 Sensors)
                </div>
                """,
                unsafe_allow_html=True,
            )
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

                preview_sensors = ["33", "130", "103", "21", "205", "510"]
                non_null_preview = pd.DataFrame(
                    [[None if pd.isna(sel_row.get(k, np.nan)) else float(sel_row[k]) for k in preview_sensors]],
                    columns=[f"Sensor {k}" for k in preview_sensors],
                    index=["Reading"],
                )
                st.caption("Preview of Top Stable Associated Sensors (out of 590):")
                st.dataframe(non_null_preview, width="stretch")

            elif s_mode == "Upload Custom Sensor CSV":
                up_s = st.file_uploader("Upload a .csv (with 590 sensor columns '0'..'589')", type=["csv"])
                if up_s is not None:
                    try:
                        df_up = pd.read_csv(up_s)
                        missing_cols = set(raw_cols) - set(df_up.columns)
                        if missing_cols:
                            st.error(f"Missing required sensor columns: {list(missing_cols)[:5]}")
                        else:
                            row0 = df_up[raw_cols].iloc[0]
                            sensor_values = [None if pd.isna(row0[c]) else float(row0[c]) for c in raw_cols]
                            st.markdown(
                                f'<div class="banner-success">✔ Loaded SECOM CSV ({len(df_up)} row(s), {len(raw_cols)} sensors, {row0.isna().mean():.1%} missing)</div>',
                                unsafe_allow_html=True,
                            )
                    except Exception as e:
                        st.error(f"Could not parse CSV: {e}")
            else:
                st.info("SECOM telemetry panel will be skipped during diagnostics.")

    st.write("")
    run_btn = st.button("✨ Run Independent Diagnostics", type="primary", width="stretch")

    # ==========================================================================
    # DIAGNOSTIC RESULTS SECTION
    # ==========================================================================
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
                    resp = get_gateway_client().post("/analyze", json=payload)
                    out = resp.json()
                except Exception as exc:
                    st.error(f"Gateway request failed: {exc}. Ensure `run_all.ps1` is running.")
                    st.stop()

            st.markdown(
                f'<div class="banner-info">ℹ️ {out.get("disclaimer", "")}</div>',
                unsafe_allow_html=True,
            )

            res_w, res_s = st.columns(2, gap="large")

            # --- LEFT RESULT CARD: WM-811K SPATIAL DIAGNOSIS ---
            with res_w:
                with st.container(border=True):
                    st.markdown(
                        """
                        <div class="card-header-strip">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2.2"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>
                            WM-811K Spatial Diagnosis
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
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
                            st.markdown(f'<div class="banner-warn">⚠️ {w_warn}</div>', unsafe_allow_html=True)

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
                            st.markdown("**🖼️ Resize CNN (B1) — Integrated Gradients Attribution (64 × 64)**")
                            fig_ig = px.imshow(
                                b1_ig,
                                color_continuous_scale="Inferno",
                            )
                            fig_ig.update_layout(
                                height=255,
                                margin=dict(l=6, r=6, t=6, b=6),
                                paper_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_ig, width="stretch")

                        if top_cause:
                            with st.expander("📖 Literature Process Hypothesis & Caveats (Classical Model)", expanded=True):
                                st.markdown(f"**Literature Hypothesis:** {top_cause.get('possible_causes', 'N/A')}")
                                st.markdown(f"**Suggested Check (Author-Inferred):** {top_cause.get('suggested_check_AUTHOR_INFERRED', 'N/A')}")
                                st.markdown(f"**Evidence Level:** `{top_cause.get('evidence_level', 'N/A')}` | **Sources:** {top_cause.get('references', 'N/A')}")
                                if top_cause.get("caveat"):
                                    st.markdown(f'<div class="banner-warn">⚠️ <strong>Caveat:</strong> {top_cause["caveat"]}</div>', unsafe_allow_html=True)
                                if top_cause.get("disclaimer"):
                                    st.caption(top_cause["disclaimer"])

            # --- RIGHT RESULT CARD: UCI SECOM TELEMETRY SCREENING ---
            with res_s:
                with st.container(border=True):
                    st.markdown(
                        """
                        <div class="card-header-strip">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2.2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                            UCI SECOM Telemetry Screening
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
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
                            st.markdown(f'<div class="banner-warn">⚠️ {s_warn}</div>', unsafe_allow_html=True)

                        sc = s_data["scores"]
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(
                                f"""
                                <div class="metric-subcard">
                                    <div class="metric-subcard-title">🎯 Fused Risk Score</div>
                                    <div class="metric-subcard-val">{sc['fused_telemetry_risk']:.4f}</div>
                                    <div class="metric-cutoff-pill">↑ Top-10% Cutoff: {s_data['threshold']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with m2:
                            st.markdown(
                                f"""
                                <div class="metric-subcard">
                                    <div class="metric-subcard-title">🌲 Balanced RF</div>
                                    <div class="metric-subcard-val">{sc['prob_balanced_rf']:.4f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with m3:
                            st.markdown(
                                f"""
                                <div class="metric-subcard">
                                    <div class="metric-subcard-title">⚡ SPE-DT (Optuna)</div>
                                    <div class="metric-subcard-val">{sc['prob_spe_dt']:.4f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        tier = s_data["risk_tier"]
                        if "High Risk" in tier:
                            st.markdown(f'<div class="banner-danger">🚨 {tier}</div>', unsafe_allow_html=True)
                        elif "Elevated" in tier:
                            st.markdown(f'<div class="banner-warn">⚠️ {tier}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="banner-success">✅ {tier}</div>', unsafe_allow_html=True)

                        expl = s_data.get("explanation")
                        if expl and expl.get("drivers"):
                            st.markdown("**📊 Local TreeSHAP Sensor Drivers (★ = Top-10 Stable Sensor)**")
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
                                color="shap_value",
                                color_continuous_scale="RdBu_r",
                                color_continuous_midpoint=0.0,
                            )
                            fig_shap.update_layout(
                                height=290,
                                margin=dict(l=6, r=6, t=6, b=6),
                                showlegend=False,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_shap, width="stretch")
                            st.caption(expl.get("note", ""))


# ==============================================================================
# TAB 2: MODEL CARDS & BENCHMARKS
# ==============================================================================
with tab_about:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin:0.2rem 0 0.7rem 0;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6D28D9" stroke-width="2.2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
            <h3 style="margin:0;color:#1E1B4B;font-weight:800;">Model Cards, Evaluation Protocols &amp; Scientific Limitations</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        cards_resp = get_gateway_client().get("/models").json()
        st.markdown(
            f'<div class="banner-info">ℹ️ {cards_resp.get("separation_policy", "")}</div>',
            unsafe_allow_html=True,
        )

        for mid, card in cards_resp.get("model_cards", {}).items():
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="card-header-strip">
                        <span style="background:#FFFFFF;color:#6D28D9;border:1px solid #C4B5FD;padding:2px 12px;border-radius:999px;font-size:0.84rem;font-weight:800;">{mid}</span>
                        <span>— {card['title']} ({card['domain']})</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c_left, c_right = st.columns([1.05, 1.15], gap="large")
                with c_left:
                    st.markdown('<div class="mc-label">⚙️ Architecture</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="mc-arch-pill">{card["architecture"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="mc-label">🎯 Evaluation Protocol &amp; Metrics</div>', unsafe_allow_html=True)
                    st.caption(card["evaluation_protocol"])
                    st.json(card["metrics"])

                with c_right:
                    lims_html = "".join(f"<li>{lim}</li>" for lim in card["limitations"])
                    st.markdown(
                        f"""
                        <div class="mc-box">
                            <div class="mc-label" style="font-size:0.98rem;margin-bottom:10px;">🚩 Known Limitations</div>
                            <ul class="lim-list">{lims_html}</ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.write("")

    except Exception as exc:
        st.warning(f"Could not fetch live model cards from Gateway: {exc}")

    st.divider()
    st.subheader("SECOM Cross-Protocol Benchmark Tables & Global SHAP Figures")
    t1_path = SECOM_DATA_DIR / "table1_protocol_a_summary.csv"
    t2_path = SECOM_DATA_DIR / "table2_protocol_b_summary.csv"
    t3_path = SECOM_DATA_DIR / "table3_shap_sensor_stability.csv"

    c_t1, c_t2 = st.columns(2, gap="large")
    if t1_path.exists():
        with c_t1:
            with st.container(border=True):
                st.markdown('<div class="card-header-strip">Table 1 — Protocol A (10 Repeated Stratified 80/20 Splits)</div>', unsafe_allow_html=True)
                st.dataframe(pd.read_csv(t1_path), width="stretch", hide_index=True)
    if t2_path.exists():
        with c_t2:
            with st.container(border=True):
                st.markdown('<div class="card-header-strip">Table 2 — Protocol B (3 Expanding-Window Chronological Splits)</div>', unsafe_allow_html=True)
                st.dataframe(pd.read_csv(t2_path), width="stretch", hide_index=True)

    st.write("")
    if t3_path.exists():
        with st.container(border=True):
            st.markdown('<div class="card-header-strip">Table 3 — Cross-Split Top SHAP Sensor Stability</div>', unsafe_allow_html=True)
            st.dataframe(pd.read_csv(t3_path), width="stretch", hide_index=True)

    st.write("")
    f1_img = ASSETS_DIR / "fig1_final_shap_global_summary.png"
    f2_img = ASSETS_DIR / "fig2_final_shap_local_waterfall.png"
    c_f1, c_f2 = st.columns(2, gap="large")
    if f1_img.exists():
        with c_f1:
            with st.container(border=True):
                st.markdown('<div class="card-header-strip">Global SHAP Summary (Protocol B Holdout)</div>', unsafe_allow_html=True)
                st.image(str(f1_img), width="stretch")
    if f2_img.exists():
        with c_f2:
            with st.container(border=True):
                st.markdown('<div class="card-header-strip">Example Local SHAP Waterfall (Reference Holdout Run)</div>', unsafe_allow_html=True)
                st.image(str(f2_img), width="stretch")