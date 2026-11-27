# Multi-Modal Semiconductor Yield Diagnostics Platform

An isolated, fault-tolerant diagnostic workbench combining **WM-811K Spatial Wafer Map Pattern Classification** and **UCI SECOM Process Telemetry Risk Screening**. Because WM-811K wafer maps and UCI SECOM sensor logs originate from independent, unpaired datasets, both pipelines execute side-by-side and are never combined into a synthetic joint score.

---

## System Architecture

| Component | Port | Environment | Role |
| :--- | :--- | :--- | :--- |
| **WM-811K Spatial Service** (`services/wafer/main.py`) | `8001` | `.venv_wm` | Runs 3 models (`wafer_bar` 137-feature GBDT, `wafer_b1` 64×64 Resize CNN, `wafer_spp` Native-Size SPP CNN), input validation, Integrated Gradients saliency, and audited literature cause lookup. |
| **SECOM Telemetry Service** (`services/secom/main.py`) | `8002` | `.venv_secom` | Runs equal-weight Balanced Random Forest + Optuna-tuned SPE-DT ensemble (`0.5 / 0.5`), KNN(`k=5`) + RobustScaler + Consensus-RFECV (`63` selected features), inspection tiering, and local TreeSHAP attributions. |
| **Unified API Gateway** (`gateway/main.py`) | `8000` | `.venv_wm` | Orchestrates parallel async calls to `:8001` and `:8002`, enforces per-panel fault isolation, and serves Model Cards. |
| **Streamlit Workbench** (`frontend/app.py`) | `8501` | `.venv_wm` | Interactive UI for demo replay, custom `.npy`/`.csv` upload, side-by-side diagnostics, and benchmark tables. |

---

## Quick Start (One-Command Launch)

From `C:\Semi_yeild_opti` in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\run_all.ps1