import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "services.yaml"
CARDS_PATH = ROOT_DIR / "model_cards" / "cards.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    SERVICES_CFG = yaml.safe_load(f)["services"]

with open(CARDS_PATH, "r", encoding="utf-8") as f:
    MODEL_CARDS = yaml.safe_load(f)["models"]

app = FastAPI(title="Semiconductor Yield Diagnostics Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class WaferInput(BaseModel):
    matrix: List[List[Any]]
    models: Optional[List[str]] = None
    explain: bool = True


class SensorInput(BaseModel):
    values: List[Optional[float]]
    columns: Optional[List[str]] = None
    explain: bool = True


class AnalyzeRequest(BaseModel):
    wafer: Optional[WaferInput] = None
    sensor: Optional[SensorInput] = None


async def _call_service(
    client: httpx.AsyncClient,
    service_key: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = SERVICES_CFG[service_key]
    if not cfg.get("enabled", True):
        return {"status": "skipped", "reason": "Service disabled in configs/services.yaml"}

    url = f"{cfg['url']}{endpoint}"
    t0 = time.perf_counter()
    try:
        if payload is None:
            resp = await client.get(url, timeout=cfg["timeout_sec"])
        else:
            resp = await client.post(url, json=payload, timeout=cfg["timeout_sec"])

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        if resp.status_code >= 400:
            return {
                "status": "error",
                "http_status": resp.status_code,
                "detail": resp.json().get("detail", resp.text),
                "latency_ms": latency_ms,
            }
        return {"status": "ok", "data": resp.json(), "latency_ms": latency_ms}
    except httpx.TimeoutException:
        return {"status": "error", "detail": f"Timed out after {cfg['timeout_sec']}s"}
    except httpx.RequestError as exc:
        return {"status": "error", "detail": f"Service unreachable ({exc.__class__.__name__})"}


@app.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        wafer_res, secom_res = await asyncio.gather(
            _call_service(client, "wafer", "/health"),
            _call_service(client, "secom", "/health"),
        )
    overall = "ok" if (wafer_res["status"] == "ok" and secom_res["status"] == "ok") else "degraded"
    return {
        "status": overall,
        "gateway": "ok",
        "services": {
            "wafer": wafer_res,
            "secom": secom_res,
        },
    }


@app.get("/models")
async def models():
    async with httpx.AsyncClient() as client:
        wafer_res, secom_res = await asyncio.gather(
            _call_service(client, "wafer", "/models"),
            _call_service(client, "secom", "/models"),
        )
    return {
        "model_cards": MODEL_CARDS,
        "live_services": {
            "wafer": wafer_res,
            "secom": secom_res,
        },
        "separation_policy": "WM-811K wafer maps and UCI SECOM sensor vectors come from separate datasets and are never fused.",
    }


@app.post("/validate")
async def validate(req: AnalyzeRequest):
    async with httpx.AsyncClient() as client:
        tasks = []
        keys = []
        if req.wafer is not None:
            keys.append("wafer")
            tasks.append(_call_service(client, "wafer", "/validate", req.wafer.model_dump()))
        if req.sensor is not None:
            keys.append("secom")
            tasks.append(_call_service(client, "secom", "/validate", req.sensor.model_dump()))

        results = await asyncio.gather(*tasks) if tasks else []

    out: Dict[str, Any] = {
        "wafer": {"status": "skipped", "reason": "No wafer matrix provided."},
        "secom": {"status": "skipped", "reason": "No sensor vector provided."},
    }
    for k, res in zip(keys, results):
        out[k] = res
    return out


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = []
        keys = []
        if req.wafer is not None:
            keys.append("wafer")
            tasks.append(_call_service(client, "wafer", "/predict", req.wafer.model_dump()))
        if req.sensor is not None:
            keys.append("secom")
            tasks.append(_call_service(client, "secom", "/predict", req.sensor.model_dump()))

        results = await asyncio.gather(*tasks) if tasks else []

    panels: Dict[str, Any] = {
        "wafer": {"status": "skipped", "reason": "No wafer matrix submitted."},
        "secom": {"status": "skipped", "reason": "No sensor telemetry row submitted."},
    }
    for k, res in zip(keys, results):
        panels[k] = res

    return {
        "panels": panels,
        "total_gateway_ms": round((time.perf_counter() - t0) * 1000, 2),
        "disclaimer": "Independent diagnostic panels: WM-811K spatial pattern classification and UCI SECOM telemetry screening are unpaired datasets and are never combined.",
    }