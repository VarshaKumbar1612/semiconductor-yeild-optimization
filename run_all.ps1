$ErrorActionPreference = "Stop"
$Root = "C:\Semi_yeild_opti"
Set-Location $Root

Write-Host "Starting WM-811K Spatial Service (:8001)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\.venv_wm\Scripts\Activate.ps1; uvicorn services.wafer.main:app --host 127.0.0.1 --port 8001"

Write-Host "Starting SECOM Telemetry Service (:8002)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\.venv_secom\Scripts\Activate.ps1; uvicorn services.secom.main:app --host 127.0.0.1 --port 8002"

Write-Host "Starting Unified Gateway (:8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\.venv_wm\Scripts\Activate.ps1; uvicorn gateway.main:app --host 127.0.0.1 --port 8000"

Write-Host "Waiting for backend services and gateway to come online..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        if ($h.status -eq "ok") {
            $ready = $true
            break
        }
    } catch {}
}

if ($ready) {
    Write-Host "All backend services healthy! Launching Streamlit UI (:8501)..." -ForegroundColor Green
} else {
    Write-Host "Warning: Backends still warming up; launching Streamlit UI (:8501) anyway..." -ForegroundColor Yellow
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\.venv_wm\Scripts\Activate.ps1; streamlit run frontend/app.py --server.port 8501"