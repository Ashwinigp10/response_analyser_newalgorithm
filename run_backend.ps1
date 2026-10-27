# Run MindWave Python FastAPI Backend on port 5000
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Please run .\setup.ps1 first."
}

Write-Host "Starting MindWave FastAPI Backend on http://127.0.0.1:5000..." -ForegroundColor Cyan
& $VenvPython -m uvicorn app:app --app-dir (Join-Path $ProjectRoot "ml") --host 127.0.0.1 --port 5000
