# ==============================================================================
# MindWave Response Analyzer - Automated Setup & Pipeline Script (XGBoost Only)
# ==============================================================================
[CmdletBinding()]
param (
    [switch]$ForceRetrain = $false,
    [switch]$SkipTests = $false
)

$ErrorActionPreference = "Stop"

# Detect project root directory automatically
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = (Get-Location).Path
}
Set-Location $ProjectRoot
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " MindWave Response Analyzer: Automated Setup (XGBoost Only)" -ForegroundColor Cyan
Write-Host " Project Directory: $ProjectRoot" -ForegroundColor Gray
Write-Host "==============================================================" -ForegroundColor Cyan

# 1. Environment & Version Checks
Write-Host "`n[1/7] Verifying System Prerequisites..." -ForegroundColor Yellow
$PythonCmd = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $PythonCmd) {
    Write-Error "Python is not found in PATH. Please install Python 3.10+ and add it to PATH."
}
$PythonVersion = python --version 2>&1
Write-Host "  Found: $PythonVersion" -ForegroundColor Green

$NodeCmd = (Get-Command node -ErrorAction SilentlyContinue)
if (-not $NodeCmd) {
    Write-Error "Node.js is not found in PATH. Please install Node.js 18+ and add it to PATH."
}
$NodeVersion = node --version 2>&1
Write-Host "  Found Node.js: $NodeVersion" -ForegroundColor Green

# 2. Virtual Environment Setup
Write-Host "`n[2/7] Checking Python Virtual Environment (.venv)..." -ForegroundColor Yellow
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "  Creating new virtual environment at .venv..." -ForegroundColor Cyan
    python -m venv .venv
}
Write-Host "  Virtual environment ready at: $VenvPython" -ForegroundColor Green

# 3. Dependency Installation
Write-Host "`n[3/7] Installing Dependencies..." -ForegroundColor Yellow
Write-Host "  Installing Python dependencies from ml/requirements.txt..." -ForegroundColor Cyan
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "ml\requirements.txt") | Out-Null
Write-Host "  Python packages verified (XGBoost, scikit-learn, fastapi, uvicorn)." -ForegroundColor Green

if (-not (Test-Path (Join-Path $ProjectRoot "node_modules"))) {
    Write-Host "  Installing Node.js dependencies via npm install..." -ForegroundColor Cyan
    npm install
} else {
    Write-Host "  Node.js dependencies already installed in node_modules." -ForegroundColor Green
}

# 4. Dataset Acquisition
Write-Host "`n[4/7] Checking Research Dataset (Akram et al. 2023, CC BY 4.0)..." -ForegroundColor Yellow
$RawExcel = Join-Path $ProjectRoot "ml\data\raw\Akram et al 2023 Student MH Data UK SciData.xlsx"
if (-not (Test-Path $RawExcel)) {
    Write-Host "  Acquiring dataset from official Figshare repository..." -ForegroundColor Cyan
    & $VenvPython (Join-Path $ProjectRoot "ml\src\download_dataset.py")
} else {
    Write-Host "  Raw dataset already present at ml/data/raw/." -ForegroundColor Green
}

# 5. Data Preprocessing
Write-Host "`n[5/7] Checking Preprocessed Data Splits..." -ForegroundColor Yellow
$PqTrain = Join-Path $ProjectRoot "ml\data\processed\pq16_train.csv"
if ($ForceRetrain -or (-not (Test-Path $PqTrain))) {
    Write-Host "  Running preprocessing and stratified data splitting..." -ForegroundColor Cyan
    & $VenvPython (Join-Path $ProjectRoot "ml\src\preprocess.py")
} else {
    Write-Host "  Processed dataset splits already exist. (Use -ForceRetrain to re-split)." -ForegroundColor Green
}

# 6. Model Training & Evaluation
Write-Host "`n[6/7] Checking Trained XGBoost Artifacts..." -ForegroundColor Yellow
$PqModel = Join-Path $ProjectRoot "ml\models\pq16_xgboost_model.json"
$MdqModel = Join-Path $ProjectRoot "ml\models\mdq_xgboost_model.json"

if ($ForceRetrain -or (-not (Test-Path $PqModel)) -or (-not (Test-Path $MdqModel))) {
    Write-Host "  Training XGBoost models for PQ-16 and MDQ tasks..." -ForegroundColor Cyan
    & $VenvPython (Join-Path $ProjectRoot "ml\src\train.py")
    Write-Host "  Evaluating models on held-out test data..." -ForegroundColor Cyan
    & $VenvPython (Join-Path $ProjectRoot "ml\src\evaluate.py")
} else {
    Write-Host "  Valid XGBoost artifacts exist at ml/models/. Skipping training." -ForegroundColor Green
    Write-Host "  (To explicitly retrain, run: .\setup.ps1 -ForceRetrain)" -ForegroundColor DarkGray
}

# 7. Model Inference Verification & Testing
Write-Host "`n[7/7] Verifying Model Inference & Tests..." -ForegroundColor Yellow
& $VenvPython (Join-Path $ProjectRoot "ml\src\predict.py")

if (-not $SkipTests) {
    Write-Host "`n  Running automated pytest test suite..." -ForegroundColor Cyan
    & $VenvPython -m pytest (Join-Path $ProjectRoot "ml\tests") -v
}

Write-Host "`n==============================================================" -ForegroundColor Green
Write-Host " Setup & Verification Complete!" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "`nTo start the services:" -ForegroundColor Cyan
Write-Host "  Backend:   .\.venv\Scripts\python.exe ml\app.py" -ForegroundColor White
Write-Host "  Frontend:  npm run dev" -ForegroundColor White
Write-Host "  Or run:    .\run_all.ps1" -ForegroundColor Yellow
