# Start both Backend and Frontend in separate PowerShell processes
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }
Set-Location $ProjectRoot

Write-Host "Launching MindWave Backend and Frontend services..." -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $ProjectRoot "run_backend.ps1")
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $ProjectRoot "run_frontend.ps1")

Write-Host "`nServices started!" -ForegroundColor Green
Write-Host "  Backend:  http://127.0.0.1:5000/api/ml/health" -ForegroundColor White
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
