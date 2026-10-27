# Run MindWave React Frontend on port 3000
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }
Set-Location $ProjectRoot

Write-Host "Starting MindWave Frontend on http://localhost:3000..." -ForegroundColor Cyan
npm run dev
