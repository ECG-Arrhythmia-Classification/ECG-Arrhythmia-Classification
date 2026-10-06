param([switch]$NoBrowser, [ValidateSet('trained','demo')][string]$Runtime = 'trained')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run .\web\scripts\setup.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'web\node_modules\vite\bin\vite.js'))) { throw 'Run setup.ps1 to install frontend dependencies.' }
foreach ($port in @(8000,5173)) {
    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $port)
    try { $listener.Start() } catch { throw "Port $port is in use. Stop the existing server before starting this demo." } finally { $listener.Stop() }
}
$logDirectory = Join-Path $projectRoot '.local'
$null = New-Item -ItemType Directory -Path $logDirectory -Force
$previousRuntime = $env:ECG_RUNTIME
$previousApi = $env:VITE_API_URL
$apiProcess = $null
$webProcess = $null
try {
    $env:ECG_RUNTIME = $Runtime
    $env:VITE_API_URL = 'http://127.0.0.1:8000'
    $apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','uvicorn','app:app','--app-dir','web/backend','--host','127.0.0.1','--port','8000') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'api.log') -RedirectStandardError (Join-Path $logDirectory 'api-error.log')
    $webProcess = Start-Process -FilePath (Get-Command node).Source -ArgumentList @('web/node_modules/vite/bin/vite.js','web','--host','127.0.0.1','--port','5173','--strictPort') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'web.log') -RedirectStandardError (Join-Path $logDirectory 'web-error.log')
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($apiProcess.HasExited -or $webProcess.HasExited) { throw "A server stopped. See logs in $logDirectory." }
        try {
            $health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2
            $null = Invoke-WebRequest 'http://127.0.0.1:5173/' -TimeoutSec 2 -UseBasicParsing
            if ($Runtime -eq 'trained' -and @($health.models | Where-Object { $_.status -ne 'ready' -or $_.is_demo }).Count -gt 0) { throw 'Trained checkpoints are unavailable.' }
            $ready = $true
            break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw "Servers did not become ready. See logs in $logDirectory; check PyTorch and checkpoint paths." }
    Write-Host 'ECG Studio: http://127.0.0.1:5173'
    Write-Host "API: http://127.0.0.1:8000/docs | Runtime: $Runtime"
    Write-Host 'Keep this terminal open. Press Ctrl+C to stop both servers.'
    if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173' }
    while (-not $apiProcess.HasExited -and -not $webProcess.HasExited) { Start-Sleep -Seconds 1 }
    throw "A server stopped. See logs in $logDirectory."
} finally {
    foreach ($process in @($apiProcess,$webProcess)) {
        if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -ErrorAction SilentlyContinue }
    }
    $env:ECG_RUNTIME = $previousRuntime
    $env:VITE_API_URL = $previousApi
}
