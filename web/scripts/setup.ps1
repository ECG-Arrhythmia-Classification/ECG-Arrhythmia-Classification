param([string]$PythonExecutable = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location $projectRoot
try {
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        if ($PythonExecutable) {
            & $PythonExecutable -m venv .venv
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -m venv .venv
        } else {
            & python -m venv .venv
        }
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create .venv. Install Python 3.12 or pass -PythonExecutable.' }
    }
    & $pythonPath -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch CPU installation failed.' }
    & $pythonPath -m pip install -r web/backend/requirements-trained.txt
    if ($LASTEXITCODE -ne 0) { throw 'API dependency installation failed.' }
    $npmCommand = if (Get-Command npm.cmd -ErrorAction SilentlyContinue) { 'npm.cmd' } else { 'npm' }
    & $npmCommand --prefix web ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    $xPath = Join-Path $projectRoot 'processed_data\split\X_test.npy'
    $dataReady = $false
    if (Test-Path -LiteralPath $xPath) {
        $stream = [System.IO.File]::OpenRead($xPath)
        try {
            $header = New-Object byte[] 6
            $null = $stream.Read($header, 0, 6)
            $dataReady = $header[0] -eq 147 -and [System.Text.Encoding]::ASCII.GetString($header, 1, 5) -eq 'NUMPY'
        } finally { $stream.Dispose() }
    }
    if (-not $dataReady) {
        & git lfs pull
        if ($LASTEXITCODE -ne 0) { throw 'Dataset download failed. Install Git LFS and run git lfs pull.' }
    }
    & $pythonPath verify_dataset.py
    if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed.' }
    Write-Host 'Setup complete. Run: .\web\scripts\run-local.ps1'
} finally { Pop-Location }
