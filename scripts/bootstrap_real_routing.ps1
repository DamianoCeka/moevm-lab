param(
    [string]$VenvPath = (Join-Path $PSScriptRoot "..\.venv-real"),
    [string]$CachePath = (Join-Path $PSScriptRoot "..\.cache\huggingface")
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONUTF8 = "1"

function Assert-NativeSuccess {
    param([string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The Python launcher is required; install Python 3.12 first."
}

py -3.12 -m venv $VenvPath
Assert-NativeSuccess "Creating the real-routing virtual environment"
$Python = Join-Path (Resolve-Path $VenvPath).Path "Scripts\python.exe"

& $Python -m pip install --upgrade pip
Assert-NativeSuccess "Updating pip"
& $Python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
Assert-NativeSuccess "Installing CUDA PyTorch"
& $Python -m pip install -e "${RepoRoot}[real-traces]"
Assert-NativeSuccess "Installing real-routing dependencies"

New-Item -ItemType Directory -Force -Path $CachePath | Out-Null
$env:HF_HOME = (Resolve-Path $CachePath).Path
& $Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
Assert-NativeSuccess "Validating CUDA"

Write-Host ""
Write-Host "Real-routing environment is ready."
Write-Host "Set: `$env:HF_HOME = '$env:HF_HOME'"
Write-Host "Run: $Python scripts\capture_real_routing.py --download-only"
