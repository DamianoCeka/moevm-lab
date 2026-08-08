$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$env:PYTHONUTF8 = "1"

function Assert-NativeSuccess {
    param([string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m venv .venv
    Assert-NativeSuccess "Creating the virtual environment"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m venv .venv
    Assert-NativeSuccess "Creating the virtual environment"
} else {
    throw "Python 3.11+ was not found."
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SitePackages = & $Python -c "import site; print(site.getsitepackages()[0])"
Set-Content -Path (Join-Path $SitePackages "moevm_lab.pth") -Value (Join-Path $RepoRoot "src") -Encoding UTF8

& $Python -m unittest discover -s tests -v
Assert-NativeSuccess "Unit tests"
& $Python -m moevm compare --config configs\toy.toml --tokens 64 --output-dir results\toy
Assert-NativeSuccess "Reference simulation"

Write-Host ""
Write-Host "MoEVM Lab is ready."
Write-Host "Run: .\.venv\Scripts\python.exe -m moevm --help"
Write-Host "Optional standard install: .\.venv\Scripts\python.exe -m pip install -e ."
