$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m venv .venv
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m venv .venv
} else {
    throw "Python 3.11+ was not found."
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SitePackages = & $Python -c "import site; print(site.getsitepackages()[0])"
Set-Content -Path (Join-Path $SitePackages "moevm_lab.pth") -Value (Join-Path $RepoRoot "src") -Encoding UTF8

& $Python -m unittest discover -s tests -v
& $Python -m moevm compare --config configs\toy.toml --output-dir results\toy

Write-Host ""
Write-Host "MoEVM Lab is ready."
Write-Host "Run: .\.venv\Scripts\python.exe -m moevm --help"
Write-Host "Optional standard install: .\.venv\Scripts\python.exe -m pip install -e ."
