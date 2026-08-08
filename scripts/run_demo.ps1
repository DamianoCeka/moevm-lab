$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run scripts\bootstrap.ps1 first."
}

& $Python -m moevm compare --config configs\toy.toml --tokens 64 --output-dir results\toy
if ($LASTEXITCODE -ne 0) {
    throw "Reference simulation failed with exit code $LASTEXITCODE."
}
