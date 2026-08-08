param(
    [string]$Repository = "DamianoCeka/moevm-lab",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is required. Install it, then run 'gh auth login'."
}

gh auth status

if (-not (Test-Path ".git")) {
    git init -b main
    git add .
    git commit -m "Initialize MoEVM Lab v0.1"
}

$Visibility = if ($Public) { "--public" } else { "--private" }
$RemoteExists = git remote 2>$null | Select-String -SimpleMatch "origin"
if (-not $RemoteExists) {
    & gh repo create $Repository $Visibility --source . --remote origin --push
} else {
    git push -u origin main
}

Write-Host "Published $Repository"
