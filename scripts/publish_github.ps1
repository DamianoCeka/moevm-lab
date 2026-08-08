param(
    [string]$Repository = "DamianoCeka/moevm-lab",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Assert-NativeSuccess {
    param([string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

git rev-parse --is-inside-work-tree *> $null
Assert-NativeSuccess "Checking the Git repository"

$DirtyEntries = @(git status --porcelain=v1 --untracked-files=all)
Assert-NativeSuccess "Checking the working tree"
if ($DirtyEntries.Count -ne 0) {
    git status --short
    throw "The working tree is not clean. Commit or remove every change before publishing."
}

$Branch = (git branch --show-current).Trim()
Assert-NativeSuccess "Reading the current branch"
if ($Branch -ne "main") {
    throw "The publisher only releases main; current branch is '$Branch'."
}

$HeadCommit = (git rev-parse HEAD).Trim()
Assert-NativeSuccess "Reading HEAD"
$MainCommit = (git rev-parse refs/heads/main).Trim()
Assert-NativeSuccess "Reading main"
if ($HeadCommit -ne $MainCommit) {
    throw "HEAD ($HeadCommit) and main ($MainCommit) do not point to the same commit."
}

$ProjectVersion = $null
$InProjectSection = $false
foreach ($Line in Get-Content -LiteralPath "pyproject.toml") {
    if ($Line -match '^\s*\[([^]]+)\]\s*$') {
        $InProjectSection = $Matches[1] -eq "project"
        continue
    }
    if ($InProjectSection -and $Line -match '^\s*version\s*=\s*"([^"]+)"\s*$') {
        $ProjectVersion = $Matches[1]
        break
    }
}
if ([string]::IsNullOrWhiteSpace($ProjectVersion)) {
    throw "Could not read [project].version from pyproject.toml."
}

$ReleaseTag = "v$ProjectVersion"
git check-ref-format "refs/tags/$ReleaseTag"
Assert-NativeSuccess "Validating release tag '$ReleaseTag'"
git show-ref --verify --quiet "refs/tags/$ReleaseTag"
if ($LASTEXITCODE -ne 0) {
    throw "Release tag '$ReleaseTag' does not exist locally. Create it on the final release commit first."
}
$TagCommit = (git rev-parse "$ReleaseTag^{commit}").Trim()
Assert-NativeSuccess "Resolving release tag '$ReleaseTag'"
if ($TagCommit -ne $HeadCommit) {
    throw "Release tag '$ReleaseTag' points to $TagCommit, not HEAD $HeadCommit."
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is required. Install it, then run 'gh auth login'."
}

gh auth status
Assert-NativeSuccess "GitHub authentication"

$Visibility = if ($Public) { "--public" } else { "--private" }
$OriginUrl = git remote get-url origin 2>$null | Select-Object -First 1
$HasOrigin = $LASTEXITCODE -eq 0

if ($HasOrigin -and $OriginUrl.EndsWith(".bundle")) {
    $SourceRemote = "source-bundle"
    if ((git remote) -contains $SourceRemote) {
        $SourceRemote = "source-bundle-$(Get-Date -Format 'yyyyMMddHHmmss')"
    }
    git remote rename origin $SourceRemote
    Assert-NativeSuccess "Preserving the bundle remote"
    Write-Host "Preserved the local bundle remote as '$SourceRemote'."
    $HasOrigin = $false
}

$EscapedRepository = [regex]::Escape($Repository)
$ExpectedOrigin = "^(https://github\.com/$EscapedRepository(?:\.git)?|git@github\.com:$EscapedRepository(?:\.git)?)$"
if ($HasOrigin -and $OriginUrl -notmatch $ExpectedOrigin) {
    throw "origin points to '$OriginUrl', not github.com/$Repository. Refusing to overwrite it."
}

if (-not $HasOrigin) {
    & gh repo create $Repository $Visibility --source . --remote origin
    Assert-NativeSuccess "Creating the GitHub repository"
}

git push --set-upstream origin HEAD:main
Assert-NativeSuccess "Pushing HEAD to main"

git push origin "refs/tags/${ReleaseTag}:refs/tags/${ReleaseTag}"
Assert-NativeSuccess "Pushing release tag '$ReleaseTag'"

Write-Host "Published $Repository from $ReleaseTag ($HeadCommit)."
