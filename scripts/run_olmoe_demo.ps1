[CmdletBinding()]
param(
    [string]$PythonPath,
    [string]$CachePath,
    [string]$OutputRoot,
    [ValidateRange(0, 15)]
    [int]$Device = 0,
    [switch]$Compare,
    [switch]$Offline,
    [switch]$Yes,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$PinnedRevision = "bd1c52f59153f724c1ad11ca1791edc77bab3806"
$ModelId = "allenai/OLMoE-1B-7B-0924"
$RequiredModelCacheBytes = 30GB
$RequiredEnvironmentBytes = 5GB
$MinimumRamBytes = 16GB
$MinimumAvailableRamBytes = 8GB
$MinimumVramMiB = 8192
$MinimumFreeVramMiB = 4096

$ShardSizes = @{
    "model-00001-of-00003.safetensors" = 4997744872L
    "model-00002-of-00003.safetensors" = 4997235176L
    "model-00003-of-00003.safetensors" = 3843741912L
}

function Format-GiB {
    param([long]$Bytes)
    return "{0:N1} GiB" -f ($Bytes / 1GB)
}

function Assert-NativeSuccess {
    param([string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

function Get-SnapshotPath {
    param([string]$HfHome)
    return Join-Path $HfHome "hub\models--allenai--OLMoE-1B-7B-0924\snapshots\$PinnedRevision"
}

function Test-SnapshotReady {
    param([string]$HfHome)
    $snapshot = Get-SnapshotPath $HfHome
    foreach ($required in @(
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json"
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $snapshot $required) -PathType Leaf)) {
            return $false
        }
    }
    foreach ($entry in $ShardSizes.GetEnumerator()) {
        $path = Join-Path $snapshot $entry.Key
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            return $false
        }
        if ((Get-Item -LiteralPath $path).Length -ne $entry.Value) {
            return $false
        }
    }
    return $true
}

function Resolve-DemoCachePath {
    param([string]$Requested)
    if ($Requested) {
        return [System.IO.Path]::GetFullPath($Requested)
    }

    $candidates = [System.Collections.Generic.List[string]]::new()
    foreach ($candidate in @(
        $env:MOEVM_HF_HOME,
        $env:HF_HOME,
        (Join-Path $RepoRoot ".cache\huggingface"),
        (Join-Path ([Environment]::GetFolderPath("UserProfile")) ".cache\huggingface")
    )) {
        if ($candidate -and -not $candidates.Contains($candidate)) {
            $candidates.Add([System.IO.Path]::GetFullPath($candidate))
        }
    }
    foreach ($drive in [System.IO.DriveInfo]::GetDrives()) {
        try {
            if ($drive.IsReady -and $drive.DriveType -eq [System.IO.DriveType]::Fixed) {
                $candidate = Join-Path $drive.RootDirectory.FullName "moevm-lab-cache\huggingface"
                if (-not $candidates.Contains($candidate)) {
                    $candidates.Add($candidate)
                }
            }
        } catch {
            continue
        }
    }
    foreach ($candidate in $candidates) {
        if (Test-SnapshotReady $candidate) {
            return $candidate
        }
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot ".cache\huggingface"))
}

function Test-DemoPythonEnvironment {
    param([string]$Candidate)
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }
    $probe = @'
import importlib.metadata as metadata
import sys
import torch

assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == '2.12.1+cu130'
assert metadata.version('accelerate') == '1.14.0'
assert metadata.version('huggingface-hub') == '1.27.0'
assert metadata.version('safetensors') == '0.8.0'
assert metadata.version('transformers') == '5.14.1'
'@
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Candidate -c $probe 1> $null 2> $null
        $probeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    return $probeExitCode -eq 0
}

function Assert-DemoPythonGpu {
    param(
        [string]$Candidate,
        [int]$DeviceIndex,
        [string]$ExpectedUuid
    )
    $probe = @'
import sys
import torch

index = int(sys.argv[1])
expected_uuid = sys.argv[2].lower().removeprefix('gpu-')
if not torch.cuda.is_available():
    print('cuda_unavailable')
    raise SystemExit(10)
if not 0 <= index < torch.cuda.device_count():
    print('device_index_unavailable')
    raise SystemExit(11)
with torch.cuda.device(index):
    properties = torch.cuda.get_device_properties(index)
    actual_uuid = str(properties.uuid).lower()
    if actual_uuid != expected_uuid:
        print('uuid_mismatch:' + actual_uuid)
        raise SystemExit(12)
    if not torch.cuda.is_bf16_supported():
        print('bf16_unsupported')
        raise SystemExit(13)
print('ok')
'@
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $probeOutput = & $Candidate -c $probe $DeviceIndex $ExpectedUuid 2> $null
        $probeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($probeExitCode -eq 0) {
        return
    }
    $detail = [string]::Join("", @($probeOutput))
    if ($probeExitCode -eq 12) {
        $mapping = $(if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "not set" })
        throw "CUDA device mapping mismatch for -Device $DeviceIndex ($detail). CUDA_VISIBLE_DEVICES=$mapping. Clear the remapping or select the matching device; no model was run."
    }
    if ($probeExitCode -eq 13) {
        throw "The CUDA device selected by PyTorch does not support BF16."
    }
    throw "The selected CUDA device is unavailable to the pinned PyTorch environment ($detail)."
}

function Get-ManagedPythonPaths {
    $managedRoot = Join-Path $RepoRoot ".cache\demo\py312-cu130"
    return @(
        (Join-Path $managedRoot "venv\Scripts\python.exe"),
        (Join-Path $managedRoot "venv-recovery-1\Scripts\python.exe"),
        (Join-Path $managedRoot "venv-recovery-2\Scripts\python.exe"),
        (Join-Path $managedRoot "venv-recovery-3\Scripts\python.exe")
    )
}

function Get-PythonCandidates {
    $candidates = [System.Collections.Generic.List[string]]::new()
    foreach ($candidate in @(
        $env:MOEVM_DEMO_PYTHON,
        $(if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" }),
        (Join-Path $RepoRoot ".venv-real\Scripts\python.exe")
    )) {
        if ($candidate -and -not $candidates.Contains($candidate)) {
            $candidates.Add([System.IO.Path]::GetFullPath($candidate))
        }
    }
    foreach ($candidate in Get-ManagedPythonPaths) {
        if (-not $candidates.Contains($candidate)) {
            $candidates.Add([System.IO.Path]::GetFullPath($candidate))
        }
    }
    return $candidates
}

function Resolve-DemoPython {
    param(
        [string]$Requested,
        [bool]$Validate
    )
    if ($Requested) {
        $resolved = [System.IO.Path]::GetFullPath($Requested)
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "PythonPath does not exist: $resolved"
        }
        if ($Validate -and -not (Test-DemoPythonEnvironment $resolved)) {
            throw "PythonPath is not a compatible Python 3.12 CUDA demo environment: $resolved"
        }
        return [pscustomobject]@{ Path = $resolved; NeedsBootstrap = $false }
    }

    foreach ($candidate in Get-PythonCandidates) {
        if (-not $Validate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return [pscustomobject]@{ Path = $candidate; NeedsBootstrap = $false }
        }
        if ($Validate -and (Test-DemoPythonEnvironment $candidate)) {
            return [pscustomobject]@{ Path = $candidate; NeedsBootstrap = $false }
        }
    }
    foreach ($target in Get-ManagedPythonPaths) {
        $venvDirectory = Split-Path (Split-Path $target -Parent) -Parent
        if (-not (Test-Path -LiteralPath $venvDirectory)) {
            return [pscustomobject]@{ Path = $target; NeedsBootstrap = $true }
        }
    }
    throw "All managed demo environment paths are present but incompatible. Move the invalid .cache\demo\py312-cu130 environments aside and retry."
}

function Get-GpuInfo {
    param([int]$Index)
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $rows = & nvidia-smi `
            --query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,driver_version `
            --format=csv,noheader,nounits 2> $null
        $probeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($probeExitCode -ne 0) {
        return $null
    }
    foreach ($row in $rows) {
        $parts = $row -split ",\s*"
        if ($parts.Count -eq 7 -and [int]$parts[0] -eq $Index) {
            return [pscustomobject]@{
                Index = [int]$parts[0]
                Uuid = $parts[1]
                Name = $parts[2]
                TotalMiB = [long]$parts[3]
                FreeMiB = [long]$parts[4]
                ComputeCapability = $parts[5]
                DriverVersion = $parts[6]
                GpuCount = @($rows).Count
            }
        }
    }
    return $null
}

function Get-SystemMemoryInfo {
    try {
        $operatingSystem = Get-CimInstance Win32_OperatingSystem
        $installedMemory = (
            Get-CimInstance Win32_PhysicalMemory |
                Measure-Object -Property Capacity -Sum
        ).Sum
        if (-not $installedMemory -or [long]$installedMemory -le 0) {
            return $null
        }
        return [pscustomobject]@{
            TotalBytes = [long]$installedMemory
            AvailableBytes = [long]$operatingSystem.FreePhysicalMemory * 1KB
        }
    } catch {
        return $null
    }
}

function Get-FreeBytesForPath {
    param([string]$Path)
    $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path))
    return ([System.IO.DriveInfo]::new($root)).AvailableFreeSpace
}

function Install-DemoEnvironment {
    param([string]$TargetPython, [string]$HfHome)
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python 3.12 was not found. Install Python 3.12, then run demo.cmd again."
    }
    $scriptsDirectory = Split-Path $TargetPython -Parent
    $venvDirectory = Split-Path $scriptsDirectory -Parent
    if (Test-Path -LiteralPath $venvDirectory) {
        throw "Refusing to modify an existing incompatible demo environment: $venvDirectory"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $venvDirectory -Parent) | Out-Null
    & py -3.12 -m venv $venvDirectory
    Assert-NativeSuccess "Creating the isolated demo environment"
    & $TargetPython -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
    Assert-NativeSuccess "Installing CUDA PyTorch"
    & $TargetPython -m pip install -e ("{0}[real-traces]" -f $RepoRoot)
    Assert-NativeSuccess "Installing pinned MoEVM demo dependencies"
    New-Item -ItemType Directory -Force -Path $HfHome | Out-Null
}

$CachePath = Resolve-DemoCachePath $CachePath
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $RepoRoot "results\demo"
} else {
    $OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}
$snapshotReady = Test-SnapshotReady $CachePath
if (-not $DryRun -and $Offline -and -not $snapshotReady) {
    throw "Offline mode was requested, but the pinned checkpoint is not complete in $CachePath."
}
$gpu = Get-GpuInfo $Device
$memory = Get-SystemMemoryInfo
if (-not $DryRun) {
    if (-not $gpu) {
        throw "An NVIDIA GPU visible to nvidia-smi is required. Check the driver and -Device index."
    }
    if ($gpu.TotalMiB -lt $MinimumVramMiB -or $gpu.FreeMiB -lt $MinimumFreeVramMiB) {
        throw "Insufficient GPU memory: at least 8192 MiB total and 4096 MiB free are required."
    }
    $computeMajor = [int]($gpu.ComputeCapability -split "\.")[0]
    if ($computeMajor -lt 8) {
        throw "The selected GPU has compute capability $($gpu.ComputeCapability); this demo requires native BF16 hardware (8.0 or newer)."
    }
    if (-not $memory) {
        throw "Windows memory information is unavailable; the demo cannot enforce its RAM safety guard."
    }
    if (
        $memory.TotalBytes -lt $MinimumRamBytes -or
        $memory.AvailableBytes -lt $MinimumAvailableRamBytes
    ) {
        throw "Insufficient RAM: at least 16 GiB total and 8 GiB available are required."
    }
}
$python = Resolve-DemoPython $PythonPath (-not $DryRun)
if (-not $DryRun -and $Offline -and $python.NeedsBootstrap) {
    throw "Offline mode cannot create the demo environment. Provide a compatible environment with -PythonPath or MOEVM_DEMO_PYTHON."
}
if (
    -not $DryRun -and
    $python.NeedsBootstrap -and
    (
        $gpu.GpuCount -ne 1 -or
        $env:CUDA_VISIBLE_DEVICES -or
        $env:CUDA_DEVICE_ORDER
    )
) {
    throw "Automatic environment setup is disabled while CUDA device mapping is ambiguous (GPU count=$($gpu.GpuCount), CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER=$env:CUDA_DEVICE_ORDER). Clear the mapping or provide a compatible environment with -PythonPath; no environment was installed."
}
if (-not $DryRun -and -not $python.NeedsBootstrap) {
    Assert-DemoPythonGpu $python.Path $Device $gpu.Uuid
}
$cacheFreeDisk = Get-FreeBytesForPath $CachePath
$environmentDirectory = Split-Path (Split-Path $python.Path -Parent) -Parent
$environmentFreeDisk = Get-FreeBytesForPath $environmentDirectory

Write-Host ""
Write-Host "MoEVM one-command OLMoE demo" -ForegroundColor Cyan
Write-Host "  Model:       $ModelId@$($PinnedRevision.Substring(0, 12))"
Write-Host "  License:     Apache-2.0 (model card: https://huggingface.co/$ModelId)"
Write-Host "  Mode:        $(if ($Compare) { 'sync + async comparison' } else { 'async quick demo' })"
Write-Host "  Cache:       $CachePath"
Write-Host "  Checkpoint:  $(if ($snapshotReady) { 'cached; SHA-256 will be verified' } else { '12.9 GiB download required' })"
Write-Host "  Python:      $($python.Path)$(if ($python.NeedsBootstrap) { ' (will be created)' })"
if ($gpu) {
    Write-Host "  GPU:         $($gpu.Name) ($($gpu.FreeMiB) / $($gpu.TotalMiB) MiB free)"
} else {
    Write-Host "  GPU:         NVIDIA GPU $Device was not detected" -ForegroundColor Yellow
}
if ($memory) {
    Write-Host "  RAM:         $(Format-GiB $memory.AvailableBytes) / $(Format-GiB $memory.TotalBytes) available"
} else {
    Write-Host "  RAM:         unavailable (dry-run warning)" -ForegroundColor Yellow
}
Write-Host "  Output:      $OutputRoot"

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run only: no environment, cache, result, network, or CUDA context was created." -ForegroundColor Green
    exit 0
}

if ($gpu.FreeMiB -lt 6144) {
    Write-Warning "Less than 6144 MiB VRAM is free; the automatic planner will reduce the expert cache."
}
$modelDiskRequired = $(if ($snapshotReady) { 0L } else { $RequiredModelCacheBytes })
$environmentDiskRequired = $(if ($python.NeedsBootstrap) { $RequiredEnvironmentBytes } else { 0L })
$cacheRoot = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($CachePath))
$environmentRoot = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($environmentDirectory))
if ($cacheRoot -eq $environmentRoot) {
    $combinedRequired = $modelDiskRequired + $environmentDiskRequired
    if ($cacheFreeDisk -lt $combinedRequired) {
        throw "The cache/environment volume has $(Format-GiB $cacheFreeDisk) free; $(Format-GiB $combinedRequired) is required for this setup."
    }
} else {
    if ($cacheFreeDisk -lt $modelDiskRequired) {
        throw "The model-cache volume has $(Format-GiB $cacheFreeDisk) free; $(Format-GiB $modelDiskRequired) is required."
    }
    if ($environmentFreeDisk -lt $environmentDiskRequired) {
        throw "The repository volume has $(Format-GiB $environmentFreeDisk) free; $(Format-GiB $environmentDiskRequired) is required for the demo environment."
    }
}

$heavyAction = $python.NeedsBootstrap -or -not $snapshotReady
if ($heavyAction -and -not $Yes) {
    Write-Host ""
    Write-Host "The first run may install about 3 GiB of dependencies and download 12.9 GiB of model files."
    $answer = Read-Host "Continue? [y/N]"
    if ($answer -notmatch "^[Yy]$") {
        Write-Host "Cancelled. No model or environment was removed."
        exit 0
    }
}

if ($python.NeedsBootstrap) {
    Install-DemoEnvironment $python.Path $CachePath
    if (-not (Test-DemoPythonEnvironment $python.Path)) {
        throw "The isolated demo environment failed its pinned package validation."
    }
    Assert-DemoPythonGpu $python.Path $Device $gpu.Uuid
}

$oldHfHome = $env:HF_HOME
$oldPythonUtf8 = $env:PYTHONUTF8
try {
    $env:HF_HOME = $CachePath
    $env:PYTHONUTF8 = "1"
    $arguments = @(
        (Join-Path $RepoRoot "scripts\olmoe_demo.py"),
        "--cache", $CachePath,
        "--output-root", $OutputRoot,
        "--device", "cuda:$Device",
        "--slots", "auto"
    )
    if ($Compare) {
        $arguments += "--compare"
    }
    if ($Offline) {
        $arguments += "--offline"
    } else {
        $arguments += "--allow-download"
    }
    Push-Location $RepoRoot
    try {
        & $python.Path @arguments
        Assert-NativeSuccess "Running the MoEVM OLMoE demo"
    } finally {
        Pop-Location
    }
} finally {
    if ($null -eq $oldHfHome) {
        Remove-Item Env:HF_HOME -ErrorAction SilentlyContinue
    } else {
        $env:HF_HOME = $oldHfHome
    }
    if ($null -eq $oldPythonUtf8) {
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONUTF8 = $oldPythonUtf8
    }
}
