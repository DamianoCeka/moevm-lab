[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Snapshot,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$BaselineDir,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputDir,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-fA-F]{40}$")]
    [string]$SourceCommit,

    [ValidateNotNullOrEmpty()]
    [string]$Python = "D:\moevm-lab-envs\m1\Scripts\python.exe",

    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$BenchmarkScript = Join-Path $RepoRoot "scripts\benchmark_paged_olmoe.py"
$WorkloadFile = Join-Path $RepoRoot "benchmarks\workloads\olmoe_m1.json"
$ExpectedRevision = "bd1c52f59153f724c1ad11ca1791edc77bab3806"
$ExpectedModelId = "allenai/OLMoE-1B-7B-0924"
$ExpectedShardSha256 = @{
    "model-00001-of-00003.safetensors" = "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e"
    "model-00002-of-00003.safetensors" = "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7"
    "model-00003-of-00003.safetensors" = "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04"
}
$WorkloadIds = @(
    "systems_en",
    "systems_it",
    "python_code",
    "math_reasoning",
    "domain_switch"
)
$CapacityOrders = @{
    1 = @(16, 24, 40, 32)
    2 = @(32, 40, 24, 16)
    3 = @(24, 16, 32, 40)
}

# These constants intentionally mirror benchmark_paged_olmoe.py and the pinned
# OLMoE checkpoint. SourceCommit prevents mixing this launcher with other code.
[int64]$ExpertBytes = 12582912
[int64]$LayerCount = 16
[int64]$NonExpertCheckpointBytes = 953421824
[int64]$VramSafetyMarginBytes = 1342177280

function Invoke-GitText {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Operation
    )

    $lines = @(& $script:GitExecutable @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $detail = ($lines -join [Environment]::NewLine).Trim()
        throw "$Operation failed with exit code $exitCode. $detail"
    }
    return ($lines -join [Environment]::NewLine).Trim()
}

function Resolve-RequiredDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $candidate = $Path
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $RepoRoot $candidate
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        throw "$Name directory does not exist: $candidate"
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Resolve-OutputDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    $candidate = $Path
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $RepoRoot $candidate
    }
    $resolved = [System.IO.Path]::GetFullPath($candidate)
    if (Test-Path -LiteralPath $resolved -PathType Leaf) {
        throw "OutputDir is an existing file: $resolved"
    }
    return $resolved
}

function Assert-RepositoryInvariant {
    $topLevel = Invoke-GitText `
        -Arguments @("-C", $RepoRoot, "rev-parse", "--show-toplevel") `
        -Operation "Resolving the Git worktree"
    $topLevel = [System.IO.Path]::GetFullPath($topLevel)
    if (-not [string]::Equals(
            $topLevel,
            $RepoRoot,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
        throw "Unexpected Git worktree: $topLevel"
    }

    $dirty = Invoke-GitText `
        -Arguments @(
            "-C",
            $RepoRoot,
            "status",
            "--porcelain=v1",
            "--untracked-files=all"
        ) `
        -Operation "Checking worktree cleanliness"
    if (-not [string]::IsNullOrWhiteSpace($dirty)) {
        throw "Refusing to run with a dirty worktree:`n$dirty"
    }

    $head = Invoke-GitText `
        -Arguments @("-C", $RepoRoot, "rev-parse", "HEAD") `
        -Operation "Resolving HEAD"
    if (-not [string]::Equals(
            $head,
            $SourceCommit,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
        throw "HEAD $head does not match SourceCommit $SourceCommit."
    }
}

function Assert-NoGameProcesses {
    $processes = @(Get-Process -ErrorAction Stop | Where-Object {
            $_.ProcessName -match "^(VALORANT|RiotClient)"
        })
    if ($processes.Count -gt 0) {
        $details = ($processes | Sort-Object ProcessName, Id | ForEach-Object {
                "{0} (PID {1})" -f $_.ProcessName, $_.Id
            }) -join ", "
        throw "Refusing to run while VALORANT/Riot processes are active: $details"
    }
}

function Get-RequiredFreeVramBytes {
    param([Parameter(Mandatory = $true)][int]$Capacity)

    return (
        ([int64]$Capacity * $LayerCount * $ExpertBytes) +
        $NonExpertCheckpointBytes +
        $VramSafetyMarginBytes
    )
}

function Assert-FreeVram {
    param([Parameter(Mandatory = $true)][int]$Capacity)

    $lines = @(& $script:NvidiaSmiExecutable `
            "--query-gpu=memory.free" `
            "--format=csv,noheader,nounits" `
            "--id=0" 2>&1)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $detail = ($lines -join [Environment]::NewLine).Trim()
        throw "Reading GPU 0 free VRAM failed with exit code $exitCode. $detail"
    }

    $values = @($lines | ForEach-Object { ([string]$_).Trim() } | Where-Object {
            -not [string]::IsNullOrWhiteSpace($_)
        })
    if ($values.Count -ne 1 -or $values[0] -notmatch "^\d+$") {
        throw "nvidia-smi returned an unexpected free-memory value: $($values -join ', ')"
    }

    [int64]$freeBytes = [int64]$values[0] * 1MB
    [int64]$requiredBytes = Get-RequiredFreeVramBytes -Capacity $Capacity
    if ($freeBytes -lt $requiredBytes) {
        throw (
            "Insufficient free VRAM for capacity {0}: {1:N2} GiB free, " +
            "{2:N2} GiB required."
        ) -f $Capacity, ($freeBytes / 1GB), ($requiredBytes / 1GB)
    }

    Write-Host (
        "GPU preflight for capacity {0}: {1:N2} GiB free; {2:N2} GiB required." -f
        $Capacity,
        ($freeBytes / 1GB),
        ($requiredBytes / 1GB)
    )
}

function Get-PromptSha256 {
    param([Parameter(Mandatory = $true)][string]$Prompt)

    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($Prompt))
        return -join ($digest | ForEach-Object { $_.ToString("x2") })
    }
    finally {
        $sha256.Dispose()
    }
}

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RequiredJsonNumber {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $property = $Object.psobject.Properties[$Name]
    if ($null -eq $property) {
        throw "$Context is missing numeric field $Name."
    }
    $value = $property.Value
    $numericTypes = @(
        [byte], [sbyte], [int16], [uint16], [int32], [uint32],
        [int64], [uint64], [single], [double], [decimal]
    )
    $isNumeric = $false
    foreach ($numericType in $numericTypes) {
        if ($value -is $numericType) {
            $isNumeric = $true
            break
        }
    }
    if (-not $isNumeric) {
        throw "$Context field $Name must be a JSON number."
    }
    [double]$number = $value
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        throw "$Context field $Name must be finite."
    }
    return $number
}

function Get-RequiredJsonInteger {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    [double]$number = Get-RequiredJsonNumber `
        -Object $Object `
        -Name $Name `
        -Context $Context
    if ($number -ne [math]::Truncate($number)) {
        throw "$Context field $Name must be an integer."
    }
    return [int64]$number
}

function Assert-BaselineMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$WorkloadId,

        [Parameter(Mandatory = $true)]
        [string]$PromptSha256,

        [Parameter(Mandatory = $true)]
        [string]$WorkloadFileSha256
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing baseline metadata for ${WorkloadId}: $Path"
    }
    try {
        $metadata = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "Invalid baseline metadata JSON for ${WorkloadId}: $Path. $($_.Exception.Message)"
    }

    if ($metadata.schema_version -ne 1) {
        throw "Baseline metadata for $WorkloadId must use schema_version 1."
    }
    if ($metadata.model.id -ne $ExpectedModelId -or
        $metadata.model.requested_revision -ne $ExpectedRevision -or
        $metadata.model.resolved_revision -ne $ExpectedRevision) {
        throw "Baseline metadata model/revision mismatch for $WorkloadId."
    }
    $actualShardNames = @($metadata.model.checkpoint_shards_sha256.psobject.Properties.Name)
    if ($actualShardNames.Count -ne $ExpectedShardSha256.Count) {
        throw "Baseline checkpoint shard set mismatch for $WorkloadId."
    }
    foreach ($shardName in $ExpectedShardSha256.Keys) {
        if ([string]$metadata.model.checkpoint_shards_sha256.$shardName -ne
            $ExpectedShardSha256[$shardName]) {
            throw "Baseline checkpoint shard hash mismatch for $WorkloadId ($shardName)."
        }
    }
    if ($metadata.workload.id -ne $WorkloadId -or
        $metadata.workload.prompt_sha256 -ne $PromptSha256 -or
        $metadata.workload.workload_file_sha256 -ne $WorkloadFileSha256) {
        throw "Baseline metadata workload/prompt mismatch for $WorkloadId."
    }
    $temperature = Get-RequiredJsonNumber `
        -Object $metadata.generation `
        -Name "temperature" `
        -Context "Baseline generation for $WorkloadId"
    if ($temperature -ne 0.0) {
        throw "Baseline metadata must use greedy temperature 0 for $WorkloadId."
    }
    if ([int]$metadata.generation.seed -ne 17) {
        throw "Baseline metadata must use seed 17 for $WorkloadId."
    }
    if (@($metadata.generation.generated_token_ids).Count -ne 32 -or
        [int]$metadata.generation.generated_tokens -ne 32) {
        throw "Baseline metadata needs exactly 32 generated token IDs for $WorkloadId."
    }
    if (@($metadata.generation.generated_token_ids) -contains 50279) {
        throw "Baseline metadata reaches EOS within 32 tokens for $WorkloadId."
    }
    $generationWall = Get-RequiredJsonNumber `
        -Object $metadata.timing_observation `
        -Name "generation_wall_seconds" `
        -Context "Baseline timing for $WorkloadId"
    $prefill = Get-RequiredJsonNumber `
        -Object $metadata.timing_observation `
        -Name "prefill_seconds" `
        -Context "Baseline timing for $WorkloadId"
    $decode = Get-RequiredJsonNumber `
        -Object $metadata.timing_observation `
        -Name "generation_decode_seconds" `
        -Context "Baseline timing for $WorkloadId"
    $peakVram = Get-RequiredJsonInteger `
        -Object $metadata.environment `
        -Name "peak_vram_bytes" `
        -Context "Baseline environment for $WorkloadId"
    if ($generationWall -le 0.0 -or $prefill -lt 0.0 -or
        $decode -lt 0.0 -or $peakVram -le 0) {
        throw "Baseline timing/environment metadata is invalid for $WorkloadId."
    }
}

function Assert-CompletedOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$WorkloadId,

        [Parameter(Mandatory = $true)]
        [int]$Capacity,

        [Parameter(Mandatory = $true)]
        [string]$ReferencePath
    )

    try {
        $result = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $reference = Get-Content -LiteralPath $ReferencePath -Raw | ConvertFrom-Json
    }
    catch {
        throw "Cannot resume from invalid JSON output ${Path}: $($_.Exception.Message)"
    }

    if ($result.schema_version -ne 1 -or $result.status -ne "ok") {
        throw "Cannot resume from incomplete or failed output: $Path"
    }
    if ($result.model.model_id -ne $ExpectedModelId -or
        $result.model.revision -ne $ExpectedRevision) {
        throw "Cannot resume from output with a different model/revision: $Path"
    }
    $actualResultShardNames = @($result.model.shards.psobject.Properties.Name)
    if ($actualResultShardNames.Count -ne $ExpectedShardSha256.Count) {
        throw "Cannot resume from output with a different checkpoint shard set: $Path"
    }
    foreach ($shardName in $ExpectedShardSha256.Keys) {
        if ([string]$result.model.shards.$shardName.sha256 -ne
            $ExpectedShardSha256[$shardName]) {
            throw "Cannot resume from output with a bad shard hash ($shardName): $Path"
        }
    }
    $budgetContext = "Paged runtime budget in $Path"
    $budgetLayers = Get-RequiredJsonInteger `
        -Object $result.runtime.budget `
        -Name "layers" `
        -Context $budgetContext
    $budgetExpertBytes = Get-RequiredJsonInteger `
        -Object $result.runtime.budget `
        -Name "expert_bytes" `
        -Context $budgetContext
    $budgetSlots = Get-RequiredJsonInteger `
        -Object $result.runtime.budget `
        -Name "slots_per_layer" `
        -Context $budgetContext
    $budgetStagingSlots = Get-RequiredJsonInteger `
        -Object $result.runtime.budget `
        -Name "staging_slots" `
        -Context $budgetContext
    $budgetCacheBytes = Get-RequiredJsonInteger `
        -Object $result.runtime.budget `
        -Name "cache_bytes" `
        -Context $budgetContext
    $budgetNonExpertBytes = Get-RequiredJsonInteger `
        -Object $result.runtime.budget `
        -Name "non_expert_checkpoint_bytes" `
        -Context $budgetContext
    $budgetExpectedWeightBytes = Get-RequiredJsonInteger `
        -Object $result.runtime.budget `
        -Name "expected_weight_vram_bytes" `
        -Context $budgetContext
    if ($result.runtime.policy -ne "lru" -or
        $budgetLayers -ne $LayerCount -or
        $budgetExpertBytes -ne $ExpertBytes -or
        $budgetSlots -ne $Capacity -or
        $budgetStagingSlots -ne 1 -or
        $budgetCacheBytes -ne ($Capacity * $LayerCount * $ExpertBytes) -or
        $budgetNonExpertBytes -ne $NonExpertCheckpointBytes -or
        $budgetExpectedWeightBytes -ne
        ($budgetCacheBytes + $budgetNonExpertBytes)) {
        throw "Cannot resume from output with a different runtime budget: $Path"
    }
    if ($result.workload.id -ne $WorkloadId -or
        [int]$result.workload.max_new_tokens -ne 32 -or
        [int]$result.workload.seed -ne 17 -or
        $result.workload.decoding -ne
        "teacher-forced reference with greedy predictions") {
        throw "Cannot resume from output with a different workload protocol: $Path"
    }
    $referenceSha256 = Get-FileSha256 -Path $ReferencePath
    if ($result.reference_comparison.available -ne $true -or
        $result.reference_comparison.mode -ne "teacher_forced" -or
        $result.reference_comparison.sha256 -ne $referenceSha256) {
        throw "Cannot resume from output bound to a different baseline: $Path"
    }

    $referenceIds = @($reference.generation.generated_token_ids)
    $cold = $result.passes.cold_expert_cache
    $retained = $result.passes.repeat_retained_expert_cache
    $coldFed = @($cold.fed_token_ids)
    $retainedFed = @($retained.fed_token_ids)
    $coldGenerated = @($cold.generated_ids)
    $retainedGenerated = @($retained.generated_ids)
    if ($referenceIds.Count -ne 32 -or
        $cold.teacher_forced -ne $true -or
        $retained.teacher_forced -ne $true -or
        $coldFed.Count -ne 32 -or
        $retainedFed.Count -ne 32 -or
        $coldGenerated.Count -ne 32 -or
        $retainedGenerated.Count -ne 32 -or
        (Compare-Object $referenceIds $coldFed -SyncWindow 0) -or
        (Compare-Object $referenceIds $retainedFed -SyncWindow 0) -or
        (Compare-Object $coldGenerated $retainedGenerated -SyncWindow 0)) {
        throw "Cannot resume from output with incomplete or divergent token sequences: $Path"
    }
}

$GitCommand = @(Get-Command git -CommandType Application -ErrorAction Stop)[0]
$script:GitExecutable = $GitCommand.Source
$NvidiaSmiCommand = @(
    Get-Command nvidia-smi -CommandType Application -ErrorAction Stop
)[0]
$script:NvidiaSmiExecutable = $NvidiaSmiCommand.Source

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
$PythonPath = (Resolve-Path -LiteralPath $Python).Path
if (-not (Test-Path -LiteralPath $BenchmarkScript -PathType Leaf)) {
    throw "Benchmark script does not exist: $BenchmarkScript"
}
if (-not (Test-Path -LiteralPath $WorkloadFile -PathType Leaf)) {
    throw "Workload file does not exist: $WorkloadFile"
}

$SnapshotPath = Resolve-RequiredDirectory -Path $Snapshot -Name "Snapshot"
$BaselinePath = Resolve-RequiredDirectory -Path $BaselineDir -Name "BaselineDir"
$OutputPath = Resolve-OutputDirectory -Path $OutputDir
if (-not [string]::Equals(
        (Split-Path -Leaf $SnapshotPath),
        $ExpectedRevision,
        [System.StringComparison]::Ordinal
    )) {
    throw "Snapshot directory must be named with pinned revision $ExpectedRevision."
}
$snapshotPrefix = $SnapshotPath.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
if ([string]::Equals(
        $OutputPath,
        $SnapshotPath,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or $OutputPath.StartsWith(
        $snapshotPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "OutputDir must not be inside the read-only snapshot."
}

Assert-RepositoryInvariant
Assert-NoGameProcesses

try {
    $workloadPayload = Get-Content -LiteralPath $WorkloadFile -Raw | ConvertFrom-Json
}
catch {
    throw "Invalid workload JSON: $WorkloadFile. $($_.Exception.Message)"
}
if ($workloadPayload.schema_version -ne 1) {
    throw "Workload file must use schema_version 1."
}

$References = @{}
$WorkloadFileSha256 = Get-FileSha256 -Path $WorkloadFile
foreach ($workloadId in $WorkloadIds) {
    $matchingWorkloads = @($workloadPayload.workloads | Where-Object {
            $_.id -eq $workloadId
        })
    if ($matchingWorkloads.Count -ne 1) {
        throw "Workload file must contain exactly one workload named $workloadId."
    }
    $referencePath = Join-Path $BaselinePath "$workloadId.metadata.json"
    $promptSha256 = Get-PromptSha256 -Prompt ([string]$matchingWorkloads[0].prompt)
    Assert-BaselineMetadata `
        -Path $referencePath `
        -WorkloadId $workloadId `
        -PromptSha256 $promptSha256 `
        -WorkloadFileSha256 $WorkloadFileSha256
    $References[$workloadId] = $referencePath
}

$Runs = @(
    foreach ($replicate in 1..3) {
        foreach ($capacity in $CapacityOrders[$replicate]) {
            foreach ($workloadId in $WorkloadIds) {
                [pscustomobject]@{
                    Replicate = $replicate
                    Capacity = $capacity
                    WorkloadId = $workloadId
                    Reference = $References[$workloadId]
                    Output = Join-Path $OutputPath (
                        "repetition-{0}\slots-{1}\{2}.json" -f
                        $replicate,
                        $capacity,
                        $workloadId
                    )
                }
            }
        }
    }
)

foreach ($run in $Runs) {
    if (Test-Path -LiteralPath $run.Output) {
        if (-not $Resume) {
            throw (
                "Refusing to overwrite an existing output: {0}. " +
                "Use -Resume to validate and skip completed outputs."
            ) -f $run.Output
        }
        Assert-CompletedOutput `
            -Path $run.Output `
            -WorkloadId $run.WorkloadId `
            -Capacity $run.Capacity `
            -ReferencePath $run.Reference
    }
    $outputParent = Split-Path -Parent $run.Output
    if (Test-Path -LiteralPath $outputParent -PathType Leaf) {
        throw "An output parent path is an existing file: $outputParent"
    }
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "Paged OLMoE capacity sweep preflight passed."
Write-Host "Source commit: $SourceCommit"
Write-Host "Snapshot: $SnapshotPath"
Write-Host "Baseline: $BaselinePath"
Write-Host "Output: $OutputPath"
Write-Host "Runs: $($Runs.Count) (3 replicates x 4 capacities x 5 workloads)"
Write-Host "Capacity orders: R1=16,24,40,32; R2=32,40,24,16; R3=24,16,32,40"

$runNumber = 0
foreach ($run in $Runs) {
    $runNumber += 1
    Write-Host ""
    Write-Host (
        "[{0}/{1}] R{2}, capacity {3}, workload {4}" -f
        $runNumber,
        $Runs.Count,
        $run.Replicate,
        $run.Capacity,
        $run.WorkloadId
    )

    Assert-RepositoryInvariant
    Assert-NoGameProcesses
    if (Test-Path -LiteralPath $run.Output) {
        if (-not $Resume) {
            throw "Refusing to overwrite an existing output: $($run.Output)"
        }
        Assert-CompletedOutput `
            -Path $run.Output `
            -WorkloadId $run.WorkloadId `
            -Capacity $run.Capacity `
            -ReferencePath $run.Reference
        Write-Host "Validated and skipped completed output: $($run.Output)"
        continue
    }
    Assert-FreeVram -Capacity $run.Capacity

    $arguments = @(
        $BenchmarkScript
        "--snapshot"
        $SnapshotPath
        "--workload-file"
        $WorkloadFile
        "--workload-id"
        $run.WorkloadId
        "--reference-metadata"
        $run.Reference
        "--teacher-force-reference"
        "--policy"
        "lru"
        "--slots-per-layer"
        [string]$run.Capacity
        "--staging-slots"
        "1"
        "--max-input-tokens"
        "64"
        "--max-new-tokens"
        "32"
        "--seed"
        "17"
        "--device"
        "cuda:0"
        "--output"
        $run.Output
    )

    & $PythonPath @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw (
            "Benchmark failed with exit code {0} at R{1}, capacity {2}, " +
            "workload {3}. Existing output was preserved."
        ) -f $exitCode, $run.Replicate, $run.Capacity, $run.WorkloadId
    }
    if (-not (Test-Path -LiteralPath $run.Output -PathType Leaf)) {
        throw "Benchmark exited successfully but did not create: $($run.Output)"
    }
    Write-Host "Completed: $($run.Output)"
}

Write-Host ""
Write-Host "Capacity sweep completed successfully: $($Runs.Count)/$($Runs.Count) runs."
