# One-command OLMoE demo

The Windows entry point is deliberately short:

```powershell
.\demo.cmd
```

It turns the validated paged-runtime components into an interactive local demo.
The default is the bounded async pipeline with LRU expert placement, two pinned
staging slots, at most 32 GPU slots per layer, at most 64 input tokens and up to
two generated tokens. An earlier EOS may shorten generation. The model and
revision are fixed; the first demo does not accept an arbitrary checkpoint.

## What the command does

1. Reads GPU, VRAM, RAM, disk, Python and cache state without administrator
   privileges.
2. Reuses a compatible Python 3.12 CUDA environment or creates one below
   `.cache/demo/py312-cu130/`, which is ignored by Git.
3. Reuses the pinned checkpoint when present. Otherwise, after one confirmation,
   it downloads `allenai/OLMoE-1B-7B-0924` at revision
   `bd1c52f59153f724c1ad11ca1791edc77bab3806`.
4. Loads only the three pinned safetensors shards and verifies their exact size
   and SHA-256. Known legacy checkpoint formats are excluded from acquisition.
5. Forces the model execution offline and never enables `trust_remote_code`.
6. Reserves at least 2 GiB or 20% of total VRAM, then selects between 2 and 32
   per-layer expert slots. The benchmark retains its independent allocation
   guard and fails before model construction when free VRAM is insufficient.
7. Runs the real OLMoE paged runtime and writes create-only local artifacts.
8. Validates the result and prints memory and speed in a compact table.

The model is public and does not require a Hugging Face token. Its weights are
not part of MoEVM Lab and remain subject to the upstream model terms documented
in [third-party models](THIRD_PARTY_MODELS.md).

## Modes

```powershell
.\demo.cmd
```

Runs the quick async demonstration. It reports the empty dynamic expert-cache
pass and the immediate repeat that retains cache state.

```powershell
.\demo.cmd -Compare
```

Runs both sync and async paths with the same model, prompt, cache budget and
token count. The local summary rejects token or logical cache/traffic-counter
differences before showing a timing ratio. A single local pair is still subject
to page-cache, thermal, clock and ordering effects.

```powershell
.\demo.cmd -DryRun
```

Shows a read-only environment preview without creating directories, installing
packages, downloading data, initializing a CUDA context or writing a result.
Final slot selection and all runtime guards execute only in the real run.

```powershell
.\demo.cmd -Offline
```

Disables all network-backed setup and acquisition. It requires both a compatible
environment and the complete pinned snapshot. File/size presence is checked
before the CUDA environment probe; full shard SHA-256 verification still occurs
before model loading.

```powershell
.\demo.cmd -Yes
```

Allows the first environment setup and model acquisition without the interactive
confirmation. It does not weaken hash, memory or output validation.

The cache and device can be selected explicitly:

```powershell
.\demo.cmd -CachePath D:\MoEVM-cache -Device 0
```

`MOEVM_HF_HOME`, `HF_HOME` and `MOEVM_DEMO_PYTHON` are also recognized. The
wrapper checks conventional MoEVM cache roots on local filesystem drives but
does not recursively scan disks. For safety it never discovers and executes an
arbitrary Python interpreter from another drive: external environments must be
selected explicitly with `-PythonPath` or `MOEVM_DEMO_PYTHON`.

On multi-GPU systems, or when `CUDA_VISIBLE_DEVICES`/`CUDA_DEVICE_ORDER` remaps
CUDA ordinals, automatic environment creation is refused before installation.
Select a verified environment explicitly with `-PythonPath` and choose the
matching `-Device` instead.

## Resource policy

The guarded minimum is:

- NVIDIA CUDA GPU with BF16 support;
- 8 GiB total VRAM and 4 GiB currently free; 6 GiB free is recommended;
- 16 GiB total system RAM and 8 GiB currently available;
- Python 3.12 and the Windows `py` launcher when automatic setup is needed;
- an NVIDIA driver compatible with the pinned CUDA 13.0 PyTorch build;
- up to 30 GiB free on the model-cache volume and 5 GiB on the repository
  volume. When both share a volume, the combined conservative guard is 35 GiB.

The three weight shards contain about 12.9 GiB. Windows systems without Hub
symlink privileges may temporarily or permanently use extra cache space. Model
acquisition is resumable: after an interruption, rerun the same command.

The automatic cache planner uses:

```text
reserve = max(2 GiB, 20% of total VRAM)
all-layer cost per slot = 16 layers * 12 MiB = 192 MiB
slots = floor((free VRAM - non-expert weights - reserve) / 192 MiB)
slots = clamp(slots, 2, 32)
```

Automatic planning fails instead of clamping when fewer than two slots fit.

The full benchmark independently checks its exact checkpoint-derived weight
budget before allocating the model.

## Outputs and resume

Local output is stored below `results/demo/<plan-id>/`. The plan ID binds model,
revision, workload, token limits, pipeline choices, device, cache capacity and
benchmark-script hash. A completed matching result is displayed again; a valid
partial run resumes only its missing stage. Existing malformed or mismatched
files are never overwritten.

Typical artifacts are:

```text
plan.json
async.json
sync.json                 # only with -Compare
summary.json
```

The summary reports:

- model-load wall time;
- empty and retained dynamic-cache wall time;
- end-to-end smoke tokens/s for each pass, including prefill but excluding model
  setup/load, plus generated token IDs/text;
- peak allocated and reserved CUDA memory;
- peak process working set;
- expert-cache and pinned-staging budgets;
- logical storage and host-to-device bytes;
- sync/async ratios when comparison mode is enabled.

Raw benchmark JSON contains the absolute local snapshot path because it is a
diagnostic artifact. It is ignored by Git and must be sanitized before sharing.
The compact console summary does not print access tokens or private environment
variables.

Before model loading, the launcher hashes about 12.9 GiB of weights. The
benchmark verifies them again after its timed passes. Verification time is not
included in the displayed pass throughput, and the first hash can warm the OS
page cache. “Empty” or “cold” therefore means only that the dynamic GPU expert
cache is empty; it never means cold physical SSD data.

## Evidence boundary

`--demo-mode` changes only provenance policy: it permits a release archive or a
modified checkout to run with best-effort Git metadata. Checkpoint integrity,
CUDA/BF16, memory guards, create-only outputs, runtime metrics and token checks
remain fail-closed. Demo output is explicitly marked as not publishable
benchmark evidence.

Two tokens and one prompt are useful for confirming that the real paging path
runs and for observing memory, but they do not establish production throughput,
physical NVMe overlap or CUDA H2D/kernel overlap. Use the committed benchmark
protocols for evidence intended for publication.

The demo checks cold-versus-retained identity and, with `-Compare`,
sync-versus-async identity. It does not run the separate CPU-offload reference,
so it is not a replacement for the reference-gated evidence harness.

## Troubleshooting

- **Python 3.12 missing:** install Python 3.12 and rerun the same command. The
  demo does not install Python or drivers.
- **Checkpoint interrupted:** rerun the command; Hugging Face resumes from its
  cache.
- **Hash mismatch:** the demo stops before model loading or model VRAM
  allocation and does not delete the file. Inspect or repair the selected cache
  explicitly.
- **Insufficient VRAM:** close GPU applications, verify `-Device`, and rerun.
  Unsafe allocation overrides are intentionally absent.
- **CUDA OOM:** no performance result is emitted and no automatic destructive
  cleanup is performed.
- **Need more detail:** inspect the preserved raw JSON and the error printed by
  the failing child process; the launcher does not overwrite partial artifacts.
