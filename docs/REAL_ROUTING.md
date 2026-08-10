# Real-routing capture: OLMoE M1

## Evidence boundary

This milestone contains two different kinds of evidence:

1. **Routing capture:** expert IDs and top-k router probabilities observed while
   the pinned OLMoE checkpoint executed real prefill and decode tokens.
2. **Trace replay:** those real decisions replayed through MoEVM Lab using
   provisional compute, RAM, PCIe and NVMe parameters.

The capture timings come from Transformers with Accelerate CPU offload. They are
recorded for auditability only and are not a production baseline. At the v0.2.0
M1 checkpoint the project still had no expert-offload execution backend. The
subsequent unreleased M3 prototype uses a deterministic two-token capture as a
controlled correctness and timing reference for its first paged-runtime smoke.

## Model and reproducibility

- Model: [`allenai/OLMoE-1B-7B-0924`](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- Revision: `bd1c52f59153f724c1ad11ca1791edc77bab3806`
- License reported by the model project: Apache-2.0
- Topology: 16 layers, 64 experts per layer, 8 selected experts per token
- Expert FFN tensor size in BF16: 12 MiB
- Capture device: NVIDIA GeForce RTX 3080 Ti (12 GiB)
- Host: AMD Ryzen 9 9950X3D, 32 GiB RAM, Windows 11
- Stack: Python 3.12.8, PyTorch 2.12.1+cu130, Transformers 5.14.1,
  Accelerate 1.14.0

The three checkpoint shards were verified against their Hugging Face LFS
SHA-256 values before capture:

```text
5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e  model-00001-of-00003.safetensors
15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7  model-00002-of-00003.safetensors
a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04  model-00003-of-00003.safetensors
```

Weights are stored only in the ignored local cache. They are not included in
the Git repository or release artifacts.

## Workloads

`benchmarks/workloads/olmoe_m1.json` contains five controlled prompts: English
systems, Italian systems, Python, mathematics and an adversarial rapid domain
switch. No user conversation, credential or private dataset is used.

Each workload was captured with sampling temperature 0.7 and seeds 17 and 29,
with 16 generated tokens after the prompt. The committed study therefore covers:

- 10 captures;
- 438 tokens;
- 7,008 token/layer steps;
- 56,064 selected-expert accesses;
- router-score coverage of 100%.

## Findings

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| Real temporal overlap | 44.09% | 38.54% | 48.33% |
| Real normalized entropy | 86.49% | 83.78% | 88.80% |
| Online predictor precision/recall | 44.01% | 40.30% | 48.20% |
| Simulated replay speedup | 1.0022× | 0.9830× | 1.0194× |
| Simulated demand-stall reduction | 0.23% | -2.24% | 2.34% |
| Simulated RAM-to-VRAM traffic change | +6.17% | +2.49% | +11.71% |

The result rejects a naive transfer of the toy benchmark claim. Real routing is
substantially less local than the synthetic headline case, and the current
predictor does not earn its extra transfer traffic consistently. This is the
main M1 result, not a failure to be hidden.

## Reproduce the capture on Windows

Create a dedicated environment. PyTorch is installed first so the optional
dependencies cannot select a CPU-only build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_real_routing.ps1 `
  -VenvPath D:\moevm-lab-envs\m1 `
  -CachePath D:\moevm-lab-cache\huggingface
```

Download and verify the pinned snapshot:

```powershell
$env:HF_HOME = 'D:\moevm-lab-cache\huggingface'
D:\moevm-lab-envs\m1\Scripts\python.exe scripts\capture_real_routing.py --download-only
```

Capture both seeds and build the portable reference:

```powershell
$env:HF_HUB_OFFLINE = '1'
D:\moevm-lab-envs\m1\Scripts\python.exe scripts\capture_real_routing.py --local-files-only --seed 17 --temperature 0.7
D:\moevm-lab-envs\m1\Scripts\python.exe scripts\capture_real_routing.py --local-files-only --seed 29 --temperature 0.7
py -3.12 scripts\summarize_real_routing.py
```

On Windows hosts without symlink privileges, the capturer automatically
completes missing snapshot files directly instead of requiring administrator
access or Developer Mode.

## Limitations and next step

- 438 tokens are enough to falsify the very-high-locality assumption, not to
  characterize all languages, tasks or sequence lengths.
- OLMoE is a base checkpoint; generation quality is not evaluated here.
- CPU-offloaded Transformers execution is a controlled reference, not the
  production baseline that a future runtime must beat.
- The original `1.0022x` M1 replay used provisional hardware parameters and
  remains the historical v0.2 result.

## Post-M1 measured follow-up

Unreleased M2 work measured pinned/pageable RAM-to-VRAM copies and expert-sized
NVMe reads. Replaying the same ten traces with the measured P310 profile and
per-expert fixed latency produces `0.9791x` aggregate speedup and +4.91%
RAM-to-VRAM traffic. This calibrated result is distinct from the historical M1
number and is still simulation.

Unreleased M3 work adds a read-only safetensors expert store, bounded per-layer
GPU slots and a synchronous Transformers expert backend. Its first full OLMoE
smoke matches the two greedy reference token IDs, uses 21.34% less observed peak
allocated VRAM, loses end-to-end with an empty expert cache, and wins on a
retained-cache repeat. One prompt and one decode interval do not establish a
general speedup. See the [hardware profile](../benchmarks/reference/hardware-rtx3080ti-p310/README.md),
[placement study](../benchmarks/reference/real-routing-olmoe-m1/placement/README.md),
and [runtime evidence](../benchmarks/reference/paged-runtime-olmoe-p310-smoke/README.md).
