# MoEVM Lab

**Expert-aware memory virtualization research for sparse Mixture-of-Experts inference.**

MoEVM Lab investigates whether a large sparse MoE can use VRAM as a small, predictive working set while colder expert weights live in RAM and NVMe storage.

> **Current status: v0.2.0 research lab.** It now captures routing from a real,
> small open MoE and replays those decisions in the simulator. It does not
> execute Kimi K3 weights, contain a weight-offload runtime, or claim real K3
> token-generation speed.

## Why this project exists

Kimi K3 publishes a 2.8T-parameter MoE topology with 104B activated parameters, 93 layers, 896 experts and 16 selected experts per token. That makes expert placement and data movement first-class inference problems. Current vLLM proposals are also exploring expert-level CPU offload, pinned memory, GPU expert caches, prediction and asynchronous pipelines.

MoEVM Lab starts one level below a production runtime: it builds a reproducible model of the memory system so that caching and scheduling ideas can be rejected or promoted before expensive CUDA work begins.

## What v0.2 implements

- Byte-capacity VRAM and RAM LRU caches with implicit NVMe backing storage.
- A protected speculative VRAM partition, so inaccurate prefetches do not directly overwrite the full demand cache.
- An online predictor combining expert popularity, same-layer temporal transitions and cross-layer transitions.
- Confidence filtering and deadline-aware admission that skips prefetches which cannot fit inside the available compute-overlap window.
- Synthetic routing traces with domains, hot sets, temporal reuse and controlled randomness.
- Configurable bandwidth, latency, compute budget and overlap efficiency.
- Baseline-versus-prefetch comparison on the **same trace**.
- JSON and Markdown reports with traffic, stall, cache and prediction metrics.
- A K3-shaped synthetic configuration that reproduces topology only, not real execution.
- Real OLMoE router capture with exact checkpoint revision, top-k probabilities,
  trace hashes, controlled workloads and environment metadata.
- Routing analysis for locality, entropy, router confidence and online predictor
  precision/recall.
- Ten committed real-routing traces and golden replay results across two seeds.

## Real-routing evidence

M1 captured `allenai/OLMoE-1B-7B-0924` at the pinned revision
`bd1c52f59153f724c1ad11ca1791edc77bab3806` on an RTX 3080 Ti. Across five
controlled workloads and two sampling seeds, the reference study contains 438
tokens, 7,008 token/layer routing steps and 56,064 expert accesses.

| Metric | Mean | Range |
|---|---:|---:|
| Real temporal expert overlap | 44.09% | 38.54–48.33% |
| Real normalized routing entropy | 86.49% | 83.78–88.80% |
| Online predictor precision/recall | 44.01% | 40.30–48.20% |
| Simulated trace-replay speedup | 1.0022× | 0.9830–1.0194× |
| Simulated RAM→VRAM traffic change | +6.17% | +2.49–11.71% |

This is deliberately reported as a mixed/negative result: the strong synthetic
locality does not transfer unchanged to the real model. Router decisions are
real; transfer timing and speedup remain simulated. See the
[full M1 study](benchmarks/reference/real-routing-olmoe-m1/README.md) and
[capture protocol](docs/REAL_ROUTING.md).

## Reference simulation

The included `configs/toy.toml` is intentionally memory constrained and routing-local. On 64 synthetic tokens, the current implementation reports:

| Metric | Baseline | Predictive prefetch |
|---|---:|---:|
| Estimated throughput | 21.699 tok/s | 30.413 tok/s |
| Demand stall | 1.413 s | 0.568 s |
| VRAM demand hit-rate | 0.00% | 71.78% |
| Total NVMe traffic | 1.39 GiB | 1.39 GiB |
| Total RAM→VRAM traffic | 24.00 GiB | 30.40 GiB |
| Prefetch precision | — | 72.92% |

That is an estimated **1.402× speedup** in this synthetic case. It also increases RAM→VRAM traffic by about 26.7%. This is the intended research trade-off: hide blocking transfers without pretending that prefetching is free. The committed [VRAM sweep](benchmarks/reference/vram_sweep.csv) also includes a negative configuration, so improvements are not presented as universal.

## Quick start on Windows

```powershell
cd moevm-lab
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

Manual setup:

```powershell
py -3 -m venv .venv
$site = .\.venv\Scripts\python.exe -c "import site; print(site.getsitepackages()[0])"
Set-Content "$site\moevm_lab.pth" "$PWD\src" -Encoding UTF8
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m moevm compare --config configs\toy.toml --tokens 64 --output-dir results\toy
```

K3-shaped memory simulation:

```powershell
.\.venv\Scripts\python.exe -m moevm doctor --config configs\k3_shape.toml
.\.venv\Scripts\python.exe -m moevm compare --config configs\k3_shape.toml --tokens 8 --output-dir results\k3-shape
```

## Commands

```text
moevm compare  Compare baseline and predictive prefetch
moevm run      Run one mode and print metrics
moevm trace    Generate a reusable JSONL routing trace
moevm doctor   Validate configuration and display cache capacity
moevm analyze-trace  Analyze locality, router scores and predictability
moevm analyze-placement  Compare audited train/test placement policies
```

A real routing trace uses one JSON object per line:

```json
{"token":0,"layer":0,"experts":[3,9,17,21],"scores":[0.31,0.22,0.14,0.09]}
```

## Repository layout

```text
src/moevm/             Simulator, cache, predictor, trace and CLI
configs/               Reproducible experiment profiles
tests/                 Zero-dependency unit tests
scripts/               Windows/Linux bootstrap, sweep and publish helpers
docs/                  Architecture, roadmap and research rules
benchmarks/reference/  Committed reference simulation output
```

## Research milestones

1. **Simulator and metrics** — complete in v0.1.
2. **Capture real routing traces** from a small open MoE — complete in v0.2.
3. **Replay traces against measured hardware** rather than assumed bandwidth — next.
4. **Pinned-RAM expert cache** with CUDA streams and a protected miss buffer.
5. **Checkpoint layout optimizer** and asynchronous NVMe reads.
6. **Tile-level streaming and out-of-order expert scheduling.**
7. **K3 checkpoint adapter**, only after smaller real models validate the design.

The current 8-token K3-shaped smoke test estimates **1.036×**, while rejecting 9,114 of 10,416 predictions that could not meet a one-layer deadline. This is still simulation, but it demonstrates why admission control matters.

See [the roadmap](docs/ROADMAP.md) and [benchmarking rules](docs/BENCHMARKING.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Reference benchmarks](benchmarks/reference/README.md)
- [Benchmarking rules](docs/BENCHMARKING.md)
- [Real-routing capture protocol and findings](docs/REAL_ROUTING.md)
- [Third-party models and tools](docs/THIRD_PARTY_MODELS.md)
- [Roadmap](docs/ROADMAP.md)
- [Research questions](docs/RESEARCH_QUESTIONS.md)
- [Italian overview](docs/README.it.md)
- [Changelog](CHANGELOG.md), [citation](CITATION.cff), [security](SECURITY.md) and [contributing](CONTRIBUTING.md)

## Important limitations

- `tokens/s` is derived from the configured compute and transfer model; it is not a hardware benchmark.
- The synthetic expert payload is not an official K3 shard size.
- The simulator currently models one transfer path and one-step predictive prefetch.
- The v0.2 OLMoE router capture is real, but its replay speedup is still simulated.
- It does not model kernels, quantization decode, attention, KV cache, PCIe contention, page faults or multi-GPU collectives.
- A positive synthetic result is only permission to build the next experiment, not proof that K3 will reach 10 tok/s.

## Release status and license

This is a private pre-release prototype. The code is currently **all rights reserved** while the public/open-core and IP strategy is evaluated. Do not make the repository public before reading [IP and release notes](docs/IP_AND_RELEASE.md).

Kimi K3 and vLLM are third-party projects. MoEVM Lab is not affiliated with Moonshot AI or the vLLM project and includes no model weights.
