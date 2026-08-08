# MoEVM Lab simulation result

> **Simulation only.** This report does not claim measured LLM token-generation performance.

## Experiment

| Field | Value |
|---|---:|
| Model shape | `toy-locality-moe` |
| Tokens | 64 |
| Layers | 12 |
| Experts per layer | 64 |
| Selected experts | 4 |
| Expert size | 8.00 MiB |
| VRAM expert cache | 256.00 MiB |
| RAM expert cache | 2.00 GiB |

## Results

| Metric | Baseline | Predictive prefetch |
|---|---:|---:|
| Estimated tokens/s | 21.699 | 30.407 |
| Estimated elapsed | 2.949 s | 2.105 s |
| Demand stall | 1.413 s | 0.569 s |
| Prefetch stall | 0.000 s | 0.000 s |
| VRAM demand hit-rate | 0.00% | 71.74% |
| RAM+VRAM demand hit-rate | 94.21% | 94.21% |
| Demand NVMe traffic | 1.39 GiB | 1.39 GiB |
| Total NVMe traffic | 1.39 GiB | 1.39 GiB |
| Total RAM→VRAM traffic | 24.00 GiB | 30.41 GiB |
| Prefetch precision | — | 72.88% |
| Predicted/admitted prefetches | — | 3,024 / 3,024 |
| Deadline rejections | — | 0 |

## Comparison

- Estimated speedup: **1.401x**
- Demand-path NVMe reduction: **0.00%**
- Total NVMe reduction: **0.00%**
- Demand-stall reduction: **59.76%**
- RAM→VRAM traffic change: **26.69%**

A useful prefetch can reduce blocking demand reads while still increasing total traffic. Both latency and traffic figures must therefore be reported.
