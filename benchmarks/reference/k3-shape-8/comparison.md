# MoEVM Lab simulation result

> **Simulation only.** This report does not claim measured LLM token-generation performance.

## Experiment

| Field | Value |
|---|---:|
| Model shape | `k3-shaped-synthetic` |
| Tokens | 8 |
| Layers | 93 |
| Experts per layer | 896 |
| Selected experts | 16 |
| Expert size | 20.00 MiB |
| VRAM expert cache | 8.00 GiB |
| RAM expert cache | 96.00 GiB |

## Results

| Metric | Baseline | Predictive prefetch |
|---|---:|---:|
| Estimated tokens/s | 0.313 | 0.325 |
| Estimated elapsed | 25.530 s | 24.638 s |
| Demand stall | 23.670 s | 22.778 s |
| Prefetch stall | 0.000 s | 0.000 s |
| VRAM demand hit-rate | 0.00% | 7.86% |
| RAM+VRAM demand hit-rate | 65.73% | 65.73% |
| Demand NVMe traffic | 79.69 GiB | 79.69 GiB |
| Total NVMe traffic | 79.69 GiB | 79.69 GiB |
| Total RAM→VRAM traffic | 232.50 GiB | 239.65 GiB |
| Prefetch precision | — | 71.89% |
| Predicted/admitted prefetches | — | 10,416 / 1,302 |
| Deadline rejections | — | 9,114 |

## Comparison

- Estimated speedup: **1.036x**
- Demand-path NVMe reduction: **0.00%**
- Total NVMe reduction: **0.00%**
- Demand-stall reduction: **3.77%**
- RAM→VRAM traffic change: **3.07%**

A useful prefetch can reduce blocking demand reads while still increasing total traffic. Both latency and traffic figures must therefore be reported.
