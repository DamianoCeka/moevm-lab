# OLMoE real-routing M1 reference study

> **Evidence boundary:** router decisions and scores were captured from the real
> pinned model. Cache speedups, stalls, and traffic are trace replays under the
> provisional simulator configuration; they are not measured runtime speedups.

## Scope

- Model: `allenai/OLMoE-1B-7B-0924` at `bd1c52f59153f724c1ad11ca1791edc77bab3806`
- License: Apache-2.0 (weights are not redistributed here)
- Workloads: 10 captures across seeds 17 and 29
- Routing evidence: 438 tokens, 7,008
  token/layer steps, 56,064 expert accesses
- Hardware: NVIDIA GeForce RTX 3080 Ti, CUDA 13.0

## Aggregate findings

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| Real temporal overlap | 44.09% | 38.54% | 48.33% |
| Real normalized entropy | 86.49% | 83.78% | 88.80% |
| Online predictor precision | 44.01% | 40.30% | 48.20% |
| Simulated replay speedup | 1.0022x | 0.9830x | 1.0194x |
| Simulated demand-stall reduction | 0.23% | -2.24% | 2.34% |
| Simulated RAM-to-VRAM traffic change | 6.17% | 2.49% | 11.71% |

The current predictor is only approximately neutral on these real traces: its
mean replay speedup is `1.0022x`, while it changes
RAM-to-VRAM traffic by `6.17%`.
This is useful negative evidence: the strong synthetic result does not transfer
unchanged to this model and workload set.

## Per capture

| Seed | Workload | Tokens | Real overlap | Predictor precision | Replay speedup | Replay traffic change |
|---:|---|---:|---:|---:|---:|---:|
| 17 | domain_switch | 44 | 39.43% | 41.17% | 1.0194x | 4.23% |
| 17 | math_reasoning | 44 | 44.80% | 48.20% | 1.0064x | 4.97% |
| 17 | python_code | 37 | 48.33% | 45.03% | 1.0104x | 2.49% |
| 17 | systems_en | 40 | 45.99% | 43.91% | 1.0021x | 4.69% |
| 17 | systems_it | 54 | 43.40% | 44.69% | 0.9927x | 8.55% |
| 29 | domain_switch | 44 | 38.54% | 40.30% | 1.0143x | 5.14% |
| 29 | math_reasoning | 44 | 43.77% | 45.68% | 0.9923x | 8.86% |
| 29 | python_code | 37 | 46.79% | 45.94% | 1.0066x | 3.82% |
| 29 | systems_en | 40 | 48.12% | 44.07% | 0.9945x | 7.24% |
| 29 | systems_it | 54 | 41.76% | 41.08% | 0.9830x | 11.71% |

## Reproduction

Every committed JSONL trace includes router scores and has its SHA-256 recorded
in `study.json`. Replay it with:

```bash
python -m moevm compare --config configs/olmoe_1b_7b_0924.toml \
  --trace benchmarks/reference/real-routing-olmoe-m1/traces/seed-17/systems_en.trace.jsonl \
  --no-write
```
