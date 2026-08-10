# OLMoE placement reference

`summary.json` is the compact, committed reference for the two reproducible
placement replays over the real OLMoE routing traces. The detailed reports stay
under ignored `results/placement/`; their SHA-256 digests make the complete
per-token, per-layer, and per-fold payloads reproducible without committing
roughly 1.2 MB of generated JSON.

The strict leave-one-workload-out protocol is the primary comparison: every
fold excludes the held-out workload from fitting, and all five fold audits show
zero shared workload IDs and zero shared step addresses. Its result is still
**exploratory**, because the 40-slot capacity and 32/8 hybrid split were not
chosen inside nested validation and the traces are short.

The cross-seed train/test protocol is a secondary stability check and is
**non-independent**. Seeds 17 and 29 use distinct files and hashes, but share
the same five workload prompts and all 3,504 `(workload, token, layer)` step
addresses. It must not be presented as workload-held-out evidence.

## Reproduce

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m moevm analyze-placement `
  --config configs/olmoe_1b_7b_0924.toml `
  --train-dir benchmarks/reference/real-routing-olmoe-m1/traces/seed-17 `
  --test-dir benchmarks/reference/real-routing-olmoe-m1/traces/seed-29 `
  --protocol train-test `
  --capacity-per-layer 40 `
  --protected-hot 32 `
  --output results/placement/seed17-to-seed29-cap40.json

.\.venv\Scripts\python.exe -m moevm analyze-placement `
  --config configs/olmoe_1b_7b_0924.toml `
  --train-dir benchmarks/reference/real-routing-olmoe-m1/traces/seed-17 `
  --test-dir benchmarks/reference/real-routing-olmoe-m1/traces/seed-29 `
  --protocol leave-one-workload-out `
  --capacity-per-layer 40 `
  --protected-hot 32 `
  --output results/placement/leave-one-workload-out-cap40.json
```

Expected full-report SHA-256 digests:

- cross-seed: `f09d3a6816b953a895d2e8335e5af233c3095c37390ada052b27ff1ef9657210`
- leave-one-workload-out: `a84955c5cb920fae74336553c81fc5fc86959ada0b614ff44cc6cb1dc4c591f8`

The hashes cover the platform-independent CLI representation: UTF-8, sorted
keys, two-space indentation, LF line endings, and no final newline. The writer
sets LF explicitly, including on Windows, and the golden test hashes its actual
output bytes.

These are trace-replay metrics, not measured runtime latency or throughput.
