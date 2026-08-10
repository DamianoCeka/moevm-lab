# Offline placement analysis

`moevm analyze-placement` compares expert-placement policies on MoEVM JSONL
routing traces. It is an offline, sequential cache replay: it reports accesses
and theoretical bytes, but does not claim measured latency or throughput.

## Policies and capacity

- `cold_lru`: an independent cold LRU cache for every layer. It does not use the
  training set.
- `static_hot`: the `capacity-per-layer` most frequent experts learned from the
  training set remain resident. A miss is modeled as an uncached scratch load:
  that scratch space is outside the stated resident capacity, and the expert is
  not inserted or counted as an eviction.
- `hybrid_hot_lru`: `protected-hot` slots per layer are permanently reserved for
  trained hot experts; all remaining slots form a cold LRU. If training observes
  fewer experts than the reserved count, the unfilled protected slots remain
  reserved rather than silently enlarging the LRU partition.

Frequency ties are resolved by expert ID. Within one routing step, accesses are
also replayed deterministically in ascending expert-ID order; router-score order
does not affect cache state. Cache state is reset between test trace files, and
static placements are preloaded again for each file. Reports separate demand,
preload, and total bytes and include per-token, per-layer, per-trace, and
whole-protocol aggregates.

With the OLMoE parameters used below, 40 slots × 16 layers × 12 MiB equals a
7.5 GiB resident cache. The hybrid split is 32 protected plus 8 LRU slots per
layer; its protected preload is 6 GiB for each independently replayed trace.

## Evaluation protocols

### Primary: strict leave-one-workload-out

`leave-one-workload-out` derives workload IDs from filenames such as
`systems_it.trace.jsonl`. For each test workload it removes every training trace
with that workload ID before fitting. Every fold reports its workload and
same-address overlap audit, both of which must be zero. Input path and SHA-256
content overlap are rejected, placements use training only, and test accesses
never update trained hot sets.

```powershell
.\.venv\Scripts\python.exe -m moevm analyze-placement `
  --config configs/olmoe_1b_7b_0924.toml `
  --train-dir benchmarks/reference/real-routing-olmoe-m1/traces/seed-17 `
  --test-dir benchmarks/reference/real-routing-olmoe-m1/traces/seed-29 `
  --protocol leave-one-workload-out `
  --capacity-per-layer 40 `
  --protected-hot 32 `
  --output results/placement/leave-one-workload-out-cap40.json
```

### Secondary: cross-seed, same-workload audit

The plain `train-test` protocol trains on all seed-17 traces and tests on all
seed-29 traces. These sets use different files and content hashes, but they share
workload IDs, prompts, and many `(workload, token, layer)` addresses. It is a
cross-seed stability check, not a workload-independent or “leakage-safe” result.
The report makes the shared workload IDs, shared same-address step count, and
exact expert-set matches explicit.

```powershell
.\.venv\Scripts\python.exe -m moevm analyze-placement `
  --config configs/olmoe_1b_7b_0924.toml `
  --train-dir benchmarks/reference/real-routing-olmoe-m1/traces/seed-17 `
  --test-dir benchmarks/reference/real-routing-olmoe-m1/traces/seed-29 `
  --protocol train-test `
  --capacity-per-layer 40 `
  --protected-hot 32 `
  --output results/placement/seed17-to-seed29-cap40.json
```

Both outputs are under `results/`, which is intentionally ignored by Git. Use
repeatable `--policy` options to select policies. Repeat
`--layer-capacity LAYER=SLOTS` to override individual layers while retaining the
default for all others.

## Interpretation limits

- Router prefill phases, concurrent requests, top-k batch materialization,
  transfer overlap, and batching are not modeled. Accesses are sequential.
- Bytes are theoretical `misses × expert_bytes` plus explicit placement preload;
  they are not measured SSD, RAM, PCIe, or CUDA traffic.
- Static misses assume scratch storage outside the configured resident capacity.
  That makes static hit-rate comparisons useful but is not a complete VRAM peak
  model.
- The committed traces contain only 37–54 tokens. Recharging preload for every
  independent trace can dominate; a persistent long session would amortize it
  differently.
- Capacity 40 and protected-hot 32 are supplied research hypotheses. They were
  not selected inside a nested validation protocol, so comparisons at those
  hyperparameters are exploratory rather than an unbiased model-selection
  estimate.
