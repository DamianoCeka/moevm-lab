# Model adapter boundary

MoEVM separates model integration from expert storage, cache policy and
scheduling.  The public adapter entry point is:

```python
from moevm.paged_runtime import attach_transformers_moe_runtime

adapter = attach_transformers_moe_runtime(model, runtime)
```

The existing `attach_transformers_olmoe_runtime` function remains available as
a backward-compatible, fail-closed OLMoE wrapper.

## Supported integration shapes

The pinned Transformers release exposes compatible packed gated-MLP expert
modules for these model types:

| Model type | Adapter | Validation level |
|---|---|---|
| `olmoe` | Supported | Tiny exact parity plus pinned full-checkpoint correctness and performance studies |
| `mixtral` | Supported experimentally | Tiny exact logits parity through the meta loader and paged runtime |
| `qwen2_moe` | Supported experimentally | Tiny exact logits parity through the meta loader and paged runtime, including the resident shared expert |

These adapters require decoder layers at `model.layers[*].mlp.experts`, SiLU
experts, and the packed `gate_up_proj` plus `down_proj` interface used by the
pinned Transformers version. OLMoE and Mixtral obtain the routed-expert width
from `intermediate_size`; Qwen2MoE uses `moe_intermediate_size`. Qwen2MoE's
shared expert and its sigmoid gate remain ordinary resident model parameters:
they are loaded by the non-expert loader and are not represented as paged cache
slots. Unsupported model types fail before the runtime is attached.

## Checkpoint boundary

The current read-only store consumes normalized per-expert safetensors entries:

```text
model.layers.<layer>.mlp.experts.<expert>.gate_proj.weight
model.layers.<layer>.mlp.experts.<expert>.up_proj.weight
model.layers.<layer>.mlp.experts.<expert>.down_proj.weight
```

An official checkpoint using another naming or packing scheme still needs an
audited, create-new conversion step before it can use this store. The adapter
does not rewrite a checkpoint, download weights or claim that an arbitrary Hub
artifact is compatible.

The pinned `Qwen/Qwen1.5-MoE-A2.7B` revision uses the normalized per-expert
names above and is therefore the selected public acceptance candidate. That
structural match is not itself full-checkpoint validation. Its separate Tongyi
Qianwen license and the no-redistribution boundary are documented in
[Third-party models and tools](THIRD_PARTY_MODELS.md).

## Evidence boundary

The Mixtral test proves that the shared runtime can replace its expert backend,
load only non-expert parameters into a meta-initialized model and reproduce the
eager model's logits on a deterministic tiny configuration. It does **not** yet
establish:

- compatibility with a specific public full-size Mixtral artifact;
- a bounded conversion tool for that artifact;
- memory reduction or speed on Mixtral;
- free-running generation, concurrency or production serving behavior.

The Qwen2MoE tiny test independently proves exact eager-versus-paged logits on a
deterministic tiny configuration whose shared expert remains resident and is
loaded through the non-expert path. Those synthetic weights are created by the
test and are not Qwen checkpoint weights.

For the pinned `Qwen/Qwen1.5-MoE-A2.7B` checkpoint, this repository has now
completed local deterministic full-checkpoint capture and sync-path parity gate:

- full-manifest integrity checks for required snapshot files;
- exact greedy-token parity against pinned reference on one controlled prompt
  (`2/2` tokens matched);
- bounded sync run accounting and fit diagnostics for that smoke prompt.

It is still a **correctness-smoke** only path: no throughput claim, no production
serving, no concurrency study, and no general benchmark across seeds/lengths/hardware.

The harness now also permits the bounded async scheduler for Qwen, but only
behind the same pinned autoregressive reference gate. `adaptive` and `auto`
remain disabled for this checkpoint. This is an experiment-enabling change,
not async full-checkpoint evidence: no Qwen async result is accepted or
published until exact output and cache/traffic invariants pass on real CUDA.

Next research steps are a paired sync/async correctness probe followed, only if
it passes, by a longer decode-focused run. Granite 4 H-Tiny and gpt-oss-20b
remain later research candidates only; neither is an implemented or validated
MoEVM adapter.

The first probe must use the same clean commit, pinned snapshot, reference,
workload and cache budget in both arms. Example (paths abbreviated):

```powershell
python scripts/benchmark_paged_olmoe.py --checkpoint qwen2-moe `
  --snapshot <PINNED_SNAPSHOT> --reference-metadata <REFERENCE_JSON> `
  --output results/qwen-sync.json --workload-file benchmarks/workloads/olmoe_m1.json `
  --workload-id systems_en --max-input-tokens 32 --max-new-tokens 2 `
  --device cuda:0 --policy lru --pipeline sync --slots-per-layer 4 `
  --staging-slots 2 --seed 17 --cuda-overlap-telemetry

python scripts/benchmark_paged_olmoe.py --checkpoint qwen2-moe `
  --snapshot <PINNED_SNAPSHOT> --reference-metadata <REFERENCE_JSON> `
  --output results/qwen-async.json --workload-file benchmarks/workloads/olmoe_m1.json `
  --workload-id systems_en --max-input-tokens 32 --max-new-tokens 2 `
  --device cuda:0 --policy lru --pipeline async --slots-per-layer 4 `
  --staging-slots 2 --seed 17 --cuda-overlap-telemetry

python scripts/compare_paged_pipeline_pair.py results/qwen-sync.json `
  results/qwen-async.json --correctness-smoke `
  --output results/qwen-sync-async-correctness.json
```

`--correctness-smoke` is deliberately separate from the normal comparison
gate. It accepts only the full-manifest Qwen profile with an exact
autoregressive reference match, emits a non-publishable result, and still
requires identical logical cache/traffic counters between sync and async.
