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

Both adapters require decoder layers at `model.layers[*].mlp.experts`, SiLU
experts, and the packed `gate_up_proj` plus `down_proj` interface used by the
pinned Transformers version. Unsupported model types fail before the runtime is
attached.

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

## Evidence boundary

The Mixtral test proves that the shared runtime can replace its expert backend,
load only non-expert parameters into a meta-initialized model and reproduce the
eager model's logits on a deterministic tiny configuration. It does **not** yet
establish:

- compatibility with a specific public full-size Mixtral artifact;
- a bounded conversion tool for that artifact;
- memory reduction or speed on Mixtral;
- free-running generation, concurrency or production serving behavior.

The next acceptance gate is one pinned public second-model checkpoint with
verified shard provenance, exact greedy-token parity and a controlled baseline.
