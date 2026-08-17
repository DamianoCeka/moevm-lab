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

The Qwen2MoE test independently proves exact eager-versus-paged logits on a
deterministic tiny configuration whose shared expert remains resident and is
loaded through the non-expert path. Those synthetic weights are created by the
test and are not Qwen checkpoint weights. This evidence does **not** yet
establish for the pinned `Qwen/Qwen1.5-MoE-A2.7B` checkpoint:

- completed acquisition and integrity verification of every required file;
- exact full-model greedy-token parity;
- bounded VRAM use or a memory reduction on a 12 GB GPU;
- speed relative to a controlled eager or offload baseline;
- free-running generation, concurrency or production serving behavior.

The next acceptance gate is that exact pinned Qwen revision with verified shard
provenance, exact greedy-token parity and a controlled baseline. Granite 4
H-Tiny and gpt-oss-20b remain later research candidates only; neither is an
implemented or validated MoEVM adapter.
