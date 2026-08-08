# Kimi K3 notes

## Published topology used by the synthetic profile

The official Kimi K3 repository describes:

- 2.8T total parameters;
- 104B activated parameters;
- 93 layers, including one dense layer;
- 896 experts;
- 16 selected experts plus shared experts;
- MXFP4 weights and MXFP8 activations;
- Kimi Delta Attention and Gated MLA;
- a 1M-token context window.

Official source: <https://github.com/MoonshotAI/Kimi-K3>

`configs/k3_shape.toml` uses only the published layer/expert/top-k shape. Its average expert payload and compute time are synthetic placeholders.

## What a real adapter must discover

- exact checkpoint shard and tensor naming;
- per-layer expert tensor sizes and alignment;
- shared-expert residency requirements;
- router output format and whether routing can be exposed early enough for prefetch;
- MXFP4 decode/kernel constraints on the target GPU;
- non-expert resident memory, including attention state and KV cache;
- legal and license conditions for redistribution, conversion and serving.

## Related current engineering direction

vLLM discussions in 2026 include expert-level CPU offload, pinned-memory storage, fixed GPU caches, cross-layer prediction and asynchronous pipelines:

- <https://github.com/vllm-project/vllm/issues/33869>
- <https://github.com/vllm-project/vllm/issues/38256>
- <https://github.com/vllm-project/vllm/issues/41447>

This means basic expert caching alone is not sufficient novelty. MoEVM Lab should focus its research contribution on combinations such as protected speculative buffers, confidence-aware admission, storage layout, tile streaming, deadline scheduling and measured auto-tuning.
