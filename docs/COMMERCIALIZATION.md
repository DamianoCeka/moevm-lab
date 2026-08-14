# Commercialization outline

GitHub is distribution and proof, not the business by itself.

## Potential users

- local-AI developers who cannot fit large MoEs in VRAM;
- research teams needing on-prem inference;
- workstation and appliance vendors;
- inference providers optimizing tokens per euro;
- cloud teams with underutilized CPU RAM and storage beside GPUs;
- runtime projects that need a specialized expert-memory backend.

## Evidence required before monetization

1. End-to-end improvement on a real public model.
2. Reproducible quality equivalence.
3. A result on more than one hardware profile.
4. A clear advantage over current vLLM/llama.cpp/offload baselines.
5. Operational stability under long contexts, batching and concurrent requests.

## Plausible product layers

```text
Community core (Apache-2.0)
  trace tools, metrics, basic cache and single-node backend

Professional
  auto-tuning, checkpoint optimizer, profiler and hardware presets

Enterprise
  multi-GPU placement, monitoring, deployment tooling and support

Hosted
  managed inference using the runtime where economics are favorable
```

## Strongest value metric

The sales metric should not be GitHub stars. It should be one of:

- same quality at lower hardware cost;
- more model capacity on the same machine;
- more tokens per second at the same cost;
- lower latency or energy per generated token.

Do not price or seek investment from simulated results alone.
