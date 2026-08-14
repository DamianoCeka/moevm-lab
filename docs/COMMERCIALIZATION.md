# Commercialization outline

GitHub is distribution and proof, not the business by itself.

## Potential users

- local-AI developers who cannot fit large MoEs in VRAM;
- research teams needing on-prem inference;
- workstation and appliance vendors;
- inference providers optimizing tokens per euro;
- cloud teams with underutilized CPU RAM and storage beside GPUs;
- runtime projects that need a specialized expert-memory backend.

## What is sellable now

MoEVM Lab can support paid, fixed-scope feasibility audits today. The customer
pays for engineering time, controlled measurements, an evidence bundle and a
decision about model/hardware fit. The community code remains Apache-2.0, and a
negative or conditional result is a valid outcome.

The current design-partner offer, boundaries and public intake path are defined
in [commercial support](../COMMERCIAL_SUPPORT.md). It does not sell a production
runtime or promise a speedup.

## Evidence required before selling a production runtime

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

Paid audits must likewise distinguish simulated, replayed, microbenchmark and
end-to-end evidence. A fee buys scoped work and a reproducible answer, not a
preselected positive result.
