# Commercial support and design-partner pilots

MoEVM Lab is an open-source research project under the
[Apache License 2.0](LICENSE). The community source remains freely available
under that license.

Paid engagements cover scoped engineering work: requirements analysis,
reproducible measurements, hardware-fit evaluation, prototype integration and
technical support. They do not buy exclusive rights to the community code or
access to a hidden, faster edition.

## MoE Model & Hardware Fit Audit

The audit answers a practical question:

> Can this sparse MoE workload run within the available GPU, RAM and storage
> budget, and what is the actual bottleneck before more hardware or integration
> work is purchased?

The result may be positive, negative or conditional. A recommendation not to
proceed is a valid audit outcome.

### Who this is for

- local-AI and on-prem teams constrained by GPU memory;
- research groups evaluating sparse MoE inference;
- workstation and inference-appliance vendors;
- runtime engineers comparing expert caching, paging and placement policies;
- teams deciding whether additional GPU capacity is necessary.

This service is not currently positioned as production inference hosting,
multi-GPU serving, a Kimi K3 deployment or an operational SLA.

### What a pilot delivers

Unless a different scope is agreed in writing, a pilot covers one checkpoint,
one hardware profile, one workload family and one existing baseline.

1. **Scope record** — exact model revision, workload, prompt/decode lengths,
   batch size and success criterion.
2. **Hardware profile** — GPU and VRAM, CPU, system RAM, storage, operating
   system and relevant runtime versions.
3. **Compatibility assessment** — what MoEVM can execute today, what requires
   an adapter and what cannot yet be evaluated reliably.
4. **Controlled measurements** — a baseline and a MoEVM candidate run when the
   model is supported, using the same documented workload and comparable
   hardware constraints, with relevant differences disclosed.
5. **Bottleneck analysis** — observed VRAM, process memory, wall time,
   throughput where valid, cache behavior and logical transfer traffic.
6. **Evidence bundle** — a concise report plus available machine-readable
   artifacts, commands, configurations and hashes needed to reproduce it.
7. **Recommendation** — proceed, stop, change configuration, collect more
   evidence or consider different hardware.

Every result is classified under the [benchmarking rules](docs/BENCHMARKING.md)
as simulation, routing capture, trace replay, microbenchmark or end-to-end
evidence. Simulated throughput is never presented as measured runtime
throughput.

## Introductory packages

These are non-binding design-partner ranges in EUR. Final scope, fee, schedule,
payment terms and applicable tax treatment are confirmed in writing before
work begins.

| Package | Scope | Target delivery window | Introductory range |
|---|---|---:|---:|
| Fit Review | 90-minute technical session and short compatibility memo | 2–3 business days | EUR 350–500 |
| Reproducible Pilot | One supported checkpoint, workstation and workload family | 5–7 business days | EUR 1,500–2,500 |
| Custom Integration | New adapter, hardware preset or expanded experiment | Agreed after discovery | From EUR 5,000 |

Delivery starts only after the written scope is accepted, required inputs and
access are available, and any agreed initial payment has been received. The
written quote states whether applicable taxes are included or excluded.
Hardware, cloud usage, paid model access and travel are not included unless the
quote says otherwise. These introductory ranges may change as the supported
scope and evidence mature.

This page is informational and is not an offer. No paid engagement exists until
a written quote or statement of work is accepted. Availability is limited, and
work is accepted only when its technical question can be evaluated
responsibly.

Design partners are asked to provide structured feedback. No customer name,
artifact, benchmark or case study is published without explicit written
permission.

Pre-existing community code remains Apache-2.0. Ownership and licensing of new
customer-specific adapters, reports, configurations or other deliverables are
defined in the written statement of work. Nothing customer-specific is
upstreamed or published without written agreement, and any requested
exclusivity must be stated explicitly before work starts.

## Current evidence boundary

MoEVM Lab provides a [one-command OLMoE demo](docs/ONE_COMMAND_DEMO.md), measured
hardware calibration and a bounded paged-runtime prototype. In the published
[multi-workload study](benchmarks/reference/paged-runtime-olmoe-p310-multiworkload/README.md),
the prototype used about 21% less peak allocated VRAM than one specific
Transformers/Accelerate CPU-offload baseline and showed positive timing results
on that controlled setup.

That study used one pinned OLMoE checkpoint, one RTX 3080 Ti, five prompts, one
seed, teacher forcing and no concurrency. The baseline was not a tuned serving
engine. The separate
[async study](benchmarks/reference/paged-runtime-olmoe-p310-async-smoke/README.md)
is an even narrower three-pair, two-token smoke. Neither result establishes a
general speedup or production-serving performance.

MoEVM Lab therefore does **not** currently guarantee:

- a particular speedup, tokens/s result or hardware-cost reduction;
- compatibility with an arbitrary MoE checkpoint;
- Kimi K3 checkpoint execution;
- production stability, long-context behavior or concurrent serving;
- multi-GPU operation;
- physical NVMe overlap or CUDA kernel/copy overlap;
- that a result from one machine transfers unchanged to another.

The [hardware reference](benchmarks/reference/hardware-rtx3080ti-p310/README.md)
and [roadmap](docs/ROADMAP.md) define the current technical boundary.

## Privacy, model access and licensing

The customer must have the right to access and evaluate the selected model,
checkpoint, prompts and data. Third-party checkpoints remain governed by their
own terms; see [third-party provenance](docs/THIRD_PARTY_MODELS.md).

GitHub issues are public. Read the [repository inquiry privacy notice](PRIVACY.md)
before submitting. Do not post proprietary checkpoints, private download links,
credentials, remote-access details, customer data, confidential prompts, logs
or benchmark artifacts. No confidential material is accepted until a written
NDA or statement of work defines the private channel, authorized access,
retention and deletion rules. State that a private process is needed without
including the material itself.

## Start a non-confidential inquiry

[Open the commercial pilot form](https://github.com/DamianoCeka/moevm-lab/issues/new?template=commercial_inquiry.yml).
Describe the decision you need to make, the public model or model family, the
hardware profile and the success criterion. Opening an issue requests a
technical-fit review; it is not automatic acceptance of an engagement or a
binding quotation.
