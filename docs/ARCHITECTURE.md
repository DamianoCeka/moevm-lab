# Architecture

## Research hypothesis

A sparse MoE inference runtime can reduce blocking weight-transfer latency when it treats experts as independently placeable memory objects and predicts the near-future working set.

The simulator tests this hypothesis without allocating model weights.

## Data flow

```text
Routing trace
     │
     ▼
Online predictor ───────► speculative expert addresses
     │                              │
     │                              ▼
     │                    protected VRAM buffer
     │                              │
     ▼                              ▼
Demand access ─────► VRAM demand cache ─► compute budget
     │                    │
     ├────────────────────┘
     ▼
RAM cache
     │
     ▼
NVMe backing store
```

## Address model

An expert is identified by `(layer_id, expert_id)`. The simulator deliberately does not assume that equal expert IDs across layers have equal semantics.

Each expert has a configured average byte size. Future versions should accept per-expert and per-tile sizes from checkpoint metadata.

## Cache hierarchy

### Demand VRAM cache

A byte-capacity LRU cache containing experts loaded by real demand. In baseline mode it receives the full configured VRAM expert budget.

### Protected prefetch VRAM buffer

When prefetch is enabled, a configured fraction of the VRAM expert budget is reserved for speculative entries. Incorrect predictions can evict other speculative entries but cannot consume the whole demand cache.

A useful speculative expert is promoted to the demand cache after access without another transfer.

### RAM cache

An inclusive byte-capacity LRU cache. Evicting an expert from RAM invalidates its VRAM copies.

### NVMe backing store

The backing store is assumed to contain every expert. It has configured sequential-equivalent bandwidth and per-batch latency. This abstraction is deliberately optimistic about file layout and I/O scheduling; real checkpoint replay must replace it.

## Predictor

The v0.1 online predictor combines:

1. target-layer expert frequency;
2. transitions from the previous token's experts at the same layer;
3. transitions from experts selected in the current layer to experts selected in the next layer;
4. a same-expert persistence prior for cold-start behavior.

Transition tables are bounded to avoid unbounded memory growth.

## Timing model

For each routing step:

1. the predictor has a ranked queue produced after the previous step;
2. confidence filtering removes weak candidates;
3. deadline-aware admission estimates current residency and transfer time;
4. candidates that cannot fit the available `previous_compute_ms × overlap_efficiency` budget are skipped;
5. admitted experts are prefetched;
6. actual experts are demanded;
7. any remaining expert transfer is blocking;
8. configured compute time is added;
9. the predictor observes the actual routing and ranks the next step.

Transfer time is modeled as:

```text
bytes / bandwidth + one batch latency
```

NVMe→RAM and RAM→VRAM times are additive in v0.1. A future pipeline model should represent concurrent engines, queue depth, cancellation and multi-step deadlines explicitly.

## Why the simulator reports traffic and latency

Prefetch can reduce demand-path stalls while increasing total PCIe traffic. Reporting only throughput would hide this cost. Every benchmark therefore includes:

- demand and prefetch stall;
- demand and total NVMe traffic;
- total RAM→VRAM traffic;
- useful and wasted prefetch entries;
- prefetch precision;
- cache hit-rates.
