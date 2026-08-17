# Roadmap

## M0 — Reproducible simulator ✅

- [x] byte-capacity VRAM/RAM hierarchy;
- [x] implicit NVMe backing store;
- [x] protected prefetch partition;
- [x] online temporal and cross-layer predictor;
- [x] deterministic synthetic routing traces;
- [x] baseline comparison and machine-readable reports;
- [x] Windows bootstrap and CI.

## M1 — Real routing evidence ✅

- [x] select a small, runnable open MoE as the first target;
- [x] capture `(token, layer, selected experts, router scores)`;
- [x] use controlled non-private prompts and store trace metadata;
- [x] compare synthetic versus real locality, entropy and transition stability;
- [x] add prediction confidence and no-prefetch admission thresholds.

**Exit criterion met:** ten trace captures across two seeds are committed with
hashes, analysis and golden replay results. The result is approximately neutral
on speedup and negative on transfer traffic, which defines the M2/M3 work.

## M2 — Hardware-calibrated replay

- [x] benchmark pageable and pinned RAM-to-VRAM paths;
- [ ] measure mapped host memory as a separate execution path;
- [x] measure NVMe random read behavior at expert-sized blocks and multiple
  queue depths;
- [ ] model PCIe copy engines and queue depth;
- [x] calibrate transfer functions from measured data;
- [x] publish a reproducible hardware profile for the development workstation.

The first calibrated P310 replay is `0.9791x` for the current predictor and
adds 4.91% RAM-to-VRAM traffic. **Exit criterion still open:** the isolated
transfer functions are reproducible, but trace-replay estimates have not yet
been shown to track end-to-end measurements within a documented error band.

## M3 — Small-MoE runtime prototype

- [x] focused PyTorch/Transformers expert-backend integration;
- [x] explicit Transformers adapter boundary with exact tiny-model parity for
  both OLMoE and Mixtral;
- [ ] verified full-checkpoint execution and measured evidence for a second MoE
  family; current full-model evidence remains OLMoE-only;
- [x] per-layer expert slot map and bounded transactional staging path;
- [x] bounded pinned-memory staging buffer;
- [ ] pinned host expert cache;
- [x] opt-in asynchronous-copy MVP with a dedicated CUDA stream and per-slot
  events;
- [x] exact tiny-model logits/generation tests and a full-model greedy-token
  parity gate;
- [x] first controlled full-model tokens/s and VRAM smoke measurement;
- [x] 36-pair RTX 6000 Ada sync/async sensitivity study across workloads,
  continuation lengths and GPU-cache capacities;
- [x] first conservative adaptive sync/async selector based on layer residency
  and free-slot state, with explicit decision counters;
- [x] fail-closed measured per-pass pipeline profiles bound to the exact GPU,
  workload, cache budget and runtime/benchmark hashes;
- [x] opt-in same-device CUDA-event telemetry for paged-expert H2D and
  paged-expert-compute intervals;
- [ ] collect and publish repeated, paired CUDA-timeline captures; the
  instrumentation is implemented, but no accepted capture exists yet and it is
  not overlap or speedup evidence on its own;
- [ ] online timing- and queue-aware adaptive policy that generalizes beyond a
  calibrated workload;

The first LRU32 OLMoE smoke reduces observed peak allocated VRAM by 21.34% and
the retained-cache pass is faster than the matching CPU-offload capture, while
the empty-cache pass is slower. The 36-pair RTX 6000 Ada follow-up finds a
positive five-workload core result but also retained-cache regressions at longer
continuations. The async work remains an opt-in, one-worker MVP over
mmap/page-cache-backed storage; it is not proof of direct physical NVMe overlap.
Opt-in CUDA-event telemetry can now collect same-device paged-expert H2D versus
expert-compute interval evidence, but no accepted paired capture has yet been
published. Even a future capture will not by itself prove physical storage
activity or a general speedup. **Exit criterion still open:** measured profiles
remain workload-specific; repeated paired timeline captures, longer free-running
generation, concurrency and cross-workload validation are still required before
a general performance claim.

## M4 — Storage-aware expert runtime

- [ ] checkpoint rewriter with expert-contiguous layout;
- [ ] asynchronous direct-I/O experiments;
- [ ] batched reads and prefetch cancellation;
- [ ] expert clusters learned from workload traces;
- [ ] admission control based on expected latency saved per byte.

## M5 — Tile streaming research

- [ ] divide experts into kernel-friendly matrix tiles;
- [ ] overlap tile arrival and GEMM;
- [ ] investigate out-of-order expert execution;
- [ ] account for partial results and deterministic reduction;
- [ ] compare against whole-expert loading.

## M6 — K3-shaped and K3 checkpoint work

- [ ] validate official checkpoint format and license requirements;
- [ ] map KDA, Gated MLA, shared experts and vision components;
- [ ] separate non-expert resident state from the expert store;
- [ ] build an adapter only after M3/M4 evidence;
- [ ] publish no 10 tok/s claim without end-to-end measurement.

## M7 — Productization

- [ ] stable runtime API;
- [ ] auto-tuner for hardware and workload;
- [ ] telemetry and profiling dashboard;
- [ ] enterprise deployment and support model;
- [x] license the community source tree under Apache-2.0; evaluate optional
  commercial layers separately.
