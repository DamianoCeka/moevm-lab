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

- [ ] benchmark pageable RAM, pinned RAM and mapped host memory;
- [ ] measure NVMe sequential and random read behavior at expert-sized blocks;
- [ ] model PCIe copy engines and queue depth;
- [ ] calibrate transfer functions from measured data;
- [ ] publish a hardware profile for the development workstation.

**Exit criterion:** trace replay estimates track microbenchmarks within a documented error band.

## M3 — Small-MoE runtime prototype

- [ ] PyTorch/C++ extension or focused vLLM integration;
- [ ] expert slot map and protected miss buffer;
- [ ] pinned-memory backing store;
- [ ] asynchronous copies with CUDA streams/events;
- [ ] correctness tests against fully resident inference;
- [ ] first end-to-end tokens/s benchmark.

**Exit criterion:** measurable improvement or capacity gain on a real model.

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
- [ ] public license/open-core decision.
