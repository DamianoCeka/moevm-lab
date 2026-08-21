# Changelog

## Unreleased

- Added an experimental `qwen2_moe` adapter shape and exact tiny-model logits
  parity including the resident shared expert. The public
  `Qwen/Qwen1.5-MoE-A2.7B` acceptance candidate is revision-pinned and its
  separate Tongyi Qianwen license is documented. A full-checkpoint sync smoke
  passed exact 2-token reference parity with bounded VRAM; the harness now also
  permits a fail-closed async correctness probe while keeping Qwen timings
  non-publishable. No model weights are redistributed.
- Added an explicit Transformers MoE adapter boundary and an exact end-to-end
  tiny-Mixtral parity test while retaining the existing OLMoE API. This proves
  model integration at small scale; it is not full-checkpoint Mixtral support
  or performance evidence.
- Added a read-only `moevm doctor --machine` report that keeps observed GPU,
  host-RAM and selected-volume capacity separate from the configuration-derived
  expert memory ledger. It does not open CUDA, load a checkpoint, benchmark
  storage or claim model fit/performance.
- Added an original memory-ledger guide and corrected the public memory-flow
  diagram to distinguish the simulator's logical RAM LRU from the runtime's
  bounded pinned staging and unobserved OS page-cache path.
- Added opt-in same-device CUDA-event telemetry for paged-expert H2D and
  expert-compute intervals, a fail-closed telemetry-aware pair gate, and a
  create-only accessible SVG timeline renderer. This instruments one model call
  at a time; it does not claim physical NVMe overlap or a general speedup.
- Added a public, non-confidential commercial inquiry path and a fixed-scope
  design-partner offer for model/hardware fit audits, while keeping the
  community core Apache-2.0 and explicitly avoiding guaranteed performance or
  production-readiness claims.
- Added a Windows `demo.cmd` entry point that detects hardware, prepares an
  isolated pinned CUDA environment, acquires and verifies the supported OLMoE
  checkpoint, selects a guarded GPU-cache capacity and prints real paged-runtime
  memory and speed observations in one command.
- Added resumable, create-only local demo plans and summaries, plus optional
  fail-closed sync/async comparison and a no-write/no-network dry-run.
- Added an explicit benchmark `--demo-mode` with best-effort Git provenance;
  scientific benchmark mode remains clean-tree and fail-closed by default.
- Added an opt-in bounded expert pipeline with one storage worker, pinned host
  staging, a dedicated CUDA H2D stream and per-slot CUDA events. The existing
  synchronous path remains the default.
- Added `--pipeline async` to the paged OLMoE benchmark; async runs require at
  least two staging slots so storage/page-cache service and H2D submission can
  progress ahead of expert compute within a routed layer.
- Added explicit pipeline draining and shutdown through `wait_idle()` and
  idempotent `close()`.
- Split logical cache misses from coalesced storage and transfer operations in
  benchmark metrics, with fail-fast bounded-queue admission. Async lookahead
  leaves demand counters and LRU recency unchanged until the expert is actually
  consumed, preserving sync-versus-async cache-policy comparability.
- Bound benchmark JSON to a clean source commit and benchmark-script SHA-256.
- Added a fail-closed sync/async pair comparator that requires identical source,
  output, cache-policy counters and transfer traffic before reporting timings.
- Added sanitized three-pair async smoke evidence and a deterministic SVG that
  shows every paired wall-time observation and its narrow evidence boundary.
- Added a sanitized 36-pair RTX 6000 Ada study with deterministic workload,
  token-length and cache-capacity charts. It records both positive empty-cache
  results and retained-cache regressions instead of reducing the study to one
  headline number.
- Added an experimental `adaptive` pipeline mode that uses bounded async fill
  for routed-layer calls that start with free eligible slots and at least two
  requested misses; calls starting with a full partition use sync. Per-forward
  and per-expert decision counters make the rule auditable in benchmark JSON.
- Added measured pipeline profiles built from at least three exact-gated
  sync/async pairs. `--pipeline auto` can select cold and retained paths
  independently, switches only at a drained pass boundary, and rejects profiles
  that do not match the GPU, model, workload, budget, environment or source
  hashes.
- Kept the profile reference gate strict but mode-aware: autoregressive runs
  require exact baseline identity, while teacher-forced runs require the exact
  pinned feed and exact sync/async prediction parity without misclassifying a
  shared numerical prediction difference as a scheduling failure.
- Added a deterministic study summarizer that recomputes pair gates, validates
  the exact experiment matrix and strips private execution paths, plus support
  for fully resident routing-capture models that do not expose an Accelerate
  `hf_device_map`.
- This MVP does not establish physical NVMe overlap or a universal end-to-end
  speedup: safetensors remains mmap/page-cache backed, and adaptive-policy
  calibration, free-running generation, concurrency and profiler evidence
  remain open.

## 0.3.0 — 2026-08-14

- Relicensed the current community source tree under Apache License 2.0, added
  explicit inbound contribution terms and preserved Transformers attribution.
- Added read-only storage and CUDA transfer benchmarks plus a reproducible
  Crucial P2/P310 and RTX 3080 Ti hardware evidence package.
- Added per-expert fixed-latency modeling; the measured P310 profile moves the
  existing real-trace prefetch replay to `0.9791x` with +4.91% RAM-to-VRAM
  traffic.
- Added audited static/hybrid placement analysis with a strict exploratory
  leave-one-workload-out protocol and explicit preload accounting.
- Added a bounded read-only safetensors expert store, per-layer LRU/static/hybrid
  GPU slots, pinned staging and a synchronous Transformers expert backend.
- Added exact tiny-model CPU/CUDA checks and the first guarded full OLMoE paged
  smoke: matching token IDs, 21.34% lower observed peak allocated VRAM, a slower
  empty-cache pass and a faster retained-cache repeat under stated limitations.

## 0.2.0 — 2026-08-09

- Captured router decisions and top-k probabilities from the pinned open
  `allenai/OLMoE-1B-7B-0924` checkpoint on a local RTX 3080 Ti.
- Added ten reproducible real-routing traces spanning five controlled workloads
  and two sampling seeds (438 tokens and 56,064 expert accesses).
- Added locality, entropy, router-confidence and online-predictor trace analysis.
- Added a scored JSONL trace schema while retaining compatibility with v0.1 traces.
- Added an OLMoE replay profile and portable real-routing reference study.
- Added golden tests that verify every committed trace hash, analysis and replay.
- Documented the negative result: the current policy averages 1.0022× simulated
  replay speedup while increasing simulated RAM-to-VRAM traffic by 6.17%.
- Added a pinned, Windows-safe real-capture environment and checkpoint workflow.

## 0.1.1 — 2026-08-08

- Bundled default configurations so installed wheel and sdist commands work outside the checkout.
- Corrected deadline admission for intra-batch cache evictions.
- Corrected speculative hit, useful-prefetch and wasted-prefetch accounting.
- Made bounded predictor tables adapt to sustained routing-domain changes.
- Avoided reserving speculative VRAM when prefetching is effectively disabled.
- Protected demand-resident VRAM entries from speculative RAM admission.
- Rejected malformed trace topology while supporting complete captured-trace windows with nonzero token offsets.
- Rejected implicit JSON coercions and non-finite configuration values.
- Hardened Windows Unicode output and native-command failure propagation.
- Hardened private GitHub publication, tag pushing and bundle-remote handling.
- Expanded tests, packaging checks and Windows/Python 3.14 CI coverage.

## 0.1.0 — 2026-08-08

- Added byte-capacity VRAM/RAM/NVMe hierarchy simulator.
- Added protected speculative VRAM buffer.
- Added online temporal and cross-layer expert predictor.
- Added confidence filtering and deadline-aware prefetch admission.
- Added deterministic synthetic routing traces and JSONL import/export.
- Added baseline comparison, JSON/Markdown reports and CLI.
- Added toy and K3-shaped configurations.
- Added tests, CI, Windows/Linux bootstrap and research documentation.
