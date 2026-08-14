# Changelog

## Unreleased

- Added an opt-in bounded expert pipeline with one storage worker, pinned host
  staging, a dedicated CUDA H2D stream and per-slot CUDA events. The existing
  synchronous path remains the default.
- Added `--pipeline async` to the paged OLMoE benchmark; async runs require at
  least two staging slots so storage/page-cache service and H2D submission can
  progress ahead of expert compute within a routed layer.
- Added explicit pipeline draining and shutdown through `wait_idle()` and
  idempotent `close()`.
- Split logical cache misses from coalesced storage and transfer operations in
  benchmark metrics, with fail-fast bounded-queue admission.
- Bound benchmark JSON to a clean source commit and benchmark-script SHA-256.
- This MVP does not establish physical NVMe overlap or an end-to-end speedup:
  safetensors remains mmap/page-cache backed, and comparative runtime evidence
  is still required.

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
