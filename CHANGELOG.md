# Changelog

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
