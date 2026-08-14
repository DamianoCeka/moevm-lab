# Reference evidence

These files are committed to make changes reviewable. Each directory states its
evidence label. Simulation, isolated hardware measurements and the controlled
runtime smoke are kept distinct; none is a production serving claim.

- `toy-64/`: 64-token run using `configs/toy.toml`.
- `k3-shape-8/`: 8-token smoke test using `configs/k3_shape.toml`.
- `vram_sweep.csv`: toy-profile sweep over several VRAM expert-cache budgets.
- `real-routing-olmoe-m1/`: real OLMoE router captures plus simulated cache
  replays, with trace hashes and two sampling seeds.
- `real-routing-olmoe-m1/placement/`: audited offline static/hybrid placement
  replay, including strict leave-one-workload-out evidence.
- `hardware-rtx3080ti-p310/`: measured read-only NVMe and RAM-to-VRAM calibration
  with sanitized per-repetition evidence and calibrated trace replay.
- `paged-runtime-olmoe-p310-smoke/`: sanitized first full-model paged-runtime
  observation, exact token-ID gate, raw-output hashes and limitations.
- `paged-runtime-olmoe-p310-async-smoke/`: three alternating-order sync/async
  pairs, exact cache/traffic gates, a deterministic comparison chart and a
  deliberately narrow two-token evidence boundary.
- `paged-runtime-olmoe-p310-multiworkload/`: five-workload, 16-token
  teacher-forced comparison against the pinned Accelerate CPU-offload path.

Regenerate with:

```bash
python -m moevm compare --config configs/toy.toml --tokens 64 --output-dir results/reference-toy
python -m moevm compare --config configs/k3_shape.toml --tokens 8 --output-dir results/k3-smoke
python scripts/sweep.py --config configs/toy.toml --sizes-mib 64,96,128,192,256,384,512 --output results/vram_sweep.csv
```
