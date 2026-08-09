# Reference simulations

These files are committed to make changes reviewable. Each directory states its
evidence label; none is a measured MoEVM end-to-end generation benchmark.

- `toy-64/`: 64-token run using `configs/toy.toml`.
- `k3-shape-8/`: 8-token smoke test using `configs/k3_shape.toml`.
- `vram_sweep.csv`: toy-profile sweep over several VRAM expert-cache budgets.
- `real-routing-olmoe-m1/`: real OLMoE router captures plus simulated cache
  replays, with trace hashes and two sampling seeds.

Regenerate with:

```bash
python -m moevm compare --config configs/toy.toml --tokens 64 --output-dir results/reference-toy
python -m moevm compare --config configs/k3_shape.toml --tokens 8 --output-dir results/k3-smoke
python scripts/sweep.py --config configs/toy.toml --sizes-mib 64,96,128,192,256,384,512 --output results/vram_sweep.csv
```
