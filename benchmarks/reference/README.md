# Reference simulations

These files are committed to make changes reviewable. They are **simulations**, not measured LLM generation benchmarks.

- `toy-64/`: 64-token run using `configs/toy.toml`.
- `k3-shape-8/`: 8-token smoke test using `configs/k3_shape.toml`.
- `vram_sweep.csv`: toy-profile sweep over several VRAM expert-cache budgets.

Regenerate with:

```bash
python -m moevm compare --config configs/toy.toml --tokens 64 --output-dir results/reference-toy
python -m moevm compare --config configs/k3_shape.toml --tokens 8 --output-dir results/k3-smoke
python scripts/sweep.py --config configs/toy.toml --sizes-mib 64,128,256,384 --output results/vram_sweep.csv
```
