# Benchmarking rules

## Labels

Every result must be labeled as exactly one of:

- **simulation** — no real model execution;
- **trace replay** — real routing decisions, simulated or measured transfers;
- **microbenchmark** — real transfer/kernel measurement without full generation;
- **end-to-end** — real model generation with quality checks.

Never place simulated and end-to-end tokens/s in the same table without an explicit label.

## Required baseline

Use the same routing trace, cache budget, expert size, bandwidth assumptions and compute budget for baseline and candidate. Only the scheduling or cache policy may change.

## Required metrics

- elapsed time and tokens/s;
- demand stall and prefetch stall;
- VRAM-only and VRAM+RAM hit-rate;
- NVMe→RAM and RAM→VRAM bytes;
- bytes per token;
- prefetch precision, useful entries and wasted entries;
- warm-up policy and random seed.

For real execution also report:

- model and exact checkpoint revision;
- quantization;
- prompt/decode lengths and batch size;
- GPU, CPU, RAM, storage, PCIe generation and operating system;
- power mode and relevant clock limits;
- quality regression versus the non-offloaded reference.

## Acceptance gates

A change advances from simulation to implementation only when it satisfies all applicable gates:

1. It improves blocking stall or memory capacity on at least two trace seeds.
2. It does not hide an excessive increase in total traffic.
3. Its predictor statistics are reported, not inferred from speedup.
4. The comparison is reproducible from committed configuration and trace metadata.
5. The result survives a less-local adversarial trace.

A real backend advances toward a K3 adapter only after it beats a public baseline on at least one smaller MoE using identical hardware and quality settings.

## Reference command

```bash
moevm compare --config configs/toy.toml --output-dir results/toy
```
