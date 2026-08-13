# Multi-workload real OLMoE paged-runtime study

This reference extends the first two-token smoke to five controlled prompts and
16 continuation tokens per prompt. MoEVM used 32 independent LRU expert slots
per layer on the RTX 3080 Ti workstation. The comparison baseline was the same
pinned OLMoE revision dispatched by Transformers and Accelerate across GPU and
CPU.

## Result

| Workload | Baseline | Paged cold | Cold vs baseline | Paged retained | Retained vs baseline | Greedy predictions |
|---|---:|---:|---:|---:|---:|---:|
| systems_en | 10.423 s | 7.283 s | 1.431x | 5.112 s | 2.039x | 15/16 |
| systems_it | 9.838 s | 7.383 s | 1.333x | 5.215 s | 1.886x | 16/16 |
| python_code | 9.799 s | 7.456 s | 1.314x | 5.037 s | 1.945x | 16/16 |
| math_reasoning | 9.856 s | 7.432 s | 1.326x | 5.407 s | 1.823x | 15/16 |
| domain_switch | 9.749 s | 7.560 s | 1.289x | 5.369 s | 1.816x | 16/16 |
| **Aggregate** | **49.665 s** | **37.114 s** | **1.338x** | **26.140 s** | **1.900x** | **78/80** |

The paged runs peaked at about 6.90 GiB allocated VRAM versus 8.77 GiB for
the CPU-offload baseline, a 21.33% reduction. Immediate retained-cache repeats
requested 60.83 GiB of logical storage traffic versus 72.08 GiB when the
dynamic expert cache started empty, a 15.61% reduction. The largest observed
paged-process RSS was 13.01 GiB.

## Correctness boundary

Timing uses explicit teacher forcing: both systems receive the 16 token IDs
produced by the pinned baseline, while MoEVM's greedy predictions are recorded
separately. All fed sequences, cold/retained predictions and post-run checkpoint
hashes passed their invariants.

MoEVM predicted 78 of 80 reference IDs. Both differences were audited with the
same forced prefix. One side of each comparison had an exact BF16 top-two tie;
the other separated those same two candidates by only 0.0625. At the mismatch
positions, the top-8 expert sets matched in every layer and logits cosine
similarity was at least 0.99993. These are numerical argmax ties, not evidence
of wrong expert bytes or cache-dependent output.

## Interpretation

This is the first controlled multi-workload evidence that the prototype can
trade less VRAM for higher throughput than this particular CPU-offload path.
The strongest honest claim is **1.34x aggregate with an empty dynamic expert
cache and 1.90x on an immediate retained-cache repeat, while allocating about
21% less VRAM**.

It is not a general serving claim. Accelerate CPU offload is not a tuned server;
the Windows file cache was not flushed; traffic counters are logical rather
than physical NVMe bytes; and the synchronous Python runtime does not overlap
I/O, transfer and compute. The study uses five prompts, one seed, one GPU and no
concurrency. See `result.json` for exact values, raw-output hashes and the full
limitations list.

## Reproduce

Create the 16-token baseline metadata first with
`scripts/capture_real_routing.py`, using greedy temperature 0, seed 17 and the
pinned local checkpoint. Then run `scripts/benchmark_paged_olmoe.py` once per
workload with:

```text
--teacher-force-reference --policy lru --slots-per-layer 32
--staging-slots 1 --max-input-tokens 64 --max-new-tokens 16 --seed 17
```

The output path is create-only, all model access is local/offline, and the
checkpoint mappings are closed before the independent SHA-256 verification.
