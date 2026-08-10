# Real OLMoE paged-runtime smoke

This reference records the first full-model execution of MoEVM Lab's bounded
expert-paging prototype. The pinned OLMoE checkpoint ran locally on an RTX 3080
Ti using 32 independent LRU slots per layer. It produced the same two greedy
token IDs as the existing Transformers/Accelerate CPU-offload capture.

## Result

| Measurement | Accelerate CPU offload | Paged, empty expert cache | Paged, retained expert cache |
|---|---:|---:|---:|
| Generated token IDs | `187, 187` | `187, 187` | `187, 187` |
| Model load | 5.055 s | 0.896 s | — (same loaded model) |
| Prefill | 1.746 s | 3.375 s | 1.602 s |
| One-token decode throughput | 1.496 tok/s | 5.362 tok/s | 6.332 tok/s |
| End-to-end throughput, including prefill | 0.828 tok/s | 0.561 tok/s | 1.136 tok/s |
| Observed peak allocated VRAM | 8.770 GiB | 6.899 GiB | 6.899 GiB |

Relative to this exact baseline, the empty-cache pass is `0.677x` end-to-end
and therefore slower. The retained-cache repeat is `1.372x` end-to-end, while
the paged run's observed peak allocation is 21.34% lower. This is useful
evidence that bounded expert paging works and that reuse can pay for data
movement. It is not evidence of a general 37% serving improvement.

Only one generation-equivalent decode interval was measured. The baseline is a
routing-capture path with Accelerate CPU offload, not a tuned inference server.
The Windows OS page cache was not flushed, and "cold expert cache" means only
that the runtime's dynamic GPU slots were empty. Storage counters are logical
requested bytes, not physical NVMe telemetry. The synchronous Python prototype
does not overlap reads, host-to-device copies, and computation.

`result.json` is the sanitized machine-readable record. The detailed ignored
outputs contain local paths and remain under `results/`; their SHA-256 digests
anchor this reference without publishing workstation-specific metadata.

## Reproduce

First create the matching two-token baseline metadata with
`scripts/capture_real_routing.py`. Then run from the repository root, replacing
the two placeholders with local paths:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
& '<PYTHON_3_12_CUDA>' .\scripts\benchmark_paged_olmoe.py `
  --snapshot '<LOCAL_PINNED_SNAPSHOT>' `
  --workload-file '.\benchmarks\workloads\olmoe_m1.json' `
  --workload-id 'python_code' `
  --reference-metadata '<TWO_TOKEN_BASELINE_METADATA>' `
  --policy lru `
  --slots-per-layer 32 `
  --staging-slots 1 `
  --max-input-tokens 64 `
  --max-new-tokens 2 `
  --seed 17 `
  --device cuda:0 `
  --output '.\results\paged-runtime-p310\lru32-python-code.json'
Remove-Item Env:PYTHONPATH
```

The harness fails closed on model revision, shard hashes, dtype, tensor
materialization, available VRAM, baseline token mismatch, and output overwrite.
