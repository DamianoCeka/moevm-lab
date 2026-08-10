# RAM-to-VRAM CUDA transfer calibration

This microbenchmark measures the isolated host-memory to GPU-memory path used by
the MoEVM simulator. It is a calibration tool, not an end-to-end model benchmark.
It loads no model weights and runs these cases sequentially:

1. pageable RAM, synchronous copy;
2. pinned RAM, synchronous copy;
3. pinned RAM, asynchronous copy on a dedicated CUDA stream.

Pageable asynchronous copies are intentionally omitted. PyTorch can accept
`non_blocking=True` for pageable memory, but a reliably asynchronous H2D DMA
transfer requires pinned host memory; including that case would give the result
a misleading label.

## Environment

Use the existing CUDA-enabled Python environment used for real routing capture.
PyTorch remains an environment-specific dependency and is not installed by the
core MoEVM package. Verify it before running the benchmark:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The last value must be `True`. A version ending in `+cpu`, or a `None` CUDA
version, cannot run this benchmark even when `nvidia-smi` sees the GPU. Install
the appropriate build using the official PyTorch installation selector rather
than adding a generic Torch dependency to `pyproject.toml`.

## Reproducible run

The default chunk is 12 MiB, matching the current synthetic expert payload:

```powershell
python -m moevm.cuda_transfer_benchmark `
  --chunk-size 12MiB `
  --operations 64 `
  --warmup 8 `
  --mode both `
  --async-depth 8 `
  --device 0 `
  --output results\hardware\ram-to-vram-12mib.json
```

Existing output files are never overwritten. Use a new path for each run and
record whether other GPU workloads were active. Run at least three times after
the GPU has reached a stable power state; compare medians instead of selecting
the best run.

## JSON interpretation

Each result contains:

- payload throughput in bytes/s, decimal GB/s, GiB/s and MiB/s;
- CUDA-event transfer latency with p50, p95 and p99;
- host submission latency;
- per-copy host completion latency for synchronous cases, or synchronized batch
  completion latency for the asynchronous case;
- warmup count and time;
- a byte-level transfer check;
- GPU, CUDA runtime, cuDNN, PyTorch, Python and host metadata.

Throughput uses total payload bytes divided by synchronized host wall time. The
sync result therefore includes one synchronization per copy; async submits up to
`--async-depth` copies before synchronizing. Cases reuse one source tensor and
one destination tensor, and execute sequentially, so depth does not multiply the
payload allocation.

## Safety and limits

- Default peak payload: 12 MiB host plus 12 MiB device for one case.
- Hard chunk limit: 256 MiB.
- Hard measured-operation limit: 4,096 transfers per case.
- At least 64 MiB of reported free VRAM is kept as a safety reserve.
- Pinned, pageable and device buffers are released after every case.
- No disk files are read except Python/package files, and no model is loaded.

Results still depend on GPU power state, Windows WDDM activity, PCIe generation
and link width, NUMA placement, other CUDA work and CPU scheduling. They calibrate
the isolated RAM-to-VRAM path only; they do not prove inference speedup.
