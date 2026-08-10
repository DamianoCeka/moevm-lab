# Storage calibration microbenchmark

This harness measures real random reads shaped like the simulator's synthetic
12 MiB expert transfers. It is a **microbenchmark**, not an end-to-end MoE result.
It never opens target files for writing and verifies their size and modification
time after each run.

## Buffered run (portable default)

Pass one or more large files. `--operations` and `--warmup` apply to every file.
The output path must not exist, which prevents accidental replacement of a target
or previous measurement.

```powershell
py -3.12 -m moevm.storage_benchmark `
  C:\moevm-lab-cache\shard-1.safetensors `
  C:\moevm-lab-cache\shard-2.safetensors `
  C:\moevm-lab-cache\shard-3.safetensors `
  --chunk-size 12MiB `
  --operations 64 `
  --warmup 4 `
  --seed 20260810 `
  --output results\storage-p310-buffered.json
```

The default mode uses normal operating-system I/O. Python's extra buffering is
disabled, but the OS page cache is active. The harness does not flush or control
that cache, so buffered results **must not be described as cold-storage speed**.
Warmup operations are excluded from throughput and latency statistics and are
reported separately.

## Windows unbuffered run

Windows can bypass the normal file cache with a second, explicit mode:

```powershell
py -3.12 -m moevm.storage_benchmark `
  C:\moevm-lab-cache\shard-1.safetensors `
  --io-mode windows-unbuffered `
  --chunk-size 12MiB `
  --operations 64 `
  --warmup 4 `
  --seed 20260810 `
  --output results\storage-p310-unbuffered.json
```

This mode uses a read-only Win32 handle with `FILE_FLAG_NO_BUFFERING`. It queries
the file's logical and physical sector requirements, uses aligned offsets and an
aligned `VirtualAlloc` buffer, and rejects incompatible chunk sizes. It fails
clearly instead of falling back to cached I/O when alignment information or the
filesystem capability is unavailable. Controller, device and SSD caches can
still affect the result; even this mode does not guarantee a physically cold SSD.

## Comparing drives

For a valid comparison:

1. Verify that both locations contain byte-identical files (for example, with
   SHA-256).
2. Use the same files, file order, chunk size, operation count, warmup and seed.
3. Keep the system power mode fixed and close unrelated disk-heavy applications.
4. Compare like with like: buffered against buffered or unbuffered against
   unbuffered.
5. Repeat runs and retain every JSON report rather than only the best result.

The JSON includes host and target metadata, all parameters, measured and warmup
operation/byte counts, aggregate and per-file throughput, and latency min/mean/
p50/p95/p99/max. Offsets are uniform random aligned positions sampled with
replacement. File order is recorded and is part of seeded reproducibility.
