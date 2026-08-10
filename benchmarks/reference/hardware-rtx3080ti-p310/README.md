# RTX 3080 Ti + Crucial P310 calibration

This directory records the first measured hardware profile for MoEVM Lab. It is
local workstation evidence, not a portable performance claim and not an
end-to-end model benchmark.

## Result

The original OLMoE replay profile assumed `22 GB/s` RAM-to-VRAM and `7 GB/s`
NVMe-to-RAM. Measurements on this host support a more conservative synchronous
profile:

| Path | Fitted bandwidth | Fitted fixed latency |
|---|---:|---:|
| Pinned RAM to RTX 3080 Ti | 13.00 GB/s | 22.3 us |
| Crucial P310 to RAM | 6.04 GB/s | 299 us |
| Crucial P2 to RAM | 1.84 GB/s | 465 us |

The P310 reaches a median `5.26 GB/s` for one outstanding random 12 MiB read and
`7.09 GB/s` at queue depth eight. The P2 reaches `1.72 GB/s` and `1.95 GB/s`
respectively. The GPU link was observed at PCIe 4.0 x8 under transfer load.

Replaying all ten real OLMoE traces with the measured P310 profile changes the
existing one-step prefetch result from approximately neutral to negative:

- aggregate speedup: `0.9791x`;
- mean per-trace speedup: `0.9797x`;
- range: `0.9582x` to `0.9973x`;
- mean RAM-to-VRAM traffic change: `+4.91%`.

The replay charges each measured fixed-latency intercept once per expert
transfer, matching the current synchronous runtime. Historical profiles retain
their original batch-level latency semantics. This result rejects the current one-step prefetch policy under the calibrated
profile. It does **not** reject expert caching or static/adaptive placement.

## Method

Storage measurements used Microsoft DiskSpd 2.2 with read-only, unbuffered,
uniform-random I/O over a verified OLMoE shard. The primary workload used 12 MiB
blocks, matching one BF16 expert. Queue-depth-one tests had three repetitions;
queue-depth-eight tests had two. Additional 1, 4 and 12 MiB measurements fit the
bandwidth-plus-fixed-latency model.

RAM-to-VRAM measurements used `python -m moevm.cuda_transfer_benchmark` with
pageable synchronous, pinned synchronous and pinned asynchronous transfers.
Long 12 MiB runs used 2,048 transfers and three repetitions. Additional 1 and
4 MiB runs provided the linear fit. Medians are reported; no best-run selection
was used.

The machine-readable record is in `calibration.json`. Raw XML and JSON are kept
locally under the ignored directory
`results/hardware-calibration/2026-08-10-p310/`.

## Safety and integrity

The 14.731 GiB cache was copied from the P2 to the P310 without deleting the
source. All 243 relative paths, sizes and SHA-256 hashes matched. The three model
shards also matched the pinned hashes already documented by the project.

Both calibration tools are read-only. The storage harness uses a read-only
Win32 handle with software caching disabled; the CUDA harness does not load model
weights and bounds all allocations.
