# Async expert pipeline

## Status and goal

The development tree includes an opt-in MVP that pipelines expert movement and
execution inside one routed MoE layer. Its purpose is to keep the compute stream
from waiting for every storage read and RAM-to-VRAM copy in strict sequence.

The synchronous path remains the default. The async path is an experiment, not
a new performance claim and not yet a production runtime contract.

## Data path

For the active experts in a routed layer, the async path uses:

1. one background I/O worker to service expert loads from the read-only
   safetensors store;
2. a bounded set of CPU staging buffers, pinned for CUDA use;
3. one dedicated CUDA H2D stream for non-blocking RAM-to-VRAM copies;
4. CUDA readiness events before an expert is consumed on the compute stream;
5. per-slot last-use events so a cache slot is not reused while prior compute
   may still read it.

Requests for the same missing expert are coalesced. Queue and staging capacity
are bounded by configuration, and expert computation retains deterministic
layer-local ordering. The current design intentionally uses one I/O worker
because safetensors shard access is serialized and additional workers would not
yet demonstrate useful physical storage parallelism.

Admission is fail-fast when the bounded queue is full; it never waits while
holding the execution lock. This keeps `resolve()`, `wait_idle()` and `close()`
able to make progress under backpressure.

With at least two staging slots, one expert can be serviced or transferred while
the GPU computes an earlier expert. In this document, "storage overlap" means
overlap of the mmap/page-cache service observed by the process. The operating
system may satisfy a load from RAM, fault pages from the SSD, or perform a mix of
both.

```text
read-only safetensors mmap
          |
          v
one bounded I/O worker -> pinned staging slots -> CUDA H2D stream
                                                   |
                                             readiness event
                                                   |
                                                   v
                                             compute stream
                                                   |
                                             last-use event
```

## Enabling it

The full OLMoE benchmark keeps the v0.3.0 synchronous behavior unless async is
requested explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_real_routing.ps1 `
  -VenvPath .\.venv-real
& '.\.venv-real\Scripts\python.exe' .\scripts\benchmark_paged_olmoe.py `
  --snapshot <pinned-local-snapshot> `
  --output .\results\paged-async.json `
  --pipeline async `
  --staging-slots 2
```

`--pipeline async` requires `--staging-slots 2` or more. Two slots are the
smallest useful overlap configuration and the initial comparison point. The
default remains `--pipeline sync`, for which one staging slot remains valid.

At the Python API level, `ExpertSlotCache(..., pipeline_mode="async")` selects
the same path. CUDA async mode requires pinned staging memory.

CUDA async consumers must execute experts through `PagedExpertRuntime`. Direct
calls to `ExpertSlotCache.get()` or `ExpertSlotCache.resolve()` are
intentionally rejected in this mode: those APIs return raw tensor views without
a way for the caller to retain the underlying slot lease until GPU consumption
has completed. `PagedExpertRuntime` owns each lease across expert compute and
records its last-use event before allowing the slot to be reused. Raw cache
access remains available for the synchronous path and for supported CPU tests.

## Lifecycle

The cache owns its worker, queue, staging-slot ownership and CUDA transfer
stream. It does not own the external `SafetensorExpertStore`.

- Call `wait_idle()` before collecting final evidence or closing the store. It
  drains submitted storage and transfer work and surfaces an asynchronous
  failure to the caller.
- Call `close()` before closing the store. `close()` is idempotent: it drains
  outstanding work, stops and joins the worker, and finishes the transfer
  stream.
- A context manager or `try/finally` should preserve that order even if a model
  forward fails.

The benchmark handles this ordering. Embedders using the cache directly are
responsible for it.

Collect metric snapshots only at drained boundaries. Logical `misses` count
non-resident requests, including coalesced requests, while `storage_loads` and
`transfer_loads` count the actual completed operations. The benchmark validates
bytes against those operation counters rather than treating every miss as a
separate read. `demand_wait_seconds` currently measures host waiting for storage
completion only; it does not include slot waits or time that the CUDA compute
stream spends behind a readiness event. `pending_loads_peak` covers all pending
keys, including queued and serviced work, while `peak_staging_in_use` is the
separate bounded staging-occupancy metric.

## What this MVP does not prove

The current storage backend is mmap/page-cache backed. Therefore CUDA events,
Python timings and an async queue can show software pipeline behavior, but they
cannot distinguish a physical NVMe read from a page-cache hit. This MVP does
not establish:

- direct or unbuffered NVMe I/O;
- physical SSD queue-depth utilization;
- a persistent pinned-RAM expert cache;
- batched reads, cancellation or multiple storage workers;
- overlap across concurrent model forwards;
- a general latency or throughput improvement.

At least two GPU cache slots are needed to overlap H2D for one expert with
compute that still reads another slot. The MVP also has no I/O watchdog: a
storage call that never returns will keep `wait_idle()` and `close()` waiting.

Page faults, Python scheduling, PCIe contention, expert hit rate, amount of
expert compute and cache state can all change whether overlap is useful. More
RAM or a faster SSD may change those terms, but neither guarantees a speedup.

## Evidence required before a speedup claim

Compare sync and async with the same checkpoint revision, prompts, token limits,
cache policy, GPU-slot capacity and exact generated or teacher-forced token
identities. Record repeated cold-dynamic-cache and retained-cache runs, while
reporting OS page-cache state as uncontrolled unless it is independently
measured.

Treat end-to-end wall time as the decision metric. Storage, transfer and wait
metrics explain the result but must not be summed as though their intervals were
serial. A physical NVMe-overlap claim requires separate OS/device tracing or a
future direct-I/O backend in addition to runtime timings.

The harness refuses to start from a dirty Git tree and records both the full
source commit and the benchmark-script SHA-256 in every JSON result. Commit the
reviewed implementation before collecting evidence; ignored result files do not
make the tree dirty.

The current tests verify bounded scheduling capability, event/lease ownership
and sync-versus-async numerical parity. They do not measure intersecting H2D and
compute intervals on a common device timeline, so passing tests alone is not
evidence of temporal overlap. That claim needs CUDA-event interval measurement
or a profiler trace showing the copy and expert kernels executing concurrently.
