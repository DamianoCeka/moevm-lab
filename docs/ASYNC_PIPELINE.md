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

Lookahead schedules only storage work. It does not update logical requests,
hit/miss counters or LRU recency. Those transitions occur when the runtime
actually consumes each expert, in the same order as the synchronous path; this
keeps cache traffic comparable in paired sync-versus-async measurements.

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

The experimental conservative selector uses the same bounded machinery:

```powershell
& '.\.venv-real\Scripts\python.exe' .\scripts\benchmark_paged_olmoe.py `
  --snapshot <pinned-local-snapshot> `
  --output .\results\paged-adaptive.json `
  --pipeline adaptive `
  --staging-slots 2
```

At the start of each routed-layer call, `adaptive` selects async only when at
least two active experts are not resident and at least one slot eligible for
those experts is empty. A call that starts with a full partition drains any
external pending work and executes synchronously. The decision is call-level:
an async-selected call may still fill its last free slot and evict later in the
same call. This conservative boundary follows the strongest signal in the RTX
6000 Ada study without pretending to be a complete cost model. Benchmark
metrics expose `adaptive_async_forwards`,
`adaptive_sync_forwards`, `adaptive_async_experts`, and
`adaptive_sync_experts` so every decision is auditable.

This selector is intentionally simple. It does not yet use learned workload
history, timing feedback, queue pressure, or a calibrated cost model.

### Measured `auto` profiles

The benchmark also accepts a fail-closed per-pass schedule built from at least
three comparable sync/async pairs. Build the profile with results from the same
GPU, model, workload and cache budget:

```powershell
& '.\.venv-real\Scripts\python.exe' .\scripts\build_paged_pipeline_profile.py `
  --pair .\results\r1\sync.json .\results\r1\async.json `
  --pair .\results\r2\sync.json .\results\r2\async.json `
  --pair .\results\r3\sync.json .\results\r3\async.json `
  --output .\results\pipeline-profile.json
```

Then run the same bound workload:

```powershell
& '.\.venv-real\Scripts\python.exe' .\scripts\benchmark_paged_olmoe.py `
  --snapshot <pinned-local-snapshot> `
  --output .\results\paged-auto.json `
  --pipeline auto `
  --pipeline-profile .\results\pipeline-profile.json `
  --staging-slots 2
```

The builder first requires exact token identity and identical cache, eviction,
storage and H2D primitives within every pair. It selects async for a pass only
when async wins every pair and the median paired `sync / async` ratio clears
the default 3% threshold. Mixed or smaller evidence selects sync. Cold and
immediate retained passes are decided independently.

The profile binds the selection to the exact GPU UUID and VRAM size, pinned
checkpoint hashes, workload/token budget, cache policy and capacity, Python
environment, benchmark-script hash and paged-runtime hash. Any mismatch aborts
before model allocation. The cache owns async infrastructure for the lifetime
of an auto run, drains all work outside the pass timers, and only then switches
the active data path. This avoids treating an old profile as transferable to a
different machine or implementation.

This is measured offline selection, not a universal online cost model. It does
not adapt inside a pass, extrapolate to unseen prompts, or replace multi-workload
validation.

At the Python API level, `ExpertSlotCache(..., pipeline_mode="async")` selects
the fixed path and `pipeline_mode="adaptive"` selects the conservative rule.
Both CUDA modes require pinned staging memory.

An async-capable cache may call `set_pipeline_mode()` at a drained pass
boundary. A cache constructed as sync-only cannot enable async later, and a
closed cache cannot be switched. The benchmark is the supported owner of this
mechanism for measured `auto` profiles.

Adaptive ticket scheduling is internal to `PagedExpertRuntime`. Public
`ExpertSlotCache.submit()` and `resolve()` remain fixed-async APIs and are
rejected in adaptive mode, preventing callers from mixing outstanding ticket
ownership with a direct synchronous `get()`.

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

The first accepted measurement is a three-pair, two-token smoke on one RTX
3080 Ti. Async used less wall time in all three pairs: the median paired
sync/async ratio is `1.274x` with an empty dynamic GPU expert cache and `1.398x`
on the immediate retained-cache repeat. The pair gate required identical token
IDs and per-scope logical cache/transfer counters; peak allocated VRAM was also
observed equal in these runs. See the
[visual comparison and sanitized evidence](../benchmarks/reference/paged-runtime-olmoe-p310-async-smoke/README.md).

A 36-pair follow-up on one RTX 6000 Ada adds five workloads, 2/8/16/32/64-token
conditions and LRU16/24/32/40 capacity points. All pair gates passed. In the
five-workload 16-token core, the median per-repetition aggregate ratio was
`1.208x` with an empty dynamic expert cache and `1.057x` on the immediate
retained repeat. Across the full matrix, async was faster in 33/36 empty-cache
comparisons but only 18/36 retained comparisons. Retained 32- and 64-token
`python_code` runs regressed to median ratios of `0.853x` and `0.869x`.

See the
[sanitized study, chart and exact evidence boundary](../benchmarks/reference/paged-runtime-olmoe-runpod-rtx6000ada-study/README.md).
This supports further pipeline work but also shows that always-on async is not
the right default. A conservative fill-aware adaptive rule is now implemented,
but it still needs full-model comparison against both fixed modes and common
CUDA-timeline evidence. Neither study is a production speedup claim.

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
