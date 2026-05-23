# Matmul Variants Profile

This profiler example is a self-contained C++ fork of the matmul programming
examples. It copies the relevant host and kernel sources under this directory,
then adds device profiler zones in the copied kernels:

- `MMVP_MULTI_CORE_{READER,COMPUTE,WRITER}`
- `MMVP_REUSE_{READER,COMPUTE,WRITER}`
- `MMVP_REUSE_MCAST_{READER,COMPUTE,WRITER}`
- `MMVP_SINGLE_CORE_{READER,COMPUTE,WRITER}` when `--include-single-core` is used

The original tutorial sources under `tt_metal/programming_examples/matmul/` are
not edited.

## Build

```bash
cmake --build build_Release --target matmul_variants_profile --parallel $(nproc)
```

## Run

Default multi-core sweep with device profiler:

```bash
rm -rf generated/profiler/.logs
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
./build_Release/programming_examples/profiler/matmul_variants_profile --warmup 1 --iters 3
```

Run a custom shape:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
./build_Release/programming_examples/profiler/matmul_variants_profile \
  --warmup 1 --iters 3 --shape 2304,128,64
```

Run one variant through all host/runtime timing boundaries:

```bash
./build_Release/programming_examples/profiler/matmul_variants_profile \
  --variant multi_core --mode all --warmup 1 --iters 3 \
  --shape 2304,128,64 --no-device-profiler-read
```

Use `--no-device-profiler-read` for a host-only smoke run. Use
`--include-single-core` only for small-shape/manual exploration; the default
sweep skips single-core because these shapes are intended for multi-core and
multicast comparisons.

Useful filters:

- `--mode eager`: current tutorial-style full call, rebuilding the program,
  buffers, kernels, semaphores, runtime args, host writes, enqueue, and blocking
  output read each measured iteration.
- `--mode prepared`: create the program, mesh workload, DRAM buffers, circular
  buffers, kernels, semaphores, and runtime args once per shape/variant. The
  measured loop only writes inputs, enqueues the already-prepared workload, and
  blocks on output readback.
- `--mode trace`: create the prepared workload once, compile/load/capture once,
  then measure input writes, mesh trace replay, and blocking output readback.
  Trace capture itself executes the workload, so the profiler buffer is drained
  after capture when device profiling is enabled.
- `--mode all`: run `eager`, `prepared`, then `trace`.
- `--variant multi_core|reuse|reuse_mcast|single_core`: restrict the sweep to
  one or more copied variants. `single_core` only supports `eager`.

## Timing Boundary

`MATMUL_PROFILE_RESULT` includes `mode=...` and times that mode's host boundary:

- `eager`: the full copied `matmul_*` host call. This includes program and
  buffer construction, `CreateKernel`, semaphore creation where applicable,
  runtime-arg setup, host writes, workload enqueue, and blocking output readback.
- `prepared`: only input writes, enqueue of a pre-built `MeshWorkload`, and
  blocking output readback.
- `trace`: only input writes, `replay_mesh_trace`, and blocking output readback.

CPU golden matmul, PCC checking, and untilization are outside all timed regions.

`MATMUL_PROFILE_STAGE_RESULT` is parsed from
`generated/profiler/.logs/profile_log_device.csv` after
`ReadMeshDeviceProfilerResults(*mesh_device)`. It reports the average, best, and
worst measured critical path for the copied reader/compute/writer zones. The
critical path is the max zone duration across cores for that stage in each
iteration. Stage rows also include `mode=...`, so device-stage comparisons do
not mix eager/prepared/trace iterations.

## Measured Results

Environment:

- Date: 2026-05-22
- Device: Blackhole P150A, chip 0
- Profiler CSV header: `ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120`
- Command: `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 ./build_Release/programming_examples/profiler/matmul_variants_profile --warmup 1 --iters 3`

Host wall-time, `mode=eager`:

| Shape | Variant | Avg ms | Best ms | Worst ms |
| --- | --- | ---: | ---: | ---: |
| 2304x128x64 | `matmul_multi_core` | 0.806 | 0.726 | 0.903 |
| 2304x128x64 | `matmul_multicore_reuse` | 0.351 | 0.334 | 0.381 |
| 2304x128x64 | `matmul_multicore_reuse_mcast` | 0.497 | 0.472 | 0.524 |
| 2400x128x64 | `matmul_multi_core` | 0.780 | 0.772 | 0.794 |
| 2400x128x64 | `matmul_multicore_reuse` | 0.403 | 0.341 | 0.446 |
| 2400x128x64 | `matmul_multicore_reuse_mcast` | 1.180 | 0.581 | 2.171 |
| 2112x192x64 | `matmul_multi_core` | 0.911 | 0.753 | 1.167 |
| 2112x192x64 | `matmul_multicore_reuse` | 0.397 | 0.355 | 0.438 |
| 1024x1024x64 | `matmul_multi_core` | 0.965 | 0.917 | 1.026 |
| 1024x1024x64 | `matmul_multicore_reuse` | 0.649 | 0.611 | 0.694 |
| 1024x2048x64 | `matmul_multi_core` | 1.315 | 1.228 | 1.360 |
| 1024x2048x64 | `matmul_multicore_reuse` | 0.945 | 0.898 | 0.980 |
| 1024x2048x64 | `matmul_multicore_reuse_mcast` | 1.194 | 1.103 | 1.266 |

Device stage critical path, average microseconds:

| Shape | Variant | Reader us | Compute us | Writer us |
| --- | --- | ---: | ---: | ---: |
| 2304x128x64 | `matmul_multi_core` | 13.023 | 13.034 | 13.408 |
| 2304x128x64 | `matmul_multicore_reuse` | 8.088 | 15.441 | 16.619 |
| 2304x128x64 | `matmul_multicore_reuse_mcast` | 6.660 | 14.013 | 14.786 |
| 2400x128x64 | `matmul_multi_core` | 13.047 | 13.060 | 13.393 |
| 2400x128x64 | `matmul_multicore_reuse` | 8.502 | 13.619 | 17.616 |
| 2400x128x64 | `matmul_multicore_reuse_mcast` | 5.969 | 11.120 | 12.720 |
| 2112x192x64 | `matmul_multi_core` | 16.995 | 17.008 | 17.383 |
| 2112x192x64 | `matmul_multicore_reuse` | 5.535 | 12.277 | 17.219 |
| 1024x1024x64 | `matmul_multi_core` | 40.027 | 40.055 | 40.411 |
| 1024x1024x64 | `matmul_multicore_reuse` | 12.926 | 25.935 | 38.396 |
| 1024x2048x64 | `matmul_multi_core` | 70.745 | 70.760 | 71.129 |
| 1024x2048x64 | `matmul_multicore_reuse` | 12.260 | 25.268 | 50.805 |
| 1024x2048x64 | `matmul_multicore_reuse_mcast` | 7.062 | 20.081 | 36.731 |

Skipped cases:

- `2112x192x64` `matmul_multicore_reuse_mcast`: mcast requires a 2D block/core range with more than one row and column.
- `1024x1024x64` `matmul_multicore_reuse_mcast`: mcast requires a 2D block/core range with more than one row and column.

## Layered Host Boundary Smoke

After adding `--mode`, host-only smoke runs were used to validate all three
boundaries:

```bash
./build_Release/programming_examples/profiler/matmul_variants_profile \
  --variant multi_core --mode all --warmup 1 --iters 2 \
  --shape 2304,128,64 --no-device-profiler-read

./build_Release/programming_examples/profiler/matmul_variants_profile \
  --variant reuse --variant reuse_mcast --mode all --warmup 1 --iters 2 \
  --shape 2304,128,64 --no-device-profiler-read
```

Results:

| Shape | Variant | Mode | Avg ms | Best ms | Worst ms |
| --- | --- | --- | ---: | ---: | ---: |
| 2304x128x64 | `matmul_multi_core` | `eager` | 2.934 | 2.934 | 2.934 |
| 2304x128x64 | `matmul_multi_core` | `prepared` | 0.125 | 0.118 | 0.133 |
| 2304x128x64 | `matmul_multi_core` | `trace` | 0.116 | 0.115 | 0.116 |
| 2304x128x64 | `matmul_multicore_reuse` | `eager` | 0.337 | 0.331 | 0.343 |
| 2304x128x64 | `matmul_multicore_reuse` | `prepared` | 0.121 | 0.114 | 0.129 |
| 2304x128x64 | `matmul_multicore_reuse` | `trace` | 0.104 | 0.102 | 0.106 |
| 2304x128x64 | `matmul_multicore_reuse_mcast` | `eager` | 0.445 | 0.429 | 0.461 |
| 2304x128x64 | `matmul_multicore_reuse_mcast` | `prepared` | 0.108 | 0.106 | 0.109 |
| 2304x128x64 | `matmul_multicore_reuse_mcast` | `trace` | 0.114 | 0.109 | 0.119 |

This smoke is not a final performance sweep; it confirms that moving
program/buffer/kernel construction out of the measured loop isolates a large
host-side component. On this small shape, `trace` removes little more than
`prepared` because host writes and the blocking output read remain inside both
timed regions.

Device-profiler smoke was also run for `matmul_multicore_reuse` with
`--iters 1` to verify that `mode=...` propagates into stage parsing:

| Shape | Variant | Mode | Host ms | Reader us | Compute us | Writer us |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2304x128x64 | `matmul_multicore_reuse` | `prepared` | 0.221 | 8.073 | 15.397 | 16.716 |
| 2304x128x64 | `matmul_multicore_reuse` | `trace` | 0.173 | 8.019 | 15.402 | 16.501 |

## Interpretation

`reuse_mcast` is not always the fastest by host wall-time in this copied
example harness. It often has a shorter device reader/compute/writer critical
path than `reuse`, but the complete host call still includes program/buffer
construction, input writes, enqueue, synchronization, and output readback. The
`2400x128x64` mcast wall-time also shows a large outlier, while its device
stage critical path is consistently shorter than the non-mcast reuse path.

So this example should be used to separate two questions:

- Device-kernel critical path: inspect `MATMUL_PROFILE_STAGE_RESULT`.
- End-to-end programming-example call cost: inspect `MATMUL_PROFILE_RESULT`.
