# Pipeline Warmup Experiments

This directory contains the separate pipeline warmup experiments. They are not
part of the compiler-managed L1 dataflow ABI experiment package.

## Contents

- `add/`: tile-add host profiler and device kernels.
- `matmul/`: matmul host profiler and device kernels.
- `sdpa/`: TTNN chunked SDPA profiling harness for checking whether long
  sequence prefill has visible pipeline warmup/startup cost.

## Build

```bash
conda run -n tt cmake --build build_Release \
  --target pipeline_warmup_experiments -j8
```

Individual targets:

```bash
conda run -n tt cmake --build build_Release \
  --target pipeline_warmup_add pipeline_warmup_matmul pipeline_warmup_sdpa -j8
```

Executables are emitted under:

```text
build_Release/programming_examples/pipeline_warmup_experiments/
```

## SDPA warmup profiling

Run the smoke profile first. It uses a small chunked-SDPA shape and prints one
CSV row per measured prefill chunk:

```bash
build_Release/programming_examples/pipeline_warmup_experiments/pipeline_warmup_sdpa \
  --preset smoke --mode all --warmup 1
```

Run the long-sequence profile with device profiler collection:

```bash
TT_METAL_DEVICE_PROFILER=1 \
build_Release/programming_examples/pipeline_warmup_experiments/pipeline_warmup_sdpa \
  --preset llama2-70b --mode all --warmup 1
```

The C++ harness compares scalar `chunk_start_idx` against device-tensor
`chunk_start_idx`, and reports copy time, op-call time, sync time, program-cache
entries, program-factory creates, and runtime-argument overrides. Device
profiler output is read from `generated/profiler/.logs/profile_log_device.csv`
when `TT_METAL_DEVICE_PROFILER=1` is set.
