# Compiler-Managed L1 Dataflow Profilers

这个目录按实验层次组织 profiler。旧的扁平路径不再作为兼容入口。

| Profiler | 路径 | 作用 | 标准 stream-register 模式 |
|---|---|---|---|
| `cb_protocol_overhead` | `microbench/cb_protocol_overhead` | Phase 0/1 协议和存储 microbenchmark 基线。 | 只使用 microbench 自己的 streamreg 模式；不是 compute-path cbregs backend。 |
| `real_copy_protocol` | `data_movement/real_copy_protocol` | Phase 1/2 dataflow-only copy fork。 | `static-streamreg-scratch`；只有一个 logical queue，没有 compute operand CB ABI。 |
| `real_tile_add_protocol` | `compute_pipeline/real_tile_add_protocol` | 当前最强正例，memory-bound compute pipeline。 | `static-streamreg-cbregs`；每个 logical CB 独立使用 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`。 |
| `static_protocol_modeling` | `compute_pipeline/static_protocol_modeling` | tile-add、eltwise-chain、matmul model case 的参数化归因。 | `static-streamreg-cbregs`；旧 `static-streamreg` compute 模式已禁用。 |
| `ttnn_binary_ng_no_bcast_protocol` | `ttnn_kernel_forks/ttnn_binary_ng_no_bcast_protocol` | 保留 TTNN TensorAccessor 和 reader/writer/compute ABI 的 binary no-bcast fork。 | `static-streamreg-cbregs`；旧 `static-streamreg` compute 模式已禁用。 |
| `ttnn_bcast_to_protocol` | `ttnn_kernel_forks/ttnn_bcast_to_protocol` | 复制 TTNN `experimental.bcast_to` row-broadcast reader/compute/writer 后做最小协议替换。 | `static-streamreg-cbregs`；用于验证 broadcast/SFPU-adjacent 路径是否暴露 CB FIFO 动态管理。 |
| `ttnn_paged_update_cache_protocol` | `ttnn_kernel_forks/ttnn_paged_update_cache_protocol` | 直接复制 TTNN `paged_update_cache` program factory 和 kernels 后做最小协议替换，用真实 decode KV-cache update 验证 CB FIFO 是否在 critical path。 | `static-streamreg-cbregs`；与 `static-runtime` 对比的是 per-CB counter backend，不改变 paged cache 语义。 |
| `real_matmul_protocol` | `ttnn_kernel_forks/real_matmul_protocol` | TTNN 风格 matmul reuse 路径。 | `static-input-only-cbregs`、`static-output-only-cbregs`、`static-input-output-cbregs`。 |
| `ttnn_llm_decode_workloads` | `ttnn_workloads/llm_decode` | 真实 TTNN RMSNorm、Softmax decode、Paged KV update host baseline，用来筛下一批 direct fork。 | 不是 static-protocol backend；只输出 `ttnn-baseline`。 |

## 构建

```bash
conda run -n tt cmake --build build_Release \
  --target compiler_managed_l1_dataflow_examples -j8
```

## Correctness smoke

```bash
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_copy_protocol \
  --mode=static-streamreg-scratch --tiles=4 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=static-streamreg-cbregs --tiles=4 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/static_protocol_modeling \
  --op=tile-add --mode=static-streamreg-cbregs --tiles=4 --num-slots=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_binary_ng_no_bcast_protocol \
  --mode=static-streamreg-cbregs --tiles=4 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_bcast_to_protocol \
  --mode=all --tiles=64 --width-tiles=8 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_paged_update_cache_protocol \
  --mode=all --users=1 --kv-heads=1 --head-dim=32 --block-size=32 \
  --max-seq-len=128 --cache-idx=0 --per-user-stride=0 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=static-input-output-cbregs --M=512 --N=512 --K=64 --num-pages=2 --repeats=1
```

## 真实 TTNN workload baseline

先用真实 TTNN op 和真实 decode-like shape 找 direct fork 优先级：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_workloads/llm_decode/run_ttnn_llm_decode_workloads.py \
  --out-dir /tmp/ttnn_llm_decode_workload_run \
  --device-id 0 \
  --warmup 3 \
  --repeats 10
```

这个 runner 测的是 `op enqueue + ttnn.synchronize_device(device)`，不是 device critical path。它只能回答“哪个真实 workload 值得下一步 fork”，不能直接回答“CB FIFO 是否已经在 critical path”。
