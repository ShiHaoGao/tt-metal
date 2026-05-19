# Compiler-Managed L1 Dataflow Profilers

这个目录按实验层次组织 profiler。旧的扁平路径不再作为兼容入口。

| Profiler | 路径 | 作用 | Level B / 标准 stream-register 模式 |
|---|---|---|---|
| `cb_protocol_overhead` | `microbench/cb_protocol_overhead` | Phase 0/1 协议和存储 microbenchmark 基线。 | 只使用 microbench 自己的 streamreg 模式；不是 compute-path cbregs backend。 |
| `real_copy_protocol` | `data_movement/real_copy_protocol` | Phase 1/2 dataflow-only copy fork。 | `static-streamreg-scratch-compiletime`；只有一个 logical queue，没有 compute operand CB ABI。 |
| `real_tile_add_protocol` | `compute_pipeline/real_tile_add_protocol` | 当前最强正例，memory-bound compute pipeline。 | Level B: `static-streamreg-cbregs`；compiletime 只作 upper-bound ablation。 |
| `static_protocol_modeling` | `compute_pipeline/static_protocol_modeling` | tile-add、eltwise-chain、matmul model case 的参数化归因。 | Level B: `static-streamreg-cbregs`；旧 `static-streamreg` compute 模式已禁用。 |
| `ttnn_binary_ng_no_bcast_protocol` | `ttnn_kernel_forks/ttnn_binary_ng_no_bcast_protocol` | 保留 TTNN TensorAccessor 和 reader/writer/compute ABI 的 binary no-bcast fork。 | Level B: `static-streamreg-cbregs`；runtime args 是多核默认路径。 |
| `ttnn_bcast_to_protocol` | `ttnn_kernel_forks/ttnn_bcast_to_protocol` | 复制 TTNN `experimental.bcast_to` row-broadcast reader/compute/writer 后做最小协议替换。 | Level B: `static-streamreg-cbregs`；收益 shape-dependent。 |
| `ttnn_paged_update_cache_protocol` | `ttnn_kernel_forks/ttnn_paged_update_cache_protocol` | 直接复制 TTNN `paged_update_cache` program factory 和 kernels 后做最小协议替换，用真实 decode KV-cache update 验证 CB FIFO 是否在 critical path。 | Level B: `static-streamreg-cbregs` on 8-user decode-like cases；single-user compiletime 是 ablation。 |
| `ttnn_transpose_wh_protocol` | `ttnn_kernel_forks/ttnn_transpose_wh_protocol` | 复制 TTNN tiled `transpose_wh` reader/compute/writer，验证 layout/transpose 路径上的 CB FIFO 动态管理是否暴露。 | `static-streamreg-cbregs`；optional layout evidence。 |
| `ttnn_embedding_lookup_protocol` | `ttnn_kernel_forks/ttnn_embedding_lookup_protocol` | 复制 TTNN embedding RM lookup 的 index->weight gather 语义，验证 lookup/read-heavy 路径上的 CB FIFO 动态管理是否暴露。 | `static-streamreg-cbregs`；当前 first-order 证据限制在 single-core BF16 row lookup。 |
| `ttnn_slice_tile_protocol` | `ttnn_kernel_forks/ttnn_slice_tile_protocol` | 收缩 TTNN tiled slice 的二维 tile 读写语义，验证 slice/layout writer path 上的 CB FIFO 动态管理是否暴露。 | `static-streamreg-cbregs`；当前是 single-core tiled slice 弱正例。 |
| `ttnn_kv_cache_load_slice_protocol` | `ttnn_kernel_forks/ttnn_kv_cache_load_slice_protocol` | 收缩 TTNN `nlp_kv_cache_load_slice` cache-read/load-slice 语义，验证 KV cache read path 上的 CB FIFO 动态管理是否暴露。 | `static-streamreg-cbregs`；当前是 single-core reader-critical 弱例/近噪声。 |
| `real_matmul_protocol` | `ttnn_kernel_forks/real_matmul_protocol` | TTNN 风格 matmul reuse 路径。 | `static-input-only-cbregs`、`static-output-only-cbregs`、`static-input-output-cbregs`；compiletime variants 是 matmul 专属 ablation。 |
| `ttnn_llm_decode_workloads` | `ttnn_workloads/llm_decode` | 真实 TTNN RMSNorm、Softmax decode、Paged KV update host baseline，用来筛下一批 direct fork。 | 不是 static-protocol backend；只输出 `ttnn-baseline`。 |

## 构建

```bash
conda run -n tt cmake --build build_Release \
  --target compiler_managed_l1_dataflow_examples -j8
```

## Correctness smoke

```bash
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_copy_protocol \
  --mode=static-streamreg-scratch-compiletime --tiles=4 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=static-streamreg-cbregs --tiles=4 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/static_protocol_modeling \
  --op=tile-add --mode=static-streamreg-cbregs --tiles=4 --num-slots=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_binary_ng_no_bcast_protocol \
  --mode=static-streamreg-cbregs --tiles=4 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_bcast_to_protocol \
  --mode=static-streamreg-cbregs --tiles=64 --width-tiles=8 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_paged_update_cache_protocol \
  --mode=static-streamreg-cbregs --users=1 --kv-heads=1 --head-dim=32 --block-size=32 \
  --max-seq-len=128 --cache-idx=0 --per-user-stride=0 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_transpose_wh_protocol \
  --mode=static-streamreg-cbregs --height-tiles=8 --width-tiles=8 --batches=1 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_embedding_lookup_protocol \
  --mode=static-streamreg-cbregs --rows=64 --vocab-size=1024 --dim=128 --num-pages=2 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_slice_tile_protocol \
  --mode=static-streamreg-cbregs --input-height-tiles=64 --input-width-tiles=64 \
  --output-height-tiles=16 --output-width-tiles=64 --start-height-tile=8 \
  --start-width-tile=0 --batches=1 --num-pages=2 --repeats=1

TT_METAL_ALLOCATOR_MODE_HYBRID=1 \
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kv_cache_load_slice_protocol \
  --mode=static-streamreg-cbregs --input-seq-tiles=128 --output-seq-tiles=32 \
  --head-dim-tiles=4 --start-seq-tile=0 --repeats=1

build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=static-input-output-cbregs --M=64 --N=64 --K=64 --num-pages=2 --repeats=1
```

Compile-time ablations remain available explicitly, for example:

```bash
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=static-streamreg-cbregs-compiletime --tiles=4 --num-pages=2 --repeats=1
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
