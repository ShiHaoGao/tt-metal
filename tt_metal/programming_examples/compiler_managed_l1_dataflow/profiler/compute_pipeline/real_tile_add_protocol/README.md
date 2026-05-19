# Real Tile Add Protocol Profiler

`real_tile_add_protocol` 比较普通 CB-based add pipeline 和 static L1 ring / semaphore protocol。它是当前最强的正例，用来验证 memory-bound/simple elementwise 路径是否能从 static schedule 中受益。

## 模式

- `cb`：标准 `cb_wait_front` / `cb_reserve_back` / `cb_pop_front` / `cb_push_back`。
- `static-runtime`：显式 L1 ring 和 semaphore，通过 runtime args 传入。
- `static-compiletime`：同样的 static protocol，但地址和 semaphore id baked into kernel defines。
- `static-streamreg-cbregs`：payload movement、L1 ring、CB descriptor、pack/unpack setup、per-tile schedule 与 `static-runtime` 对齐；per-CB queue state 使用 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`，launch rendezvous 使用 idle stream register。
- `static-streamreg-cbregs-compiletime`：compile-time config ablation / upper bound，在 `static-streamreg-cbregs` 上继续把 ring/layout/config baked into kernel defines。

旧 `static-streamreg` compute 模式已禁用。它把一个 idle stream 当成多个 logical CB 的共享 scratch register，不是有效 compute-path 对比。

## 构建和运行

```bash
cmake --build build_Release --target real_tile_add_protocol -j8
```

```bash
TT_METAL_DEVICE_PROFILER=1 \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=all --tiles=1024 --num-pages=2 --repeats=5
```

稳定 per-case sweep：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/real_tile_add_protocol_cases \
  --tiles 256 1024 4096 \
  --num-pages 2 4 \
  --repeats 3
```

2x2 multi-core sweep：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/real_tile_add_protocol_multicore \
  --tiles 1024 4096 \
  --num-pages 2 \
  --repeats 3 \
  --modes cb static-runtime static-streamreg-cbregs \
  --core-grid-x 2 \
  --core-grid-y 2
```

Level B smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/real_tile_add_level_b_smoke \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=static-streamreg-cbregs --tiles=4 --num-pages=2 --repeats=1
```

## 当前结论

- `static-runtime` 相对 CB 有稳定大收益，约 `435-525 cycles/tile`，device critical path speedup 约 `1.66x-1.89x`。
- `static-streamreg-cbregs` 保留这部分收益，但相对 `static-runtime` 基本是 near parity；差异通常只有几个 cycles/tile。
- `static-streamreg-cbregs-compiletime` 用来隔离 per-CB counter backend 后剩余 runtime/config 成本；当前只作为 single-core upper-bound ablation。
- 因此 cbregs 的意义主要是验证 per-CB ABI 边界和 launch/control placement，不是新的 steady-state 大优化来源。

Profiler CSV：

```text
generated/profiler/.logs/profile_log_device.csv
```
