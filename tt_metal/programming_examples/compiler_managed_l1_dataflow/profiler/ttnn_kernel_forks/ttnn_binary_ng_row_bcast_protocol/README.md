# TTNN Binary NG Row-Bcast Static Protocol Fork

这个 profiler fork 使用 TTNN `binary_ng` tiled row-broadcast add kernel shape，保留真实 TTNN 风格 reader/writer/compute ABI。

当前版本是 **reader-side 软件 row-bcast add**：

- `SRC_BCAST=1`
- `BCAST_LLK=0`
- reader 在读入 LHS tile 后调用 `fill_tile_with_first_row_bfloat16`
- compute 仍使用普通 `add_tiles`

也就是说，它不是 LLK `unary_bcast<ROW>` 路径，也不是 SFPU-heavy 路径。之前尝试的 LLK/SFPU row-bcast compute path 会超时，尚未形成可用 profiler 证据。

- reader source：`reader_interleaved_row_bcast_protocol.cpp`
- writer source：`writer_interleaved_row_bcast_protocol.cpp`
- compute source：`eltwise_binary_row_bcast_protocol.cpp`

## 模式

- `cb`：保留正常 TTNN circular-buffer FIFO calls。
- `static-runtime`：保留 LLK/compute API 所需的 CB descriptor view，但绕过 `reserve/push/wait/pop`，改用 compiler-managed L1 rings 和 generation counters。
- `static-streamreg-cbregs`：保留同样 TTNN reader/writer/compute ABI 和 TensorAccessor shape，但 generation counters 使用 per-CB `tiles_received` / `tiles_acked` stream registers。

旧 `static-streamreg` 已禁用，不是有效 compute-path baseline。

## 构建和运行

```bash
cmake --build build_Release --target ttnn_binary_ng_row_bcast_protocol -j8
```

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_ALLOCATOR_MODE_HYBRID=1 \
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_binary_ng_row_bcast_protocol \
  --mode=all --op=add --tiles=4096 --width-tiles=8 --num-pages=2 --repeats=3
```

sweep：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_binary_ng_row_bcast_protocol/run_ttnn_binary_ng_row_bcast_protocol_cases.py \
  --out-dir /tmp/ttnn_binary_ng_row_bcast_protocol_cases \
  --ops add \
  --tiles 1024 4096 16384 --width-tiles 8 --num-pages 2 --repeats 3 \
  --modes cb static-runtime static-streamreg-cbregs
```

主要对比文件：

```text
device_mode_comparison.csv
```

## 当前结论

- 这是 TTNN-style binary row-broadcast add 正例：static protocol 在真实 reader/writer/compute ABI 中仍然能赢。
- 本轮 `/tmp/ttnn_binary_ng_row_bcast_protocol_cases` 的 device profiler 显示 critical stage 全部是 writer。
- `1024/4096/16384` tiles、`width_tiles=8`、`num_pages=2` 下，`static-runtime` 约节省 `936-941 cycles/local tile`，`static-streamreg-cbregs` 约节省 `938-943 cycles/local tile`，speedup 约 `1.132x-1.133x`。
- `static-streamreg-cbregs` 应解读为 per-CB ABI 验证和控制项，不是主要收益来源。
- 该结果不能推广到 SFPU-heavy；SFPU-heavy row-bcast 仍需要单独 fork 并检查 device profiler critical stage。
