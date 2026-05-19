# TTNN Binary NG No-Bcast Static Protocol Fork

这个 profiler fork 使用 TTNN `binary_ng` tiled no-broadcast add kernel shape，保留真实 TTNN 风格 reader/writer/compute ABI：

- reader source：`reader_interleaved_no_bcast.cpp`
- writer source：`writer_interleaved_no_bcast.cpp`
- compute source：`eltwise_binary_no_bcast.cpp`

## 模式

- `cb`：保留正常 TTNN circular-buffer FIFO calls。
- `static-runtime`：保留 LLK/compute API 所需的 CB descriptor view，但绕过 `reserve/push/wait/pop`，改用 compiler-managed L1 rings 和 generation counters。
- `static-streamreg-cbregs`：Level B 标准模式，保留同样 TTNN reader/writer/compute ABI 和 TensorAccessor shape，但 generation counters 使用 per-CB `tiles_received` / `tiles_acked` stream registers。
- `static-streamreg-cbregs-compiletime`：compile-time config ablation / upper bound，只把可 baked 的 queue/layout/config 字段转成 kernel defines；TTNN TensorAccessor/reader/writer/compute ABI 保持不变。

旧 `static-streamreg` 已禁用，不是有效 compute-path baseline。

## 构建和运行

```bash
cmake --build build_Release --target ttnn_binary_ng_no_bcast_protocol -j8
```

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_ALLOCATOR_MODE_HYBRID=1 \
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_binary_ng_no_bcast_protocol \
  --mode=all --tiles=4096 --num-pages=2 --repeats=3
```

sweep：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_binary_ng_no_bcast_protocol/run_ttnn_binary_ng_no_bcast_protocol_cases.py \
  --out-dir /tmp/ttnn_binary_ng_no_bcast_protocol_cases \
  --tiles 1024 4096 16384 --num-pages 2 --repeats 3 \
  --modes cb static-runtime static-streamreg-cbregs
```

主要对比文件：

```text
device_mode_comparison.csv
```

## 当前结论

- 这是 TTNN-style 正例对照：static protocol 在真实 reader/writer/compute ABI 中仍然能赢。
- 收益约 `18-23 cycles/local tile`，大约 `1.03x`，明显小于 standalone tile add。
- `static-streamreg-cbregs` 应解读为 per-CB ABI 验证和控制项，不是主要收益来源。
