# TTNN Binary NG SFPU No-Bcast Static Protocol Fork

这个 profiler fork 使用 TTNN `binary_ng` tiled no-broadcast SFPU `div` kernel shape，保留真实 TTNN 风格 reader/writer/compute ABI：

- reader source：`reader_interleaved_no_bcast.cpp`
- writer source：`writer_interleaved_no_bcast.cpp`
- compute source：`eltwise_binary_sfpu_no_bcast.cpp`

## 模式

- `cb`：保留正常 TTNN circular-buffer FIFO calls。
- `static-runtime`：保留 LLK/compute API 所需的 CB descriptor view，但绕过 `reserve/push/wait/pop`，改用 compiler-managed L1 rings 和 generation counters。
- `static-streamreg-cbregs`：保留同样 TTNN reader/writer/compute ABI 和 TensorAccessor shape，但 generation counters 使用 per-CB `tiles_received` / `tiles_acked` register path。

旧 `static-streamreg` 已禁用，不是有效 compute-path baseline。

## 构建和运行

```bash
cmake --build build_Release --target ttnn_binary_ng_sfpu_no_bcast_protocol -j8
```

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_ALLOCATOR_MODE_HYBRID=1 \
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_binary_ng_sfpu_no_bcast_protocol \
  --mode=all --tiles=4096 --num-pages=2 --repeats=3
```

sweep：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_binary_ng_sfpu_no_bcast_protocol/run_ttnn_binary_ng_sfpu_no_bcast_protocol_cases.py \
  --out-dir /tmp/ttnn_binary_ng_sfpu_no_bcast_protocol_cases \
  --tiles 1024 4096 16384 --num-pages 2 --repeats 3 \
  --modes cb static-runtime static-streamreg-cbregs
```

主要对比文件：

```text
device_mode_comparison.csv
critical_stage_summary.csv
zone_summary.csv
```

## 解读原则

这个 fork 用来回答：当 binary elementwise 从简单 FPU add 变成 SFPU-heavy `div` 后，CB FIFO 动态管理是否仍在 device critical path 上。

结论必须看 `device_mode_comparison.csv` 中的 critical stage 和 device cycles；host timing 只能作为辅助现象，不能用于宣称收益。
