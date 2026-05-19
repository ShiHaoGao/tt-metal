# TTNN bcast_to Row-Bcast Protocol Fork

这个目录是 `ttnn.experimental.broadcast_to` row-broadcast 路径的真实 device-kernel fork。

已复制并最小改造的 TTNN 原始路径：

- host 形状和 runtime-args 语义来自 `ttnn/cpp/ttnn/operations/experimental/bcast_to/device/bcast_to_program_factory.cpp`。
- reader 源自 `reader_interleaved_row_bcast_to.cpp`。
- compute 源自 `compute_interleaved_row_bcast_to.cpp`，保留 `unary_bcast<BroadcastType::ROW>`。
- writer 源自 `writer_interleaved_row_bcast_to.cpp`。

本 fork 只比较数据流协议，不改变 row-broadcast 算子语义。输入 shape 等价于 `[1, 1, 1, width_tiles * 32]`，输出 shape 等价于 `[1, 1, height_tiles * 32, width_tiles * 32]`。

## 构建

```bash
conda run -n tt cmake --build build_Release --target ttnn_bcast_to_protocol -j8
```

## Correctness smoke

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/ttnn_bcast_to_protocol_smoke_cache \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_bcast_to_protocol \
  --mode=static-streamreg-cbregs --tiles=64 --width-tiles=8 --num-pages=2 --repeats=1 --device-id=0
```

已验证输出：

- `cb`、`static-runtime`、`static-streamreg-cbregs`、`static-streamreg-cbregs-compiletime` 均 `max_abs_error=0`。

## Device profiler sweep

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_bcast_to_protocol/run_ttnn_bcast_to_protocol_cases.py \
  --out-dir /tmp/ttnn_static_protocol_suite_bcast_execute_check/runs/ttnn_bcast_to_row \
  --tiles 1024 4096 16384 \
  --width-tiles 8 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb static-runtime static-streamreg-cbregs \
  --device-id 0
```

当前 Blackhole 单卡 sweep 结果：

| tiles | mode | CB critical stage | Static critical stage | delta cycles/tile | speedup |
|---:|---|---|---|---:|---:|
| 1024 | `static-runtime` | writer | writer | +5.03 | 1.0089 |
| 1024 | `static-streamreg-cbregs` | writer | writer | +11.44 | 1.0204 |
| 4096 | `static-runtime` | writer | writer | +1.04 | 1.0018 |
| 4096 | `static-streamreg-cbregs` | writer | writer | +10.85 | 1.0193 |
| 16384 | `static-runtime` | writer | writer | +0.27 | 1.0005 |
| 16384 | `static-streamreg-cbregs` | writer | writer | +2.74 | 1.0048 |

解读：row-broadcast 的 critical stage 仍在 writer，static protocol 有小幅 device-side 正收益，但 `static-runtime` 随 shape 变大很快摊薄到噪声区间。`static-streamreg-cbregs` 在 1024/4096 tiles 上更稳定，16384 tiles 上也明显摊薄。这个结果支持继续测 binary broadcast / SFPU-heavy eltwise，但不能单独推广为所有 broadcast 算子都稳定加速。
