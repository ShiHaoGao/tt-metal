# TTNN paged_update_cache Protocol Fork

这个目录是 `ttnn.experimental.paged_update_cache` 的真实 kernel fork，用来验证 `CB FIFO` 动态管理是否处在 decode KV-cache update 的关键路径上。

实现方式以 TTNN 原始代码为母版：

- host 侧复刻 `paged_update_cache_program_factory.cpp` 的 paged cache 几何、page table、`update_idxs_tensor`、per-user L1 sharded input。
- device 侧从原始 reader/writer/compute kernel 复制后加 profiler zone。
- `cb` 模式保留原始 CB FIFO 路径。
- `static-runtime` / `static-streamreg-cbregs` / `static-streamreg-cbregs-compiletime` 模式只替换 hot cache/intermediate/output 流水的 CB 动态 FIFO 管理，仍保留原始 byte-row update、untilize/tilize 和 paged address math。
- `static-streamreg-cbregs` 是 Level B 标准模式；8-user decode-like evidence 使用 `static-runtime` / `static-streamreg-cbregs`。`static-streamreg-cbregs-compiletime` 是 single-user compile-time ablation。
- fork 中绑定到 CB 的 per-core L1 ring 使用 top-down L1 分配，避免和普通静态 CB scratch 区重叠；这是为了保持独立 profiler harness 可运行，不改变 device kernel 的更新语义。
- 默认 core 选择使用二维 sub-grid，例如 8 users 默认 `4x2`、32 users 默认 `4x8`。实验中确认横向长 row（例如 `8x1`）会让当前 static fork 在多 core 下卡住，因此不作为有效 profiler 证据；需要复现该布局问题时可显式传 `--core-grid-x=8 --core-grid-y=1`。

当前第一版为了保证归因干净，只覆盖 BF16 cache/input，并限制 `head_dim <= 256`。更宽 head_dim 需要补齐 static untilize/tilize 的 block-splitting 路径后再纳入结论。

## 构建

```bash
cmake --build build_Release --target ttnn_paged_update_cache_protocol -j8
```

## 单 case smoke

设备运行需要 device 0 空闲，并且需要 HYBRID allocator：

```bash
TT_METAL_ALLOCATOR_MODE_HYBRID=1 \
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_CACHE=/tmp/ttnn_paged_update_cache_protocol_smoke \
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_paged_update_cache_protocol \
  --mode=static-streamreg-cbregs \
  --users=1 \
  --kv-heads=1 \
  --head-dim=32 \
  --block-size=32 \
  --max-seq-len=128 \
  --cache-idx=0 \
  --per-user-stride=0 \
  --repeats=1 \
  --device-id=0
```

## 批量 profiler

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_paged_update_cache_protocol/run_ttnn_paged_update_cache_protocol_cases.py \
  --users 8 \
  --kv-heads 8 \
  --head-dims 128 \
  --block-sizes 64 128 \
  --max-seq-lens 2048 \
  --cache-idxs 127 1057 \
  --repeats 2 \
  --out-dir /tmp/ttnn_paged_update_cache_protocol_u8_real_shapes
```

主要输出：

- `host_results.csv`
- `host_summary.csv`
- `zone_summary.csv`
- `critical_stage_summary.csv`
- `device_mode_comparison.csv`
- `host_mode_comparison.csv`

分析时优先看 `device_mode_comparison.csv` 的 critical stage 和 `delta_cycles_cb_minus_static`。如果 static 只在非 critical stage 变快，端到端收益可能不会出现。

## 当前实验结论

已验证的真实 TTNN-style case：

```text
users=8, kv_heads=8, head_dim=128, block_size=64/128,
max_seq_len=2048, cache_idx=127/1057, num_pages=2, repeats=2
```

所有 `cb`、`static-runtime`、`static-streamreg-cbregs` case 都通过正确性检查，`max_abs_error=0`。结果保存在：

```text
/tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18/runs/ttnn_paged_update_cache
```

Device critical-path 对比：

| Case | Static mode | CB critical | Static critical | Saved cycles | Speedup |
|---|---|---:|---:|---:|---:|
| `b64 idx127` | `static-runtime` | 12845 | 12461 | 384 | 1.031x |
| `b64 idx127` | `static-streamreg-cbregs` | 12845 | 12402 | 443 | 1.036x |
| `b64 idx1057` | `static-runtime` | 12840 | 12485 | 355 | 1.028x |
| `b64 idx1057` | `static-streamreg-cbregs` | 12840 | 12527 | 313 | 1.025x |
| `b128 idx127` | `static-runtime` | 12877 | 12413 | 464 | 1.037x |
| `b128 idx127` | `static-streamreg-cbregs` | 12877 | 12342 | 535 | 1.043x |
| `b128 idx1057` | `static-runtime` | 12778 | 12394 | 384 | 1.031x |
| `b128 idx1057` | `static-streamreg-cbregs` | 12778 | 12365 | 413 | 1.033x |

解读：

- 这是真实 TTNN C++ factory/kernel fork，不是 synthetic 模型；结论可以作为 KV-cache update family 的第一条 direct static evidence。
- `CB` 模式 critical stage 是 `compute-input-untilize`；static 模式 critical stage 变成 `compute-pack`，说明被移除的 CB FIFO 动态管理确实在原路径上影响了 critical path。
- 收益稳定但不大，本轮复跑约 `313-535 cycles`，median speedup 约 `1.032x`。这符合真实 TTNN op 中 page table、NoC read/write、untilize/tilize 和 writer overwrite 共同占比更高的预期。
- `static-streamreg-cbregs` 和 `static-runtime` 接近；它主要验证 per-CB stream-register ABI，不是主要收益来源。

当前限制：

- 32-user 真实 shape 的 static path 还会卡住，当前记录为 fork scalability bug，尚不能作为 32-user 性能结论。
- 横向长 row layout（例如 `8x1`）会让当前 static fork 卡住；默认 runner 已改成二维 sub-grid，复现该问题时显式传 `--core-grid-x=8 --core-grid-y=1`。
- `head_dim > 256` 尚未进入结论，因为 static untilize/tilize 的宽 Wt block-splitting 还没有补齐。
