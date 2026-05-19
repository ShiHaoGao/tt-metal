# Static Protocol Modeling Profiler

`static_protocol_modeling` 是参数化 attribution profiler，用来比较 TT-Metal CB FIFO 管理和 static L1 ring protocol，同时保留 unpacker/packer/LLK APIs 所需的 CB descriptor path。

## Workloads

- `tile-add`：每个输出 tile 做一次 BF16 tile add。
- `eltwise-chain`：在 pack 前重复 tile add，用 `--chain-depth` 增加 compute pressure。
- `matmul-single`：单核 matmul，`Mt`、`Nt`、`Kt` 可配置。
- `matmul-block`：多核 matmul，每个 core 负责不重叠的 output tile block；v1 不做 cross-core reduction。

## 模式

- `cb`：标准 CB FIFO API。
- `static-runtime`：显式 L1 ring 和 static counters，通过 runtime args 传入。
- `static-compiletime`：同样 protocol，但地址和计数 baked into kernel defines。
- `static-serialized`：负向控制项，破坏 pipeline overlap，用来证明收益不是简单来自少调用 CB API。
- `static-streamreg-cbregs`：同样 static ring protocol，但 queue counters 使用 per-logical-CB `tiles_received` / `tiles_acked` registers。
- `static-streamreg-cbregs-compiletime`：compile-time config ablation / upper bound，在 per-CB stream-register counters 上继续静态化 ring/layout/config。

旧 `static-streamreg` 在 compute-path workloads 中禁用。有效模型必须是 per logical CB：`c_0`、`c_1`、`c_16` 各自使用独立的 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`。

## 构建和 smoke

```bash
cmake --build build_Release --target static_protocol_modeling -j8
```

```bash
TT_METAL_CACHE=/tmp/static_protocol_modeling_smoke \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/static_protocol_modeling \
  --op=tile-add --mode=static-streamreg-cbregs --tiles=4 --num-slots=2 --repeats=1
```

```bash
TT_METAL_CACHE=/tmp/static_protocol_modeling_matmul \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/static_protocol_modeling \
  --op=matmul-single --mode=all \
  --matmul-m-tiles=2 --matmul-n-tiles=4 --matmul-k-tiles=2 \
  --num-slots=2 --repeats=1
```

## 常用 sweep

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/static_protocol_modeling/run_static_protocol_cases.py \
  --out-dir /tmp/static_protocol_modeling_memory_bound \
  --case-set memory-bound \
  --repeats=3
```

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/static_protocol_modeling/run_static_protocol_cases.py \
  --out-dir /tmp/static_protocol_modeling_device_cases \
  --repeats=3
```

per-case runner 会避免 TT-Metal profiler source-location hash collisions，并写出合并后的 host/device CSV。

## 输出和结论

- host 输出 `RESULT` CSV rows。
- device profiler zones 使用 `SPM_` 前缀。
- per-case runner 会生成 `combined_device_critical_summary.csv` 和 `combined_device_critical_comparison.csv`。
- 当前结果支持：memory-bound tile-add / eltwise-chain 有稳定收益；matmul model case 只适合作为 shape-specific attribution，不能 broad promote。
