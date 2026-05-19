# Compiler-Managed L1 Dataflow 实验入口

这个目录是 TTNN static-protocol / compiler-managed L1 dataflow ABI 实验的统一入口。

当前结论不是“直接移除 TT-Metal CB”，而是：**把 CB 从 compiler IR 的核心内存抽象，降级为可选 backend lowering target**。编译器侧应该表达显式 L1 allocation、view、queue/sync 和 compute operand；后端再选择降到 CB、pseudo-CB、per-CB stream-register 或 descriptor-only 方案。

独立的 pipeline warmup 实验不在这里，路径是：

```text
tt_metal/programming_examples/pipeline_warmup_experiments
```

## 目录结构

- `docs/`：当前研究问题、阶段状态、关键结论和下一步实验顺序。
- `profiler/`：CB overhead、copy、tile add、matmul、TTNN binary no-bcast 等直接 profiler fork。
- `profiler/ttnn_workloads/`：真实 TTNN workload baseline，用来筛下一批 direct static-protocol fork。
- `suite/`：阶段感知的 suite runner，以及当前 root-cause attribution 工具。

## 构建

```bash
conda run -n tt cmake --build build_Release \
  --target compiler_managed_l1_dataflow_examples -j8
```

单独构建主要 profiler：

```bash
conda run -n tt cmake --build build_Release \
  --target real_copy_protocol real_tile_add_protocol real_matmul_protocol \
           static_protocol_modeling ttnn_binary_ng_no_bcast_protocol \
           ttnn_bcast_to_protocol \
           ttnn_paged_update_cache_protocol -j8
```

可执行文件输出在：

```text
build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/
```

## 推荐入口

先看主结论文档：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/docs/ttnn_compiler_managed_l1_dataflow_abi_2026_05_18.md
```

TTNN 算子横向总结报告：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/docs/ttnn_operator_family_static_protocol_report_2026_05_18.md
```

跑覆盖计划：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier coverage --out-dir /tmp/ttnn_static_protocol_suite_coverage
```

生成根因归因报告：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/analyze_protocol_attribution.py \
  --real-copy-dir /tmp/real_copy_protocol_streamreg_single \
  --real-tile-add-dir /tmp/real_tile_add_protocol_cbregs_phase \
  --real-matmul-dir /tmp/real_matmul_protocol_ttnn_sweep_2026_05_18 \
  --ttnn-add-dir /tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18/runs/ttnn_binary_ng_no_bcast \
  --ttnn-bcast-to-row-dir /tmp/ttnn_bcast_to_protocol_smoke_profile \
  --ttnn-paged-update-cache-dir /tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18/runs/ttnn_paged_update_cache \
  --out-dir /tmp/compiler_managed_l1_attribution_final_2026_05_18
```

跑真实 TTNN LLM decode workload baseline，筛选下一批 direct fork：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier core \
  --phases phase3 \
  --families normalization_softmax embedding_kv_cache \
  --skip-build \
  --family-sweep-mode none \
  --pytest-mode none \
  --ttnn-workload-mode execute \
  --out-dir /tmp/ttnn_static_protocol_suite_llm_decode
```

当前 compute-path stream-register 控制项 smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rtadd_cbregs_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=static-streamreg-cbregs --tiles=4 --num-pages=2 --repeats=1 --device-id=0
```

## 当前结论

- stream-register 方向已经锁定为 per-CB ABI：每个 logical CB 使用自己的 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`。
- dataflow-only copy 可以用 `static-streamreg-scratch` 做 ablation，但不能把它推广成 compute-path ABI。
- 大收益来自 memory-bound / simple elementwise 路径上的 static ring/schedule 替代 CB FIFO 动态管理。
- `static-streamreg-cbregs` 验证 ABI 边界，但在当前 tile add 上相对 `static-runtime` 几乎没有新的 steady-state 收益。
- 真实 TTNN fork 应优先复制原始 C++ program factory 和 device kernels，再做最小协议替换；`ttnn_paged_update_cache_protocol` 已按这个路线验证。
- `ttnn_binary_ng_no_bcast` 复跑仍为正，`static-runtime` 约 `19.8-23.4 cycles/local-tile`，median speedup 约 `1.035x`。
- `ttnn_bcast_to_protocol` row-broadcast 已通过 device profiler；critical stage 是 writer。1024/4096/16384 tiles 上，`static-runtime` 约 `+5.03/+1.04/+0.27 cycles/tile`，`static-streamreg-cbregs` 约 `+11.44/+10.85/+2.74 cycles/tile`。这是 broadcast direct fork 的第一条证据，但还不能推广到所有 broadcast/SFPU-heavy 算子。
- `paged_update_cache` 的 8-user decode-like 真实形状上，static protocol 有稳定 device critical-path 收益，本轮复跑约 `313-535 cycles`、median speedup 约 `1.032x`；32-user static path 当前仍是 fork 的 scalability 待修项。
- matmul 仍然 mixed，不能 broad promote。
- 下一步已经落到真实 TTNN binary broadcast/SFPU-heavy、RMSNorm、Softmax decode、Paged KV read / embedding lookup、layout movement fork；baseline 只用于选择 direct fork，不用于直接宣称 static protocol speedup。
