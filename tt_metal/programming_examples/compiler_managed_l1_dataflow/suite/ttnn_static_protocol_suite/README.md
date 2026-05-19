# TTNN Static Protocol Operator-Family Suite

这个 suite 是 static-protocol profiler forks 的 operator-family 伴随工具。

它做三件事：

1. 构建并运行现有 protocol profilers。
2. 生成 TTNN sweep / pytest entrypoints 的 family-level coverage matrix。
3. 把 protocol-bound 收益、compute-bound 行为和 bandwidth-bound 行为分开记录。

## 运行

只生成 coverage：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier coverage \
  --out-dir /tmp/ttnn_static_protocol_suite_coverage
```

Phase-limited smoke：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier smoke \
  --phases phase0 phase1 \
  --out-dir /tmp/ttnn_static_protocol_suite_phase01
```

只生成 smoke plan，不执行：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier smoke \
  --dry-run \
  --out-dir /tmp/ttnn_static_protocol_suite_smoke
```

在当前 workspace 的 TTNN Python 环境里跑 core Phase 3：

```bash
conda run -n tt python tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier core \
  --phases phase3 \
  --skip-build \
  --out-dir /tmp/ttnn_static_protocol_suite_phase3_core
```

`smoke` 默认不执行 sweep-framework 和 pytest，因为它们需要完整 TTNN Python 开发环境。若环境已配置，可以加：

```bash
--family-sweep-mode dry-run --pytest-mode collect-only
```

单卡系统上，CCL sweep 可能没有适用 vectors；这些会记录为 `skip`，不是 Phase 3 失败。

只跑真实 TTNN LLM decode workload baseline：

```bash
conda run -n tt python tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier core \
  --phases phase3 \
  --families normalization_softmax embedding_kv_cache \
  --skip-build \
  --family-sweep-mode none \
  --pytest-mode none \
  --ttnn-workload-mode execute \
  --out-dir /tmp/ttnn_static_protocol_suite_llm_decode
```

如果只想确认计划和 shape，不打开 device：

```bash
--ttnn-workload-mode dry-run
```

## 输出

- `operator_family_matrix.csv`：family coverage 和 conclusion policy。
- `phase_matrix.csv`：Phase 0-3 计划和 exit criteria。
- `family_decision_matrix.csv`：每个 family 的当前 decision 和 next experiment。
- `path_validation.csv`：引用的 sweep / pytest 文件是否存在。
- `task_plan.csv`：计划执行的命令。
- `task_results.csv`：执行或计划状态。
- `static_protocol_summary.csv`：可用时聚合 static-profiler CSV rows。
- `ttnn_workload_summary.csv`：真实 TTNN workload baseline 的 host timing 汇总；它不是 static-protocol speedup 证据。
- `suite_summary.md`：人读摘要。

## 根因归因报告

focused profiler sweeps 跑完后，用现有 CSV 输出生成中文 root-cause report：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/analyze_protocol_attribution.py \
  --real-copy-dir /tmp/real_copy_protocol_streamreg_single \
  --real-tile-add-dir /tmp/real_tile_add_protocol_cbregs_phase \
  --real-matmul-dir /tmp/real_matmul_protocol_ttnn_sweep_2026_05_18 \
  --ttnn-add-dir /tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18/runs/ttnn_binary_ng_no_bcast \
  --ttnn-bcast-to-row-dir /tmp/ttnn_bcast_to_protocol_smoke_profile \
  --ttnn-paged-update-cache-dir /tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18/runs/ttnn_paged_update_cache \
  --out-dir /tmp/compiler_managed_l1_attribution_final_2026_05_18
```

输出包括：

- `root_cause_report.md`：中文结论报告。
- `ttnn_operator_family_static_protocol_report.md`：TTNN family 横向总结、收益归因和复现方案。
- `attribution_summary.csv`：每个 case 的 device/host 方向和 root-cause class。
- `stage_delta_matrix.csv`：stage-level CB-vs-static deltas。
- `attribution_mode_stats.csv` 和 `stage_delta_stats.csv`：聚合 median。

单独复现 broadcast direct fork：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_bcast_to_protocol/run_ttnn_bcast_to_protocol_cases.py \
  --out-dir /tmp/ttnn_bcast_to_protocol_smoke_profile \
  --tiles 256 1024 \
  --width-tiles 8 \
  --num-pages 2 \
  --repeats 2 \
  --modes cb static-runtime static-streamreg-cbregs \
  --device-id 0
```

## 解读规则

- device critical path 以 protocol profilers 为准。
- sweep / pytest coverage 只说明 TTNN family surface 已覆盖，不代表该 family 已有 speedup claim。
- decode-like、prefill-like、multi-core 结果必须分开看。
- stream-register 模式必须严格区分：
  - dataflow-only copy/layout ablation 使用 `static-streamreg-scratch`。
  - compute-path 使用 `static-streamreg-cbregs`，每个 logical CB 独立使用 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`。
  - 一个 idle stream 作为多个 compute CB 的共享 scratch register 不是有效对比。
- TTNN workload baseline 只用来筛 direct fork 优先级；最终结论仍然必须回到 device critical-path stage delta。
