# TTNN LLM Decode Workload Baseline

这个目录用于跑真实 TTNN decode-like workload，目标是筛选下一批直接 static-protocol fork 的优先级。

它不是 static protocol 性能证明。这里测的是：

```text
TTNN op enqueue + ttnn.synchronize_device(device)
```

因此结果只能说明某个真实 workload 的端到端 latency 是否值得继续拆 device profiler；不能直接说明 CB FIFO 动态管理已经在 critical path。

## 当前覆盖

| Workload | 默认真实形状 | 对应 family | 用途 |
|---|---|---|---|
| `rmsnorm` | `B=1, S=1/16, hidden=8192` | `normalization_softmax` | LLM decode/prefill RMSNorm 候选。 |
| `softmax-decode` | `B=1, heads=64, q=1, kv=128/1024/4096/8192` | `normalization_softmax` | Decode attention softmax 候选。 |
| `paged-update-cache` | `users=1/16, kv_heads=8, head_dim=128, block_size=32, max_seq=2048` | `embedding_kv_cache` | Paged KV cache update 候选。 |

## 只生成 workload matrix

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_workloads/llm_decode/run_ttnn_llm_decode_workloads.py \
  --dry-run \
  --out-dir /tmp/ttnn_llm_decode_workload_dryrun
```

## 执行真实 TTNN workload

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_workloads/llm_decode/run_ttnn_llm_decode_workloads.py \
  --out-dir /tmp/ttnn_llm_decode_workload_run \
  --device-id 0 \
  --warmup 3 \
  --repeats 10
```

## 输出

- `workload_matrix.csv`：本次要跑的 workload 和 shape。
- `raw_host_times.csv`：每次 repeat 的 host end-to-end latency。
- `host_summary.csv`：每个 case 的 median/mean/min/max/p10/p90。
- `llm_decode_workload_report.md`：中文摘要和下一步解读规则。

## 解读规则

- 如果某个 case 的 host latency 高，只说明它值得做 direct static fork，不说明 static protocol 一定有收益。
- RMSNorm/Softmax decode 下一步需要拆 reader/reduce/SFPU/write 的 device stage。
- Paged update cache 更接近 dataflow/write-heavy decode helper；若 token-step latency 暴露，应优先 fork。
- 直接 fork 仍然必须比较 `cb`、`static-runtime`、`static-streamreg-cbregs`；纯 dataflow helper 才使用 `static-streamreg-scratch`。
