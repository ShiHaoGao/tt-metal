# TTNN 算子横向 Static Protocol 实验总结

日期：2026-05-18

## 总结论

- 不能把 compiler-managed static ring/schedule 作为所有 TTNN 算子的统一加速结论；它只在 CB FIFO 动态管理处于 device critical path 或接近 critical path 时稳定收益。
- 已有 direct profiler 正例是 memory-bound/simple elementwise、TTNN row-broadcast 小扫和 paged KV update；已知负例/弱例是 dataflow-only L1-semaphore copy 和当前 matmul reuse path。
- `static-streamreg-cbregs` 是正确的 compute-path ABI 控制面：每个 logical CB 使用自己的 `tiles_received/tiles_acked` stream register。它不是主要收益来源；主要收益来自消除 CB FIFO 动态管理和静态化 schedule。
- 对还没有 direct static fork 的 family，当前结果只完成 sweep/pytest/TTNN baseline 覆盖，不能宣称 static speedup。

## 本轮复现实验

全类别 coverage 复现：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier coverage \
  --out-dir /tmp/ttnn_static_protocol_suite_coverage_repro_2026_05_18
```

结果：

- `operator_family_matrix.csv`：12 行，覆盖全部 family。
- `path_validation.csv`：114 行，覆盖 sweep/pytest 引用路径。
- `ttnn_llm_decode_norm_softmax`、`ttnn_llm_decode_paged_update_cache` dry-run 通过。

稳定 direct TTNN profiler 复现：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier core \
  --phases phase3 \
  --families eltwise embedding_kv_cache \
  --skip-build \
  --family-sweep-mode none \
  --pytest-mode none \
  --ttnn-workload-mode none \
  --repeats 3 \
  --out-dir /tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18
```

结果：

- `ttnn_binary_ng_no_bcast`：pass，46.400s。
- `ttnn_binary_ng_no_bcast_multicore`：pass，30.492s。
- `ttnn_paged_update_cache`：pass，61.223s。

归因报告生成：

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

补充说明：一次全量 Phase 3 复跑在 `real_matmul_low_k` 的 `profiled-cb` device profiler 启动后没有继续输出；最终 matmul 结论仍使用今天早些时候完成的同配置输出 `/tmp/real_matmul_protocol_ttnn_sweep_2026_05_18`。该现象记录为复现稳定性问题，不改变已有 matmul mixed 结论。

## Direct Profiler 证据

| 类别 | Direct case | 结果 | 根因 | 当前决策 |
|---|---|---|---|---|
| data_movement/layout | `real_copy_protocol` | `static-runtime` -115.52 cycles/work，`static-streamreg-scratch` -4.05 cycles/work | L1 counter/semaphore 同步状态存储成本超过 CB；scratch register 说明 payload 搬运不是主因 | 继续做 tilize/untilize/transpose direct fork，不能从 copy 推广正收益 |
| eltwise | `real_tile_add_protocol` | `static-runtime` +450.39 cycles/work，`static-streamreg-cbregs` +451.45 cycles/work | writer/queue path 暴露，static schedule 替代 CB FIFO 动态管理带来大收益 | 推广到 memory-bound/simple elementwise direct forks |
| eltwise/TTNN | `ttnn_binary_ng_no_bcast_protocol` | `static-runtime` +23.22 cycles/local-tile，speedup 1.0350 | 真实 TTNN reader/writer/compute ABI 中 writer 仍有协议成本，但被真实地址/NoC/compute 稀释 | 作为 TTNN-style 正例，下一步扩展 unary/bcast/SFPU-heavy |
| eltwise/broadcast TTNN | `ttnn_bcast_to_protocol` row-bcast | 1024/4096/16384 tiles：`static-runtime` +5.03/+1.04/+0.27 cycles/tile；`static-streamreg-cbregs` +11.44/+10.85/+2.74 cycles/tile | critical stage 仍是 writer；static protocol 小幅降低 writer critical cycles，但大 shape 的 runtime 收益接近噪声 | 作为 broadcast direct fork 的第一条证据，继续测 binary broadcast/SFPU-heavy，不能直接推广 |
| matmul/linear | `real_matmul_protocol` | 正负混合，部分 output-static shape 为正，部分 shape 回退 | reuse/compute/writeback critical path 掩盖 queue-only 收益 | 不 broad promote；只测 low-K/GEMV/multicast/decode-like |
| embedding/KV-cache | `ttnn_paged_update_cache_protocol` | 8-user decode-like saved 313.0 到 535.0 cycles，speedup median 1.0322 | CB critical stage 为 `compute-input-untilize`，static 后转到 `compute-pack`；CB FIFO 管理触到 critical path | shape-specific promote；32-user/cache-read/embedding lookup 未推广 |

## 全部 TTNN Family 当前状态

| Family | 当前证据 | 是否可宣称收益 | 结论 | 下一步 |
|---|---|---|---|---|
| `eltwise` | direct TTNN no-bcast、bcast_to row-bcast + real tile-add + sweep/pytest coverage | 是，限 memory-bound/simple/exposed broadcast | 稳定小到大收益，取决于真实 op 中协议成本占比 | fork unary、binary broadcast、SFPU-heavy chain |
| `data_movement_layout` | direct copy 负例 + sweep/pytest coverage | 否 | L1-semaphore static 不如 CB；streamreg scratch 只是证明同步存储成本 | fork tilize/untilize、transpose、slice、concat |
| `embedding_kv_cache` | direct paged_update_cache update + sweep/pytest coverage | 是，限 8-user paged update | 真实 TTNN KV update 有小幅稳定 device critical-path 收益 | 修 32-user static scalability，补 cache-read/embedding lookup |
| `matmul_linear` | direct low-K matmul + sweep/pytest coverage | 否，只有 shape-specific 候选 | 当前 reuse path 混合，不能推广 | GEMV/low-K/multicast/decode-like fork |
| `normalization_softmax` | sweep/pytest + TTNN workload baseline 计划 | 否 | 需要 direct RMSNorm/softmax fork 才能判断 CB 是否在 critical path | fork RMSNorm/LayerNorm、softmax decode |
| `reduction` | sweep/pytest coverage | 否 | 小 reduction 可能暴露协议，但尚未 profile | fork sum/mean/max，分 small/long/cross-core |
| `transformer_attention` | sweep/pytest coverage | 否 | prefill attention 不适合作为证明点；decode helper 更可能暴露 | fork SDPA decode、rotary、QKV split、concat heads |
| `ccl` | sweep generation 可跑；单卡 runner 无适用 vectors | 否 | 必须分离本地协议成本和 fabric/同步瓶颈 | 多设备 all-gather/reduce-scatter direct fork |
| `conv_pool` | sweep/pytest coverage | 否 | 作为 vision/CNN control lane；多数可能 bandwidth/compute dominated | 小/depthwise/pool baseline profile 后再 fork |
| `creation_fill_typecast` | sweep/pytest coverage | 否 | 主要用于 host/runtime 和 writer accounting | fill/typecast 小 shape direct fork |
| `backward_moreh_experimental` | sweep/pytest coverage | 否 | 不能聚合，需要按底层瓶颈拆分 | 先用 profiler 找 CB-heavy backward kernel |

## 为什么有收益

收益出现的条件是：CB FIFO 动态管理在 reader/writer/compute input/output 队列路径上暴露，并且这个队列路径处于 device critical path 或接近 critical path。

已验证正例：

- `real_tile_add_protocol`：writer 是 critical stage；static ring/schedule 避开 CB FIFO 动态管理后，每 work item 节省约 450 cycles。
- `ttnn_binary_ng_no_bcast_protocol`：仍然是 writer critical；真实 TTNN 地址、NoC 和 compute ABI 稀释收益，但仍稳定节省约 20-23 cycles/local-tile。
- `ttnn_bcast_to_protocol`：row-broadcast 的 writer 仍是 critical stage；1024/4096 tiles 上 `static-streamreg-cbregs` 小幅稳定正向，16384 tiles 上也明显摊薄，说明 broadcast 需要继续按 shape/family 测。
- `ttnn_paged_update_cache_protocol`：CB 模式 critical stage 是 `compute-input-untilize`，static 后 critical stage 转为 `compute-pack`；说明 CB FIFO 管理确实影响原 critical path。

## 为什么没有收益

无收益或回退通常有三类原因：

- 同步状态存储成本更高：`real_copy_protocol` 的 L1 counter/semaphore static-runtime 比 CB 更慢，说明静态协议如果把状态放在 L1 并频繁轮询，成本可能高于原 CB。
- critical path 不在 CB FIFO：matmul reuse 路径里 compute/reuse/writeback 成本占主导，局部 queue 静态化会被掩盖，甚至因 schedule/layout 改动回退。
- 真实 op 的其它成本稀释收益：paged KV update 里 page table、NoC read/write、untilize/tilize、writer overwrite 都仍存在，所以收益稳定但只有约 1.03x。

## 如何判断 CB FIFO 是否在 Critical Path

1. 必须用 device profiler 看 stage-level critical cycles，而不是只看 host enqueue/finish。
2. 如果 CB baseline 的最大 stage 是 reader/writer/compute input/output queue 附近，并且 static 后同一 shape 的 critical cycles 下降，才说明 CB FIFO 动态管理在 critical path 上。
3. 如果 static 只让非 critical stage 变快，或者 critical stage 仍是 compute math/reuse/NoC bandwidth，那么端到端不会稳定收益。
4. 如果收益随 local tile/user/page 数稳定缩放，可信度高；如果正负随 shape 翻转，必须标成 shape-specific。

## 下一步

1. `eltwise`：从 no-broadcast add 和 bcast_to row-broadcast 扩展到 unary、binary broadcast、SFPU-heavy chain。
2. `data_movement_layout`：复制真实 tilize/untilize、transpose、slice、concat C++/kernel 路径，保留 CB、L1-semaphore static、streamreg scratch 三条基线。
3. `embedding_kv_cache`：修 32-user static scalability，补 paged cache read 和 embedding lookup。
4. `normalization_softmax`：复制 TTNN RMSNorm/LayerNorm 和 softmax decode 路径，做 direct static fork。
5. `matmul_linear`：只追 low-K/GEMV-like/multicast/decode-like，不把 large prefill GEMM 当 proof point。
