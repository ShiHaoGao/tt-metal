# TTNN Compiler-Managed L1 Dataflow ABI 研究计划 - 2026-05-18

这是当前 TTNN static-protocol / compiler-managed L1 dataflow ABI 实验的主结论文档。

核心结论：**研究方向成立，但要表述为 compiler / ABI 问题，而不是单个 runtime 优化或局部 CB micro-optimization**。目标不是消灭所有 queue / operand slot / L1 view 这类硬件必需概念，而是把 TT-Metal 的 `TensorAccessor`、`CircularBufferConfig`、`CreateCircularBuffer`、`cb_*` FIFO API、firmware `CBInterface` 初始化和手写 program factory 从 compiler path 中整体替换掉。新的研究对象是 **compiler-managed CB / L1Queue / OperandView**：CB-like 概念仍然存在，但由编译器静态规划和维护，而不是由 TT-Metal runtime CB 维护。

本文档现在是 `compiler_managed_l1_dataflow/docs` 下唯一主结论文档。原 `ttnn_operator_family_static_protocol_report_2026_05_18.md` 的横向算子报告、direct profiler 证据、收益/负例解释和下一步 family 优先级已经合并到本文档；后续不再维护第二份结论源。

## 当前仓库保护状态

当前实验分支是：

```text
exp/cb-runtime-vs-static-protocol
```

保护动作已经完成到本地：

- 已将当前全部 tracked / untracked 实验状态提交为 `WIP compiler-managed L1 dataflow experiments`。
- 已 `fetch upstream` 并成功 `rebase upstream/main`。
- 当前 HEAD 基于 `upstream/main` 之后。
- `git push -u origin exp/cb-runtime-vs-static-protocol` 被 SSH host key verification 拦住；这是远端 SSH trust 问题，不是代码或 rebase 冲突。

继续推送前需要先让本机信任 GitHub host key，例如：

```bash
ssh-keyscan github.com >> ~/.ssh/known_hosts
git push -u origin exp/cb-runtime-vs-static-protocol
```

## 研究问题

更准确的问题不是“是否删除所有 CB-like 概念”，而是：

> TT-Metal runtime-managed CB 是否应该继续作为 compiler IR、firmware launch ABI 和 op kernel 的核心抽象？还是应该被 compiler-managed CB / L1Queue / OperandView 替代，并把 TT-Metal CB 降级成兼容 backend 或 baseline？

目标 compiler IR 应避免直接表达 TT-Metal CB / TensorAccessor 语义，而表达：

- `l1.alloc`：compiler-owned L1 storage，包含 lifetime、size、layout、bank constraints。
- `l1.view`：带 tile/page/stride metadata 的 L1 view。
- `async.noc_read` / `async.noc_write`：显式数据搬运。
- `queue.acquire` / `queue.release`：producer/consumer 同步，由 compiler 选择 static schedule、stream register、L1 counter、semaphore 或 fallback CB。
- `compute.operand`：unpacker / packer / math 所需的 operand view descriptor，包含 format、tile shape、page size、base、stride 和可选 sync signal。
- `compute.unpack` / `compute.op` / `compute.pack`：compute engine 操作，不暴露 TT-Metal CB id 作为 IR 语义。

必须被替换的 TT-Metal 软件边界：

- Host API：`CreateCircularBuffer`、`CircularBufferConfig`、`TensorAccessorArgs`。
- Dispatch / launch descriptor：`local_cb_mask`、`local_cb_offset`、`remote_cb_offset`、`min_remote_cb_start_index`。
- Firmware：`CBInterface`、`setup_local_cb_read_write_interfaces`、`setup_remote_cb_interfaces` 的 TT-Metal CB 初始化路径。
- Kernel dataflow API：`cb_reserve_back`、`cb_wait_front`、`cb_push_back`、`cb_pop_front`、`get_read_ptr`、`get_write_ptr`、`get_tile_size`。
- Compute operand bridge：`pack_tile`、`copy_tile`、`binary_tiles_init`、`matmul_tiles_init` 等 API 中“CB id 就是 operand descriptor”的语义。

可以保留但必须重命名/重定义的硬件必要概念：

- operand slot / operand id：硬件仍需要 operand selector，但它应指向 compiler-owned `OperandView`，不是 TT-Metal CB。
- queue / pipe / ring：producer-consumer 仍需要容量和同步，但其状态由 compiler ABI 管理。
- L1 storage：仍需要 base、size、bank placement、alignment 和 lifetime。
- tile descriptor：仍需要 data format、tile shape、face layout、page size。

然后比较多个 backend：

- 现有 TT-Metal CB backend。
- compiler-managed CB on TT-Metal-compatible launch：kernel 不调用 `cb_*`，但 firmware launch 仍可临时提供兼容 descriptor。
- compiler-owned descriptor-table backend：host / dispatch / firmware 不再通过 TT-Metal `local_cb_mask` 初始化 CB，而加载 compiler descriptor table。
- LLK-compatible `OperandView` backend：继续复用 LLK 的 pack/unpack/math helper，但 LLK 读取的是 compiler-owned operand metadata。
- mini-LLK / raw Tensix instruction backend：直接生成 pack/unpack/math 指令或宏，只在 firmware ABI 和 `OperandView` 稳定后推进。

## 当前收益测量的目的

当前 profiler 不是为了证明“每个 op 都应该替换 CB 后更快”，也不是为了把局部 cycles savings 当成最终目标。它们服务于 compiler 研究的四个问题：

1. **确认哪些路径暴露了 runtime-managed CB 的成本。** 如果 memory-bound / simple elementwise 明显受益，说明这些路径适合由 compiler 静态规划 L1 queue、counter 和 schedule。
2. **定位 TT-Metal CB 的成本来自哪一层。** 需要区分 FIFO rd/wr pointer 更新、producer/consumer counter 存储、firmware launch 初始化、TensorAccessor 地址计算、NoC traffic、pack/unpack/math 本身。
3. **约束新的 ABI 设计。** 正例告诉我们 compiler-managed CB 至少要表达哪些字段；负例告诉我们哪些场景不能用单一 backend 硬替换。
4. **决定替换顺序。** 如果只测 kernel-level `cb_*` 替换，会漏掉 firmware `CBInterface` 和 launch descriptor 成本；如果一开始直接写 mini-LLK，又无法判断收益来自 CB 替换、LLK 重写还是 schedule 改变。

因此，当前探究收益的直接用途是：为 compiler-managed CB 的 ABI、firmware launch descriptor 和 backend selection policy 提供证据，而不是单纯追求某个 profiler 数字最大。

## 已合并的 TTNN 横向实验结论

### 本轮复现实验

全类别 coverage 复现：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier coverage \
  --out-dir /tmp/ttnn_static_protocol_suite_coverage_repro_2026_05_18
```

结果：

- `operator_family_matrix.csv`：12 行，覆盖全部 family。
- `path_validation.csv`：114 行，覆盖 sweep / pytest 引用路径。
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

### Direct profiler 证据

| 类别 | Direct case | 结果 | 根因 | 当前决策 |
|---|---|---|---|---|
| data_movement/layout | `real_copy_protocol` | `static-runtime` -115.52 cycles/work，`static-streamreg-scratch` -4.05 cycles/work | L1 counter/semaphore 同步状态存储成本超过 CB；scratch register 说明 payload 搬运不是主因 | 继续做 tilize/untilize/transpose direct fork，不能从 copy 推广正收益 |
| eltwise | `real_tile_add_protocol` | `static-runtime` +450.39 cycles/work，`static-streamreg-cbregs` +451.45 cycles/work | writer/queue path 暴露，static schedule 替代 CB FIFO 动态管理带来大收益 | 推广到 memory-bound/simple elementwise direct forks |
| eltwise/TTNN | `ttnn_binary_ng_no_bcast_protocol` | `static-runtime` +23.22 cycles/local-tile，speedup 1.0350 | 真实 TTNN reader/writer/compute ABI 中 writer 仍有协议成本，但被真实地址/NoC/compute 稀释 | 作为 TTNN-style 正例，下一步扩展 unary/bcast/SFPU-heavy |
| eltwise/broadcast TTNN | `ttnn_bcast_to_protocol` row-bcast | 1024/4096/16384 tiles：`static-runtime` +5.03/+1.04/+0.27 cycles/tile；`static-streamreg-cbregs` +11.44/+10.85/+2.74 cycles/tile | critical stage 仍是 writer；static protocol 小幅降低 writer critical cycles，但大 shape 的 runtime 收益接近噪声 | 继续测 binary broadcast/SFPU-heavy，不能直接推广 |
| matmul/linear | `real_matmul_protocol` | 正负混合，部分 output-static shape 为正，部分 shape 回退 | reuse/compute/writeback critical path 掩盖 queue-only 收益 | 不 broad promote；只测 low-K/GEMV/multicast/decode-like |
| embedding/KV-cache | `ttnn_paged_update_cache_protocol` | 8-user decode-like saved 313.0 到 535.0 cycles，speedup median 1.0322 | CB critical stage 为 `compute-input-untilize`，static 后转到 `compute-pack`；CB FIFO 管理触到 critical path | shape-specific promote；32-user/cache-read/embedding lookup 未推广 |

### 全部 TTNN family 当前状态

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

### 为什么有收益

收益出现的条件是：CB FIFO 动态管理在 reader/writer/compute input/output 队列路径上暴露，并且这个队列路径处于 device critical path 或接近 critical path。

已验证正例：

- `real_tile_add_protocol`：writer 是 critical stage；static ring/schedule 避开 CB FIFO 动态管理后，每 work item 节省约 450 cycles。
- `ttnn_binary_ng_no_bcast_protocol`：仍然是 writer critical；真实 TTNN 地址、NoC 和 compute ABI 稀释收益，但仍稳定节省约 20-23 cycles/local-tile。
- `ttnn_bcast_to_protocol`：row-broadcast 的 writer 仍是 critical stage；1024/4096 tiles 上 `static-streamreg-cbregs` 小幅稳定正向，16384 tiles 上明显摊薄，说明 broadcast 需要继续按 shape/family 测。
- `ttnn_paged_update_cache_protocol`：CB 模式 critical stage 是 `compute-input-untilize`，static 后 critical stage 转为 `compute-pack`；说明 CB FIFO 管理确实影响原 critical path。

### 为什么没有收益

无收益或回退通常有三类原因：

- 同步状态存储成本更高：`real_copy_protocol` 的 L1 counter/semaphore static-runtime 比 CB 更慢，说明静态协议如果把状态放在 L1 并频繁轮询，成本可能高于原 CB。
- critical path 不在 CB FIFO：matmul reuse 路径里 compute/reuse/writeback 成本占主导，局部 queue 静态化会被掩盖，甚至因 schedule/layout 改动回退。
- 真实 op 的其它成本稀释收益：paged KV update 里 page table、NoC read/write、untilize/tilize、writer overwrite 都仍存在，所以收益稳定但只有约 1.03x。

### 如何判断 CB FIFO 是否在 critical path

1. 必须用 device profiler 看 stage-level critical cycles，而不是只看 host enqueue/finish。
2. 如果 CB baseline 的最大 stage 是 reader/writer/compute input/output queue 附近，并且 static 后同一 shape 的 critical cycles 下降，才说明 CB FIFO 动态管理在 critical path 上。
3. 如果 static 只让非 critical stage 变快，或者 critical stage 仍是 compute math/reuse/NoC bandwidth，那么端到端不会稳定收益。
4. 如果收益随 local tile/user/page 数稳定缩放，可信度高；如果正负随 shape 翻转，必须标成 shape-specific。

## Stream-Register 方向已经锁定

compute-path 的正式规则如下：

- 不同 logical CB 不能共享一个 queue signal。
- compute-path stream-register backend 必须映射为 `CB i -> stream i`。
- ready/produced counter 使用 `get_cb_tiles_received_ptr(i)`。
- free/consumed counter 使用 `get_cb_tiles_acked_ptr(i)`。

二元 compute pipeline 的目标映射：

| Logical CB | Producer publishes | Consumer publishes |
|---|---|---|
| `c_0` input A | `get_cb_tiles_received_ptr(0)` | `get_cb_tiles_acked_ptr(0)` |
| `c_1` input B | `get_cb_tiles_received_ptr(1)` | `get_cb_tiles_acked_ptr(1)` |
| `c_16` output | `get_cb_tiles_received_ptr(16)` | `get_cb_tiles_acked_ptr(16)` |

重要边界：

- 现阶段不是 physical CB-less。CB descriptor 仍可用于 unpacker/packer/LLK operand setup。
- 旧方案“一个 idle stream 作为多个 compute CB 的共享 scratch register”已经被否定，不是研究方向。
- dataflow-only copy 可以继续使用 `static-streamreg-scratch` 做 ablation，因为它只有一个 logical queue，不涉及 compute CB ABI。
- shared start-gate register 只能作为 launch rendezvous，不能替代 per-CB produced/consumed counters。

命名约定：

- `static-runtime`：显式 L1 rings + L1 semaphore/counter storage。
- `static-compiletime`：同样 protocol，但地址和 shape 常量 baked into kernel defines。
- `static-streamreg-scratch`：dataflow-only scratch-register ablation。
- `static-streamreg-cbregs`：正式 compute-path stream-register backend，使用 per-CB `tiles_received` / `tiles_acked` registers。

## 执行计划

新的执行计划按 ABI 依赖顺序推进。关键调整是：**firmware / launch descriptor 替换必须早于 mini-LLK**。如果 firmware 仍通过 `local_cb_mask` 和 `CBInterface` 初始化 TT-Metal CB，那么 kernel 源码层面的 CB-less 只能算局部实验，不能声称已经绕过 TT-Metal CB 概念。

| Phase | 目标 | 关键产物 | 当前状态 |
|---|---|---|---|
| Phase 0: Evidence baseline | 保留 CB/runtime/static 基线，区分 protocol-bound 与非 protocol-bound family。 | `cb_protocol_overhead`、`real_copy_protocol`、`real_tile_add_protocol`、TTNN fork sweeps。 | 已足够指导后续替换顺序。 |
| Phase 1: Compiler IR and descriptor schema | 定义不含 TT-Metal CB/TensorAccessor 语义的 compiler IR。 | `CompilerCBDescriptor`、`OperandViewDescriptor`、`TensorLayoutDescriptor`、`QueueSyncDescriptor`。 | 需要从当前实验约束中固化字段。 |
| Phase 2: Host and launch descriptor replacement | 替换 `CreateCircularBuffer` / `CircularBufferConfig` / `TensorAccessorArgs` / `local_cb_mask`。 | compiler descriptor table、per-core L1 allocation table、launch descriptor extension 或 side-table。 | 尚未进入；应优先于 mini-LLK。 |
| Phase 3: Firmware descriptor loader | firmware 加载 compiler descriptor table，不再走 TT-Metal CB 初始化路径。 | `compiler_cb_interface` 或 `operand_view_table` loader；跳过 `setup_local_cb_read_write_interfaces` 的实验路径。 | 尚未进入；这是判断“是否真正绕过 TT-Metal CB”的边界。 |
| Phase 4: Kernel dataflow CompilerCB backend | kernel 不调用 `cb_*`，由 compiler 生成 queue acquire/release、NoC、L1 pointer、sync。 | `compiler_cb_wait/acquire/release` 或直接展开的 generated code。 | 现有 static protocol fork 是局部原型。 |
| Phase 5: LLK-compatible OperandView backend | 继续复用 LLK，但 LLK 看到的是 compiler-owned operand metadata，不是 TT-Metal CB state。 | `OperandView -> LLK operand setup` 兼容层。 | 尚未进入；这是从 CB descriptor 解耦 compute 的主路径。 |
| Phase 6: Operator-family migration | 按 family 迁移真实 op，建立 backend selection policy。 | elementwise、layout、KV cache、softmax、reduction、matmul/CCL 的 direct forks。 | Phase 3 coverage 已给出优先级。 |
| Phase 7: mini-LLK / raw Tensix backend | 对少数核心 op 绕过 LLK，直接生成 Tensix 指令/MOP/config。 | raw pack/unpack/math codegen；Tensix instruction schedule。 | 只应在 Phase 3/5 稳定后推进。 |
| Phase 8: Full compiler path | 高层 op 直接 lowering 到 compiler-managed L1 dataflow，不依赖 TTNN program factory。 | end-to-end compiler pipeline、validation suite、性能报告。 | 长期目标。 |

## 源码链路和阅读顺序

在开始修改前，先把当前 TT-Metal CB descriptor 的完整路径读清楚。目标不是先删除 CB，而是明确哪些字段由 host 生成、哪些由 dispatch 搬运、哪些由 firmware materialize 成每个 RISC 本地状态、哪些被 kernel/LLK 当作 operand metadata 使用。

### Host / dispatch descriptor 链路

关键路径：

```text
tt_metal/impl/program/program.cpp
tt_metal/impl/program/dispatch.cpp
tt_metal/hw/inc/hostdev/dev_msgs.h
tt_metal/api/tt-metalium/circular_buffer_constants.h
```

当前流程：

1. op / program factory 调用 `CreateCircularBuffer` 和 `CircularBufferConfig`。
2. `ProgramImpl` 在 kernel group 级别汇总每个 core 使用的 local / remote CB，生成 `local_cb_mask` 和 `min_remote_cb_start_index`。
3. `finalize_cbs` 计算 CB config 在 kernel config ring 里的 `local_cb_offset` / `remote_cb_offset` 和大小。
4. dispatch 为每个 local CB 写 4 个 word：`addr, size, num_pages, page_size`。
5. remote CB payload 只在 launch config 中放 `config_address, page_size`，真正 remote config 在 global circular buffer config 中。
6. launch message 通过 `kernel_config_msg_t` 把 offset、mask 和 kernel text/RTA/CRTA 信息交给 firmware。

这一层要先理解的关键点：

- `local_cb_mask` 不是 descriptor 本体，它只是告诉 firmware 哪些 CB slot 需要初始化。
- descriptor 本体在每个 core 的 kernel config L1 区中。
- 当前 descriptor schema 以 TT-Metal CB 为中心，compiler-owned descriptor table 需要从这里开始替换或 side-load。

### Firmware materialization 链路

关键路径：

```text
tt_metal/hw/inc/internal/circular_buffer_interface.h
tt_metal/hw/inc/internal/circular_buffer_init.h
tt_metal/hw/firmware/src/tt-1xx/brisc.cc
tt_metal/hw/firmware/src/tt-1xx/ncrisc.cc
tt_metal/hw/firmware/src/tt-1xx/trisc.cc
```

当前流程：

1. BRISC、NCRISC、TRISC firmware image 各自定义自己的 `CBInterface cb_interface[NUM_CIRCULAR_BUFFERS]`。
2. `CBInterface` 是 local / remote sender / remote receiver 三种 view 的 union。
3. firmware 每次 launch 从 `kernel_config_base + local_cb_offset` 找到 local CB payload。
4. `setup_local_cb_read_write_interfaces` 按 `local_cb_mask` 遍历 CB slot，把 4-word payload materialize 成本 RISC 本地 `LocalCBInterface`。
5. BRISC/NCRISC 一般按 read+write 初始化；TRISC 根据 `UCK_CHLKC_UNPACK` / `UCK_CHLKC_PACK` 只初始化读端或写端；MATH-only TRISC 不定义这份表。
6. remote CB 初始化通过 `setup_remote_cb_interfaces`，BRISC 还负责 remote setup barrier。

这一层要先理解的关键点：

- `CBInterface[]` 不是一个全局共享表，而是每个 RISC firmware image 的本地全局状态。
- kernel 代码和 LLK wrapper 通过 firmware symbol / local state 使用这张表。
- 只在 kernel 源码里不调用 `cb_*`，但 firmware 仍按 `local_cb_mask` 初始化 `CBInterface[]`，不能算真正绕过 TT-Metal CB descriptor。

### Kernel / LLK operand 链路

关键路径：

```text
tt_metal/hw/inc/api/dataflow/dataflow_api.h
tt_metal/hw/inc/api/compute/cb_api.h
tt_metal/hw/ckernels/blackhole/metal/llk_api
tt_metal/tt-llk/tt_llk_blackhole/llk_lib
tt_metal/jit_build/genfiles.cpp
```

当前流程：

1. dataflow kernel 用 `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` 管理 queue state。
2. compute API 和 LLK wrapper 用 `get_local_cb_interface(cbid)` 得到 `fifo_rd_ptr` / `fifo_wr_ptr` / `fifo_page_size`。
3. `jit_build/genfiles.cpp` 根据 CB config 生成 `unpack_src_format`、`unpack_dst_format`、`pack_dst_format`、`unpack_tile_size`、`pack_tile_size` 等数组。
4. 现有 experimental helper 中，底层 `_llk_unpack_A_custom_(address)` 和 `_llk_pack_block_contiguous_(tile_index, address, num_tiles)` 已经能吃 raw L1 address，但 public wrapper 仍把 `cbid` 转成地址。

这一层要先理解的关键点：

- Level 3 不只是“不调用 cb API”，还要让 format/tile/operand metadata 不再由 TT-Metal CB descriptor 派生。
- raw-address unpack/pack 是可行性证据，但现有 wrapper 不能直接作为 Level 3 证明。
- mini-LLK 之前必须先把 compiler-owned `OperandViewDescriptor` 打通，否则收益归因会混在 LLK 重写里。

### 现有实验目录定位

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler
tt_metal/programming_examples/compiler_managed_l1_dataflow/suite
```

当前已可复用的实验：

- `cb_protocol_overhead`：CB FIFO 和 static protocol microbenchmark。
- `real_copy_protocol`：data movement / copy 负例。
- `real_tile_add_protocol`：memory-bound tile pipeline 正例。
- `static_protocol_modeling`：建模不同 static protocol 变体。
- `ttnn_binary_ng_no_bcast_protocol`：真实 TTNN-style no-bcast elementwise 正例。
- `ttnn_bcast_to_protocol`：row broadcast 小幅正例。
- `ttnn_paged_update_cache_protocol`：真实 KV-cache update 正例。
- `real_matmul_protocol`：matmul reuse mixed/负例。

这些实验继续作为 Phase 0 baseline；下一步不再扩大 broad coverage，而是把这些证据转成 compiler descriptor 和 firmware launch ABI 原型。

### Phase 1: descriptor schema

最低需要三类 descriptor：

```text
TensorLayoutDescriptor:
  memory_space, base_addr, page_size, shard_map, bank_map, strides, shape, dtype

CompilerCBDescriptor:
  l1_base, total_size, slot_count, page_size, bank_policy,
  producer_thread, consumer_thread, sync_kind, produced_addr, consumed_addr,
  static_schedule_id, lifetime_start, lifetime_end

OperandViewDescriptor:
  operand_id, l1_base, tile_shape, data_format, page_size,
  read_stride, write_stride, pack_unpack_role, queue_ref
```

设计规则：

- IR 中不能出现 TT-Metal `CBIndex`、`TensorAccessorArgs` 或 `CircularBufferConfig`。
- `operand_id` 可以存在，但它是硬件 operand selector，不是 TT-Metal CB id。
- `queue_ref` 可以 lowering 到 compiler counter、stream register、static ordering 或 fallback TT-Metal CB。
- descriptor 必须能表达 dataflow-only、compute input、compute output、remote/multicast 四类路径。

### Phase 2: host / dispatch 替换

目标是在新 compiler path 中不调用：

- `CreateCircularBuffer`
- `CircularBufferConfig`
- `TensorAccessorArgs`
- TT-Metal program factory 中手写的 CB allocation logic

替代方案：

- compiler 先完成 per-core L1 allocation，并生成 `CompilerCBDescriptor[]`。
- compiler 生成 tensor layout / bank mapping，不再让 kernel 用 `TensorAccessor` 动态解释布局。
- dispatch 只负责把 descriptor table 放到每个 core 可见的 L1/config 区域。
- launch descriptor 中增加或 side-load `compiler_descriptor_offset`、`compiler_descriptor_count`、`operand_view_offset`。

验收标准：

- 一个 unary/binary elementwise op 的 host path 不再创建 TT-Metal CB。
- `local_cb_mask` 可以为 0，或者只用于 legacy fallback。
- kernel 仍能通过 compiler descriptor 完成读、算、写。

### Phase 3: firmware descriptor loader

这是优先级必须提前的阶段。当前 TT-Metal firmware 在 BRISC/TRISC/NCRISC/ERISC 中都存在 `CBInterface` 和 CB 初始化路径；如果不替换这里，compiler path 仍然依赖 TT-Metal CB 概念。

目标：

- 新增 experimental launch mode，例如 `DISPATCH_MODE_COMPILER_L1`。
- firmware 检测该 mode 后跳过 `setup_local_cb_read_write_interfaces` / `setup_remote_cb_interfaces`。
- firmware 加载 compiler descriptor table，初始化 compiler-owned operand view / queue state。
- 对 legacy TT-Metal program 保持原路径，不破坏现有 op。

最小实验：

1. 单 core unary/binary elementwise，不使用 remote CB。
2. `local_cb_mask=0` 或 descriptor-only sentinel。
3. firmware 只初始化 `OperandViewDescriptor` 所需 metadata。
4. kernel 通过 compiler descriptor 获取 L1 base/page size/format/sync。

验收标准：

- firmware profiler 中不出现 TT-Metal local/remote CB init zone，或该 zone 在 compiler mode 下为空。
- kernel correctness 通过。
- device critical path 可与 legacy CB、kernel-only static protocol 对比。

### Phase 4: kernel CompilerCB backend

目标是让 generated kernel 不出现 TT-Metal dataflow API：

- 不调用 `cb_reserve_back` / `cb_wait_front` / `cb_push_back` / `cb_pop_front`。
- 不调用 `get_read_ptr` / `get_write_ptr` / `get_tile_size` 获取 TT-Metal CB state。
- reader/writer/compute 使用 compiler descriptor 中的 L1 地址、page size 和 sync state。

backend 变体：

- `compiler-static-schedule`：producer/consumer 次序完全静态，适合小 pipeline / known shape。
- `compiler-streamreg`：每个 compiler queue 使用独立 stream register/counter。
- `compiler-l1-counter`：L1 counter/semaphore fallback。
- `legacy-cb`：只作为 baseline/fallback，不是 compiler IR 语义。

验收标准：

- `real_tile_add_protocol` 风格 workload 保持当前 static protocol 的正向收益。
- `real_copy_protocol` 负例不被隐藏；必须继续记录 l1-counter 负面结果。
- TTNN binary no-bcast 风格 workload 能穿过真实 reader/writer/compute ABI。

### Phase 5: LLK-compatible OperandView

这一阶段不急于绕过 LLK，而是先把 LLK 和 TT-Metal CB descriptor 解耦。

目标：

- `pack_tile`、`copy_tile`、`binary_tiles_init`、`matmul_tiles_init` 的 operand metadata 来自 `OperandViewDescriptor`。
- LLK 可以继续负责架构相关 pack/unpack/math 配置。
- TT-Metal `CBInterface` 不再是 LLK operand metadata 的唯一来源。

原因：

- LLK 隐含大量硬件细节：tile format、unpacker/packer config、addrmod、MOP、thread hazard、架构差异。
- 直接绕过 LLK 会把“CB 替换收益”和“LLK 重写收益/风险”混在一起。
- 先做 LLK-compatible backend 可以证明 compiler-managed CB ABI 是否足够表达真实 op。

### Phase 7: mini-LLK / raw Tensix backend

这一阶段放在 firmware 和 `OperandView` 稳定之后。

目标：

- 对少数核心 op 直接生成 Tensix instruction / MOP / config sequence。
- 优先选择 unary/binary elementwise、copy/layout、低 K matmul 或 GEMV-like microkernel。
- 不以完整 TTNN op 覆盖为第一目标。

拒绝条件：

- 如果 Phase 3/5 尚未稳定，不开始全量 mini-LLK。
- 如果一个 op 的性能差异无法区分来自 schedule、firmware launch、operand descriptor 或 raw instruction codegen，不作为 thesis proof point。

## 当前根因归因结论

归因工具：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/analyze_protocol_attribution.py \
  --real-copy-dir /tmp/real_copy_protocol_streamreg_single \
  --real-tile-add-dir /tmp/real_tile_add_protocol_cbregs_phase \
  --real-matmul-dir /tmp/real_matmul_protocol_ttnn_sweep_2026_05_18 \
  --ttnn-add-dir /tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18/runs/ttnn_binary_ng_no_bcast \
  --ttnn-paged-update-cache-dir /tmp/ttnn_static_protocol_suite_direct_eltwise_kv_repro_2026_05_18/runs/ttnn_paged_update_cache \
  --out-dir /tmp/compiler_managed_l1_attribution_final_2026_05_18
```

归因摘要：

| Case | 结论 | 关键数字 |
|---|---|---|
| `real_copy_protocol` | L1 semaphore static runtime 是负例；scratch-register sync 几乎追平 CB。主因更像同步状态存储/协议成本，不是 payload movement。 | `static-runtime` median `-115.52 cycles/work`；`static-compiletime` `-65.01`；`static-streamreg-scratch` `-4.05`。 |
| `real_tile_add_protocol` | static schedule 替代 CB FIFO 有大收益；cbregs 保留收益但相对 runtime 只有极小差异。 | `static-runtime` median `+450.39 cycles/work`；`static-streamreg-cbregs` `+451.45`。 |
| `real_matmul_protocol` | mixed / shape-specific；不能 broad promote。 | 本轮 sweep `4` 个正向、`5` 个负向。 |
| `ttnn_binary_ng_no_bcast_protocol` | TTNN-style 正例对照；static protocol 能穿过真实 TTNN 风格 ABI，但收益远小于 standalone tile add。 | 本轮复跑 `static-runtime` median `+23.22 cycles/work`；`static-streamreg-cbregs` `+21.40`。 |
| `ttnn_paged_update_cache_protocol` | 真实 `paged_update_cache` C++ factory/kernel fork 正例；KV-cache update 中 CB FIFO 动态管理会影响 device critical path，但收益幅度是 TTNN-style 小幅稳定收益。 | 本轮复跑 8-user decode-like 真实形状 `+313` 到 `+535 cycles`，median speedup 约 `1.032x`；32-user static path 仍是 fork scalability 待修项。 |

结论一句话：**stream-register 不是万能收益来源；per-CB stream-register 是正确 ABI 控制面，但真正的大收益来自 static ring/schedule 消除 CB FIFO 动态管理，并且只在 protocol cost 暴露的路径上明显。**

## Phase 0/1 Microbenchmark

Phase 0/1 microbenchmark 已通过 phase-aware suite：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier smoke \
  --phases phase0 phase1 \
  --out-dir /tmp/ttnn_static_protocol_suite_phase01_smoke
```

Phase 0：system-level CB vs static protocol。

| Scenario | Pair max cycles/page | Net vs empty | CB - static-runtime | CB - static-compiletime |
|---|---:|---:|---:|---:|
| `system-cb` | 43.94 | 39.90 | 8.46 | 9.55 |
| `system-static-runtime` | 35.47 | 31.43 |  |  |
| `system-static-compiletime` | 34.38 | 30.34 |  |  |

Phase 1：显式 L1 ring + semaphore 的 DRAM traffic。

| Scenario | Pair max cycles/page | Net vs empty | CB - static-runtime | CB - static-compiletime |
|---|---:|---:|---:|---:|
| `dram-cb` | 550.61 | 546.57 | 14.71 | 29.47 |
| `dram-static-runtime` | 535.90 | 531.86 |  |  |
| `dram-static-compiletime` | 521.14 | 517.10 |  |  |

解释：

- CB overhead 可测，但在保留 launch 和可靠 semaphore synchronization 后，microbenchmark 上幅度不大。
- compile-time static variant 比 runtime-address variant 再省一点。
- 这只验证 Phase 0/1 边界，不能直接推广为 TTNN op speedup。

## Production-Shaped Copy 负例

`real_copy_protocol` 路径：

```text
DRAM -> reader -> L1 staging ring -> writer -> DRAM
```

关键结果：

| Tiles | Pages | Grid | Static mode | CB critical | Static critical | Saved cycles/local tile | Speedup |
|---:|---:|:---:|---|---:|---:|---:|---:|
| 256 | 2 | 1x1 | static-runtime | 143023 | 171825 | -112.5 | 0.832x |
| 1024 | 2 | 1x1 | static-runtime | 571427 | 691448 | -117.2 | 0.826x |
| 4096 | 2 | 1x1 | static-runtime | 2284966 | 2769279 | -118.2 | 0.825x |
| 16384 | 2 | 1x1 | static-runtime | 9135148 | 11129001 | -121.7 | 0.821x |
| 256 | 2 | 1x1 | static-compiletime | 143023 | 160326 | -67.6 | 0.892x |
| 1024 | 2 | 1x1 | static-compiletime | 571427 | 639651 | -66.6 | 0.893x |

stream-register scratch follow-up：

| Tiles | Pages | Grid | Static mode | CB critical | Static critical | Saved cycles/local tile | Speedup |
|---:|---:|:---:|---|---:|---:|---:|---:|
| 256 | 2 | 1x1 | static-streamreg-scratch | 143421 | 144745 | -5.2 | 0.991x |
| 1024 | 2 | 1x1 | static-streamreg-scratch | 573299 | 576306 | -2.9 | 0.995x |
| 4096 | 2 | 1x1 | static-streamreg-scratch | 2294333 | 2303130 | -2.1 | 0.996x |
| 16384 | 2 | 1x1 | static-streamreg-scratch | 9184229 | 9336224 | -9.3 | 0.984x |
| 4096 | 2 | 2x2 | static-streamreg-scratch | 575795 | 576461 | -0.7 | 0.999x |

解释：

- pure copy 不适合作为 compiler-managed L1 的证明点。
- L1 semaphore static runtime 明显慢于 CB；compile-time 地址常量只能缓解一部分。
- scratch-register sync 把差距缩小到近似 noise floor，说明同步状态存储/协议成本是核心负例原因。
- 下一步 data movement/layout 应看 tilize/untilize、transpose、slice、concat 这类可能受 lifetime/layout freedom 影响的路径，而不是继续只看 pure copy。

## Memory-Bound Tile Pipeline 正例

`real_tile_add_protocol` 是当前最强正例：

| Tiles | Pages | Static mode | CB critical | Static critical | Saved cycles/tile | Speedup |
|---:|---:|---|---:|---:|---:|---:|
| 256 | 2 | static-runtime | 284317 | 160107 | 485.2 | 1.776x |
| 256 | 2 | static-compiletime | 284317 | 150177 | 524.0 | 1.893x |
| 1024 | 2 | static-runtime | 1136286 | 672711 | 452.7 | 1.689x |
| 4096 | 2 | static-runtime | 4541710 | 2691857 | 451.6 | 1.687x |
| 4096 | 4 | static-runtime | 4480498 | 2697757 | 435.2 | 1.661x |

多核趋势也稳定：

| Grid | Active cores | Tiles | Max tiles/core | CB critical | Static runtime | Saved cycles/local tile | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x2 | 2 | 1024 | 512 | 567748 | 338640 | 447.5 | 1.677x |
| 2x2 | 4 | 1024 | 256 | 284042 | 168631 | 450.8 | 1.684x |
| 2x2 | 4 | 4096 | 1024 | 1132973 | 680650 | 441.7 | 1.665x |

解释：

- memory-bound / simple elementwise pipeline 可以稳定从 static schedule 中受益。
- `static-serialized` 负向控制项更慢，说明收益依赖 producer/consumer overlap，不是简单少调用几个 API。
- 这是后续 elementwise 扩展的主要正例依据。

## Compute-Path `static-streamreg-cbregs`

`static-streamreg-cbregs` 已在 `real_tile_add_protocol` 实现并通过 1x1 / 2x2 sweeps。

与 CB 对比：

| Tiles | Pages | Static mode | CB critical | Static critical | Saved cycles/tile vs CB | Speedup vs CB |
|---:|---:|---|---:|---:|---:|---:|
| 256 | 2 | static-runtime | 284283 | 159851 | 486.1 | 1.778x |
| 256 | 2 | static-streamreg-cbregs | 284283 | 158718 | 490.5 | 1.791x |
| 1024 | 2 | static-runtime | 1135581 | 673778 | 451.0 | 1.685x |
| 1024 | 2 | static-streamreg-cbregs | 1135581 | 672849 | 451.9 | 1.688x |
| 4096 | 2 | static-runtime | 4541048 | 2698686 | 449.8 | 1.683x |
| 4096 | 2 | static-streamreg-cbregs | 4541048 | 2693691 | 451.0 | 1.686x |

直接与 `static-runtime` 对比：

| Tiles | Pages | Runtime critical | Cbregs critical | Runtime - cbregs | Cycles/tile | Cbregs speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2 | 159851 | 158718 | 1133 | 4.43 | 1.0071x |
| 1024 | 2 | 673778 | 672849 | 929 | 0.91 | 1.0014x |
| 4096 | 2 | 2698686 | 2693691 | 4995 | 1.22 | 1.0019x |
| 4096 | 4 | 2698391 | 2698277 | 114 | 0.03 | 1.0000x |

解释：

- cbregs 保留 static 相对 CB 的大收益。
- cbregs 相对 `static-runtime` 没有新的大收益，因为 steady-state queue counters 在 static runtime 路径里已经等价地使用 per-CB stream-register counters。
- `static-streamreg-cbregs` 主要验证 ABI boundary；不能宣传成“把 compute sync 移到 stream regs 就会自动大幅变快”。

## TTNN Binary No-Broadcast Add

这是 TTNN-kernel-level 正例对照，保留 TTNN TensorAccessorArgs、tiled reader/writer ABI 和 `eltwise_binary_no_bcast` compute 结构。

| Tiles | Grid | Max tiles/core | CB cycles | Static cycles | Saved cycles/local tile | Speedup |
|---:|:---:|---:|---:|---:|---:|---:|
| 1024 | 1x1 | 1024 | 703853 | 680519 | 22.79 | 1.0343x |
| 4096 | 1x1 | 4096 | 2813895 | 2718286 | 23.34 | 1.0352x |
| 16384 | 1x1 | 16384 | 11253491 | 10941406 | 19.05 | 1.0285x |
| 4096 | 2x2 | 1024 | 705825 | 686500 | 18.87 | 1.0282x |
| 16384 | 2x2 | 4096 | 2829605 | 2749813 | 19.48 | 1.0290x |

解释：

- static protocol 在 TTNN-style kernel 中仍然有收益。
- 幅度约 `18-23 cycles/local tile`，大约 `1.03x`，远小于 standalone memory-bound tile-add。
- 原因是 TTNN kernel 中 reader/writer、TensorAccessor、compute、pack/unpack 和 scheduling 已经占据更多 critical path。

## TTNN Paged Update Cache Direct Fork

这是第一轮按“直接复制 TTNN C++ program factory 和 kernels，再做最小协议替换”的真实 KV-cache update fork。

母版路径：

```text
ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_program_factory.cpp
ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp
ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp
ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp
```

fork 路径：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_paged_update_cache_protocol
```

已验证 case：

```text
users=8, kv_heads=8, head_dim=128, block_size=64/128,
max_seq_len=2048, cache_idx=127/1057, num_pages=2, repeats=2
```

结果：

| Case | Static mode | CB critical stage | CB critical | Static critical stage | Static critical | Saved cycles | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| `b64 idx127` | `static-runtime` | compute-input-untilize | 12845 | compute-pack | 12461 | 384 | 1.031x |
| `b64 idx127` | `static-streamreg-cbregs` | compute-input-untilize | 12845 | compute-pack | 12402 | 443 | 1.036x |
| `b64 idx1057` | `static-runtime` | compute-input-untilize | 12840 | compute-pack | 12485 | 355 | 1.028x |
| `b64 idx1057` | `static-streamreg-cbregs` | compute-input-untilize | 12840 | compute-pack | 12527 | 313 | 1.025x |
| `b128 idx127` | `static-runtime` | compute-input-untilize | 12877 | compute-pack | 12413 | 464 | 1.037x |
| `b128 idx127` | `static-streamreg-cbregs` | compute-input-untilize | 12877 | compute-pack | 12342 | 535 | 1.043x |
| `b128 idx1057` | `static-runtime` | compute-input-untilize | 12778 | compute-pack | 12394 | 384 | 1.031x |
| `b128 idx1057` | `static-streamreg-cbregs` | compute-input-untilize | 12778 | compute-pack | 12365 | 413 | 1.033x |

解读：

- 这条路线比 synthetic-only 更可靠：host 侧保留 paged cache geometry、page table、`update_idxs_tensor`、per-user L1 sharded input，device 侧保留原始 reader/writer/compute 语义。
- `CB` 模式 critical stage 是 `compute-input-untilize`；static 模式 critical stage 变成 `compute-pack`。这说明 static ring/schedule 确实移走了原 critical path 上的一部分 CB FIFO 动态管理成本。
- 收益稳定但不大，本轮复跑约 `313-535 cycles`，median speedup 约 `1.032x`。这是合理结果：真实 KV-cache update 还包含 page table lookup、DRAM/NOC、byte-row overwrite、untilize/tilize，CB FIFO 不是唯一成本。
- `static-streamreg-cbregs` 与 `static-runtime` 基本同级。它验证 per-CB stream-register ABI，但不能单独解释收益。

当前限制：

- 32-user 真实 shape 的 static path 仍会卡住；该问题记录为 fork scalability bug，不能把 32-user 性能纳入结论。
- 横向长 row layout，例如 `8x1`，会触发当前 static fork 卡住；默认 runner 已改成二维 sub-grid。
- `head_dim > 256` 还没有纳入 direct conclusion，因为 static untilize/tilize 的宽 Wt block-splitting 还没补齐。

## Real Matmul Reuse

matmul evidence 目前是负例/不确定例。

| M=N | K | static-input-only | static-output-only | static-input-output |
|---:|---:|---:|---:|---:|
| 512 | 64 | -35.0 | -227.5 | -150.5 |
| 512 | 128 | -39.5 | 26.5 | 15.0 |
| 512 | 256 | -64.0 | 344.5 | 174.5 |
| 1024 | 64 | -83.0 | 187.5 | 348.5 |
| 1024 | 128 | 214.5 | -69.5 | -23.0 |

解释：

- delta 小且随 shape / static mode 变号。
- compute、pack/reload、writer traffic、reuse schedule 会隐藏 local FIFO-management savings。
- 不能用 broad matmul 支撑 thesis；只能继续看 low-K、GEMV-like、multicast、decode-like exposed shapes。

## Phase 3 Operator-Family Coverage

Phase 3 core suite 已在单卡 Blackhole 系统上跑过：

```bash
conda run -n tt python tt_metal/programming_examples/compiler_managed_l1_dataflow/suite/ttnn_static_protocol_suite/run_ttnn_static_protocol_suite.py \
  --tier core \
  --phases phase3 \
  --skip-build \
  --out-dir /tmp/ttnn_static_protocol_suite_phase3_core_conda_v6
```

任务状态：

| 状态 | 数量 | 含义 |
|---|---:|---|
| pass | 73 | direct profiler tasks、sweep dry-run、pytest collect-only 大多通过。 |
| skip | 3 | 单卡系统上 CCL sweep 没有适用 vectors。 |
| fail | 1 | `pytest_ccl` collect-only 会 import 多设备/demo 依赖；runner 后续把它按 single-card skip 处理。 |

family coverage：

| Family | Modules dry-run | Vectors dry-run | Empty modules |
|---|---:|---:|---:|
| `eltwise` | 3 | 2784 | 0 |
| `data_movement_layout` | 3 | 3682 | 0 |
| `reduction` | 3 | 271872 | 0 |
| `normalization_softmax` | 3 | 884 | 0 |
| `matmul_linear` | 3 | 132 | 0 |
| `transformer_attention` | 3 | 11248 | 0 |
| `embedding_kv_cache` | 3 | 10808 | 0 |
| `conv_pool` | 3 | 1629 | 0 |
| `creation_fill_typecast` | 3 | 384 | 0 |
| `backward_moreh_experimental` | 3 | 3328 | 0 |
| `ccl` | 3 | 0 | 3 |

Phase 3 结论：

- broad survey 已足够指导下一步，不需要继续无差别扩大 coverage。
- 只有 elementwise / memory-bound 路径可以基于直接正向数据 promote。
- matmul 不能 broad promote。
- KV-cache update 已有真实 direct fork 正例，但目前只覆盖 8-user decode-like shape；32-user scalability 需要先修 fork，再讨论更真实批量结论。
- CCL 必须在 multi-device / fabric-capable setup 上测。
- 其他 family 只是 coverage-confirmed，必须先做 direct static fork 才能声称 speedup。

## 决策规则

- Phase 1 production-shaped fork 如果慢于 CB，则该操作保持 CB，并只推进可能受 lifetime/layout freedom 影响的变体。
- Phase 2 如果匹配或超过 CB，可把 CB 视为该路径的 backend detail。
- 如果一个 family 只在 decode-like 或 small-shape 赢，结论必须限定在该 shape class。
- matmul 如果继续不稳定，不得作为 thesis proof point。
- compute-path stream-register 变体如果没有 per-CB `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`，直接判定为无效对比。
- static backend 如果靠破坏 overlap 才赢，直接拒绝；`static-serialized` 已经证明 pipeline overlap 是必要条件。

## 下一步顺序

下一步不再是继续扩大 broad Phase 3 survey，而是把实验结果转化为 compiler ABI 和 firmware launch 原型。

### Step 0: 保护分支并固定文档入口

目标：

- 当前实验状态先本地 commit、rebase 到 `upstream/main`。
- 远端 push 到 fork；如果 SSH host key 未信任，先修 `known_hosts`。
- `docs/` 只保留本文档作为唯一结论入口。

验收：

- `git status` clean。
- `git log` 显示 WIP commit 在 `upstream/main` 之后。
- `README.md` 指向本文档，不再指向独立 operator-family report。

### Step 1: 固化 descriptor schema

修改范围：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/docs
tt_metal/programming_examples/compiler_managed_l1_dataflow/include   # 新增 experimental prototype header
```

目标：

- 在 prototype header 中定义 `TensorLayoutDescriptor`、`CompilerCBDescriptor`、`OperandViewDescriptor`、`QueueSyncDescriptor`。
- 字段只表达 compiler-owned L1 storage、queue sync、operand view、format/tile metadata。
- 不出现 TT-Metal `CBIndex`、`CircularBufferConfig`、`TensorAccessorArgs` 作为 IR 语义。

验收：

- header 可被 host 和 kernel-side experimental code include。
- 文档中 descriptor 字段和 header 保持一致。

### Step 2: host-side descriptor table 原型

修改范围：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow
tt_metal/impl/program/dispatch.cpp                 # 只加 experimental side-table path
tt_metal/hw/inc/hostdev/dev_msgs.h                 # 只在需要 launch flag/offset 时加 experimental 字段
```

目标：

- 选择单 core unary/binary elementwise 作为第一条 host path。
- 不调用 `CreateCircularBuffer` / `CircularBufferConfig` / `TensorAccessorArgs`。
- compiler-side helper 生成 per-core L1 allocation 和 operand views。
- dispatch 把 compiler descriptor table side-load 到每个 core 可见的 kernel config/L1 区。

验收：

- `local_cb_mask=0` 或 descriptor-only sentinel 时，host 仍能启动 experimental kernel。
- legacy TT-Metal CB examples 不受影响。

### Step 3: firmware experimental mode

修改范围：

```text
tt_metal/hw/firmware/src/tt-1xx/brisc.cc
tt_metal/hw/firmware/src/tt-1xx/ncrisc.cc
tt_metal/hw/firmware/src/tt-1xx/trisc.cc
tt_metal/hw/inc/internal
```

目标：

- 新增只影响实验 target 的 launch mode、compile-time flag 或 sentinel。
- 在 compiler-managed mode 下跳过 `setup_local_cb_read_write_interfaces` 和 `setup_remote_cb_interfaces`。
- 加载 compiler descriptor table，初始化 compiler-owned operand view / queue state。
- legacy mode 保持原 CB init 路径。

验收：

- firmware profiler 中 compiler mode 下 `CBP_FW_LOCAL_CB_INIT` / `CBP_FW_REMOTE_CB_INIT` 为空或不执行有效 work。
- legacy CB target 仍能跑。
- experimental target 不依赖 `local_cb_mask` 驱动 `CBInterface[]` 初始化。

### Step 4: no-`cb_*` raw-L1 tile copy

修改范围：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/level3_raw_l1_tile_copy
tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental       # 只新增 raw-address wrapper 时才改
tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental   # 只新增 raw-address API 时才改
```

目标：

- 单 core，一个或少量 tile。
- reader 写固定 L1 input staging。
- compute 使用 raw L1 address unpack、datacopy、raw L1 pack。
- writer 从固定 L1 output staging 写回 DRAM。
- kernel 中禁止 `cb_reserve_back`、`cb_wait_front`、`cb_push_back`、`cb_pop_front`、`get_local_cb_interface`、`copy_tile`、`pack_tile`。

验收：

- 输出正确。
- grep experimental kernel 不出现 TT-Metal CB API。
- raw-address helper 不通过 public wrapper 从 `cbid` 反查地址。

### Step 5: binary/unary op 扩展

目标：

- 从 raw-L1 copy 扩展到 binary add 或 unary SFPU。
- 验证 descriptor 可以表达两个 input operand 和一个 output operand。
- 仍然不碰 matmul。

验收：

- 正确性通过。
- profiler 能分离 legacy CB、kernel-only no-`cb_*`、firmware skip-CB-init、compiler descriptor + raw-L1。

### Step 6: mini-LLK / raw Tensix backend

目标：

- 在 Phase 3/5 稳定后，对单个 microkernel 直接生成或调用 raw Tensix instruction / MOP / config sequence。
- 优先 unary/binary elementwise 或 copy/layout microkernel。
- 不以完整 TTNN op 覆盖为第一目标。

验收：

- raw Tensix microkernel 正确。
- 性能归因能区分 CB/firmware/descriptor 替换收益和 LLK 重写收益。

### Step 7: operator-family migration

目标顺序：

1. `eltwise`：unary、binary broadcast、SFPU-heavy chain。
2. `data_movement_layout`：tilize/untilize、transpose、slice、concat。
3. `embedding_kv_cache`：修 32-user static scalability，再补 paged cache read / embedding lookup。
4. `normalization_softmax`：RMSNorm/LayerNorm、softmax decode。
5. `reduction`：sum/mean/max，分 small/long/cross-core。
6. `matmul_linear`：只看 GEMV-like、low-K、multicast、decode-like。
7. `ccl`：只在 multi-device / fabric-capable setup 上做。

## 执行验收矩阵

| Milestone | 必须证明 | 不足以证明 |
|---|---|---|
| M1: Descriptor schema | IR 不包含 TT-Metal CB/TensorAccessor 语义，descriptor 能表达 L1 storage、queue、operand view。 | 只把字段从 `CircularBufferConfig` 改名。 |
| M2: Host path replacement | 新 compiler path 不调用 `CreateCircularBuffer` / `TensorAccessorArgs`。 | 只在 kernel 里绕开 `cb_*`。 |
| M3: Firmware mode | compiler mode 下 firmware 不走 TT-Metal CB init，或 CB init 只作为 legacy fallback。 | `local_cb_mask` 仍驱动 `CBInterface` 初始化。 |
| M4: Kernel CompilerCB | reader/writer/compute 不使用 TT-Metal CB API，正确性通过。 | 只把 `cb_*` 包一层 wrapper。 |
| M5: LLK-compatible OperandView | LLK 的 operand metadata 来自 compiler descriptor，而不是 TT-Metal CB state。 | 仍依赖 `get_local_cb_interface(cbid)` 作为事实来源。 |
| M6: Operator policy | 能按 family/shape 给出 backend selection rule。 | 用一个全局平均 speedup 宣称所有 op 都应替换。 |
| M7: mini-LLK | raw Tensix backend 对选定 microkernel 正确且可归因。 | 结果无法区分来自 LLK 重写还是 CB/firmware 替换。 |

## 研究产出定位

这条线最终要产出的不是“某个 op 快了多少”的局部优化报告，而是一个新的编译器后端论证：

- 抽象层：TT-Metal runtime-managed CB / TensorAccessor 不应是 compiler IR 的核心抽象。
- ABI 层：firmware launch 应接收 compiler descriptor table，而不是只认识 `local_cb_mask` 和 CB config blob。
- Backend 层：compiler 可以根据 op family 和 shape 选择 static schedule、stream register、L1 counter、legacy CB fallback 或 raw Tensix backend。
- 评估层：收益要拆成 host/firmware/kernel steady-state/compute operand setup，而不是只看 end-to-end latency。

当前所有 profiler 收益/负例都服务于这个 thesis：**哪些 TT-Metal runtime-managed 数据流职责应该被提升到 compiler，并且这个提升在什么 op family、shape 和硬件路径上值得做。**
