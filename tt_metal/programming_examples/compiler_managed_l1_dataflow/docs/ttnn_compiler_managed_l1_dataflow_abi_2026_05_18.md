# TTNN Compiler-Managed L1 Dataflow ABI 研究计划 - 2026-05-18

这是当前 TTNN static-protocol / compiler-managed L1 dataflow ABI 实验的主结论文档。

核心结论：**研究方向成立，但要表述为 compiler / ABI 问题，而不是单个 runtime 优化或局部 CB micro-optimization**。目标不是让用户手动管理 TT-Metal `LocalCBInterface`，也不是在 device code 中换一个 runtime CB struct。目标是把 TT-Metal 的 `TensorAccessor`、`CircularBufferConfig`、`CreateCircularBuffer`、`cb_*` FIFO API、firmware `CBInterface` 初始化和手写 program factory 从 compiler path 中整体替换掉。新的研究对象是 **compiler-time L1Queue / OperandView / Schedule abstraction**：CB-like queue 只存在于编译器 IR 和 lowering 决策中，lowering 后直接生成 L1 地址计算、同步、NoC 和 pack/unpack 所需代码。

本文档现在是 `compiler_managed_l1_dataflow/docs` 下唯一主结论文档。原 `ttnn_operator_family_static_protocol_report_2026_05_18.md` 的横向算子报告、direct profiler 证据、收益/负例解释和下一步 family 优先级已经合并到本文档；后续不再维护第二份结论源。

## Mode Taxonomy

本轮收尾把 mode 名称固定下来，避免把 dataflow-only ablation、compute-path control 和下一阶段 Level C 混在一起。

| Mode | 定义 | 验证目标 | 边界 |
|---|---|---|---|
| `cb` | TT-Metal runtime-managed CB baseline；host 创建 `CircularBufferConfig`，firmware 初始化 `CBInterface`，kernel 使用 `cb_*`/LLK CB operand ABI。 | baseline。 | 不是 compiler-managed path。 |
| `static-runtime` | 显式 L1 ring + runtime args + L1 semaphore/counter；保留 correctness synchronization。 | 测量把 FIFO queue 明确化后，相对 CB 的收益或回退。 | L1 sync 状态本身可能比 CB 慢。 |
| `static-compiletime` | `static-runtime` 的地址/shape/layout/config baked into kernel defines。 | 隔离 runtime args 和 generic config 成本。 | 仍是 L1 semaphore/counter，不代表 compute-path stream-register backend。 |
| `static-streamreg-scratch` | dataflow-only 单 queue scratch-register ablation。 | 判断 copy 负例是否来自 payload movement 还是同步状态存储。 | 只适用于 copy 这类单 logical queue；不用于 compute-path。 |
| `static-streamreg-scratch-compiletime` | dataflow-only 单 queue + compile-time config ablation。 | 验证在保留 ready/consumed sync 时，compile-time queue/layout 是否能转正。 | 不推广到 compute operand CB ABI。 |
| `static-streamreg-cbregs` | **Level B 正式目标**：compute-path per-CB stream-register backend；每个 logical CB 使用自己的 `tiles_received` / `tiles_acked` register，per-core L1 address、work partition 和 shape/config 通过 runtime args 传入。 | 验证在 TT-Metal/LLK CB operand ABI 内替换 queue counter backend，评估真实多核工程路径的收益。 | 仍保留 `CreateCircularBuffer`/CB descriptor/LLK CB operand setup；runtime args 不是当前主要瓶颈。 |
| `static-streamreg-cbregs-compiletime` | compile-time config ablation / upper bound：在 `static-streamreg-cbregs` 上继续把可 baked 的 queue/layout/config 放进 kernel defines。 | 估计 runtime args/config bake-in 在 TT-Metal/LLK CB ABI 内部还能多省多少。 | 不是 Level B 完成条件；direct TTNN fork 多数只适合 single-core 或 per-core kernel variant。 |
| `static-input-only-cbregs-compiletime` / `static-output-only-cbregs-compiletime` / `static-input-output-cbregs-compiletime` | `real_matmul_protocol` 专属 compile-time ablation 命名，分别对应 input/output/input+output 三种替换范围。 | 在 matmul reuse 路径里隔离 input queue、output queue 和双端替换的上界效果。 | 当前只在 single-core low-K/GEMV-like 形状上有效，不能 broad promote，也不是全局 Level B 标准。 |
| `level-c-generated-static` | Level C first-proof hook：把 tile-add 的 L1 ring/layout/work partition 当作 compiler-lowered static schedule，用 compile-time constants 和独立 device-profiler zone 在真实 device path 上验证。 | 证明 `AxeTensor/L1Queue/OperandView` 方向可以落到真实 tile-add device path，并与 CB / Level B mode 做同 shape critical-stage 对比。 | 当前仍复用 TT-Metal `CreateCircularBuffer`、firmware `CBInterface` 初始化和 LLK CB-derived operand metadata；不是完整 CB-less / firmware-less Level C。 |
| `level-c-llk-direct` | Level C LLK direct-address proof：复用 static L1 ring/sync，但 compute kernel 手工展开 raw-address LLK unpack/math/pack 入口，不再通过 `add_tiles()` / `matmul_tiles()` / `pack_tile()` wrapper 推导 operand 地址。 | 先证明 compiler-lowered L1 address 可以直接喂给 LLK 的 unpack/math/pack 组合；add 和 matmul 都有对应 proof hook。 | 当前仍保留 host `CircularBufferConfig`、firmware `CBInterface` 初始化和部分 LLK format/tile metadata bring-up；不是 firmware/launch descriptor 替换。matmul 第一版限 `K=64,num_blocks=1,single-core`。 |
| `level-c-llk-direct-fw-skip-cb-init` | Level C firmware / launch descriptor gate：沿用 raw-address LLK proof，但 host 不注册 CB，launch descriptor 不设置 local CB mask，firmware 因 descriptor sentinel 跳过 local/remote CB init。 | 证明 experimental path 不再 materialize `CBInterface[]` 作为 operand/queue metadata 事实来源，同时 correctness 和 device critical-stage 对比仍成立。 | 本轮没有新增全局 `DISPATCH_MODE_COMPILER_L1` enum；当前是实验 executable 的 descriptor-empty/sentinel gate。它证明 ownership boundary，但不代表完整 TTNN runtime 已迁移。 |
| `compiler-owned operand/queue lowering IR` | Level C：替换 TT-Metal CB/LLK operand ABI，由 compiler IR 生成具体地址计算、同步和 operand-view 代码；descriptor 只是 bring-up / runtime-dynamic 字段载体，不是新的 runtime CB object。 | 判断是否真正绕过 TT-Metal CB 概念。 | 不在本轮实现完整 profiler。 |

## Level 2 Completion Definition

Level B / Level 2 本轮正式定义为：**保留 TT-Metal CB/LLK operand ABI，保留 correctness synchronization，使用 per-CB stream-register counter 替代 compute-path queue counter backend；多核 per-core L1 address、work partition 和 shape/config 默认通过 runtime args 传入。**

本轮只回答一个问题：在 TT-Metal/LLK CB ABI 内部，`static-streamreg-cbregs` 是否能稳定、省得可预测，并足够指导下一步 Level C 的源码级替换。`static-streamreg-cbregs-compiletime` 只回答 runtime args/config bake-in 的上界，不回答 Level C 的问题，即 compiler-time operand/queue/schedule abstraction 能否 lowering 成不依赖 firmware `CBInterface`、launch `local_cb_mask` 和 LLK CB-derived operand metadata 的具体代码。

Level 2 完成条件：

- Level B 标准 mode `static-streamreg-cbregs` 在 compute-path 正例、TTNN-style fork、KV-cache update fork、modeling harness 和 matmul input/output variants 中有明确覆盖状态。
- `static-streamreg-scratch-compiletime` 只作为 dataflow-only copy ablation，不作为 compute-path 结论源。
- `static-streamreg-cbregs-compiletime` 只作为 compile-time ablation / upper bound，不能作为多核工程路径或 Level B 完成条件。
- 结果按 family/shape 报告，正收益定义仍是 `CB critical - candidate critical > 0`。
- 负例保留：L1 semaphore static runtime、matmul reuse mixed、真实 op 中收益被 NoC/page table/compute 稀释。
- 进入 Level 3/Level C 前，必须先有结论说明哪些 family 值得迁移，哪些只作为 shape-specific candidate，哪些不推广。

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
- compiler-generated L1Queue lowering on TT-Metal-compatible launch：kernel 不调用 `cb_*`，queue/schedule 在编译器中规划，lowering 后直接生成 L1 ring 地址、sync 和 operand-view 代码；firmware launch 只允许作为临时 transport/fallback。
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
| data_movement/layout | `real_copy_protocol` | `static-runtime` median -124.38 cycles/work，`static-streamreg-scratch` median -11.66，`static-streamreg-scratch-compiletime` median +20.35 | L1 counter/semaphore 同步状态存储成本超过 CB；scratch register 说明 payload 搬运不是主因；compile-time scratch 说明 correctness sync 保留时仍能去掉 runtime/config 成本 | copy 只证明 compiler-known queue/layout 有收益空间；仍需 tilize/untilize/transpose direct fork 验证真实 layout op |
| eltwise | `real_tile_add_protocol` | `static-runtime` +450.39 cycles/work，`static-streamreg-cbregs` +451.45 cycles/work | writer/queue path 暴露，static schedule 替代 CB FIFO 动态管理带来大收益 | 推广到 memory-bound/simple elementwise direct forks |
| eltwise/TTNN | `ttnn_binary_ng_no_bcast_protocol` | `static-runtime` +23.22 cycles/local-tile，speedup 1.0350 | 真实 TTNN reader/writer/compute ABI 中 writer 仍有协议成本，但被真实地址/NoC/compute 稀释 | 作为 TTNN-style 正例，下一步扩展 unary/bcast/SFPU-heavy |
| eltwise/broadcast TTNN | `ttnn_bcast_to_protocol` row-bcast | 1024/4096/16384 tiles：`static-runtime` +5.03/+1.04/+0.27 cycles/tile；`static-streamreg-cbregs` +11.44/+10.85/+2.74 cycles/tile | critical stage 仍是 writer；static protocol 小幅降低 writer critical cycles，但大 shape 的 runtime 收益接近噪声 | 继续测 binary broadcast/SFPU-heavy，不能直接推广 |
| matmul/linear | `real_matmul_protocol` | 正负混合，部分 output-static shape 为正，部分 shape 回退 | reuse/compute/writeback critical path 掩盖 queue-only 收益 | 不 broad promote；只测 low-K/GEMV/multicast/decode-like |
| embedding/KV-cache | `ttnn_paged_update_cache_protocol` | 8-user decode-like saved 313.0 到 535.0 cycles，speedup median 1.0322 | CB critical stage 为 `compute-input-untilize`，static 后转到 `compute-pack`；CB FIFO 管理触到 critical path | shape-specific promote；32-user/cache-read/embedding lookup 未推广 |

### 全部 TTNN family 当前状态

| Family | 当前证据 | 是否可宣称收益 | 结论 | 下一步 |
|---|---|---|---|---|
| `eltwise` | direct TTNN no-bcast、bcast_to row-bcast + real tile-add + sweep/pytest coverage | 是，限 memory-bound/simple/exposed broadcast | 稳定小到大收益，取决于真实 op 中协议成本占比 | fork unary、binary broadcast、SFPU-heavy chain |
| `data_movement_layout` | direct copy L1 semaphore 负例 + streamreg compile-time ablation + sweep/pytest coverage | 仅限 dataflow-only copy ablation，不能推广到真实 layout op | L1-semaphore static 不如 CB；runtime scratch 接近但仍负；compile-time scratch 转正，说明 runtime/config 成本可以被 compiler-known queue/layout 去掉 | fork tilize/untilize、transpose、slice、concat |
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

- 同步状态存储成本更高：`real_copy_protocol` 的 L1 counter/semaphore static-runtime 比 CB 更慢，说明静态协议如果把状态放在 L1 并频繁轮询，成本可能高于原 CB；但 `static-streamreg-scratch-compiletime` 转正也说明 correctness sync 本身不是问题，问题是 runtime/config/generic FIFO 成本没有被 compiler 静态化。
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
- `static-streamreg-cbregs-compiletime`：compile-time ablation，在 per-CB registers 上继续 bake queue/layout/config；当前 direct TTNN fork 只用于 single-core 或 per-core kernel variant。

## Level 2 Profiler Coverage

| Profiler | Level B coverage | Mode | 备注 |
|---|---|---|---|
| `real_copy_protocol` | not applicable | `static-streamreg-scratch-compiletime` | dataflow-only 单 queue ablation；不作为 compute-path Level B。 |
| `real_tile_add_protocol` | done | `static-streamreg-cbregs` | memory-bound compute-path 正例；1x1/2x2 都用 runtime args 传 per-core 参数，compiletime 只作为上界消融。 |
| `static_protocol_modeling` | done | `static-streamreg-cbregs` | tile-add、eltwise-chain、matmul model 统一归因；`static-streamreg-cbregs-compiletime` 只在显式 ablation run 中使用。 |
| `ttnn_binary_ng_no_bcast_protocol` | done | `static-streamreg-cbregs` | 保留 TTNN TensorAccessor/reader/writer/compute ABI；single-core 和 multicore 都以 runtime args 路径为主。 |
| `ttnn_bcast_to_protocol` | done | `static-streamreg-cbregs` | row-broadcast direct fork；收益 shape-dependent，compiletime 只看 single-core upper bound。 |
| `ttnn_paged_update_cache_protocol` | done | `static-streamreg-cbregs` | 8-user decode-like evidence 是正式 Level B 证据；单用户 compiletime run 只作为 upper-bound ablation。 |
| `ttnn_binary_ng_row_bcast_protocol` | optional | `static-streamreg-cbregs` | untracked optional fork；作为 binary row-bcast add evidence，不作为 Level 2 必须结论源。 |
| `real_matmul_protocol` | done, shape-specific | `static-input-only-cbregs` / `static-output-only-cbregs` / `static-input-output-cbregs` | matmul 专属 input/output 范围命名；`*-compiletime` variants 只作为 single-core low-K/GEMV-like 上界消融，不能 broad promote。 |

## Level C 开展原则

Level C 的文档分两层：本节先固定收益假设和 ABI 依赖边界；后文 `Level C Kickoff Plan` 是实际开工顺序。关键调整是：**firmware / launch descriptor 替换必须早于 mini-LLK**。如果 firmware 仍通过 `local_cb_mask` 和 `CBInterface` 初始化 TT-Metal CB，那么 kernel 源码层面的 CB-less 只能算局部实验，不能声称已经绕过 TT-Metal CB 概念。

## Level C 收益猜想

Level C 的收益假设不是“把 runtime args 全部改成 compile-time defines”。Level B 已经说明，runtime args 用来传 per-core L1 address、work partition 和 shape/config 是合理的多核工程路径，当前数据中它不是主要瓶颈。Level C 应验证的是：**当 TT-Metal CB 不再是 host / firmware / LLK operand ABI 的事实来源后，compiler 是否能拿到更低的结构性成本和更大的 L1/schedule 优化空间。**

预期收益来源按优先级排列：

1. **移除 TT-Metal CB descriptor / operand ABI 兼容层。**
   - 不再由 `CreateCircularBuffer`、`CircularBufferConfig`、`local_cb_mask` 和 firmware `CBInterface` 驱动 operand/queue metadata。
   - 小 op、短 pipeline、many-CB op 可能从更少的 descriptor materialization、CB slot 遍历和 CB-derived metadata lookup 中受益。
   - 这部分收益要拆成 host construction、dispatch payload、firmware init 和 kernel steady-state，而不是只看 end-to-end latency。

2. **减少 generic CB/FIFO steady-state 路径。**
   - Level B 已经绕过一部分 `cb_wait/reserve/push/pop`，但仍保留 CB descriptor 和 LLK CB operand setup。
   - Level C 如果让 reader/writer/compute 使用 compiler-lowered 地址表达式、sync 和 operand view，可以减少泛化 pointer/counter 维护、CB id 到 operand metadata 的适配，以及不必要的 queue capability checks。
   - 预期首先体现在 memory-bound / simple elementwise 和 writer-bound 小 op 上。

3. **释放 L1 allocation、lifetime 和 bank placement。**
   - Level B 仍在 TT-Metal CB ABI 内部迁就 slot/page/layout 结构。
   - Level C 可以让 compiler 全局规划 L1 buffer 复用、queue materialization、bank conflict、tile/page layout、producer/consumer lifetime。
   - 这可能比“少几次 runtime arg load”更重要，尤其是 layout、broadcast、KV-cache update 和 decode helper kernels。

4. **跨 stage 的 static schedule / queue-depth 优化。**
   - compiler-owned queue descriptor 可以按 shape 选择 static schedule、stream-register counter、L1 counter fallback 或 legacy CB fallback。
   - 可以把 reader prefetch、compute operand lifetime、writer flush 顺序和 queue depth 作为 lowering 决策，而不是手写 program factory 的固定模式。

5. **减少 host/program-factory/firmware launch 的结构性开销。**
   - 对大 GEMM 这通常不是主因；对 decode step、小 batch、小 tensor、many-op graphs 可能可见。
   - Level C 目标是 compiler-generated L1Queue/OperandView lowering；descriptor table 只承载 runtime-dynamic 字段或 bring-up 验证，不是新的 device runtime CB struct。

按当前 Level B 数据，Level C 首批收益假设如下：

| Family / path | Level C 预期 | 原因 |
|---|---|---|
| memory-bound / simple elementwise | 最可能稳定收益。 | Level B 已经有强正例，writer/queue path 暴露。 |
| TTNN no-bcast / simple binary | 小而稳定的真实收益，适合作为 first source-change proof。 | 保留真实 TTNN reader/writer/compute ABI 时仍有约 `19 cycles/tile`。 |
| KV-cache update / decode helper | 小幅 shape-specific 收益，适合作为真实 workload proof。 | 真实 `paged_update_cache` 中 CB/protocol cost 触到 critical path，但被 page table、untilize/pack、NoC 稀释。 |
| broadcast / layout / transpose | 作为第二批候选。 | writer-bound 小正例存在，但接近噪声区间，必须按 shape 和 stage 判断。 |
| matmul reuse | 不作为首批收益目标。 | reuse/compute/writeback 掩盖 queue-only 收益，Level B mixed。 |
| pure copy L1 semaphore static-runtime | 作为负例保留。 | 说明错误的 sync storage/backend 可能比 TT-Metal CB 更慢。 |

因此 Level C 的首要 thesis 是：**收益来自 compiler 拥有 operand view、queue topology、L1 lifetime 和 schedule，并把它们 lowering 成具体代码；不是来自把 per-core 参数全部编译期常量化，也不是来自用户手动维护 `LocalCBInterface`。**

## Level C ABI 依赖路线图

下表是 Level C 的 ABI dependency map 和设计约束，不是另一套并行执行计划；实际开工顺序以后文 `Level C Kickoff Plan` 为准。

| Phase | 目标 | 关键产物 | 当前状态 |
|---|---|---|---|
| Phase 0: Evidence baseline | 保留 CB/runtime/static 基线，区分 protocol-bound 与非 protocol-bound family。 | `cb_protocol_overhead`、`real_copy_protocol`、`real_tile_add_protocol`、TTNN fork sweeps。 | 已足够指导后续替换顺序。 |
| Phase 1: Compiler IR and descriptor schema | 定义不含 TT-Metal CB/TensorAccessor 语义的 compiler IR，并区分 IR 与后续 lowering/ABI descriptor。 | `AxeTensor`、`AxeStorage`、`AxeLayout`、`AxeIter`；以及 lowering descriptor：`L1QueueDescriptor`、`OperandViewDescriptor`、`TensorLayoutDescriptor`、`QueueSyncDescriptor`。 | `level_c_ir.hpp` 是 compiler IR，可用 `std::vector` / `std::string`；`level_c_lowering_descriptors.hpp` 是后续 lowering / dynamic-field carrier，不是 runtime CB object。 |
| Phase 2: Host and launch descriptor replacement | 替换 `CreateCircularBuffer` / `CircularBufferConfig` / `TensorAccessorArgs` / `local_cb_mask`。 | compiler descriptor table、per-core L1 allocation table、launch descriptor extension 或 side-table。 | 尚未进入；应优先于 mini-LLK。 |
| Phase 3: Firmware metadata transport | firmware 搬运/暴露 compiler lowering 所需 metadata，不再走 TT-Metal CB 初始化路径。 | `operand_view_table` / lowering metadata loader；跳过 `setup_local_cb_read_write_interfaces` 的实验路径。 | 尚未进入；这是判断“是否真正绕过 TT-Metal CB”的边界。 |
| Phase 4: Kernel L1Queue codegen backend | kernel 不调用 `cb_*`，由 compiler 直接生成 queue acquire/release、NoC、L1 pointer、sync。 | 直接展开的 generated code；静态 schedule 下生成 slot/address expression。 | 现有 static protocol fork 是局部原型。 |
| Phase 5: LLK-compatible OperandView backend | 继续复用 LLK，但 LLK 看到的是 compiler-owned operand metadata，不是 TT-Metal CB state。 | `OperandView -> LLK operand setup` 兼容层。 | 尚未进入；这是从 CB descriptor 解耦 compute 的主路径。 |
| Phase 6: Operator-family migration | 按 family 迁移真实 op，建立 backend selection policy。 | elementwise、layout、KV cache、softmax、reduction、matmul/CCL 的 direct forks。 | Level B operator-family evidence 已给出优先级。 |
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
- 当前 launch schema 以 TT-Metal CB 为中心，compiler-owned lowering metadata 需要从这里开始替换或 side-load。

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

- Level C / Level 3 不只是“不调用 cb API”，还要让 format/tile/operand metadata 不再由 TT-Metal CB descriptor 派生。
- raw-address unpack/pack 是可行性证据，但现有 wrapper 不能直接作为 Level C / Level 3 证明。
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

下面 Phase 1-7 是设计约束说明，不是新的执行顺序；执行顺序见 `Level C Kickoff Plan`。

### Phase 1: compiler IR 与 lowering descriptor 分层

当前 prototype header 已拆成两层：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/include/level_c_ir.hpp
tt_metal/programming_examples/compiler_managed_l1_dataflow/include/level_c_lowering_descriptors.hpp
```

当前要先区分两层：`AxeTensor` / `AxeLayout` 是 compiler IR，表达 tensor view 和 layout algebra 本体；`TensorLayoutDescriptor`、`L1QueueDescriptor`、`OperandViewDescriptor`、`QueueSyncDescriptor` 和 `DescriptorTableHeader` 是后续 lowering / runtime-dynamic field carrier，不是 device-side runtime CB object。

```text
AxeTensor:
  name, element_type,
  storage: AxeStorage{
    name, kind, memory_space, ownership,
    size_bytes, base_address, root_storage_name, device_scope
  },
  layout: AxeLayout

AxeStorage:
  kind: owned_device_allocation | owned_mesh_allocation |
        borrowed_host_storage | borrowed_device_address |
        view_of_storage | register_fragment
  ownership: owned | borrowed | view | temporary

AxeLayout:
  d_iters: vector<AxeIter{extent, stride, axis=AxeAxis{name, kind}}>
  r_iters: vector<AxeIter{extent, stride, axis=AxeAxis{name, kind}}>
  o_entries: vector<AxeCoordinateEntry{value, axis=AxeAxis{name, kind}}>

TensorLayoutDescriptor:
  schema_version, memory_space, address_policy, data_format,
  base_addr, page_size_bytes, page_count,
  logical_shape(rank, dims[4]), physical_shape(rank, dims[4]),
  stride_pages[4],
  shard_map_index, bank_map_index

QueueSyncDescriptor:
  schema_version, sync_kind, producer, consumer,
  produced_addr, consumed_addr,
  produced_stream_reg, consumed_stream_reg,
  static_schedule_id, initial_produced, initial_consumed

L1QueueDescriptor:
  schema_version, l1_base_addr, total_size_bytes, slot_count,
  page_size_bytes, bank_policy, bank_map_index, sync_descriptor_index,
  lifetime_start_step, lifetime_end_step, flags

OperandViewDescriptor:
  schema_version, operand_id, operand_role, pack_unpack_role,
  data_format, tile_shape, page_size_bytes,
  queue_descriptor_index, tensor_layout_index,
  l1_base_addr, read_stride_pages, write_stride_pages,
  tiles_per_page, flags

DescriptorTableHeader:
  schema_version, tensor_layout_count, queue_sync_count,
  l1_queue_count, operand_view_count,
  tensor_layout_offset_bytes, queue_sync_offset_bytes,
  l1_queue_offset_bytes, operand_view_offset_bytes,
  total_size_bytes
```

设计规则：

- IR 中不能出现 TT-Metal `CBIndex`、`TensorAccessorArgs` 或 `CircularBufferConfig`。
- `AxeTensor` 不单独保存 `logical_shape`。layout 是 tensor domain 和 mapping 的唯一事实源：`D` 中 iter 的 extent 表达可访问 domain，stride/axis 表达坐标映射，`R` 表达 replica，`O` 表达 fixed coordinate offset。
- `AxeTensor` 必须有明确的 `AxeStorage`，不能是没有 storage engine 的悬空 tensor。这个 storage 可以在编译早期是 symbolic storage；`base_address=0` 表示尚未物理分配，但 kind、memory space、ownership、root/view/device scope 等资源语义仍然明确，后续再 lowering 到 L1/DRAM/base address/queue slot。
- `operand_id` 可以存在，但它是硬件 operand selector，不是 TT-Metal CB id。
- `queue_ref` 可以 lowering 到 compiler counter、stream register、static ordering 或 fallback TT-Metal CB。
- 参照 CuTe 的 `Tensor = Engine + Layout` 和 MLIR `memref` 的分层方式，compiler semantic type 可以表达 element type、storage、layout、memory space 和 schedule 约束；真实生成代码不应依赖 `ShapeDescriptor` 这个 C++ 类型。
- `ShapeDescriptor{rank, dims[4]}` 只是 Phase 1 descriptor-table bring-up 的 lowered metadata carrier，用来承载 lowering 后仍需要运行时传递的 shape 字段；如果某个 shape/layout 已经静态确定，应被折叠成常量地址表达式、loop bounds 或 compile-time schedule。
- 固定上限仍保留为 4 维，维度不命名为 `n/c/h/w`，避免把 NCHW 或某一种布局语义写进 descriptor ABI。host / TTNN 层可以继续使用已有 `tt::tt_metal::Shape` 或未来 MLIR-like 类型，lowering 到 descriptor-table backend 时再转换成固定上限数组。
- `page_size_bytes` / `page_count` / `stride_pages` 描述的是物理存储单位和地址步进，不是 tensor 维度本身；如果采用 Axe 风格的统一 layout，这一层可以直接看成 block/atom 级 layout 的一个实现细节。它们存在的原因是 lowering 后需要生成可执行的 L1 地址算式、ring slot 算式和布局访问算式，而不是因为我们必须把 page 当成独立语义层。
- `AxeLayout` 直接按 Axe 论文 Figure 1 建模：`AxeIter = (extent, stride, axis)` 是显式 IR 类型，`D` 是 ordered list of iters，`R` 是 replication iters 的 set，`O` 是一个 fixed coordinate offset。CuTe/CUTLASS 的 `Shape + Stride` 只说明单个 iter 的 extent/stride 地址含义，不再作为顶层 layout IR 形状。
- 因为 `AxeTensor` / `AxeLayout` 是 compiler IR，不是 device ABI descriptor，所以不需要 `schema_version`、`d_iter_count`、`r_iter_count`、`o_count` 或固定容量数组；`AxeLayout` 顶层直接是 `d_iters`、`r_iters`、`o_entries` 三个 vector。`o_entries` 表达单个 coordinate `O` 的稀疏 axis-value entries；未出现的 axis 默认 offset 为 0，它不是多个 offset。只有进入序列化、kernel config 或跨 host/device ABI 时，才需要另行定义固定数组和 count。
- `AxeAxis` 是独立的 compiler IR 对象，不只是 enum 包装。`name` 表达轴身份，可以区分 `row_outer` / `row_inner` / `tile_row` 这类不同轴；`kind` 只是 TT-Metal lowering 需要的分类 hint。shard、replica、offset 的语义分别来自 iter 所在的 `D`、`R`、`O` 容器，而不是来自 axis 内部的 role。
- tile、face、page、block、stick 不应成为互相平行的特殊 layout 字段；它们应被看成嵌套 layout mode 和 named axis。当前 `TensorLayoutDescriptor` 保留 page 字段，是为了兼容 lowering 后的 TT-Metal allocator / TensorAccessor 风格地址算式，不是因为 page 必须作为独立 compiler IR 概念长期存在。
- `block_size_bytes`、`atom_size_bytes`、`tile_height`、`tile_width`、`face_height`、`face_width`、`buffer_layout` 这类字段已经从 layout algebra 中移除。它们要么能由 mode extent/stride 推导，要么属于 TT-Metal lowering/backend policy，不应污染统一 layout 类型。
- `TensorLayout`、`PageConfig`、`Tile`、`ShardSpec`、`ShardSpecBuffer`、`BufferShardingArgs`、`NdShardSpec`、`MeshBuffer`、`TensorAccessorArgs` 这些现有概念可以映射到统一 layout algebra；`Buffer` / `MeshBuffer` / `Tensor` 的 allocation、ownership、lifetime、device visibility、view/root 和 API interop 语义则映射到 `AxeStorage` 这层 CuTe-style storage engine。
- 因此，“一网打尽”的正确边界更新为：`AxeLayout` 统一 logical index 到 physical address / bank / core / mesh / tile / face 的 mapping 语义；`AxeStorage` 统一 resource/storage engine 语义；`AxeTensor` 统一承载 element type、storage 和 layout。当前 proof 不等价于 TTNN public API 已经迁移或删除。
- descriptor 必须能表达 dataflow-only、compute input、compute output、remote/multicast 四类路径。
- 后续要下发到 device 的 lowering descriptor 必须是 POD/trivially-copyable，不能依赖 `std::vector`、`std::string`、TT-Metal host object 或 device 不可解释的 C++ runtime state；但 compiler IR 层的 `AxeTensor` / `AxeLayout` 可以使用 `std::vector` 和 `std::string`。
- Level C generated code 中不应出现 `LocalCBInterface`、`CBInterface[]`、`get_local_cb_interface()`、`cb_wait_front()`、`cb_reserve_back()`、`cb_push_back()`、`cb_pop_front()`、CB API 形式的 `get_read_ptr()` / `get_write_ptr()`。

### Phase 1.5: layout algebra coverage sanity

当前新增的可执行 proof target：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_axe_layout_coverage_sanity.cpp
tt_metal/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_tt_metal_layout_adapter_sanity.cpp
```

它们不是性能实验，而是 coverage sanity：第一层把 tech report 和当前类型系统里最核心的 layout family 都落到同一个 `AxeLayout` 上，并检查每个 case 至少具备对应的 named axis、`D` iter、`R` iter 或 `O` offset 信息；第二层模拟 TT-Metal / TTNN 真实类型的语义输入，证明这些输入可以经 adapter 生成 `AxeTensor + AxeStorage + AxeLayout`，并对关键地址/placement 公式和资源语义做等价检查。

覆盖项：

| TT-Metal 概念 | Axe-like 表达 |
|---|---|
| row-major | `D=[(64,64,row),(64,1,col)]` |
| tiled layout | `D=[tile_row,tile_col,face_row,face_col]`，每个 iter 携带自己的 extent/stride |
| face order / tile atom | `D` 中嵌套 `tile_*` + `face_*` iter；atom size 由 nested extent/stride 推导，不再是字段 |
| interleaved memory | `D` 中同时出现 `storage` 和 `bank` iter，表达 physical placement |
| height sharding | `D` 中出现 `core_y` shard placement iter，local `height/width` 仍是普通 iter |
| width sharding | `D` 中出现 `core_x` shard placement iter，local `height/width` 仍是普通 iter |
| block sharding | `D` 中出现 `core_y/core_x` placement iter，local block `height/width` 仍是普通 iter |
| ND sharding | `D` 中组合多个 logical axis、`core_x/core_y` placement axis 和 `storage` axis |
| mesh replicated | `D=[row,col]`，`R=[mesh_x,mesh_y]` |
| mesh sharded | `D` 中包含 `mesh_x/mesh_y` placement iter |
| TensorAccessor 地址表达 | 从统一 layout 的 `D + O` 生成 `base + offset + index * stride` 类地址算式 |

TT-Metal adapter proof 覆盖项：

| TT-Metal / TTNN 类型或语义 | Proof 中的 adapter 结论 |
|---|---|
| `TensorLayout + RowMajorPageConfig` | `AxeTensor` 持有确定 storage，`AxeLayout D=[row,col]`，row-major offset 与 `row * width + col` 等价。 |
| `TensorLayout + TilePageConfig + Tile` | tile、face、face 内坐标都表达为 named axes；`transpose_tile=true` 通过不同 stride 表达 face/value 转置，不需要额外特殊字段。 |
| `MemoryConfig(INTERLEAVED, DRAM/L1)` / `InterleavedAddrGen` | storage 落到 `AxeStorage.memory_space`，page 到 bank 的 round-robin 和 bank-local page offset 由 `bank` / `bank_page` axes 表达。 |
| `TensorMemoryLayout::HEIGHT_SHARDED + ShardSpecBuffer` | core placement 用 `core_y/core_x` axes 表达，local shard page 用 `local_page_row/local_page_col` 表达；page->core/page_offset 与现有 height sharded addrgen 公式等价。 |
| `TensorMemoryLayout::WIDTH_SHARDED + ShardSpecBuffer` | width-fractured placement 用 `core_x/core_y` 和 local page axes 表达；page->core/page_offset 与 width sharded addrgen 公式等价。 |
| `TensorMemoryLayout::BLOCK_SHARDED + ShardSpecBuffer` | 2D block placement 用 `core_y/core_x` + local block axes 表达；page->core/page_offset 与 block sharded addrgen 公式等价。 |
| `TensorMemoryLayout::ND_SHARDED + NdShardSpec` | 每个 sharded dimension 拆成 `shard_dim_i` 和 `local_dim_i`；round-robin bank distribution 用 `bank` / `bank_shard` axes 表达，覆盖 regular、uneven distribution 和 single-dimension sharding。 |
| `MeshBuffer` replicated | device mesh replication 落到 `R=[mesh_x, mesh_y]`。 |
| `MeshBuffer` sharded | mesh placement 落到 `D` 中的 `mesh_x/mesh_y` axes。 |
| `Buffer::view` / padded-slice offset | fixed base/view offset 落到单个 `O` 的 sparse coordinate entries。 |
| `Buffer` owned allocation | owned device allocation 落到 `AxeStorage{kind=owned_device_allocation, ownership=owned, memory_space, base_address, device_scope}`。 |
| `Buffer::view` ownership | view/root lifetime 语义落到 `AxeStorage{kind=view_of_storage, ownership=view, root_storage_name}`。 |
| `MeshBuffer` owned allocation | mesh backing allocation 和 mesh scope 落到 `AxeStorage{kind=owned_mesh_allocation, ownership=owned, device_scope="mesh[...]"} `。 |
| `HostTensor` / borrowed data | host-side borrowed storage 落到 `AxeStorage{kind=borrowed_host_storage, ownership=borrowed, memory_space=system_memory}`。 |
| externally owned device address | 外部 L1/DRAM address 落到 `AxeStorage{kind=borrowed_device_address, ownership=borrowed, base_address}`。 |
| register fragment | 临时 compute fragment 落到 `AxeStorage{kind=register_fragment, ownership=temporary, memory_space=register_file}`。 |

当前 proof 输出 layout 和 resource 两类计数：`layout_covered_cases=14`，`resource_covered_cases=6`，`covered_cases=20`。这证明的是“layout mapping + storage/resource engine 语义可表达”，不是证明当前 TT-Metal 源码已经迁移，也不是证明 TTNN public API 类型已经删除。

这证明的是“表达能力覆盖”，不是证明当前 TT-Metal 代码已经重构完成。真正迁移时应先把现有 `TensorAccessorArgs` / sharding helper 变成 `AxeLayout` adapter，再逐步让 kernel lowering 从统一 layout 直接生成地址表达式。

### Phase 1.75: lowering adapter 与 CB 建模 sanity

当前新增的 lowering adapter proof target：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_lowering_adapter_sanity.cpp
tt_metal/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_tile_add_codegen_sanity.cpp
```

这一步证明的是：CB 可以在 compiler IR 中拆成几件独立职责，而不是作为 `LocalCBInterface` runtime object 继续存在：

| 原 CB / TensorAccessor 职责 | Level C 表达 |
|---|---|
| L1 payload storage | `AxeStorage{memory_space=l1, base_address, size_bytes}` + `AxeLayout D=[ring_slot, byte]` |
| queue/ring 容量和 slot 地址 | `L1QueueDescriptor{l1_base_addr, slot_count, page_size_bytes}` |
| producer/consumer 同步 | `QueueSyncDescriptor{sync_kind=stream_register_pair/static_schedule/...}` |
| compute operand selector 和 pack/unpack 角色 | `OperandViewDescriptor{operand_id, operand_role, pack_unpack_role, data_format, tile_shape}` |
| DRAM tensor page 地址 | `AxeTensor` + `TensorLayoutDescriptor{base_addr, page_size_bytes, page_count}` |
| view/root offset | `AxeStorage{kind=view_of_storage, root_storage_name}` + `AxeLayout O` |

`level_c_lowering_adapter_sanity` 的当前输出为 `lowering_adapter_cases=7`，覆盖 CB payload、CB sync、CB operand、Axe ring address evaluator、TensorAccessor DRAM address、Buffer view offset 和 generated pseudo-code no-runtime-CB-token 检查。`level_c_tile_add_codegen_sanity` 进一步证明 tile-add 的 ring schedule 可以从 descriptor 生成等价地址序列，例如 tile 0/2 复用 slot 0，tile 1/3 复用 slot 1。

这一步仍是 host-side / codegen sanity，不是 device profiler proof。它的意义是把“CB 能否建模出来”回答清楚：**能建模，但要建模成 storage + layout + queue + sync + operand view + schedule，而不是建模成新的 runtime CB 结构体。**

### Phase 2: host / dispatch 替换

目标是在新 compiler path 中不调用：

- `CreateCircularBuffer`
- `CircularBufferConfig`
- `TensorAccessorArgs`
- TT-Metal program factory 中手写的 CB allocation logic

替代方案：

- compiler 先完成 per-core L1 allocation，并生成 `L1QueueDescriptor[]` 或直接生成等价 lowering code。
- compiler 生成 tensor layout / bank mapping，不再让 kernel 用 `TensorAccessor` 动态解释布局。
- dispatch 只负责把 descriptor table 放到每个 core 可见的 L1/config 区域。
- launch descriptor 中增加或 side-load `compiler_descriptor_offset`、`compiler_descriptor_count`、`operand_view_offset`。

验收标准：

- 一个 unary/binary elementwise op 的 host path 不再创建 TT-Metal CB。
- `local_cb_mask` 可以为 0，或者只用于 legacy fallback。
- kernel 仍能通过 compiler lowering metadata 或已经展开的 codegen 完成读、算、写。

### Phase 3: firmware descriptor loader

这是优先级必须提前的阶段。当前 TT-Metal firmware 在 BRISC/TRISC/NCRISC/ERISC 中都存在 `CBInterface` 和 CB 初始化路径；如果不替换这里，compiler path 仍然依赖 TT-Metal CB 概念。

目标：

- 新增 experimental launch mode，例如 `DISPATCH_MODE_COMPILER_L1`。
- firmware 检测该 mode 后跳过 `setup_local_cb_read_write_interfaces` / `setup_remote_cb_interfaces`。
- firmware 只加载或搬运 compiler lowering 所需的 runtime-dynamic metadata；不初始化新的 runtime CB object。
- 对 legacy TT-Metal program 保持原路径，不破坏现有 op。

最小实验：

1. 单 core unary/binary elementwise，不使用 remote CB。
2. `local_cb_mask=0` 或 descriptor-only sentinel。
3. firmware 只暴露 `OperandViewDescriptor` 等 lowering metadata 所需的 runtime-dynamic 字段。
4. kernel 通过 compiler lowering metadata 或已经展开的 generated code 获取 L1 base/page size/format/sync。

验收标准：

- firmware profiler 中不出现 TT-Metal local/remote CB init zone，或该 zone 在 compiler mode 下为空。
- kernel correctness 通过。
- device critical path 可与 legacy CB、kernel-only static protocol 对比。

### Phase 4: kernel L1Queue codegen backend

目标是让 generated kernel 不出现 TT-Metal dataflow API：

- 不调用 `cb_reserve_back` / `cb_wait_front` / `cb_push_back` / `cb_pop_front`。
- 不调用 `get_read_ptr` / `get_write_ptr` / `get_tile_size` 获取 TT-Metal CB state。
- reader/writer/compute 使用 compiler lowering 后的 L1 地址表达式、page size 和 sync code。

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

### Phase 6: operator-family migration policy

这一阶段不在 descriptor / firmware / `OperandView` bring-up 前启动。它只在 first profiler fork 能稳定区分 legacy CB、Level B `static-streamreg-cbregs`、compiler descriptor、firmware skip-CB-init 和 `OperandView` 后开始。

目标：

- 用 Level B family 结论决定迁移顺序，而不是按 op 名单全量铺开。
- 首批只迁移 memory-bound / simple elementwise 和一个 KV-cache update / decode helper proof。
- broadcast、layout、transpose 放在第二批；matmul reuse 只保留 low-K / GEMV-like / decode-like shape-specific candidate。
- 每个 family 都保留 legacy CB fallback，并输出 backend selection policy。

验收标准：

- 每个迁移 family 都有同 shape 的 `cb`、`static-streamreg-cbregs` 和 Level C mode device-profiler 对比。
- 能说明收益来自 descriptor/firmware/operand view、queue backend、L1 lifetime，还是其它 stage 稀释。
- 如果某 family 只有噪声区间或 mixed 结果，文档必须记录为负例或 shape-specific candidate，不能 broad promote。

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
| `real_copy_protocol` | L1 semaphore static runtime 是负例；runtime scratch 仍接近但慢于 CB；scratch-register + compile-time queue/layout 转正。主因不是 payload movement，而是同步状态存储和 runtime/config/generic FIFO 成本的组合。 | 本轮 `/tmp/real_copy_protocol_compiletime_ablation_2026_05_19`：`static-runtime` median `-124.38 cycles/work`；`static-compiletime` `-64.68`；`static-streamreg-scratch` `-11.66`；`static-streamreg-scratch-compiletime` `+20.35`。 |
| `real_tile_add_protocol` | static schedule 替代 CB FIFO 有大收益；cbregs 保留收益但相对 runtime 只有极小差异。 | `static-runtime` median `+450.39 cycles/work`；`static-streamreg-cbregs` `+451.45`。 |
| `real_matmul_protocol` | mixed / shape-specific；不能 broad promote。 | 本轮 sweep `4` 个正向、`5` 个负向。 |
| `ttnn_binary_ng_no_bcast_protocol` | TTNN-style 正例对照；static protocol 能穿过真实 TTNN 风格 ABI，但收益远小于 standalone tile add。 | 本轮复跑 `static-runtime` median `+23.22 cycles/work`；`static-streamreg-cbregs` `+21.40`。 |
| `ttnn_paged_update_cache_protocol` | 真实 `paged_update_cache` C++ factory/kernel fork 正例；KV-cache update 中 CB FIFO 动态管理会影响 device critical path，但收益幅度是 TTNN-style 小幅稳定收益。 | 本轮复跑 8-user decode-like 真实形状 `+313` 到 `+535 cycles`，median speedup 约 `1.032x`；32-user static path 仍是 fork scalability 待修项。 |

Level B 主线 device-profiler 证据来自 `static-streamreg-cbregs` 复跑和前置 direct sweeps。本轮主线输出记录在：

```text
/tmp/levelb_cbregs_completion_2026_05_19
```

该目录每个 profiler 都保留 `host_results.csv`、`host_summary.csv`、`zone_summary.csv`、`critical_stage_summary.csv`、`device_mode_comparison.csv` 和 `host_mode_comparison.csv`，并在 `attribution/` 下生成 `level_b_summary.csv`、`compiletime_ablation_summary.csv`、`root_cause_report.md`。

| Profiler | Level B case | CB critical stage | Level B critical stage | Saved cycles | Saved cycles/local work | Speedup |
|---|---|---|---|---:|---:|---:|
| `real_tile_add_protocol` | `tiles=256 pages=2` | writer | writer | 123118 | 480.93 / tile | 1.770x |
| `real_tile_add_protocol` | `tiles=1024 pages=2` | writer | writer | 457144 | 446.43 / tile | 1.680x |
| `real_tile_add_protocol` | `tiles=4096 pages=2` | writer | writer | 1824931 | 445.54 / tile | 1.678x |
| `ttnn_binary_ng_no_bcast_protocol` | `tiles=1024 pages=2` | writer | writer | 20057 | 19.59 / tile | 1.029x |
| `ttnn_binary_ng_no_bcast_protocol` | `tiles=4096 pages=2` | writer | writer | 78095 | 19.07 / tile | 1.029x |
| `ttnn_bcast_to_protocol` | `tiles=1024 width_tiles=8 pages=2` | writer | writer | 7821 | 7.64 / tile | 1.014x |
| `ttnn_bcast_to_protocol` | `tiles=4096 width_tiles=8 pages=2` | writer | writer | 25400 | 6.20 / tile | 1.011x |
| `ttnn_bcast_to_protocol` | `tiles=16384 width_tiles=8 pages=2` | writer | writer | 92188 | 5.63 / tile | 1.010x |
| `ttnn_paged_update_cache_protocol` | `users=8 kv_heads=8 head_dim=128 block=64/128 seq=2048` | compute-input-untilize | compute-pack | 190 to 330 | 23.75 to 41.25 / user | 1.016x to 1.027x |

主线结论：

- `real_tile_add_protocol` 是强正例：Level B `static-streamreg-cbregs` 稳定节省约 `445-481 cycles/tile`。
- `ttnn_binary_ng_no_bcast_protocol` 是真实 TTNN-style 正例：收益稳定但被真实 reader/writer/compute ABI 稀释到约 `19 cycles/tile`。
- `ttnn_bcast_to_protocol` 是小幅 writer-bound 正例：约 `5.6-7.6 cycles/tile`，接近噪声边界，不能单独推广到所有 broadcast/SFPU-heavy。
- `ttnn_paged_update_cache_protocol` 是真实 KV update 小幅正例：8-user decode-like shapes 仍为正，但本轮幅度约 `190-330 cycles/case`，小于旧跑；结论应写成 shape-specific promote。
- host enqueue/finish 方向在多个短 case 中与 device delta 不一致，因此 Level B 和 Level C gate 只以 device critical stage 为准。

compile-time 上界消融输出记录在：

```text
/tmp/level2_completion_2026_05_19
```

该目录每个 profiler 都保留 `host_results.csv`、`host_summary.csv`、`zone_summary.csv`、`critical_stage_summary.csv`、`device_mode_comparison.csv` 和 `host_mode_comparison.csv`。旧表中的 2026-05-18 数字仍是 Level A/B 前置 evidence；下表只记录本轮新增 `*-compiletime` ablation 的 compact core-shape 结果，不能替代 Level B 主线 `static-streamreg-cbregs` 结论。

| Profiler | Compile-time ablation case | CB critical stage | Ablation critical stage | Saved cycles | Saved cycles/local work | Speedup |
|---|---|---|---|---:|---:|---:|
| `real_tile_add_protocol` | `tiles=256 pages=2` | writer | writer | 132656 | 518.19 / tile | 1.883x |
| `real_tile_add_protocol` | `tiles=1024 pages=2` | writer | writer | 489717 | 478.24 / tile | 1.765x |
| `ttnn_binary_ng_no_bcast_protocol` | `tiles=1024 pages=2` | writer | writer | 34085 | 33.29 / tile | 1.051x |
| `ttnn_binary_ng_no_bcast_protocol` | `tiles=4096 pages=2` | writer | writer | 133395 | 32.57 / tile | 1.050x |
| `ttnn_bcast_to_protocol` | `tiles=1024 width_tiles=8 pages=2` | writer | writer | 21479 | 20.98 / tile | 1.038x |
| `ttnn_bcast_to_protocol` | `tiles=4096 width_tiles=8 pages=2` | writer | writer | 87124 | 21.27 / tile | 1.039x |
| `ttnn_paged_update_cache_protocol` | `users=1 kv_heads=1 head_dim=32 block=32 seq=128 idx=0 pages=2` | compute-input-untilize | compute-pack | 726 | 726 / user | 1.433x |

结论一句话：**Level B 可以用 `static-streamreg-cbregs` 作为主线完成；stream-register 不是万能收益来源，per-CB stream-register 是正确 ABI 控制面，真正的大收益来自 static ring/schedule 消除 CB FIFO 动态管理，并且只在 protocol cost 暴露的路径上明显。compile-time bake-in 有上界价值，但不是多核工程路径的前置条件。**

## Level B Completion And Level C Gate

当前目录可以宣称完成 Level B 的核心实验闭环，但结论必须按 family/shape 限定：

- 可以进入 Level C 的首批对象：memory-bound / simple elementwise，以及 KV-cache update 这类 writer/queue 或 compute-input/output protocol cost 暴露的路径。
- 可以作为第二批候选：broadcast 和 layout/transpose 弱正例；它们需要按 shape 和 critical stage 继续分开判断。
- 暂不进入 Level C 首批：matmul reuse、pure copy L1 semaphore static runtime、SFPU-heavy 噪声区间路径。
- Level C 修改源码时应优先替换 compiler/firmware/launch descriptor 中的 TT-Metal CB dependency，而不是追求把 runtime args 全部 bake 成 compile-time defines。

## Level C First Proof: Tile Add

当前第一条真实 device proof 不是完整 CB-less kernel，而是 `real_tile_add_protocol` 中新增的 `level-c-generated-static` mode。它的作用是把 Level C IR/lowering 方向落到一个真实可运行的 tile-add device path 上，并保留独立 profiler zone，便于与 `cb` 和 Level B `static-streamreg-cbregs` 做同 shape device critical-stage 对比。

复现命令：

```bash
conda run -n tt cmake --build build_Release \
  --target compiler_managed_l1_dataflow_level_c_examples real_tile_add_protocol -j8

build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_descriptor_schema_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_axe_layout_coverage_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_tt_metal_layout_adapter_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_lowering_adapter_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_tile_add_codegen_sanity

TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rtadd_level_c_generated_static_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=level-c-generated-static --tiles=4 --num-pages=2 --repeats=1 --device-id=0

conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/level_c_tile_add_first_proof_2026_05_19 \
  --tiles 256 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb static-streamreg-cbregs level-c-generated-static
```

本轮验证结果：

| Check | 结果 |
|---|---|
| `level_c_descriptor_schema_sanity` | pass；示例 descriptor table `280 bytes`。 |
| `level_c_axe_layout_coverage_sanity` | pass；row-major、tile/face、interleaved、height/width/block/ND sharding、mesh replicated/sharded、offset view、TensorAccessor-style address 全部 covered。 |
| `level_c_tt_metal_layout_adapter_sanity` | pass；`layout_covered_cases=14`，`resource_covered_cases=6`，`covered_cases=20`。 |
| `level_c_lowering_adapter_sanity` | pass；`lowering_adapter_cases=7`。 |
| `level_c_tile_add_codegen_sanity` | pass；tile 0/2 复用 slot 0，tile 1/3 复用 slot 1。 |
| `real_tile_add_protocol --mode=level-c-generated-static --tiles=4` | pass；`max_abs_error=0`。 |

Device-profiler first proof：

| Mode | Critical stage | Critical cycles | Delta vs CB | Delta / tile | Speedup vs CB |
|---|---|---:|---:|---:|---:|
| `cb` | writer | 283809 | 0 | 0 | 1.000x |
| `static-streamreg-cbregs` | writer | 159605 | 124204 | 485.17 | 1.778x |
| `level-c-generated-static` | writer | 150432 | 133377 | 521.00 | 1.887x |

结论边界：

- 这条 proof 证明 `AxeTensor/AxeLayout/AxeStorage -> lowering descriptor/address expression -> real tile-add device path` 的第一段链路可行。
- 它也证明真实 device profiler 能识别 `RTADD_LEVEL_C_GENERATED_STATIC_*` zones，并能按 critical stage 与 CB / Level B 对比。
- 它没有证明 firmware 已经跳过 `CBInterface` 初始化，也没有证明 LLK operand metadata 已经摆脱 `get_local_cb_interface(cbid)`。当前 compute helper 仍通过 `add_tiles(kCbIn0,kCbIn1,...)` 和 `pack_tile(...,kCbOut)` 复用 LLK 的 CB-derived operand ABI。
- 因此当前 Level C 进展应描述为 **IR/lowering/真实 device first-proof 已打通**，而不是完整 Level C ABI 已实现。

## Level C LLK Direct-Address Proof: Add + Matmul

用户提出 add 例子过于简单，因此本轮把 proof 扩展到 `real_matmul_protocol`。这个扩展不是复制 `programming_examples/matmul` 的教学 target，而是在现有 TTNN-style matmul profiler fork 中加入同类 Level C hook：它复用 `matmul_common/bmm_op.hpp` 的 blocking 和 `bmm_large_block_zm` compute 结构，同时保留 reader/writer/profiler/analyzer 的真实协议实验环境。

新增模式：

- `real_tile_add_protocol --mode=level-c-llk-direct`：手工展开 `_llk_unpack_AB_`、`_llk_math_eltwise_binary_`、`_llk_pack_`。
- `real_matmul_protocol --mode=level-c-llk-direct`：手工展开 `_llk_unpack_AB_matmul_`、`_llk_math_matmul_`、`_llk_pack_`。

关键设计：

- static input/output L1 ring、per-CB stream-register counter 和 writer/reader 同步逻辑沿用 Level B `static-streamreg-cbregs`。
- compute kernel 不再通过高层 compute API wrapper 推导当前 tile 地址；raw LLK 入口直接接收 compiler-lowered L1 地址表达式。
- raw LLK 地址遵循现有 TT-Metal wrapper 的 Tensix 约定：传入 `(l1_addr >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT) - 1`，tile size 传入 16B word count，而不是 byte count。
- matmul 第一版限制为 `K=64` / `num_blocks=1` / single-core active shape。原因是 `K>64` 会进入 partial-sum spill/reload，需要继续展开 `copy_tile(kCbInterm)` 等价 LLK 路径；如果第一步把 reload 也混入，无法分清主 matmul unpack/math/pack 是否已经可行。

复现命令：

```bash
conda run -n tt cmake --build build_Release \
  --target real_tile_add_protocol real_matmul_protocol -j8

TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rtadd_level_c_llk_direct_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=level-c-llk-direct --tiles=4 --num-pages=2 --repeats=1 --device-id=0

TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rmp_level_c_llk_direct_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=level-c-llk-direct --M=64 --N=64 --K=64 --num-pages=2 --repeats=1 --device-id=0
```

device-profiler 对比：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/level_c_tile_add_llk_direct_proof_2026_05_19 \
  --tiles 256 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb level-c-generated-static level-c-llk-direct

conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py \
  --out-dir /tmp/level_c_matmul_llk_direct_proof_2026_05_19 \
  --dims 64 \
  --Ks 64 \
  --num-pages 2 \
  --repeats 1 \
  --modes profiled-cb static-input-output-cbregs level-c-llk-direct
```

结论边界：

- 这一步回答的是“能不能先不改 firmware，证明 LLK unpack/math/pack 可以不依赖 CB wrapper 接收 compiler-lowered raw L1 address”。这是进入 firmware / launch descriptor 改造前的必要 proof。
- 它仍不是完整 Level C，因为 host 仍创建 `CircularBufferConfig`，firmware 仍初始化 `CBInterface`，LLK 的 format/tile shape setup 仍由现有 operand metadata bring-up 辅助完成。
- 当前 smoke/JIT/device profiler 已通过两个 proof case：
  - `real_tile_add_protocol --mode=level-c-llk-direct --tiles=4`：`max_abs_error=0`。
  - `real_matmul_protocol --mode=level-c-llk-direct --M=64 --N=64 --K=64`：`pcc=0.999049`，`max_abs_error=0.015625`。
- 当前 device-profiler 结果：

| Proof | Mode | Critical stage | Critical cycles | Delta vs CB |
|---|---|---|---:|---:|
| tile-add `tiles=256,num_pages=2` | `cb` | writer | 283532 | 0 |
| tile-add `tiles=256,num_pages=2` | `level-c-generated-static` | writer | 149239 | 134293 |
| tile-add `tiles=256,num_pages=2` | `level-c-llk-direct` | writer | 149072 | 134460 |
| matmul `M=N=K=64,num_pages=2` | `profiled-cb` | writer | 2703 | 0 |
| matmul `M=N=K=64,num_pages=2` | `static-input-output-cbregs` | writer | 2461 | 242 |
| matmul `M=N=K=64,num_pages=2` | `level-c-llk-direct` | writer | 2472 | 231 |

- 结论是：add 和低 K matmul 的 compute wrapper dependency 可以被移除，raw LLK direct-address path 可以正确运行并被 device profiler 观察到。
- 性能归因不能说“raw LLK 带来额外收益”。add 中 `level-c-llk-direct` 与 `level-c-generated-static` 基本持平，matmul 中 `level-c-llk-direct` 与 `static-input-output-cbregs` 基本持平。收益主要仍来自 static L1 ring/schedule 替代 CB FIFO 动态管理；raw LLK direct 的价值是证明 compiler-lowered address 可以穿过 LLK 边界。
- matmul 仍不能 broad promote：该 proof 只覆盖 `K=64,num_blocks=1`，不覆盖 partial-sum reload、多核 large GEMM、multicast reuse 等路径。

## Level C Firmware / Launch Descriptor Gate: Add + Matmul

本轮目标不再是继续证明 raw LLK 能跑，而是跨过 Level C 的 firmware / launch ownership boundary：experimental path 不能再把 host `CreateCircularBuffer`、launch `local_cb_mask` 和 firmware materialized `CBInterface[]` 当作 operand/queue metadata 的事实来源。

新增模式：

- `real_tile_add_protocol --mode=level-c-llk-direct-fw-skip-cb-init`
- `real_matmul_protocol --mode=level-c-llk-direct-fw-skip-cb-init`

实现边界：

- host path 在该 mode 下直接返回，不调用 `CreateCircularBuffer` / `CircularBufferConfig` 注册 `c_0`、`c_1`、`c_16`/`c_24`。
- 因为没有注册 local/remote CB，launch descriptor 的 `local_cb_mask=0`，remote CB range 为空。
- `tt_metal/hw/firmware/src/tt-1xx/brisc.cc`、`ncrisc.cc`、`trisc.cc` 改为只有在 `local_cb_mask != 0` 时才进入 `CBP_FW_LOCAL_CB_INIT` / `setup_local_cb_read_write_interfaces()`，只有在 remote range 非空时才进入 `CBP_FW_REMOTE_CB_INIT` / `setup_remote_cb_interfaces()`。
- kernel 的 L1 base、page size、format/tile words、tile shape 和 sync binding 来自 compile-time descriptor / runtime args / 显式 stream-register binding。fw-skip kernel variant 的 sync counter 地址用 `STREAM_REG_ADDR(OPERAND_START_STREAM + cbid, ...)` 展开，不再通过 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)` 作为事实来源。
- 本轮没有新增全局 `DISPATCH_MODE_COMPILER_L1` enum；这是刻意保守的实验切分。当前 proof 使用实验 executable 的 descriptor-empty/sentinel gate 先证明“不注册 CB 时 firmware 可以跳过 CB init 且 kernel 正确”。后续产品化可以再把这个 gate 提升为显式 launch mode。

复现命令：

```bash
conda run -n tt cmake --build build_Release \
  --target real_tile_add_protocol real_matmul_protocol level_c_fw_skip_cb_init_contract_sanity -j8

build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_fw_skip_cb_init_contract_sanity

TT_METAL_ALLOCATOR_MODE_HYBRID=1 \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=level-c-llk-direct-fw-skip-cb-init --tiles=4 --num-pages=2 --repeats=1 --device-id=0

TT_METAL_ALLOCATOR_MODE_HYBRID=1 \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=level-c-llk-direct-fw-skip-cb-init --M=64 --N=64 --K=64 --num-pages=2 --repeats=1 --device-id=0
```

device-profiler 对比：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/level_c_tile_add_fw_skip_cb_init_proof_2026_05_19_rerun \
  --tiles 256 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb static-streamreg-cbregs level-c-llk-direct level-c-llk-direct-fw-skip-cb-init

python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py \
  --out-dir /tmp/level_c_matmul_fw_skip_cb_init_proof_2026_05_19_rerun \
  --dims 64 \
  --Ks 64 \
  --num-pages 2 \
  --repeats 1 \
  --modes profiled-cb static-input-output-cbregs level-c-llk-direct level-c-llk-direct-fw-skip-cb-init
```

正确性：

| Proof | Mode | Result |
|---|---|---|
| tile-add `tiles=4,num_pages=2` | `level-c-llk-direct-fw-skip-cb-init` | pass；`max_abs_error=0` |
| matmul `M=N=K=64,num_pages=2` | `level-c-llk-direct-fw-skip-cb-init` | pass；`pcc=0.999049`，`max_abs_error=0.015625` |
| contract sanity | `level_c_fw_skip_cb_init_contract_sanity` | pass；`local_cb_mask=0`，`min_remote_cb_start_index=64`，即当前 Blackhole/host build 的 remote-empty sentinel。 |

firmware profiler 证据：

| Proof | Mode | `CBP_FW_LOCAL_CB_INIT` / `CBP_FW_REMOTE_CB_INIT` |
|---|---|---|
| tile-add | `cb` | present |
| tile-add | `static-streamreg-cbregs` | present |
| tile-add | `level-c-llk-direct` | present |
| tile-add | `level-c-llk-direct-fw-skip-cb-init` | absent |
| matmul | `profiled-cb` | present |
| matmul | `static-input-output-cbregs` | present |
| matmul | `level-c-llk-direct` | present |
| matmul | `level-c-llk-direct-fw-skip-cb-init` | absent |

device critical-stage 结果：

| Proof | Mode | Critical stage | Critical cycles | Delta vs CB |
|---|---|---|---:|---:|
| tile-add `tiles=256,num_pages=2` | `cb` | writer | 284846 | 0 |
| tile-add `tiles=256,num_pages=2` | `static-streamreg-cbregs` | writer | 159337 | 125509 |
| tile-add `tiles=256,num_pages=2` | `level-c-llk-direct` | writer | 149240 | 135606 |
| tile-add `tiles=256,num_pages=2` | `level-c-llk-direct-fw-skip-cb-init` | writer | 160302 | 124544 |
| matmul `M=N=K=64,num_pages=2` | `profiled-cb` | writer | 2660 | 0 |
| matmul `M=N=K=64,num_pages=2` | `static-input-output-cbregs` | writer | 2559 | 101 |
| matmul `M=N=K=64,num_pages=2` | `level-c-llk-direct` | writer | 2555 | 105 |
| matmul `M=N=K=64,num_pages=2` | `level-c-llk-direct-fw-skip-cb-init` | writer | 2575 | 85 |

结论：

- 这一步已经证明 experimental path 中 `CBInterface[]` 不再是 operand/queue metadata 的事实来源。host 没有注册 CB，launch descriptor 不再用 `local_cb_mask` 驱动该路径的 CB 初始化，firmware profiler 也证明 CB init path 被跳过。
- kernel correctness 通过，device critical stage 仍可与 legacy CB、Level B `static-streamreg-cbregs` 和旧 `level-c-llk-direct` 对比。
- fw-skip 相对 Level B / 旧 `level-c-llk-direct` 没有新的 steady-state writer critical 收益。tile-add 中 fw-skip 与 `static-streamreg-cbregs` 基本同级，但慢于旧 `level-c-llk-direct`；matmul 中 fw-skip 与 Level B / 旧 LLK-direct 只差几十 cycles。这里不能把差异解释为 firmware skip 回退，因为 fw-skip 当前把 L1 base/page/sync metadata 作为 runtime args 运输，而旧 `level-c-llk-direct` 仍是 compile-time protocol args upper-bound。真正结论是：firmware CB init 是 launch-time path，不是 steady-state writer loop 的主要成本。
- 因此本轮的价值不是“又快了一大截”，而是 ownership boundary 已经从 TT-Metal runtime/firmware CB materialization 切到 compiler descriptor / static schedule。性能大头仍来自 static L1 ring/schedule 替代 CB FIFO 动态管理。

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

## Production-Shaped Copy: Runtime 负例与 Compile-Time 转正

`real_copy_protocol` 路径：

```text
DRAM -> reader -> L1 staging ring -> writer -> DRAM
```

关键结果来自 `/tmp/real_copy_protocol_compiletime_ablation_2026_05_19`，所有模式 `max_abs_error=0`。这里的 delta 是 `CB critical - static critical`，正数表示 static 减少 cycles。

| Tiles | Pages | Grid | Static mode | CB critical | Static critical | Saved cycles/local tile | Speedup |
|---:|---:|:---:|---|---:|---:|---:|---:|
| 256 | 2 | 1x1 | static-runtime | 144021 | 171908 | -108.93 | 0.838x |
| 1024 | 2 | 1x1 | static-runtime | 573508 | 699254 | -122.80 | 0.820x |
| 4096 | 2 | 1x1 | static-runtime | 2293250 | 2809192 | -125.96 | 0.816x |
| 16384 | 2 | 1x1 | static-runtime | 9171088 | 11248931 | -126.82 | 0.815x |
| 256 | 2 | 1x1 | static-compiletime | 144021 | 160429 | -64.09 | 0.898x |
| 1024 | 2 | 1x1 | static-compiletime | 573508 | 639870 | -64.81 | 0.896x |
| 4096 | 2 | 1x1 | static-compiletime | 2293250 | 2558109 | -64.66 | 0.896x |
| 16384 | 2 | 1x1 | static-compiletime | 9171088 | 10230947 | -64.69 | 0.896x |

stream-register scratch follow-up：

| Tiles | Pages | Grid | Static mode | CB critical | Static critical | Saved cycles/local tile | Speedup |
|---:|---:|:---:|---|---:|---:|---:|---:|
| 256 | 2 | 1x1 | static-streamreg-scratch | 144021 | 145423 | -5.48 | 0.990x |
| 1024 | 2 | 1x1 | static-streamreg-scratch | 573508 | 584926 | -11.15 | 0.980x |
| 4096 | 2 | 1x1 | static-streamreg-scratch | 2293250 | 2343065 | -12.16 | 0.979x |
| 16384 | 2 | 1x1 | static-streamreg-scratch | 9171088 | 9376476 | -12.54 | 0.978x |
| 256 | 2 | 1x1 | static-streamreg-scratch-compiletime | 144021 | 138498 | +21.57 | 1.040x |
| 1024 | 2 | 1x1 | static-streamreg-scratch-compiletime | 573508 | 552851 | +20.17 | 1.037x |
| 4096 | 2 | 1x1 | static-streamreg-scratch-compiletime | 2293250 | 2210127 | +20.29 | 1.038x |
| 16384 | 2 | 1x1 | static-streamreg-scratch-compiletime | 9171088 | 8836642 | +20.41 | 1.038x |

解释：

- pure copy 不适合作为真实 layout op 收益的唯一证明点，因为它没有 tilize/untilize/transpose/slice/concat 的 layout freedom。
- L1 semaphore static runtime 明显慢于 CB；compile-time 地址常量只能把损失收敛到约 `-65 cycles/local tile`，说明 L1 semaphore/generic FIFO 同步仍是主要负担。
- runtime `static-streamreg-scratch` 把 L1 semaphore storage 换成 stream scratch register 后仍为 `-5` 到 `-13 cycles/local tile`，说明 CB baseline 本身已经是 stream-register backed counter，scratch 不是天然更快。
- `static-streamreg-scratch-compiletime` 保留 ready/consumed/start 同步但移除 runtime args 和动态 layout 配置后稳定转正，约 `+20 cycles/local tile`。这支持用户提出的修正：correctness synchronization 不应减少，但 runtime/config/generic FIFO 成本应该由 compiler 静态化。
- 下一步 data movement/layout 应看 tilize/untilize、transpose、slice、concat 这类可能受 lifetime/layout freedom 影响的路径，同时保留 `static-streamreg-scratch-compiletime` 作为 dataflow-only ablation。

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

## Level C Kickoff Plan

Level C 从现在开始不再扩大 broad operator-family survey，而是把 Level B 证据转化为 compiler ABI 和 firmware launch 原型。首个 proof point 固定为 **single-core simple elementwise / tile-add style path**，原因是它在 Level B 中收益最稳定、critical stage 清楚、正确性容易验证。KV-cache update 作为第二个真实 workload proof，不作为第一步 firmware/descriptor bring-up 的阻塞项。

Level C 的修改原则：

- 所有改动默认放在 experimental path / flag / sentinel 下，legacy TT-Metal CB path 必须保持可运行。
- 第一目标是证明“不依赖 TT-Metal CB descriptor / `CBInterface` / CB-derived LLK operand metadata”，不是追求最高 speedup。
- runtime args 可以继续传 per-core L1 address、work partition 和 descriptor table offset；不把 compile-time bake-in 作为目标。
- 每个 milestone 都必须能和 Level B 的 `cb`、`static-runtime`、`static-streamreg-cbregs` 做同 shape device-profiler 对比。

### Step 0: 保护分支并冻结 Level B baseline

目标：

- 当前 Level B 状态先本地 commit，保留 `/tmp/levelb_cbregs_completion_2026_05_19` 作为 baseline artifact。
- README 和本文档都指向 `static-streamreg-cbregs` 作为 Level B 主线。
- 在新分支上开始 Level C，避免和 Level B profiler fork 的收尾修改混在一起。

验收：

- `git diff --check` 和 Level B smoke 通过。
- 新分支的第一条 commit 只包含 Level B 文档/runner 收敛，不包含 Level C 源码改造。

### Step 1: 固化 compiler IR 与 lowering descriptor 边界

修改范围：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/docs
tt_metal/programming_examples/compiler_managed_l1_dataflow/include   # experimental prototype headers
```

目标：

- 在 `level_c_ir.hpp` 中定义 `AxeTensor`、`AxeStorage`、`AxeLayout`、`AxeIter`；在 `level_c_lowering_descriptors.hpp` 中定义 `TensorLayoutDescriptor`、`L1QueueDescriptor`、`OperandViewDescriptor`、`QueueSyncDescriptor`。
- `AxeTensor` 是 `element_type + storage + layout`，其中 storage 是 CuTe-style engine，必须明确存在，可表达 owned device allocation、owned mesh allocation、borrowed host storage、borrowed device address、view-of-storage 和 register fragment；layout 同时表达 domain 和 mapping，不再单独维护 `logical_shape`。`AxeLayout` 直接用 `d_iters`、`r_iters`、`o_entries` 三个 vector 表达 `D/R/O`；`O` 是单个 fixed coordinate offset，`o_entries` 是它的稀疏 entries，不携带 `schema_version` 或固定数组 count。其它 descriptor 字段只表达 compiler-owned L1 storage、queue topology/sync、operand view、format/tile metadata，是后续 lowering / dynamic-field carrier，不是新的 runtime CB object。
- 不出现 TT-Metal `CBIndex`、`CircularBufferConfig`、`TensorAccessorArgs` 作为 IR 语义。
- tensor shape 在高层 compiler IR 中来自 `AxeLayout` 的 domain，即 `D` iter 的 extent；`rank + dims[4]` 只用于 descriptor-table backend 的 lowered metadata，不代表最终 kernel 必须读取这个结构体。
- 完全静态的 shape/layout 应在 codegen 时消失，变成常量地址表达式、loop bounds、tile count 或 schedule；只有运行时仍动态的字段才进入 descriptor table。
- descriptor table 不直接嵌入 TT-Metal host-side `Shape` 或 `AxeLayout` 的 `std::vector`，也不使用带 `n/c/h/w` 语义的结构体；需要跨 host/device ABI 时应另行定义 serialized lowering form。
- compiler IR / descriptor schema 明确区分：
  - L1 storage/lifetime：base、size、bank policy、lifetime。
  - queue/sync：capacity、producer/consumer、sync kind、counter/register binding。
  - operand view：format、tile shape、page size、stride、pack/unpack role。
  - tensor layout：global tensor rank/dims、page layout、shard mapping、NoC address policy。
  - unified tensor/layout algebra：`AxeTensor{name, element_type, storage, layout}` + `AxeStorage{kind, memory_space, ownership, size/address/root/device scope}` + `AxeAxis{name, kind}` + `AxeIter{extent, stride, axis}` + `D(list)` / `R(set)` / `O(offset)`。

验收：

- `level_c_ir.hpp` 可被 host-side experimental compiler code include；`level_c_lowering_descriptors.hpp` 可被 host 和 kernel-side experimental lowering code include。
- 文档中 descriptor 字段和 header 保持一致。
- 不能只是把 `CircularBufferConfig` 字段原样改名；必须能表达 non-CB L1 storage 和 operand view。
- `level_c_descriptor_schema_sanity` 可以单独构建和运行，证明 `AxeTensor` / `AxeLayout` 是 vector/string-based IR，同时输出 lowering descriptor size 和一个示例 descriptor table size。
- `level_c_axe_layout_coverage_sanity` 可以单独构建和运行，覆盖 row-major、tile/face、interleaved、height/width/block/ND sharding、mesh replicated/sharded 和 TensorAccessor-style address expression。
- `level_c_tt_metal_layout_adapter_sanity` 可以单独构建和运行，覆盖 tech report / 源码中的 `TensorLayout`、`PageConfig`、`Tile`、`MemoryConfig`、`ShardSpec`、`ShardSpecBuffer`、`NdShardSpec`、`MeshBuffer`、`Buffer::view` 语义，并额外覆盖 `Buffer` owned allocation、view/root ownership、`MeshBuffer` owned allocation、`HostTensor` borrowed storage、externally owned device address、register fragment 等资源语义。

### Step 1.5: 证明 layout 语义覆盖边界

目标：

- 以 TT-Metal 当前 tech report 中的 tensor layout / memory layout / tensor sharding 为输入，证明它们都能被 Axe Figure 1 的 `AxeIter(extent, stride, axis)`、`D(shard list)`、`R(replica set)`、`O(offset)` 表达。
- 把 `page` 从高层语义中降级为 block/atom/storage unit：它可以继续出现在 lowered metadata 中，但不再是独立于 layout algebra 的 compiler IR 概念。
- axis 在 compiler IR 中是 `AxeAxis{name, kind}` 对象；它不是 device runtime object。kernel lowering 只消费 iter 的 extent/stride/axis 和 `D/R/O` 归属生成地址算式，必要时再把 axis lowering 成紧凑 id 或固定数组。
- 明确资源层边界：`AxeLayout` 不单独替代 `Buffer`、`MeshBuffer`、`Tensor`；这些对象中的 allocation、ownership、lifetime、device/API interop 语义应由 `AxeStorage` 表达，最终由 `AxeTensor + AxeStorage + AxeLayout` 统一承载。
- 以 adapter 方式逐步迁移 `TensorLayout`、`PageConfig`、`Tile`、`ShardSpec`、`ShardSpecBuffer`、`BufferShardingArgs`、`TensorAccessorArgs`，避免一次性改动 TTNN public API。

验收：

- `level_c_axe_layout_coverage_sanity` 输出所有 coverage case 为 `covered`。
- `level_c_tt_metal_layout_adapter_sanity` 输出所有 adapter case 为 `covered`，并报告 `layout_covered_cases=14`、`resource_covered_cases=6`、`covered_cases=20`。
- README 和本文档明确写出“`AxeLayout` 统一 layout mapping，`AxeStorage` 统一 resource/storage engine，`AxeTensor` 统一 tensor 语义；当前 proof 不等价于 TTNN public API 已经删除”的边界。
- 下一步 adapter 设计必须能从现有 TT-Metal/TTNN 类型生成 `AxeLayout`，并能从该 IR 生成 TensorAccessor 等价地址表达式。

### Step 1.6: TT target dialect v0

新增文档：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/docs/tt_target_dialect_design_2026_05_19.md
```

目标：

- 把 Level C 的底层 dialect 定义为硬件动作 dialect，而不是 TT-Metal CB / TensorAccessor 的改名版本。
- 分清三层：`AxeTensor/AxeLayout/AxeStorage` 表达 tensor/layout/storage；schedule / lowering IR 表达 per-core work、L1 lifetime、ring slot 复用和 producer/consumer 依赖；`tt.target` 表达 NoC、stream register、L1 地址、tile register、unpack/math/pack。
- `TensorAccessor`、`CircularBuffer`、`CircularBufferConfig`、`CreateCircularBuffer`、`cb_*`、`get_local_cb_interface`、`CBInterface[]` 只能作为 legacy adapter 输入或 baseline 对照，不能进入 canonical bottom IR。
- v0 op set 只覆盖 add / matmul 已验证 proof：`tt.target.addr.*`、`tt.target.noc.*`、`tt.target.reg.*`、`tt.target.wait.*`、`tt.target.tile.*`、`tt.target.unpack.*`、`tt.target.math.*`、`tt.target.pack`。

验收：

- `level_c_tt_target_dialect_sanity` 可以单独构建和运行。
- checker 内置 tile-add 和 real-matmul pseudo IR，并验证没有 legacy CB/TensorAccessor token、reader/compute/writer stage 存在、producer/consumer token 配对、ring slot 地址可静态求值、关键 `tt.target` op family 全部覆盖。

### Step 2: host-side descriptor table 原型

修改范围：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow
tt_metal/impl/program/dispatch.cpp                 # 只加 experimental side-table path
tt_metal/hw/inc/hostdev/dev_msgs.h                 # 只在需要 launch flag/offset 时加 experimental 字段
```

目标：

- 选择 single-core tile-add / binary elementwise 作为第一条 host path。
- 不调用 `CreateCircularBuffer` / `CircularBufferConfig` / `TensorAccessorArgs`。
- compiler-side helper 生成 per-core L1 allocation 和 operand views。
- dispatch 把 compiler descriptor table side-load 到每个 core 可见的 kernel config/L1 区。
- runtime args 只传 descriptor table offset、per-core work range、必要的 tensor base addresses；不传 TT-Metal CB ids。

验收：

- `local_cb_mask=0` 或 descriptor-only sentinel 时，host 仍能启动 experimental kernel。
- legacy TT-Metal CB examples 不受影响。
- grep host experimental path 不出现 `CreateCircularBuffer`、`CircularBufferConfig`、`TensorAccessorArgs`。

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
- 加载或搬运 compiler lowering 所需 metadata，不初始化 compiler-owned runtime queue object。
- legacy mode 保持原 CB init 路径。
- firmware 不再以 `local_cb_mask` 决定 compiler-managed operand/queue metadata；`local_cb_mask` 只允许作为 legacy fallback 或 disabled sentinel。

验收：

- firmware profiler 中 compiler mode 下 `CBP_FW_LOCAL_CB_INIT` / `CBP_FW_REMOTE_CB_INIT` 为空或不执行有效 work。
- legacy CB target 仍能跑。
- experimental target 不依赖 `local_cb_mask` 驱动 `CBInterface[]` 初始化。
- 这是 Level C 的第一条硬边界：如果 firmware 仍 materialize `CBInterface[]` 作为事实来源，只能算 Level B+，不能算 Level C。

### Step 4: no-`cb_*` L1Queue codegen backend

修改范围：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/level_c
tt_metal/programming_examples/compiler_managed_l1_dataflow/include
```

目标：

- single-core tile-add / binary elementwise。
- reader/writer/compute 从 compiler lowering metadata 或展开后的 codegen 获取 L1 base、page size、tile count、sync binding。
- kernel 中禁止 `cb_reserve_back`、`cb_wait_front`、`cb_push_back`、`cb_pop_front`、`get_read_ptr`、`get_write_ptr`、`get_tile_size`。
- compute 先允许继续走 LLK-compatible path，但 operand metadata 的事实来源必须是 compiler OperandView/L1Queue lowering，不是 TT-Metal CB。

验收：

- 输出正确。
- grep experimental kernel 不出现 TT-Metal CB API。
- profiler 能产生四条对比：legacy `cb`、Level B `static-streamreg-cbregs`、compiler descriptor + legacy LLK-compatible path、compiler descriptor + firmware skip-CB-init。

### Step 5: LLK-compatible OperandView

目标：

- 让 `pack_tile`、`copy_tile`、`binary_tiles_init`、`matmul_tiles_init` 的 operand metadata 来自 `OperandViewDescriptor`。
- 优先实现 binary add 所需的两个 input operand 和一个 output operand。
- 保留 LLK 硬件细节处理，先不重写 mini-LLK。

验收：

- 正确性通过。
- profiler 能区分 descriptor 替换收益和 LLK 重写收益；本阶段不允许把两者混成一个数字。
- `get_local_cb_interface(cbid)` 不再是 operand format/page/ptr metadata 的事实来源。

### Step 6: Level C first profiler fork

目标：

- 建立 `level_c_tile_add_protocol` 或等价 target。
- 同 shape 比较：
  - `cb`
  - `static-runtime`
  - `static-streamreg-cbregs`
  - `compiler-descriptor-fw-cb-init-skip`
  - `compiler-descriptor-operandview`
- 输出 host/firmware/kernel steady-state 分层 profiler 数据。

验收：

- 正确性 `max_abs_error=0`。
- device critical stage 和 saved cycles/work 可与 `/tmp/levelb_cbregs_completion_2026_05_19` 对照。
- 如果收益不超过 Level B，也要记录为有效结论：Level C 的价值可能在 ABI/lifetime/schedule，而不是单 kernel steady-state。

### Step 7: mini-LLK / raw Tensix backend

目标：

- 只在 Step 3/5/6 稳定后，对单个 microkernel 直接生成或调用 raw Tensix instruction / MOP / config sequence。
- 优先 unary/binary elementwise 或 copy/layout microkernel。
- 不以完整 TTNN op 覆盖为第一目标。

验收：

- raw Tensix microkernel 正确。
- 性能归因能区分 CB/firmware/descriptor 替换收益和 LLK 重写收益。
- 如果 raw Tensix 结果无法归因，不作为 Level C thesis proof。

### Step 8: operator-family migration

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
| M1: Compiler IR / descriptor schema | IR 不包含 TT-Metal CB/TensorAccessor 语义；`AxeTensor` 固定为 element type + storage + layout；lowering descriptor 能表达 L1 storage、queue、operand view。 | 只把字段从 `CircularBufferConfig` 改名，或在 tensor 里维护第二份 logical shape。 |
| M2: Host path replacement | 新 compiler path 不调用 `CreateCircularBuffer` / `TensorAccessorArgs`。 | 只在 kernel 里绕开 `cb_*`。 |
| M3: Firmware mode | compiler mode 下 firmware 不走 TT-Metal CB init，或 CB init 只作为 legacy fallback。 | `local_cb_mask` 仍驱动 `CBInterface` 初始化。 |
| M4: Kernel L1Queue codegen | reader/writer/compute 不使用 TT-Metal CB API，正确性通过；queue lowering 后是具体地址/同步代码。 | 只把 `cb_*` 包一层 wrapper，或引入新的 runtime CB struct。 |
| M5: LLK-compatible OperandView | LLK 的 operand metadata 来自 compiler descriptor，而不是 TT-Metal CB state。 | 仍依赖 `get_local_cb_interface(cbid)` 作为事实来源。 |
| M6: Operator policy | 能按 family/shape 给出 backend selection rule。 | 用一个全局平均 speedup 宣称所有 op 都应替换。 |
| M7: mini-LLK | raw Tensix backend 对选定 microkernel 正确且可归因。 | 结果无法区分来自 LLK 重写还是 CB/firmware 替换。 |

## 研究产出定位

这条线最终要产出的不是“某个 op 快了多少”的局部优化报告，而是一个新的编译器后端论证：

- 抽象层：TT-Metal runtime-managed CB / TensorAccessor 不应是 compiler IR 的核心抽象。
- ABI 层：firmware launch 应接收 compiler lowering metadata / descriptor table，而不是只认识 `local_cb_mask` 和 CB config blob。
- Backend 层：compiler 可以根据 op family 和 shape 选择 static schedule、stream register、L1 counter、legacy CB fallback 或 raw Tensix backend。
- 评估层：收益要拆成 host/firmware/kernel steady-state/compute operand setup，而不是只看 end-to-end latency。

当前所有 profiler 收益/负例都服务于这个 thesis：**哪些 TT-Metal runtime-managed 数据流职责应该被提升到 compiler，并且这个提升在什么 op family、shape 和硬件路径上值得做。**
