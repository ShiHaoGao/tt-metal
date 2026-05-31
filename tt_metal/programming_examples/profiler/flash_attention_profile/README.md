# FlashAttention / SDPA Profile

这个目录是唯一的 FlashAttention/SDPA 性能实验目录，也是
`matmul_variants_profile` 的 FlashAttention/SDPA 对应版本。它的目的不是先发明一个新算子，
而是把 TTNN 现有 SDPA 的真实 host/device 路径复制到 profiler example 中，建立同一进程、
同一 shape、同一计时边界下的官方 TTNN baseline 和 copied fork 对照。后续所有
reader/compute/writer 或 host 调度实验都在本目录逐步迭代，不再维护第二套
`flash_attention_profile_experiments` 目录或 binary。

## 复制的是哪个 TTNN SDPA 版本

TTNN 里现在有多类 attention 路径，至少包括：

- `ttnn/cpp/ttnn/operations/transformer/sdpa/`
  - 普通 prefill / full-sequence SDPA。
  - public API 是 `ttnn::transformer::scaled_dot_product_attention`。
  - 底层 device op 是 `ttnn::prim::sdpa`。
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/`
  - decode 阶段专用路径，不是这里的主目标。
- `ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/`
  - windowed/sliding-window 变体，不是这里的主目标。
- `sdpa/device/*joint*`、`*ring*`
  - 多设备或 joint/ring 变体，不是这里的主目标。

本目录复制的是最通用、最常见的 prefill/full SDPA 路径：

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.*`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.*`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
- 相关 shared kernel helper，如 `compute_common.hpp`、
  `dataflow_common.hpp`、`sdpa_streaming_qktv.hpp`。

复制后的 namespace 是 `ttnn::prim::flash_attention_profile_sdpa`，kernel 路径指向本目录下的
`kernels/`。因此 `copied_sdpa` 和 TTNN baseline 的算法/tiling 仍然是同一类东西；当前差异主要是
instrumentation、kernel path、host wrapper 边界，不应把它误读成已经完成的新 FlashAttention 算子。

chunked/paged prefill 也通过同一个 copied
`ttnn::prim::flash_attention_profile_sdpa::sdpa` 路径覆盖。
`copied_chunked` 使用 page table 和 device `chunk_start_idx` tensor。
它不是 `sdpa_decode`。

## 变体

- `ttnn_sdpa_baseline`
  - 官方 `ttnn::transformer::scaled_dot_product_attention`。
  - 这是常见 full prefill SDPA 的主基线。
- `copied_sdpa`
  - 本目录复制出来的 `ttnn::prim::sdpa` device op + local kernels。
  - 这是后续优化的主要 fork 点。
- `ttnn_chunked_baseline`
  - 官方 `ttnn::transformer::chunked_scaled_dot_product_attention`。
  - 用于长上下文 paged K/V prefill。
- `copied_chunked`
  - 本目录复制出来的 chunked/paged prefill 路径。
  - 支持 `trace`，因为 chunk start 是 device tensor，可以在 trace replay 时更新。

## Build

从 `third_party/tt-metal` 目录运行：

```bash
cmake --build build_Release --target flash_attention_profile --parallel $(nproc)
```

如果机器上的 `/tmp` 空间不足，OpenMPI 可能会在启动前失败。此时给运行命令加上：

```bash
export TMPDIR=/wafer/gsh/tmp/fa_profile_tmp
export OMPI_MCA_orte_tmpdir_base=/wafer/gsh/tmp/fa_profile_tmp
export PRTE_MCA_prte_tmpdir_base=/wafer/gsh/tmp/fa_profile_tmp
mkdir -p /wafer/gsh/tmp/fa_profile_tmp
```

## Run

快速 correctness + smoke：

```bash
./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset smoke --variant all --mode prepared --warmup 1 --iters 1 \
  --check-correctness --no-device-profiler-read
```

2K full SDPA 主基线：

```bash
./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset llama_prefill_2k --variant all --mode prepared --warmup 2 --iters 5 \
  --no-device-profiler-read
```

设备阶段 profile：

```bash
rm -rf generated/profiler/.logs
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset llama_prefill_2k --variant copied_sdpa --mode prepared \
  --warmup 1 --iters 1
```

做细粒度 profile 前先清空 `generated/profiler/.logs`，否则
`ReadMeshDeviceProfilerResults` 可能会因为 source location hash 冲突直接失败。

可选参数：

- `--preset`:
  `smoke`, `llama_prefill_2k`, `llama_prefill_2k_q128_k128`,
  `llama_prefill_2k_q256_k256`, `llama_prefill_16k_chunked`,
  `llama_prefill_16k_chunked_q128_k128`,
  `llama_prefill_16k_chunked_q256_k256`, `all`
- `--shape B,H,KVH,S,D,prefill,q,k,page`
- `--chunks Q,K`：覆盖所选 preset/custom shape 的 q/k chunk size，用来扫分块。
- `--experiment copied|split_compute_v1|llk_microflow_v1`
  - `copied`：默认路径，运行当前 copied TTNN SDPA fork。
  - `split_compute_v1`：运行独立 `host/split_compute_sdpa/device/` +
    `kernels/split_compute_sdpa/` 路径。v1 先把 producer SDPA compute 放在
    生产者 core grid 上，并额外挂一个独立 consumer probe compute kernel 到保留的一个
    core 上，用于验证“两个 compute kernel / producer-consumer program boundary”
    能否被 TT-Metal program 和 profiler 可靠表达。它还没有把 QK 中间结果真正交给
    consumer 做 `PV`，因此不要把 v1 性能当作最终 split 方案结论。
  - `llk_microflow_v1`：运行独立 `host/llk_microflow_sdpa/device/` +
    `kernels/llk_microflow_sdpa/` 路径。v1 保持数值路径不变，先在
    `fa3_pipe_detail` 下增加更贴近 LLK 片段的 profile zone，把
    `exp_max_diff`、previous-group SALAD、drain SALAD 和 `PV` group matmul 拆开。
    目标是找出后续应该移动哪一个小片段，而不是再把整段 `PV` 粗粒度塞进 QK phase。
- `--variant ttnn_sdpa_baseline|ttnn_chunked_baseline|copied_sdpa|copied_chunked|all`
- `--mode eager|prepared|prepared_no_q_copy|trace|all`
- `--pipeline auto|stream_h1|qktv_h1|salad_first|qktv_h1_salad_first|non_streaming`
  - `auto`：当前 copied SDPA 默认路径。
  - `stream_h1`：只作用于 copied 变体，强制 streaming output drain group height 为 1。
  - `qktv_h1`：只作用于 copied 变体，强制 Phase-2 `QKTV` matmul/drain row group height 为 1。
  - `salad_first`：只作用于 copied 变体，在最后 K chunk 上尝试更早做 SALAD+normalize+push。
  - `qktv_h1_salad_first`：同时启用 `qktv_h1` 和 `salad_first`。
  - `non_streaming`：只作用于 copied 变体，关闭 streaming compute，作为对照组。
  - 非 `auto` 时官方 TTNN baseline 会被跳过，correctness 仍用官方 baseline 输出对比 copied 输出。
- `--pipeline-depth N`
  - 只作用于 copied 变体，改变 streaming output CB 的 row-group slot 数。
  - 默认值是 2，也就是复制 TTNN 路径时的原始深度。
  - 非默认值时官方 TTNN baseline 会被跳过，避免把 copied-only 编译参数误套到官方路径。
- `--qk-subblock H,W`：只作用于 copied 变体，覆盖 QK matmul 的
  `qk_out_subblock_h/w`，用于扫 DST subblock 形状。
- `--q-buffer-factor N`：只作用于 copied 变体，覆盖 Q 输入 CB buffer factor。
- `--dst-full-sync` / `--dst-half-sync`：只作用于 copied 变体，覆盖 compute
  kernel 的 DST sync 模式。
- `--qk-softmax-profile none|wait_max|sub_math|wait_sub|exp_sfpu|pack`
  - 只作用于 copied compute kernel 的 QK softmax/sub-exp 路径。
  - 一次只编译一个子阶段；如果把所有子 zone 同时打开，Blackhole 上 compute kernel 会超过
    TENSIX kernel config buffer 上限。
- `--qk-softmax-schedule before_matmul|after_matmul|after_matmul_except_final_kt`
  - 只作用于 copied compute kernel 的 QK phase。
  - `before_matmul` 是默认顺序：`softmax(prev q_subblock) -> QK matmul(cur q_subblock)`。
  - `after_matmul` 是实验顺序：先做当前 QK matmul，再延后 previous row softmax。
  - `after_matmul_except_final_kt` 只在非 final kt 使用 matmul-first；
    final kt 保持 `softmax -> matmul -> max_reduce`，用于验证 reduce handoff 假设。
- `--qk-detail-profile none|qk_threads|q_reader|qk_init|phase_timeline|qktv_detail|qktv_barrier_split|qktv_matmul_detail|qktv_math_detail|qktv_pack_detail|qktv_pipeline_detail|writer_pipeline_detail|first_output_handoff|fa3_pipe_detail|llk_microflow_detail`
  - 只作用于 copied profiler fork，一次只打开一类细粒度测点，避免把 kernel code size 推爆。
  - `qk_threads`：把 QK body 中发给 `UNPACK`、`MATH`、`PACK` 的真实执行段按 TRISC 拆开。
  - `q_reader`：把 reader 侧 Q subblock path 拆成 reserve、NOC read、read barrier、push。
  - `qk_init`：把 `FAP_COMPUTE_QK_INIT` 拆成 format reconfig、`mm_no_mop_init_short`、
    pack width 配置。
  - `phase_timeline`：打开 QK/QKV phase boundary、QK row、KT step、partial handoff
    等时间线测点。它不会同时打开 QK body 的 thread 细分，否则 profiler kernel 会超过
    TENSIX kernel config buffer。
  - `qktv_detail`：打开 Phase 2 / `QKT@V` 相关的高价值测点：
    pack/unpack barrier、final-row sub-exp drain、split-drain matmul kt、
    `QKTV_MATMUL_PACK`、V subblock、QKT/V wait。reserve/reconfig/mm-init/pack-width
    这类小片段不保留，避免再次超过 TENSIX kernel config buffer。
  - `qktv_barrier_split`：只保留 `FAP_QKTV_BARRIER_TOTAL`，用于确认
    in-place softmax PACK 写入到 `QKT@V` UNPACK 读取之间的显式 semaphore
    handoff 是否真是瓶颈。
  - `qktv_matmul_detail`：只拆 `QKT@V` matmul group 的线程级路径：
    `UNPACK`、`MATH`、`PACK` 和 V subblock。
  - `qktv_math_detail`：只拆 `QKT@V` matmul math 侧 enclosure：
    tile regs acquire/commit/wait/release、matmul loop、UNPACK/MATH。
  - `qktv_pack_detail`：只拆 `QKT@V` pack 侧 enclosure：
    pack rows 和 pack thread。该档位不和 math 细分同时打开。
  - `qktv_pipeline_detail`：只拆 Phase 2 流水线时间线，包括 group0
    split-drain、main group `QKT@V`、`exp_max_diff` 和 final drain。
    它不同时打开 `QKT@V` body 线程细分；V/QKT wait、SALAD、row norm
    继续使用常驻 zone。
  - `writer_pipeline_detail`：只拆 streaming writer 的 row-group drain，
    区分首个 row group 和后续 row group 的 wait/store/flush。
  - `first_output_handoff`：只拆首个 writer-visible output 的 compute
    handoff 链路，包括 group0 drain+`QKT@V`、group0 后 barrier、
    next group `QKT@V`、首个 `exp_max_diff`、首个 SALAD 和首个
    row normalize。它用于判断首个 output 为什么不能更早 push 给 writer。
  - `fa3_pipe_detail`：只拆 FA3-style 交错实验关心的时间线，包括
    group0 drain、当前 softmax、提前 `PV`、后续 `PV`、softmax state、
    handoff gap 和首个 output push。它用于判断把 previous-row `PV`
    插入 QK phase 后是否真的形成 overlap。
  - `llk_microflow_detail`：只保留 `llk_microflow_v1` 小片段实验需要的
    LLK/microflow zone，避免 `fa3_pipe_detail` 在 16K q128/k256 profile 下超过
    TENSIX kernel config buffer。
- `--device-profiler-by-risc`
  - 在普通 `FLASH_ATTN_PROFILE_STAGE_RESULT` 之外，额外输出
    `FLASH_ATTN_PROFILE_STAGE_RISC_RESULT`，按 `RISC processor type` 分组。
- `--q-reader-schedule default|first_before_k|first_during_k_read`
  - 只作用于 copied reader kernel 的 `use_q_subblock_push` 路径。
  - `default`：K read/push 之后再推 Q subblocks。
  - `first_before_k`：第一个 Q subblock 在 K loop 之前读，定位 Q wait 来源。
  - `first_during_k_read`：chunked/paged 第一段 K read 的最后 barrier
    前发射第一个 Q subblock，验证是否能把 Q NOC read 藏进 K read 尾部。
- `--qk-first-body-warmup none|tiny_matmul|same_config_init`
  - copied-only 首段 QK body 实验旋钮。
  - `tiny_matmul`：正式 QK loop 前跑一个 1x1 scratch matmul，验证 math/pack 首段 warmup。
  - `same_config_init`：第一次等待 K/Q 前预置真实 QK matmul 配置，第一次
    `QK_INIT` 直接复用该配置。
- `--compute-pipeline-schedule SCHEDULE`
  - local SDPA fork 的 compute pipeline 实验旋钮，不作用于官方 TTNN baseline。
  - `default`：保持 copied TTNN SDPA 的当前顺序。
  - `partial_handoff_v1`：当上一 row group 的 QK/softmax/max reduce 已完成后，
    在 Phase 1 内提前做它的 `QKT@V`，验证是否能缩短 Phase 1/Phase 2 边界。
  - `qktv_drain_all_then_matmul`：只改变 Phase 2 第一行 group。先 drain 完最后一行
    所有 kt 的 `sub_exp`，再做一次完整 inner-dim 的 `QKT@V`，验证 split-drain
    的分段 matmul/accumulation 是否是瓶颈。
  - `group0_early_push_v1`：只在 last K chunk 上，把 group0 的
    SALAD+normalize+push 提到 next-group `QKT@V` 之前。它是比
    `salad_first` 更窄的 copied-only 实验，只验证“首个输出更早可见”是否能降低
    writer wait。
  - `fa3_pv_softmax_v1`：FA3-style 负实验。当前 row group 做 QK 时，
    previous row group 的每个 kt softmax 分片完成后，立刻尝试做该分片的
    partial `P@V`，最后在 Phase 2 跳过已提前完成的 row group。它只用于验证
    “把 previous `PV` 塞进 QK phase”能否让 FPU/MATH 更连续；当前数据证明这个
    粗粒度做法会拉长 critical path。
  - `llk_drain_exp_before_pv_v1`：`llk_microflow_v1` 专用，把 group0 drain 的
    `sub_exp` 全部放到 `PV` matmul 前，验证 split-drain 局部性交错是否必要。
  - `llk_prev_exp_after_pv_v1`：`llk_microflow_v1` 专用，把 previous-group
    `exp_max_diff` 延后到当前 `PV` matmul 后，验证 softmax state 是否能从
    `PV` 前路径挪开。
  - `llk_salad_before_pv_v1`：`llk_microflow_v1` 专用，把 previous-group
    SALAD 放到当前 `PV` matmul 前，验证 SALAD 是否应当更早完成。
  - `llk_v_ready_prefetch_v1`：`llk_microflow_v1` 专用，在 QK phase 内更早等待
    V ready，验证 V wait 是否能被藏起来。
  - `split_signal_only_v1`：`split_compute_v1` 专用，只做 producer writer 到
    consumer dataflow 的信号链路。producer core0 在启动和首个 output chunk 写出后
    发 token；consumer dataflow 只等 semaphore，不再额外启动 consumer compute。
    它不搬 QK/P 数据，只测 split handoff 的基础同步成本。
  - `split_output_stream_signal_v1`：`split_compute_v1` 专用，producer core0
    对自己写出的每个 output chunk 都发 ready token。它用于测连续小粒度 token
    handoff，而不是只看首个 output。
  - `split_l1_ready_signal_v1`：`split_compute_v1` 专用，writer 在
    `cb_out` row group 已经可读、但还没有写 DRAM 前发 token。它用于测 final
    normalized output 在 L1 可见是否明显早于 writer-after-store。
  - `split_state_ready_signal_v1`：`split_compute_v1` 专用，producer compute 在
    final K chunk 的 QK/softmax state ready 后先向本地 writer CB 发 token，
    再由 writer 转发给 consumer。它用于测把 downstream consumer 接到 softmax
    state 后、而不是接到 final output 后，能提前多少。
  - `split_state_consumer_probe_v1`：`split_compute_v1` 专用，在
    `split_state_ready_signal_v1` 的 state-ready token 后，让 consumer compute
    执行一个 `PV` 形状的 dummy matmul probe。这个 probe 不写最终 output，只测
    第二 compute kernel 的工作能不能被 producer 后续窗口隐藏。
  - `split_state_consumer_probe_x4_v1` /
    `split_state_consumer_probe_x8_v1`：同一 consumer probe 的 4x/8x stress
    版本，用来测隐藏窗口容量边界。
  - `split_state_consumer_vprefetch_x8_v1`：`split_compute_v1` 专用，consumer
    在 start token 后先读取真实 `V` tiles 到本地 CB，再等 state-ready token 做
    x8 `PV` probe。它测“V 可提前预取”时的真实 V 搬运成本。
  - `split_state_consumer_vafter_state_x8_v1`：`split_compute_v1` 专用，consumer
    等 state-ready token 到达后才读取真实 `V` tiles，再做 x8 `PV` probe。它测最保守
    情况下 V 读是否会进入 critical path。
  - `split_state_real_p_vprefetch_x8_v1`：`split_compute_v1` 专用，把 dummy `P`
    换成 producer softmax 后的真实 `P` tiles。producer 在 group0 softmax 完成后把
    `P` pack 到 state mailbox，consumer 先预取真实 `V`，state-ready 后从 producer
    L1 remote-read `P`，再做 x8 `PV` probe。这个实验仍然保留 producer 原本的
    `PV`，所以它测的是 handoff 成本和 overlap 空间，不是最终 split-PV 正确性路径。
  - `split_state_real_p_vafter_state_x8_v1`：同上，但 consumer 等 state-ready 后才读
    真实 `V`，用于测最保守的 real-`P` + real-`V` critical path。
  - `split_state_real_p_kt_stream_v1`：`split_compute_v1` 专用，把 real-`P`
    handoff 从整块 `P` 降到 `kt_subblock` packet。producer 每个 packet 的
    `sub_exp` 完成后 pack 到 `c_15`，writer 逐包发 state-ready，consumer
    逐包做小 `P_kt@V_kt` probe。producer 仍保留原始输出路径，所以该实验支持
    correctness smoke。
  - `split_pv_owner_probe_v1`：`split_compute_v1` 专用的 profile-only 上界实验。
    producer 跳过 group0 的本地 `PV`，consumer 做 real-`P/V` probe，但 consumer
    输出还没有接回最终 output tensor，因此这个 schedule 不支持 correctness 结论；
    它只用于估算“如果 `PV` 真迁出 producer”能省多少 producer critical path。
  - `split_state_mailbox_ring_v1`：`split_compute_v1` 专用，在
    `split_state_real_p_kt_stream_v1` 的基础上把 `c_15` real-`P` mailbox 从单槽
    一包一 ack 改成最多 4-slot ring。writer 对一批 packet 先发 ready token，
    consumer 按 ring slot remote-read，writer 延迟到批末再等 ack 和 pop。producer
    仍保留原始输出路径，所以该实验支持 correctness smoke。
  - `split_pv_owner_output_v1`：`split_compute_v1` 专用，把
    `split_pv_owner_probe_v1` 推进成最小 correctness-capable 路径。producer
    只在非 chunked causal、`q_chunk0` 只有一个 K chunk 的情况下交出
    core0/head0/q_chunk0 的第一个 row group：handoff real `P` 和 `sum`，跳过
    该 row group 的本地 `PV` 与 writer 输出；consumer 读取 real `P/sum/V`，
    做 `P@V`、normalize，并直接写回同一个 output tensor。它是窄作用域
    ownership 实验，不是当前最快配置。
  - `split_pv_owner_output_no_ack_v1`：`split_compute_v1` 专用，保持
    `split_pv_owner_output_v1` 的同一 first-row-group ownership，但 writer
    发出 state-ready 后不再等待 consumer ACK，consumer remote-read `P/sum`
    后也不回 ACK。它用于验证 `c_15` handoff 的 ack/backpressure 是否还值得优化；
    当前 q128/k128 correctness 已通过，但它不是最快配置。
  - 除 `split_*` 信号实验外，这个模式只允许 streaming compute、默认
    SALAD handoff，并要求 `QKTV` row group height 等于 QK row group height；
    不满足条件时直接报错，不做 fallback。
- `--grid-policy default|copied_balanced_q`
  - `default`：不改 grid 选择。
  - `copied_balanced_q`：只作用于 copied 变体，按 shape 自动选择更规则的 Q chunk split。
- `--grid X,Y`：显式覆盖 `SDPAProgramConfig.compute_with_storage_grid_size`，优先级高于 `--grid-policy`。
- `--high-precision`
- `--check-correctness`
- `--no-device-profiler-read`

## 时间边界

`FLASH_ATTN_PROFILE_RESULT` 是 host-side wall time，不等于纯 kernel 时间：

- `eager`
  - 每次 measured iteration 里创建 Q / chunk-start runtime tensor，然后调用 op 并同步。
- `prepared`
  - K/V/page-table 常驻，measured iteration 里复制 Q、复制 chunk-start，然后调用 op 并同步。
- `prepared_no_q_copy`
  - K/V/page-table/Q/chunk-start 都常驻，measured iteration 不复制 Q，也不复制 chunk-start。
  - chunked 模式下固定 `chunk_start=0`，所以它是 host-copy ablation，不是完整长上下文 chunk sweep 的端到端替代。
- `trace`
  - 只支持 chunked 变体；先 capture 一次 chunked SDPA，再在 measured iteration 里复制 Q / chunk-start、
    replay trace、同步。

字段含义：

- `avg_ms`：整个 measured iteration。
- `call_avg_ms`：op call 或 trace replay 的 host 调用时间。
- `sync_avg_ms`：`distributed::Synchronize` 等待设备完成的时间。
- `copy_q_avg_ms`：measured iteration 中 Q runtime input copy。
- `copy_start_avg_ms`：chunk-start tensor copy，非 chunked 为 0。
- `grid`：`auto` 表示使用设备默认 compute grid；`8x8` 等表示通过 `--grid X,Y` 指定。
- `grid_policy`：`default` 表示不改 grid policy；`copied_balanced_q` 表示 copied 变体会自动选 grid。
- `pipeline_depth`：copied streaming output CB 深度；官方 TTNN 路径只使用默认值 2。

`FLASH_ATTN_PROFILE_STAGE_RESULT` 来自 `generated/profiler/.logs/profile_log_device.csv`，
是 device profiler zone，不是 host wall time。当前 copied kernels 先有粗粒度 zone，
再有更细的内部拆分：

- `FAP_READER`
- `FAP_COMPUTE`
- `FAP_WRITER`

## 当前最快 vs baseline 汇总

本节维护“当前最快 copied 配置”相对官方 TTNN baseline 的单页入口。
更新时只使用 correctness 通过、同一 shape、同一计时边界的结果。

当前 copied tuned 配置：

```text
q=128,k=256,grid=8x8,pipeline_depth=2,pipeline=auto,
qk_softmax_schedule=before_matmul,
q_reader_schedule=default,qk_first_body_warmup=none,
compute_pipeline_schedule=default
```

刷新状态：2026-05-30 15:02-15:10 已重跑 16K chunked tuned host
no-profiler 对照。`first_before_k` 和 `first_during_k_read` 都只是定位实验，
没有成为当前最快配置。2026-05-30 15:52-16:05 继续重跑
`partial_handoff_v1`，它也没有成为当前最快配置。
2026-05-30 16:22-16:24 新增 `qktv_detail` 细分复核后，结论不变：
`partial_handoff_v1` 缩短 Phase 2 只是因为把 `QKT@V` 前移，未形成真正 overlap。
2026-05-30 17:05-17:06 继续完成 `qktv_barrier_split` 和
`qktv_drain_all_then_matmul` 实验：显式 pack/unpack semaphore handoff 只有约
0.06 us，不是主要瓶颈；drain-all schedule 正确但 device critical path 回退，
因此没有成为当前最快配置。
2026-05-30 19:10-19:28 完成 `fa3_pv_softmax_v1` 真触发复测：
correctness 通过，但 host sync 和 device critical path 明显回退，因此也不更新
current fastest。
2026-05-30 20:20-20:26 新增 `split_compute_v1` 和 `llk_microflow_v1`
独立实验 fork 的 smoke/profile/host 对照。它们用于验证实验边界和更细粒度测点，
不改变 current fastest 结论。
2026-05-30 21:55-22:04 继续完成 split handoff 两步实验：
`split_l1_ready_signal_v1` 和 `split_state_ready_signal_v1`。它们证明同步成本很小，
并且 state-ready handoff 能把 writer 的 final output wait 从约 128 us 移到更早的
state wait，但还没有实现真实 `PV` consumer，因此也不更新 current fastest。
2026-05-30 22:55-23:01 新增并测试
`split_state_real_p_vprefetch_x8_v1` / `split_state_real_p_vafter_state_x8_v1`：
correctness smoke 通过，6 个主 shape（不含 smoke）的 host/no-copy 扫描和
q128/k128 device profile 完成。结论是 real-`P` handoff 当前没有超过 baseline；
真实 `P` remote read 和 ack 本身小于 1 us，但“整块 `P` 完成后再 handoff”的
producer 排布把 TRISC/WRITER critical path 拉长约 12 us。因此 current fastest 不变。
2026-05-31 10:34-10:37 继续完成 `split_state_real_p_kt_stream_v1` 和
`split_pv_owner_probe_v1`。kt-stream correctness smoke 通过，但 host/device
仍慢于 copied default；pv-owner probe 在 device critical path 上显示“移出
producer PV”有潜在收益，但它还没有把 consumer output 接回最终输出，所以不能更新
current fastest。
2026-05-31 12:01-12:07 继续完成 `split_state_mailbox_ring_v1` 和
`split_pv_owner_output_v1`。mailbox ring 没有降低 kt-stream 的关键等待；
`split_pv_owner_output_v1` 已经在受限非 chunked causal 作用域内通过
q128/k128、q256/k256 correctness，但 host 和 device critical path 都慢于
copied default / TTNN baseline，因此 current fastest 仍不变。
2026-05-31 15:04-15:25 尝试 `split_pv_owner_output_all_groups_v1`。该方向没有
产出可用性能数据：带 profiler 初版超过 TENSIX program size，删减新增 debug zones
后仍 timeout 且没有 kernel zone；因此它不是 current fastest 候选。
2026-05-31 16:48-16:52 继续修正 split signal/real-P handoff 路径，并验证
`split_pv_owner_output_no_ack_v1`。signal-only schedules 现在只启动 dataflow
consumer；real-P schedules 不再额外依赖 dummy token CB，而是直接等真实 `P/V/sum`
CB。no-ack 版本 correctness 通过，host avg 比 schedule 23 好，但 device critical
path 没有下降；ack/backpressure 不是当前主瓶颈，因此 current fastest 仍不变。

相对官方 TTNN tuned baseline，主要看 device critical path：

| shape | metric | TTNN tuned baseline | copied fastest | speedup | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 2K full q128/k256 | device critical us | 134.956 | 134.727 | 1.002x | 基本持平，不能宣称明确超越 |
| 16K chunked q128/k256 | device critical us | 137.081 | 139.271 | 0.984x | 最新轻量 device profiler 中 copied default 略慢 |

同一 tuned 配置的 host no-copy 辅助指标：

| shape | metric | TTNN tuned baseline | copied fastest | speedup | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| 2K full q128/k256 | avg ms | 0.175 | 0.181 | 0.967x | copied avg 略慢 |
| 2K full q128/k256 | sync ms | 0.140 | 0.140 | 1.000x | device completion 基本相同 |
| 16K chunked q128/k256 | avg ms | 0.157 | 0.163 | 0.963x | 最新 no-profiler 复测 copied avg 略慢 |
| 16K chunked q128/k256 | sync ms | 0.145 | 0.145 | 1.000x | sync 基本持平 |

2026-05-30 最新 16K chunked tuned host no-profiler 对照，单位 ms：

| variant / schedule | avg | best | worst | call | sync | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TTNN chunked baseline | 0.158 | 0.157 | 0.160 | 0.010 | 0.146 | 官方 tuned 边界 |
| copied default | 0.159 | 0.157 | 0.163 | 0.011 | 0.146 | 当前 copied 最快，仍是 parity |
| copied `first_during_k_read` | 0.161 | 0.158 | 0.176 | 0.012 | 0.146 | 不胜过 default |
| copied `first_before_k` | 0.167 | 0.158 | 0.272 | 0.021 | 0.144 | 扰动较大，不作为 tuned 路径 |
| copied `fa3_pv_softmax_v1` | 0.170 | 0.169 | 0.177 | 0.010 | 0.158 | 真触发后明显慢于 default |

2026-05-30 `partial_handoff_v1` 复测，仍是
`q=128,k=256,grid=8x8,pipeline_depth=2,prepared_no_q_copy`，单位 ms：

| variant / schedule | avg | best | worst | call | sync | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TTNN chunked baseline | 0.157 | 0.157 | 0.162 | 0.010 | 0.145 | 最新官方 tuned 边界 |
| copied default | 0.163 | 0.158 | 0.214 | 0.015 | 0.145 | sync 持平，avg 略慢 |
| copied `partial_handoff_v1` | 0.161 | 0.159 | 0.169 | 0.011 | 0.148 | avg 有噪声级改善，但 sync 和 device critical 回退 |

同一轮轻量 device profiler 显示，`partial_handoff_v1` 把 Phase 2 的
`QKV` drain 变短，但把提前 `QKT@V` 放进 Phase 1，导致 QK phase 和总
writer/compute critical path 变长；因此不更新 current fastest。

相对官方默认 baseline，`copied_balanced_q` 在 q128/k128 shape 上有明确 host 收益：

| shape | official default avg ms | copied policy avg ms | avg speedup | official default sync ms | copied policy sync ms | sync speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full q128/k128 | 0.200 | 0.184 | 1.09x | 0.157 | 0.144 | 1.09x |
| 16K chunked q128/k128 | 0.209 | 0.173 | 1.21x | 0.162 | 0.148 | 1.09x |

解释：

- 对默认 baseline 的收益主要来自 grid scheduling policy，不是新数学 kernel。
  官方 TTNN 如果手动传同样 `grid=8x8` 也能获得类似收益。
- 对 tuned baseline 的 device critical path 目前不能宣称超越。2K 历史数据只是
  parity；16K 最新轻量复测中 copied default 略慢，`partial_handoff_v1`
  和 `fa3_pv_softmax_v1` 都进一步回退。
- 后续只有当 copied fastest 的 device critical path 稳定低于 TTNN tuned baseline
  2% 以上，才把本节结论改成明确性能胜出。

## 2026-05-30 real-P / softmax-state handoff 实验

本轮问题：把 split consumer probe 的 dummy `P` 换成 producer 真实 softmax 后的
`P` / state handoff，确认第二个 compute kernel 是否能在真实数据依赖下获得可用
overlap。

实现边界：

- 不改 official TTNN baseline，不改 copied default。
- 新增 `split_state_real_p_vprefetch_x8_v1` 和
  `split_state_real_p_vafter_state_x8_v1` 两个 `split_compute_v1` schedule。
- producer compute 在 group0 softmax 完成后，把 `P` tiles pack 到 `c_15`
  state mailbox。
- producer writer 等 `c_15` ready 后发 state-ready signal；consumer dataflow 从
  producer L1 remote-read `P` 到本地 `c_1`，再发 token 给 consumer compute。
- consumer compute 用真实 `P` 和真实 `V` 做 x8 `PV` probe；producer 仍然执行原
  copied SDPA 的 `PV` 和 output，保证 correctness 路径不变。

correctness smoke：

```bash
./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset smoke --variant all --experiment split_compute_v1 \
  --mode prepared_no_q_copy --warmup 1 --iters 1 --grid 8,8 \
  --compute-pipeline-schedule split_state_real_p_vprefetch_x8_v1 \
  --check-correctness --no-device-profiler-read
```

结果：

| pair | elements | max abs diff | passed |
| --- | ---: | ---: | --- |
| full SDPA | 131072 | 0.000000 | true |
| chunked SDPA | 32768 | 0.000000 | true |

host/no-copy 主 shape 对照，`warmup=1 iters=3 grid=8x8`，单位 ms：

| shape | copied default avg/sync | dummy P + V prefetch avg/sync | real P + V prefetch avg/sync | real P + V after-state avg/sync | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2K q128/k128 | 0.176 / 0.148 | 0.206 / 0.177 | 0.198 / 0.170 | 0.200 / 0.169 | real P 仍慢于 default |
| 2K q256/k128 | 0.274 / 0.246 | 0.279 / 0.248 | 0.290 / 0.259 | 0.292 / 0.261 | real P 明显回退 |
| 2K q256/k256 | 0.230 / 0.197 | 0.229 / 0.200 | 0.244 / 0.209 | 0.247 / 0.206 | dummy P 近似持平，real P 回退 |
| 16K chunked q128/k128 | 0.184 / 0.151 | 0.187 / 0.158 | 0.217 / 0.169 | 0.201 / 0.172 | real P 回退，vafter avg 略好但 sync 不好 |
| 16K chunked q256/k128 | 0.278 / 0.247 | 0.298 / 0.249 | 0.292 / 0.262 | 0.293 / 0.263 | real P 没有收益 |
| 16K chunked q256/k256 | 0.234 / 0.198 | 0.232 / 0.202 | 0.241 / 0.212 | 0.244 / 0.212 | dummy P 近似持平，real P 回退 |

同一轮 official tuned baseline 参考，`prepared_no_q_copy grid=8x8`：

| shape | TTNN baseline avg/sync | copied default avg/sync |
| --- | ---: | ---: |
| 2K q128/k128 | 0.171 / 0.144 | 0.176 / 0.148 |
| 2K q256/k128 | 0.285 / 0.235 | 0.274 / 0.246 |
| 2K q256/k256 | 0.220 / 0.186 | 0.230 / 0.197 |
| 16K chunked q128/k128 | 0.175 / 0.146 | 0.184 / 0.151 |
| 16K chunked q256/k128 | 0.275 / 0.242 | 0.278 / 0.247 |
| 16K chunked q256/k256 | 0.249 / 0.186 | 0.234 / 0.198 |

device profiler 细分，`q128/k128 grid=8x8 warmup=1 iters=1`，单位 us：

| shape / schedule | TRISC kernel | FAP_COMPUTE | FAP_WRITER | consumer PV x8 | state/local wait | real P pack | real P remote read | ack wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K dummy P + V prefetch | 151.336 | 144.142 | 148.341 | 112.180 | 119.956 | - | - | - |
| 2K real P + V prefetch | 163.621 | 156.053 | 161.564 | 122.723 | 130.987 | 1.369 | 0.588 | 0.607 |
| 16K chunked dummy P + V prefetch | 153.681 | 146.209 | 150.353 | 112.140 | 120.117 | - | - | - |
| 16K chunked real P + V prefetch | 165.837 | 158.396 | 164.002 | 122.380 | 130.484 | 1.351 | 0.487 | 0.507 |

第一性原理判断：

- 一个 producer Tensix 当前仍在同一条 copied SDPA compute 路径里做
  `QK -> softmax -> PV -> SALAD/normalize`。新增 consumer core 只是额外执行
  x8 `PV` probe，尚未替代 producer 的 `PV`，所以端到端不可能直接变快。
- 真实 `P` 的 L1 remote read 和 semaphore ack 本身不是主瓶颈，critical 都小于
  1 us；`P` pack 约 1.35 us，也不是 12 us 回退的全部来源。
- 主要回退来自 schedule 形态：当前 real-`P` 版本为了交出完整 `P`，让 producer
  先完成整块 group0 softmax/handoff，再进入本地 `PV`。这破坏了原本 split-drain
  的局部交错，导致 `FAP_COMPUTE`、`FAP_WRITER`、consumer token wait 一起后移。
- 因此下一步不应该继续只扫 `V` prefetch 或简单增大 consumer probe，而应该把
  handoff 粒度从“整块 `P`”降到 `kt_subblock`，并让 consumer 的 `PV` 真正接管一部分
  producer `PV`，否则只是复制计算。

Tensix 任务划分草案：

| 角色 | 当前 `split_state_real_p_*` 做法 | 下一步应该验证的做法 | 目的 |
| --- | --- | --- | --- |
| Producer compute Tensix | `QK -> softmax -> pack P -> 本地 PV -> SALAD/normalize` | `QK -> softmax -> 按 kt 交出 P_kt`，并跳过被 consumer 接管的本地 `PV` row group | 减少重复 `PV`，让 producer 更快回到下一段 `QK` |
| Producer dataflow/writer | 等完整 `c_15` 后发 state-ready，再等 ack，最后继续写自己的 output | 对每个 `P_kt` 或 row group 发 ready token；final output writer 只负责未拆出的 row group | 避免 writer 等整块 `P` 完成才启动下游 |
| Consumer dataflow Tensix | state-ready 后 remote-read 整块 `P`，`V` 可提前或延后读取 | 在 producer 每个 `P_kt` ready 后读取 `P_kt`，同时提前保持对应 `V_kt` 在 L1 | 把 remote read 从 phase boundary 移进 producer 计算窗口 |
| Consumer compute Tensix | 用真实 `P/V` 做 x8 `PV` probe，但结果不进入最终 output | 对接管的 row group 做真实 `P_kt @ V_kt` accumulation，并产出可 normalize 的 partial output | 把 probe 变成真正减少 producer 工作的 split-PV 路径 |

按当前 q128/k128 device profile 的时间预算看，优先级不是继续优化 remote read：

| 项 | 2K real P + V prefetch | 16K chunked real P + V prefetch | 判断 |
| --- | ---: | ---: | --- |
| `FAP_SPLIT_REAL_P_REMOTE_READ` | 0.588 us | 0.487 us | 小于 1 us，不是首要瓶颈 |
| `FAP_SPLIT_REAL_P_ACK_WAIT` | 0.607 us | 0.507 us | 小于 1 us，不是首要瓶颈 |
| `FAP_SPLIT_REAL_P_HANDOFF_PACK` | 1.369 us | 1.351 us | 有成本，但不足以解释约 12 us 回退 |
| `FAP_COMPUTE` 回退 | 144.142 -> 156.053 us | 146.209 -> 158.396 us | producer 排布变差是主问题 |
| `FAP_WRITER` 回退 | 148.341 -> 161.564 us | 150.353 -> 164.002 us | writer 被整块 state handoff 重新排队 |

从第一性原理看，FlashAttention 优化不是“再多一个 core 计算就更快”。`P`
是 `softmax(QK)` 的结果，`PV` 必须消费已经归一化或可正确累计的 softmax state。
如果 producer 已经做完本地 `PV`，consumer 再做一次 `PV` 只会增加工作量；
如果 producer 必须先攒完整 `P` 才交出去，又会破坏原本 `sub_exp` 与 partial
`PV` 的局部流水。真正可能提升性能的条件是：consumer 接管的 `PV` 必须替代
producer 的同一段 `PV`，并且 handoff 必须细到 `kt_subblock` 或 row group，
这样 producer compute 才能更早进入下一段 `QK`，consumer compute 才能在 producer
处理后续 `QK/softmax` 时消化上一段 `PV`。

下一步实验计划：

1. `split_state_real_p_kt_stream_v1`
   - producer 每个 `kt_subblock` softmax 后立刻 handoff 该片 `P_kt`。
   - consumer 对 `P_kt @ V_kt` 做 L1 accumulation，最后 pack partial output。
   - 目标：恢复 split-drain 的局部性，把 consumer 启动点提前 8-12 us。
2. `split_pv_owner_v1`
   - 对一个 row group，producer 只做 `QK/softmax/state`，跳过本地 `PV`；
     consumer 成为该 row group 的 `PV` owner。
   - 目标：去掉当前 probe 的重复计算，验证两个 compute kernel 是否能降低总 TRISC
     critical path。
3. 大 shape 摊销实验
   - 在 q128/k128、q256/k256 之外增加更大 `D` 或更多 `V` columns 的 custom shape。
   - 判断 handoff 固定成本是否能被更大的 `PV` 工作量摊薄；如果大 shape 仍不收益，
     说明需要更细 LLK/microflow 级交错，而不是只做 core 间 split。

## 2026-05-31 kt-stream / PV-owner 实验

本轮问题：验证上一节提出的两个方向。

- `split_state_real_p_kt_stream_v1`：把 real-`P` handoff 粒度从整块 `P`
  降到 `kt_subblock` packet，仍保留 producer 原始输出路径，所以 correctness
  路径不变。
- `split_pv_owner_probe_v1`：让 producer 跳过 group0 的本地 `PV`，consumer 做
  real-`P/V` probe。这个实验没有把 consumer output 接回最终 output tensor，
  所以是 profile-only 上界，不跑 correctness。

correctness / smoke：

| schedule | correctness | 说明 |
| --- | --- | --- |
| `split_state_real_p_kt_stream_v1` | full/chunked smoke max abs diff = 0 | producer 仍写正确 output |
| `split_pv_owner_probe_v1` | 不适用 | producer 故意跳过 group0 `PV`，output 不是正确 SDPA |

host/no-copy 主 shape 对照，`warmup=1 iters=3 grid=8x8`，单位是
`avg / sync ms`：

| shape | copied default | real P full block | kt-stream | pv-owner probe | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2K q128/k128 | 0.176 / 0.148 | 0.198 / 0.170 | 0.198 / 0.164 | 0.180 / 0.148 | kt-stream 仍慢；pv-owner 仅是 invalid-output 上界 |
| 2K q256/k128 | 0.274 / 0.246 | 0.290 / 0.259 | 0.288 / 0.262 | 0.273 / 0.245 | pv-owner 接近 default，但不代表正确性能 |
| 2K q256/k256 | 0.230 / 0.197 | 0.244 / 0.209 | 0.241 / 0.214 | 0.225 / 0.195 | pv-owner 上界显示 producer PV 可省时间 |
| 16K chunked q128/k128 | 0.184 / 0.151 | 0.217 / 0.169 | 0.195 / 0.165 | 0.184 / 0.150 | kt-stream 改善 full-block 但仍慢于 default |
| 16K chunked q256/k128 | 0.278 / 0.247 | 0.292 / 0.262 | 0.312 / 0.259 | 0.277 / 0.248 | kt-stream 对 q256/k128 反而更差 |
| 16K chunked q256/k256 | 0.234 / 0.198 | 0.241 / 0.212 | 0.244 / 0.214 | 0.227 / 0.199 | pv-owner 上界略好，但仍非正确路径 |

device profiler，`q128/k128 grid=8x8 warmup=1 iters=1`，单位 us：

| shape / schedule | TRISC kernel | FAP_COMPUTE | FAP_WRITER | consumer PV | state/local wait | P pack | P remote read | ack wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K dummy P + V prefetch | 151.336 | 144.142 | 148.341 | 112.180 | 119.956 | - | - | - |
| 2K real P full block | 163.621 | 156.053 | 161.564 | 122.723 | 130.987 | 1.369 | 0.588 | 0.607 |
| 2K kt-stream | 157.750 | 157.676 | 162.887 | 136.022 | 132.063 | 1.355 | 0.554 | 0.578 |
| 2K pv-owner probe | 141.277 | 141.202 | 145.494 | 120.970 | 118.093 | 1.379 | 0.561 | 0.580 |
| 16K dummy P + V prefetch | 153.681 | 146.209 | 150.353 | 112.140 | 120.117 | - | - | - |
| 16K real P full block | 165.837 | 158.396 | 164.002 | 122.380 | 130.484 | 1.351 | 0.487 | 0.507 |
| 16K kt-stream | 159.958 | 159.878 | 165.618 | 136.261 | 132.304 | 1.335 | 0.461 | 0.488 |
| 16K pv-owner probe | 142.882 | 142.803 | 148.135 | 120.009 | 117.629 | 1.379 | 0.501 | 0.521 |

本轮判断：

- kt-stream 没有解决问题。它把 `P` handoff 拆成 packet，但当前实现仍由 writer
  串行等待 packet、发送 token、等待 ack，再 pop `c_15`；consumer 的
  `FAP_SPLIT_SIGNAL_OUTPUT_WAIT` 仍在 136 us 左右。结果是 producer 和 writer 没有
  形成真正异步流水，只是把 full-block handoff 的大包拆成小包并增加 barrier/pack
  频率。
- pv-owner probe 证明“移走 producer 本地 `PV`”方向有价值：2K q128/k128 的
  `FAP_COMPUTE` 从 real-P full block 的 156.053 us 降到 141.202 us，16K chunked
  从 158.396 us 降到 142.803 us。但这是 invalid-output 上界，因为 consumer 还没有
  接管所有 K chunk 的 `PV` 状态，也没有把 output 写回最终 tensor。
- 因此下一步不应该继续只拆 packet 或加小 buffer；真正方向是做 correctness-capable
  PV ownership。consumer 必须拥有某个 row group 横跨所有 K chunk 的
  `P_kt @ V_kt` accumulation，以及对应的 max/sum/SALAD/normalize 状态，否则 producer
  一旦跳过本地 `PV` 就会丢失正确输出。

下一步实验计划：

1. `split_pv_owner_output_v1`
   - consumer 不再只是 probe，而是对一个 row group 维护跨 K chunk 的 partial
     output、max/sum state 和最终 normalize。
   - consumer 直接写该 row group 的 output tensor，producer writer 跳过同一 row
     group 的写出。
   - 目标：把 pv-owner probe 的 141-143 us 上界变成 correctness-capable 路径。
2. `split_state_mailbox_ring_v1`
   - 如果还要保留 kt-stream，`c_15` 不能是一包一 ack 的单槽邮箱。
   - 改成 2-4 slot ring，writer 批量或延迟 ack，consumer dataflow 先 remote-read
     后消费，避免 producer 在每个 packet 后被 writer/ack 反压。
   - 目标：验证 kt-stream 的失败是同步拓扑问题，还是 packet 化本身就不值得。
3. `consumer_state_minimal_v1`
   - 先只给 consumer 接管最后一个 row group，并显式搬运它需要的 `cur.max` /
     `cur.sum` / partial output state。
   - 如果这个最小 correctness path 仍慢于 default，再回到单 Tensix 内部
     LLK/microflow 交错，而不是继续扩大跨 core split。

### 2026-05-31 mailbox ring 实验结果

实现内容：

- 新增 `split_state_mailbox_ring_v1`。
- producer compute 仍按 `kt_subblock` 把 real-`P` packet pack 到 `c_15`。
- `c_15` 容量从 1 个 packet 扩大到最多 4 个 packet。
- writer 不再每包立刻 `ack -> pop`，而是在一个 ring batch 内先发多个
  state-ready token，再等 consumer ack，最后批量 pop。
- consumer dataflow 用 `output_index % ring_slots` 计算 remote-read offset，从
  producer `c_15` ring slot 读取对应 packet。

correctness / smoke：

| schedule | correctness | 说明 |
| --- | --- | --- |
| `split_state_mailbox_ring_v1` | full/chunked smoke max abs diff = 0 | producer 仍写正确 output，ring 只改变 handoff 同步 |

host/no-copy 主 shape 对照，`warmup=1 iters=3 grid=8x8`，单位是
`avg / sync ms`。2K 采用 `copied_sdpa`，16K 采用 `copied_chunked`：

| shape | copied default | kt-stream | mailbox ring | 结论 |
| --- | ---: | ---: | ---: | --- |
| 2K q128/k128 | 0.176 / 0.148 | 0.198 / 0.164 | 0.196 / 0.165 | sync 基本不变 |
| 2K q256/k128 | 0.274 / 0.246 | 0.288 / 0.262 | 0.294 / 0.261 | 仍慢于 default |
| 2K q256/k256 | 0.230 / 0.197 | 0.241 / 0.214 | 0.266 / 0.212 | sync 接近 kt-stream，avg 有 host 抖动 |
| 16K chunked q128/k128 | 0.184 / 0.151 | 0.195 / 0.165 | 0.210 / 0.166 | 没有改善关键等待 |
| 16K chunked q256/k128 | 0.278 / 0.247 | 0.312 / 0.259 | 0.291 / 0.263 | avg 好于 kt-stream，但 sync 未下降 |
| 16K chunked q256/k256 | 0.234 / 0.198 | 0.244 / 0.214 | 0.248 / 0.214 | 与 kt-stream 基本持平 |

device profiler，`q128/k128 grid=8x8 warmup=1 iters=1`，单位 us：

| shape / schedule | TRISC kernel | FAP_COMPUTE | FAP_WRITER | consumer PV/probe | state/local wait | P pack | P remote read | ack wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K kt-stream | 157.750 | 157.676 | 162.887 | 136.022 | 132.063 | 1.355 | 0.554 | 0.578 |
| 2K mailbox ring | 157.823 | 157.748 | 163.282 | 135.720 | 132.259 | 1.353 | 0.547 | 0.548 |
| 16K kt-stream | 159.958 | 159.878 | 165.618 | 136.261 | 132.304 | 1.335 | 0.461 | 0.488 |
| 16K mailbox ring | 160.018 | 159.941 | 165.798 | 136.260 | 132.216 | 1.325 | 0.554 | 0.557 |

结论：

- ring 没有解决 kt-stream 的主瓶颈。`FAP_SPLIT_SIGNAL_OUTPUT_WAIT` 仍在
  约 136 us，`FAP_SPLIT_STATE_READY_LOCAL_WAIT` 仍在约 132 us；这说明慢点不在
  单包 ack 的纳秒级/亚微秒级成本，而在 state-ready token 本身到达太晚。
- `P` remote read 和 ack 仍小于 1 us。即使把 ack 延迟到 batch 末尾，device critical
  path 也没有明显下降，说明当前 packet 化路径的收益被“producer 完成后才由 writer
  统一转发”吃掉了。
- 因此 kt-stream 失败更像是同步拓扑和 state ownership 问题，不是 mailbox
  单槽容量问题。下一步应转向真正的 `PV` output ownership，而不是继续扩大 ring。

### 2026-05-31 split_pv_owner_output_v1 实验结果

本轮问题：把 `split_pv_owner_probe_v1` 的 invalid-output 上界推进成一个真正写回
output tensor 的最小 ownership 路径，验证“consumer 接管一段 `PV`”在正确性成立后
是否还能保留 probe 中看到的 producer critical path 收益。

实现边界：

- 新增 `split_pv_owner_output_v1`，编号 23，只作用于 `split_compute_v1`。
- 作用域刻意收窄：只支持 causal、非 chunked prefill，且 `Sq_chunk_t <= Sk_chunk_t`，
  即 `q_chunk0` 只有一个 K chunk。这样 consumer 只需要处理第一段 K，不需要先实现
  跨多 K chunk 的 online max/sum state。
- producer 只把 core0/head0/q_chunk0 的第一个 row group 交给 consumer：handoff
  softmax 后的 real `P` 和 `sum`，跳过该 group 的本地 `PV` 和 writer output。
  其余 row group 仍走原 copied SDPA 路径。
- consumer dataflow 读取 producer `c_15` 中的 `P/sum`，预取真实 `V`，consumer
  compute 做 `P@V`、按 `sum` normalize，并通过 consumer dataflow 写回同一个
  output tensor 的第一个 row group。
- 当前写回 tile id 只覆盖第一个 output group，因此这个 schedule 明确不是通用
  split-PV；它只是验证 ownership 机制是否可行。

correctness，`prepared_no_q_copy warmup=2 iters=5 grid=8x8`：

| shape | elements | max abs diff | mean abs diff | tolerance | passed |
| --- | ---: | ---: | ---: | ---: | --- |
| 2K q128/k128 | 2097152 | 0.000000 | 0.000000 | 0.125000 | true |
| 2K q256/k256 | 2097152 | 0.005859 | 0.000102 | 0.125000 | true |

host/no-profiler 对照，`warmup=2 iters=5 grid=8x8 prepared_no_q_copy`，单位 ms：

| shape | TTNN baseline avg/sync | copied default avg/sync | split owner output avg/sync | 结论 |
| --- | ---: | ---: | ---: | --- |
| 2K q128/k128 | 0.158 / 0.145 | 0.162 / 0.148 | 0.193 / 0.163 | 正确但明显慢于 default |
| 2K q256/k256 | 0.201 / 0.187 | 0.212 / 0.199 | 0.235 / 0.210 | 正确但慢于 default 和 TTNN |

device profiler 对照，`q128/k128 grid=8x8 warmup=0 iters=1`，单位 us。device
profiler 运行的 host `avg_ms` 包含 mid-run dump 开销，本表只使用 device zone：

| schedule | TRISC kernel max | BRISC kernel | FAP_COMPUTE | FAP_WRITER | FAP_READER | writer wait output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| copied default | 144.175 | 147.370 | 144.087 | 147.333 | 128.766 | 124.399 |
| split owner output | 156.573 | 160.636 | 156.482 | 160.596 | 139.495 | 136.981 |

owner-output 专属 zone，`q128/k128`，单位 us：

| zone | critical us | 判断 |
| --- | ---: | --- |
| `FAP_SPLIT_COMPUTE_V1_CONSUMER_PROBE` | 17.403 | consumer 真实 `P@V` + normalize 已进入关键路径附近 |
| `FAP_SPLIT_CONSUMER_PROBE_PV_GROUP` | 17.126 | 单个接管 group 的 `PV` 成本 |
| `FAP_SPLIT_CONSUMER_OUTPUT_TOKEN_WAIT` | 17.209 | consumer 等 output token，说明启动点仍偏晚 |
| `FAP_SPLIT_SIGNAL_OUTPUT_WAIT` | 12.107 | dataflow 等 producer state-ready/output token |
| `FAP_SPLIT_STATE_READY_LOCAL_WAIT` | 10.625 | writer-mediated state-ready 仍是固定等待 |
| `FAP_SPLIT_OWNER_P_SUM_HANDOFF_PACK` | 1.530 | handoff pack 不是主瓶颈 |
| `FAP_SPLIT_REAL_P_REMOTE_READ` | 0.909 | remote read 小于 1 us |
| `FAP_SPLIT_REAL_P_ACK_WAIT` | 0.947 | ack 小于 1 us |
| `FAP_SPLIT_OWNER_OUTPUT_WRITE` | 3.481 | consumer 直接写 output 有成本但不是 12 us 回退的全部 |
| `FAP_SPLIT_CONSUMER_OWNER_NORMALIZE` | 1.163 | normalize 成本可控 |
| `FAP_SPLIT_OWNER_DISCARD_LOCAL_GROUP0` | 0.099 | producer 丢弃本地 group0 缓冲不是瓶颈 |

结果判断：

- `split_pv_owner_output_v1` 已经是 correctness-capable，但只是受限最小路径：
  非 chunked causal、只接管第一个 q chunk 的第一个 row group、只覆盖一个 K chunk。
- 它没有保留 `split_pv_owner_probe_v1` 的上界收益。default 的 device critical path
  是 writer 约 147 us / compute 约 144 us；owner-output 变成 writer 约 161 us /
  compute 约 156 us。
- 原因不是 L1 remote read 或 ack。`P` remote read、ack 都小于 1 us，handoff pack
  约 1.5 us。真正新增的是：consumer `PV`/normalize/output write 进入路径，同时
  producer 为了交出这个 group 进入 full-handoff 风格排布，state-ready 和 output
  token 到达偏晚。
- v1 只移走一个 row group 的 producer `PV`，省下的本地 `PV` 小于新增的
  consumer compute、writer-mediated signal、output write 固定成本。因此当前不能把它
  列为最快配置。

下一步实验方向：

1. state-ready 不应继续完全由 writer 转发；需要试一个 compute-to-consumer 或
   shared-L1 semaphore 变体，让 consumer 在 producer softmax group ready 后更早启动。
2. 真正面向 q256/k128 / chunked 的版本必须实现跨多 K chunk 的 online state：
   consumer 拥有 `prev.max`、`prev.sum`、partial `out`，每个 K chunk 消费
   `P_kt@V_kt` 后再最终 normalize。当前 v1 为了先证明 output ownership，故意没有跨
   K chunk。
3. 不继续扩大 mailbox ring，也不继续做 all-groups ownership。ring 已经证明单槽
   ack 不是关键瓶颈；all-groups 初版同时扩大 handoff、writer token/ack、consumer
   output write 和 producer code size，粒度过大。
4. 后续如果再接管更多 row groups，必须先拆出更小的 state-ready 拓扑实验，并保证
   schedule 23 的 correctness-capable 路径不回退。
   关键是 ownership 粒度和 state-ready 启动时间。

结果文件：

- `results/pv_owner_output_q128_k128_host_20260531.txt`
- `results/pv_owner_output_q256_k256_host_20260531.txt`
- `results/pv_owner_output_default_q128_k128_host_20260531.txt`
- `results/pv_owner_output_default_q256_k256_host_20260531.txt`
- `results/pv_owner_output_q128_k128_device_20260531.txt`
- `results/pv_owner_output_default_q128_k128_device_20260531.txt`

### 2026-05-31 split_pv_owner_output_all_groups_v1 负结果

本轮问题：`split_pv_owner_output_v1` 只接管一个 row group，固定同步和 output write
成本大于省下的 producer `PV`。下一步把 q_chunk0 的所有 compute `qktv_h` row
groups 都交给 consumer，验证接管更多工作能不能摊薄固定成本。

尝试内容：

- 曾临时新增 `split_pv_owner_output_all_groups_v1`（编号 24），只作用于
  `split_compute_v1`。
- 计划作用域仍限制为 causal、非 chunked prefill，并要求 `q_chunk0` 只有一个 K
  chunk；也就是先不解决跨 K chunk online max/sum ownership。
- producer 对 q_chunk0 的每个 `qktv_h` row group handoff real `P` 和 `sum`，
  并跳过这些 groups 的本地 `PV`。
- consumer 每收到一个 group packet 就做 `P@V + normalize`，并按 group index
  写回 output tensor 对应 tile range。
- producer writer 对 q_chunk0 不再 drain 本地 output；只负责等待 state packets
  并转发 state-ready。

过程中先修正了两个同步/粒度问题：

| 问题 | 现象 | 修正 |
| --- | --- | --- |
| writer packet 数仍按 full-block handoff 的 1 个 packet | consumer 等多个 row-group token 时会挂住 | schedule 24 下 `split_state_ready_packets = row_group_count` |
| owner group 数按 writer drain 的 `out_out_subblock_h` 推导 | 可能和 compute 实际 handoff 的 `qktv_h` 不一致 | owner-output 的 packet/CB/consumer q_rows 改为使用 compute `qktv_h` |

验证结果：

| 项 | 结果 |
| --- | --- |
| focused host build | 通过：`cmake --build build_Release --target flash_attention_profile --parallel $(nproc)` |
| `git diff --check` | 通过 |
| copied-only q128/k128 + device profiler | 初版失败：producer TENSIX program size 71792 > 70656 |
| 删除新增 all-groups debug zones 后 copied-only q128/k128 + device profiler | 80s timeout，`profile_log_device.csv` 只有 header，没有 kernel zone |
| q128/k128 correctness | 未通过验证；当前 schedule 24 不能作为 correctness-capable 结果 |
| q256/k256 correctness/profile | 未跑；q128/k128 已经失败，继续扩大 shape 没有意义 |
| 当前代码状态 | schedule 24 可执行入口已删除；README 只保留负实验记录 |

失败判断：

- 一次性把所有 row groups 迁出 producer 不是一个好的下一步粒度。它同时扩大了
  producer handoff、writer token/ack、consumer output write 的状态空间，还把 producer
  compute kernel 推到 program-size 边界。
- all-groups 版本没有给出任何可用性能数据，不能和 baseline 比较，也不能替代
  `split_pv_owner_output_v1` 的正确性结论。按“无用代码不保留”的原则，相关
  可执行 schedule 已删除。
- 下一步不继续扩大 ownership 范围，而应先做更小的拓扑实验：仍接管一个 row group，
  但绕过 writer-mediated state-ready，验证 compute/producer 能否更早触发 consumer。
  也就是先解决 v1 中 10-17 us 的 token/state wait，再考虑接管更多 row groups。

### 2026-05-31 split_pv_owner_output_no_ack_v1 实验结果

本轮问题：schedule 23 的 owner-output 路径已经正确，但慢于 copied default。上一轮
profile 里 `P` remote read 和 ACK 都小于 1 us，因此需要验证：去掉 ACK/backpressure
后，是否能降低 host sync 或 device critical path。如果不能，下一步就不该继续做
mailbox/ring/ack 方向。

实现和修正：

- `split_signal_only_v1` / `split_output_stream_signal_v1` /
  `split_l1_ready_signal_v1` / `split_state_ready_signal_v1` 改回 dataflow-only：
  host 不再为这些纯信号 schedule 创建 consumer compute kernel。
- real-`P` handoff schedules 不再让 consumer compute 等 `c_6` dummy token；consumer
  直接等真实 `P/V/sum` CB。`c_6` token 只保留给 dummy probe schedules。
- 新增并验证 `split_pv_owner_output_no_ack_v1`，编号 25。它和 schedule 23 一样只接管
  first row group output，但 writer 发 state-ready 后不等 ACK，consumer remote-read
  `P/sum` 后也不回 ACK。

correctness：

| schedule | shape | elements | max abs diff | mean abs diff | tolerance | passed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `split_pv_owner_output_v1` | 2K q128/k128 | 2097152 | 0.000000 | 0.000000 | 0.125000 | true |
| `split_pv_owner_output_v1` | 2K q256/k256 | 2097152 | 0.005859 | 0.000102 | 0.125000 | true |
| `split_pv_owner_output_no_ack_v1` | 2K q128/k128 | 2097152 | 0.000000 | 0.000000 | 0.125000 | true |

host/no-profiler 对照，`llama_prefill_2k_q128_k128 warmup=2 iters=5 grid=8x8
prepared_no_q_copy`，单位 ms：

| 路径 | avg | best | worst | call avg | sync avg | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TTNN baseline | 0.160 | 0.157 | 0.166 | 0.013 | 0.145 | 当前 shape 的官方对照 |
| split default | 0.161 | 0.160 | 0.165 | 0.012 | 0.147 | copied split fork 的默认对照 |
| split owner output | 0.189 | 0.182 | 0.200 | 0.024 | 0.163 | 正确但明显慢 |
| split owner output no-ack | 0.177 | 0.176 | 0.181 | 0.012 | 0.164 | avg 比 schedule 23 好，sync 没有改善 |

device profiler 对照，`llama_prefill_2k_q128_k128 grid=8x8 warmup=0 iters=1`，
单位 us。应用层 summary 仍显示 `zones=0`，但 `profile_log_device.csv` 有有效 zone，
本表用 CSV 手工解析：

| zone | schedule 23 | schedule 25 no-ack | 判断 |
| --- | ---: | ---: | --- |
| `FAP_WRITER` | 160.514 | 160.631 | critical path 没降 |
| `FAP_COMPUTE` | 155.893 | 156.204 | compute 也没有改善 |
| `FAP_READER` | 138.826 | 139.103 | reader 不受 ACK 影响 |
| `FAP_WRITER_WAIT_OUTPUT` | 136.083 | 136.821 | final output 等待仍在 |
| `FAP_SPLIT_COMPUTE_V1_CONSUMER_PROBE` | 16.507 | 16.766 | consumer `PV`/normalize 仍进入尾部路径 |
| `FAP_SPLIT_CONSUMER_PROBE_PV_GROUP` | 16.308 | 16.567 | 接管 group 的真实 `PV` 成本未降低 |
| `FAP_SPLIT_CONSUMER_PROBE_WAIT_INPUTS` | 14.020 | 14.261 | consumer 仍等真实 `P/V/sum` |
| `FAP_SPLIT_SIGNAL_OUTPUT_WAIT` | 11.653 | 11.910 | state-ready 到达仍偏晚 |
| `FAP_SPLIT_STATE_READY_LOCAL_WAIT` | 10.128 | 10.387 | writer-mediated state-ready 仍是固定等待 |
| `FAP_SPLIT_OWNER_P_SUM_HANDOFF_PACK` | 1.561 | 1.559 | handoff pack 不变 |
| `FAP_SPLIT_REAL_P_REMOTE_READ` | 0.884 | 0.870 | remote read 不是瓶颈 |
| `FAP_SPLIT_REAL_P_ACK_WAIT` | 0.930 | - | ACK 已移除，但 critical path 没降 |
| `FAP_SPLIT_OWNER_OUTPUT_WRITE` | 2.843 | 3.673 | consumer 写 output 成本仍存在 |

结果判断：

- no-ack 版本证明 schedule 23 的 ACK/backpressure 不是主瓶颈。ACK wait 消失后，
  device critical path 基本不动，host avg 的改善主要来自 `call_avg_ms` 回到 0.012 ms，
  `sync_avg_ms` 仍约 0.164 ms。
- 当前慢点仍然是 state-ready 启动太晚，加上 consumer `PV`/normalize/output write
  被放在尾部路径上。也就是说，consumer 拿到真实 `P/sum` 的时刻已经晚了，去掉 ACK
  只能省掉一个小同步点，不能创造 overlap 窗口。
- 下一步不再继续做 mailbox ring 或 no-ack 微调。更有价值的是：让 producer compute
  在 row group 的 softmax state ready 后直接唤醒 consumer，绕过 writer-mediated
  转发；同时准备一个 consumer-owned online state 版本，后续扩展到跨 K chunk 的
  `P_kt@V_kt` accumulation。

结果文件：

- `/wafer/gsh/tmp/fa_profile_20260531_ttnn_2k_q128_k128.log`
- `/wafer/gsh/tmp/fa_profile_20260531_split_default_2k_q128_k128.log`
- `/wafer/gsh/tmp/fa_profile_20260531_split_owner_2k_q128_k128.log`
- `/wafer/gsh/tmp/fa_profile_20260531_split_owner_no_ack_2k_q128_k128.log`
- `/wafer/gsh/tmp/fa_profile_20260531_split_owner_2k_q128_k128_device.csv`
- `/wafer/gsh/tmp/fa_profile_20260531_split_owner_no_ack_2k_q128_k128_device.csv`
- `/wafer/gsh/tmp/fa_profile_20260531_split_owner_no_ack_correctness.log`

## 2026-05-29 实验记录

机器和构建：

- 工作目录：`/wafer/gsh/wallfacer/third_party/tt-metal`
- 设备：单卡 Blackhole，`chip_freq_mhz=1350.000`
- firmware bundle：`19.5.0`
- KMD：`2.7.0`
- 构建命令：
  `cmake --build build_Release --target flash_attention_profile --parallel $(nproc)`

correctness smoke：

| pair | elements | max abs diff | passed |
| --- | ---: | ---: | --- |
| full SDPA | 131072 | 0.000000 | true |
| chunked SDPA | 32768 | 0.000000 | true |

其中 full SDPA 对比的是 `ttnn_sdpa_baseline` vs `copied_sdpa`；
chunked SDPA 对比的是 `ttnn_chunked_baseline` vs `copied_chunked`。

2K full prefill，prepared：

- shape：`B=1 H=8 KVH=1 S=2048 D=128 q=256 k=128`
- run：`warmup=2 iters=5`

| variant | avg ms | best ms | worst ms | call ms | sync ms | copy Q ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ttnn_sdpa_baseline` | 0.611 | 0.529 | 0.776 | 0.030 | 0.252 | 0.325 |
| `copied_sdpa` | 0.643 | 0.543 | 0.827 | 0.044 | 0.241 | 0.353 |
| `ttnn_chunked_baseline` | 0.939 | 0.681 | 1.319 | 0.085 | 0.238 | 0.601 |
| `copied_chunked` | 0.694 | 0.579 | 0.838 | 0.049 | 0.242 | 0.391 |

2K full prefill 的设备阶段，`copied_sdpa`，prepared，`warmup=1 iters=1`：

| zone | count | avg cycles | max cycles | critical us |
| --- | ---: | ---: | ---: | ---: |
| `FAP_READER` | 220 | 85826 | 274268 | 203.161 |
| `FAP_COMPUTE` | 660 | 107388 | 313571 | 232.275 |
| `FAP_WRITER` | 220 | 109451 | 314705 | 233.115 |
| `CBP_FW_LOCAL_CB_INIT` | 880 | 545 | 603 | 0.447 |

同一个 shape 下，官方 `ttnn_sdpa_baseline` 的 coarse firmware/kernel critical path 也在同一量级：

| zone | critical us |
| --- | ---: |
| `BRISC-KERNEL` | 233.562 |
| `TRISC-KERNEL` | 232.695 |
| `NCRISC-KERNEL` | 203.505 |
| `CBP_FW_LOCAL_CB_INIT` | 0.445 |

16K paged/chunked prefill：

- shape：`B=1 H=8 KVH=1 S=16384 D=128 prefill=2048`
- chunk：`q=256 k=128 page=128`

| variant | mode | avg ms | best ms | worst ms |
| --- | --- | ---: | ---: | ---: |
| `ttnn_chunked_baseline` | prepared | 1.094 | 0.925 | 1.278 |
| `copied_chunked` | prepared | 1.049 | 0.839 | 1.274 |
| `ttnn_chunked_baseline` | trace | 0.963 | 0.778 | 1.160 |
| `copied_chunked` | trace | 0.946 | 0.757 | 1.146 |

| variant | mode | call ms | sync ms | copy Q ms |
| --- | --- | ---: | ---: | ---: |
| `ttnn_chunked_baseline` | prepared | 0.041 | 0.462 | 0.581 |
| `copied_chunked` | prepared | 0.052 | 0.488 | 0.498 |
| `ttnn_chunked_baseline` | trace | 0.005 | 0.473 | 0.478 |
| `copied_chunked` | trace | 0.003 | 0.468 | 0.468 |

16K chunked trace 的设备阶段，`copied_chunked`，`warmup=1 iters=1`：

| zone | count | avg cycles | max cycles | critical us |
| --- | ---: | ---: | ---: | ---: |
| `FAP_READER` | 110 | 89384 | 278814 | 206.529 |
| `FAP_COMPUTE` | 330 | 110311 | 318007 | 235.561 |
| `FAP_WRITER` | 110 | 112143 | 319288 | 236.510 |
| `CBP_FW_LOCAL_CB_INIT` | 440 | 574 | 631 | 0.467 |

### 最新单次 host 诊断

这两条和下面的细分 zone 来自同一批 clean logs，方便把 host 和 device 对上。

| shape | variant | mode | avg ms | call ms | sync ms | copy Q ms | copy start ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llama_prefill_2k` | `copied_sdpa` | prepared | 0.894 | 0.088 | 0.243 | 0.558 | 0.000 |
| `llama_prefill_16k_chunked` | `copied_chunked` | trace | 0.887 | 0.033 | 0.253 | 0.591 | 0.008 |

### 细粒度 zone 拆分

下面两张表都是单次 `warmup=1 iters=1` 的诊断跑，用来找尾部谁在等谁，不拿来做稳定均值比较。

#### 2K full prefill，`copied_sdpa`，`prepared`

| part | zone | count | critical us | first start us | last end us | span us |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reader | `FAP_READER` | 220 | 205.401 | 0.797 | 1140.567 | 1139.770 |
| reader | `FAP_READER_Q` | 128 | 10.563 | 5.192 | 951.784 | 946.593 |
| reader | `FAP_READER_K_READ` | 1152 | 7.459 | 1.010 | 1134.015 | 1133.004 |
| reader | `FAP_READER_V_READ` | 1152 | 6.845 | 14.649 | 1140.496 | 1125.847 |
| compute | `FAP_COMPUTE` | 660 | 234.785 | 0.950 | 1170.187 | 1169.237 |
| compute | `FAP_COMPUTE_QK_PHASE` | 3408 | 17.801 | 1.183 | 1144.602 | 1143.419 |
| compute | `FAP_COMPUTE_QKV_PHASE` | 3376 | 3.864 | 15.888 | 1145.670 | 1129.782 |
| compute | `FAP_COMPUTE_QKTV_MATMUL_PACK` | 10080 | 1.048 | 19.484 | 1139.819 | 1120.335 |
| compute | `FAP_COMPUTE_ROW_NORM` | 1264 | 1.434 | 33.403 | 1141.557 | 1108.154 |
| compute | `FAP_COMPUTE_SALAD_CORRECT` | 11856 | 0.613 | 31.631 | 1140.656 | 1109.024 |
| writer | `FAP_WRITER` | 220 | 235.792 | 0.810 | 1170.987 | 1170.177 |
| writer | `FAP_WRITER_PREPARE` | 220 | 0.478 | 0.879 | 935.808 | 934.929 |
| writer | `FAP_WRITER_MASK_TEMPLATE` | 220 | 2.483 | 1.359 | 938.286 | 936.927 |
| writer | `FAP_WRITER_STORE_TILES` | 512 | 0.693 | 34.652 | 1170.750 | 1136.099 |
| writer | `FAP_WRITER_STORE_FLUSH` | 512 | 0.032 | 35.325 | 1170.799 | 1135.473 |
| writer | `FAP_WRITER_STORE_BARRIER` | 128 | 0.124 | 42.710 | 1170.947 | 1128.237 |
| writer | `FAP_WRITER_WAIT_OUTPUT` | 512 | 224.752 | 3.921 | 1170.103 | 1166.182 |
| cbp | `CBP_FW_LOCAL_CB_INIT` | 880 | 0.447 | 0.205 | 935.236 | 935.031 |

#### 16K chunked trace，`copied_chunked`

| part | zone | count | critical us | first start us | last end us | span us |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reader | `FAP_READER` | 110 | 209.201 | 0.767 | 210.035 | 209.268 |
| reader | `FAP_READER_CHUNK_START` | 110 | 2.641 | 0.861 | 3.707 | 2.845 |
| reader | `FAP_READER_PAGE_TABLE` | 64 | 0.920 | 2.745 | 4.581 | 1.836 |
| reader | `FAP_READER_Q` | 64 | 10.816 | 7.884 | 20.345 | 12.461 |
| reader | `FAP_READER_K_READ` | 576 | 7.610 | 3.767 | 203.559 | 199.792 |
| reader | `FAP_READER_V_READ` | 576 | 6.790 | 17.121 | 209.933 | 192.812 |
| compute | `FAP_COMPUTE` | 330 | 238.665 | 0.959 | 239.763 | 238.804 |
| compute | `FAP_COMPUTE_QK_PHASE` | 1704 | 19.264 | 1.848 | 214.107 | 212.259 |
| compute | `FAP_COMPUTE_QKV_PHASE` | 1688 | 4.006 | 18.381 | 215.173 | 196.792 |
| compute | `FAP_COMPUTE_QKTV_MATMUL_PACK` | 5040 | 1.012 | 22.195 | 208.851 | 186.656 |
| compute | `FAP_COMPUTE_ROW_NORM` | 632 | 1.442 | 35.874 | 210.615 | 174.741 |
| compute | `FAP_COMPUTE_SALAD_CORRECT` | 5928 | 0.613 | 34.313 | 209.700 | 175.387 |
| writer | `FAP_WRITER` | 110 | 239.625 | 0.802 | 240.485 | 239.683 |
| writer | `FAP_WRITER_CHUNK_START` | 110 | 0.076 | 3.826 | 4.131 | 0.305 |
| writer | `FAP_WRITER_MASK_TEMPLATE` | 110 | 2.485 | 1.334 | 4.033 | 2.699 |
| writer | `FAP_WRITER_PREPARE` | 110 | 0.495 | 0.856 | 1.550 | 0.693 |
| writer | `FAP_WRITER_STORE_TILES` | 256 | 0.676 | 37.156 | 240.284 | 203.127 |
| writer | `FAP_WRITER_STORE_FLUSH` | 256 | 0.031 | 37.827 | 240.333 | 202.506 |
| writer | `FAP_WRITER_STORE_BARRIER` | 64 | 0.132 | 45.168 | 240.433 | 195.265 |
| writer | `FAP_WRITER_WAIT_OUTPUT` | 256 | 228.524 | 3.992 | 239.630 | 235.639 |
| cbp | `CBP_FW_LOCAL_CB_INIT` | 440 | 0.467 | 0.187 | 0.904 | 0.717 |

## 2026-05-29 继续实验：host no-copy 和 q/k chunk sweep

本轮只改本目录 copied fork 和 profile harness，没有改 TTNN official SDPA。新增内容：

- `prepared_no_q_copy` mode：在 measured loop 中跳过 Q 和 chunk-start copy。
- q/k chunk preset：在同一 shape 上扫 `q_chunk` / `k_chunk`。
- 实验目录中的 `CreateKernel` 和 kernel helper include 都指向
  `flash_attention_profile/kernels`，确保 copied fork 使用本目录的 local kernels。
- device profiler 在 warmup 后先 drain，再记录 measured CSV offset；
  `critical_us` 可用于本轮判断。跨 core 的 `span_us` 仍只作为参考，不作为结论依据。

host no-Q-copy 对照：

| shape | variant | mode | avg ms | best ms | worst ms | call ms | sync ms | copy Q ms | copy start ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama_prefill_2k` | `copied_sdpa` | `prepared` | 0.577 | 0.534 | 0.771 | 0.021 | 0.266 | 0.286 | 0.000 |
| `llama_prefill_2k` | `copied_sdpa` | `prepared_no_q_copy` | 0.258 | 0.257 | 0.264 | 0.013 | 0.243 | 0.000 | 0.000 |
| `llama_prefill_16k_chunked` | `copied_chunked` | `prepared` | 1.393 | 0.752 | 2.302 | 0.027 | 0.877 | 0.480 | 0.003 |
| `llama_prefill_16k_chunked` | `copied_chunked` | `trace` | 1.392 | 0.750 | 2.286 | 0.002 | 0.899 | 0.485 | 0.003 |
| `llama_prefill_16k_chunked` | `copied_chunked` | `prepared_no_q_copy` | 0.261 | 0.260 | 0.264 | 0.013 | 0.246 | 0.000 | 0.000 |

注意：chunked 的 `prepared` / `trace` measured loop 会按 iteration 改变
`chunk_start`，后面的 chunk 有更长 prefix；`prepared_no_q_copy`
固定 `chunk_start=0`。所以 chunked 行用来说明 host copy 可以被剥离，
不能直接宣称完整 16K chunked 端到端已经降到 0.261 ms。

q/k chunk sweep，host 侧使用 `prepared_no_q_copy` 去掉 Q copy 噪声：

| shape | q | k | variant | avg ms | best ms | worst ms | sync ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `llama_prefill_2k_q128_k128` | 128 | 128 | `copied_sdpa` | 0.175 | 0.172 | 0.181 | 0.160 |
| `llama_prefill_2k` | 256 | 128 | `copied_sdpa` | 0.258 | 0.257 | 0.263 | 0.243 |
| `llama_prefill_2k_q256_k256` | 256 | 256 | `copied_sdpa` | 0.219 | 0.216 | 0.226 | 0.190 |
| `llama_prefill_16k_chunked_q128_k128` | 128 | 128 | `copied_chunked` | 0.178 | 0.178 | 0.184 | 0.164 |
| `llama_prefill_16k_chunked` | 256 | 128 | `copied_chunked` | 0.286 | 0.260 | 0.368 | 0.253 |
| `llama_prefill_16k_chunked_q256_k256` | 256 | 256 | `copied_chunked` | 0.210 | 0.208 | 0.215 | 0.195 |

对应 device critical path，单次 `warmup=1 iters=1`：

| shape | q | k | reader us | compute us | writer us | writer wait us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama_prefill_2k_q128_k128` | 128 | 128 | 142.992 | 148.888 | 151.898 | 127.833 |
| `llama_prefill_2k` | 256 | 128 | 205.667 | 235.284 | 236.346 | 225.276 |
| `llama_prefill_2k_q256_k256` | 256 | 256 | 142.428 | 182.318 | 183.342 | 170.521 |
| `llama_prefill_16k_chunked_q128_k128` | 128 | 128 | 148.417 | 154.217 | 157.593 | 129.472 |
| `llama_prefill_16k_chunked` | 256 | 128 | 209.335 | 238.901 | 239.915 | 228.729 |
| `llama_prefill_16k_chunked_q256_k256` | 256 | 256 | 145.177 | 185.181 | 186.223 | 173.330 |

本轮最直接的调度结论：

- `q=128,k=128` 在 full 和 chunked-first-chunk 两条路径上都最快。
- `q=256,k=256` 比当前 `q=256,k=128` 好，说明只把 K chunk 放大能减少部分 reader/compute/writer critical path，
  但仍不如把 Q chunk 降到 128。
- `FAP_WRITER_WAIT_OUTPUT` 仍然贴着 `FAP_WRITER` critical path：
  当前 writer 不是主要卡在 store/flush，而是在等 compute 产出。
- reader critical path 也跟着 chunk 变小而下降，但尾部仍是 compute/writer；
  下一步优化应围绕 compute 输出节奏和 writer 消费节奏，而不是先优化 page table 或 chunk-start 读取。

收尾复验，`warmup=1 iters=5`，关闭 device profiler read，只确认方向：

| shape | q | k | variant | mode | avg ms | best ms | worst ms | sync ms |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `llama_prefill_2k_q128_k128` | 128 | 128 | `copied_sdpa` | `prepared_no_q_copy` | 0.182 | 0.174 | 0.210 | 0.159 |
| `llama_prefill_2k` | 256 | 128 | `copied_sdpa` | `prepared_no_q_copy` | 0.340 | 0.308 | 0.404 | 0.241 |
| `llama_prefill_2k_q256_k256` | 256 | 256 | `copied_sdpa` | `prepared_no_q_copy` | 0.251 | 0.234 | 0.295 | 0.196 |
| `llama_prefill_16k_chunked_q128_k128` | 128 | 128 | `copied_chunked` | `prepared_no_q_copy` | 0.191 | 0.178 | 0.234 | 0.163 |
| `llama_prefill_16k_chunked` | 256 | 128 | `copied_chunked` | `prepared_no_q_copy` | 0.271 | 0.261 | 0.303 | 0.246 |
| `llama_prefill_16k_chunked_q256_k256` | 256 | 256 | `copied_chunked` | `prepared_no_q_copy` | 0.217 | 0.208 | 0.246 | 0.194 |

同一轮里，2K full `copied_sdpa` 的 host-copy ablation 复验：

| shape | variant | mode | avg ms | best ms | worst ms | call ms | sync ms | copy Q ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama_prefill_2k` | `copied_sdpa` | `prepared` | 0.634 | 0.539 | 0.800 | 0.036 | 0.253 | 0.342 |
| `llama_prefill_2k` | `copied_sdpa` | `prepared_no_q_copy` | 0.266 | 0.257 | 0.299 | 0.021 | 0.243 | 0.000 |

device profiler 入口也用 `llama_prefill_2k_q128_k128` /
`copied_sdpa` / `prepared_no_q_copy` 复验了一次。`critical_us`
和上面的 q128/k128 记录一致，仍然只使用 `critical_us` 判断：

| zone | critical us |
| --- | ---: |
| `FAP_READER` | 142.231 |
| `FAP_COMPUTE` | 148.106 |
| `FAP_WRITER` | 150.916 |
| `FAP_WRITER_WAIT_OUTPUT` | 127.018 |

## 2026-05-29 pipeline v1：输出 drain group 高度实验

本轮继续只改本目录 copied fork 和 profile harness，没有改 TTNN official SDPA。
新增 `--pipeline` 是实验 binary 的开关：

- `auto`：原 copied 行为。
- `stream_h1`：如果当前 shape 支持 streaming compute，则强制
  `out_out_subblock_h` 的上限从 2 改成 1，让 writer 以更小 row group drain `cb_out`。
- `non_streaming`：关闭 streaming compute，作为确认 streaming 是否必要的对照。
- `pipeline_mode` 已进入 copied SDPA 的 program hash，避免不同编译参数误用同一个 cached program。

correctness smoke：

| pipeline | full SDPA max abs diff | full passed | chunked max abs diff | chunked passed |
| --- | ---: | --- | ---: | --- |
| `auto` | 0.000000 | true | 0.000000 | true |
| `stream_h1` | 0.000000 | true | 0.000000 | true |
| `non_streaming` | 0.015625 | true | 0.015625 | true |

host 侧 no-Q-copy 对比，`warmup=1 iters=5`：

| shape | variant | pipeline | avg ms | best ms | worst ms | call ms | sync ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llama_prefill_2k_q128_k128` | `copied_sdpa` | `auto` | 0.193 | 0.177 | 0.206 | 0.031 | 0.159 |
| `llama_prefill_2k_q128_k128` | `copied_sdpa` | `stream_h1` | 0.185 | 0.174 | 0.214 | 0.022 | 0.159 |
| `llama_prefill_2k_q128_k128` | `copied_sdpa` | `non_streaming` | 0.246 | 0.211 | 0.270 | 0.055 | 0.184 |
| `llama_prefill_16k_chunked_q128_k128` | `copied_chunked` | `auto` | 0.187 | 0.178 | 0.215 | 0.021 | 0.164 |
| `llama_prefill_16k_chunked_q128_k128` | `copied_chunked` | `stream_h1` | 0.201 | 0.179 | 0.221 | 0.030 | 0.168 |
| `llama_prefill_16k_chunked_q128_k128` | `copied_chunked` | `non_streaming` | 0.214 | 0.185 | 0.245 | 0.043 | 0.167 |

device critical path，单次 `warmup=1 iters=1`：

| shape | pipeline | reader us | compute us | writer us | writer wait us | store tiles us |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2K full q128/k128 | `auto` | 143.198 | 149.085 | 152.034 | 127.727 | 4.974 |
| 2K full q128/k128 | `stream_h1` | 142.725 | 148.569 | 151.287 | 126.795 | 3.420 |
| 2K full q128/k128 | `non_streaming` | 142.796 | 151.937 | 156.766 | 131.928 | 6.084 |
| 16K chunked q128/k128 | `auto` | 148.710 | 154.338 | 157.251 | 130.038 | 5.807 |
| 16K chunked q128/k128 | `stream_h1` | 148.011 | 153.749 | 156.656 | 128.666 | 4.431 |
| 16K chunked q128/k128 | `non_streaming` | 148.584 | 156.821 | 161.289 | 132.725 | 6.564 |

这个实验的结论：

- `stream_h1` 只把 `FAP_WRITER_WAIT_OUTPUT` 降低约 0.9-1.4 us，
  幅度不到 1.1%；host 侧 full shape 只有小幅波动级改善，chunked-first-chunk 反而变慢。
  所以它不能作为新的 tuned baseline。
- `non_streaming` 明确更慢，说明当前 streaming compute/drain 路径不能删除。
- store 本身仍然很小，`STORE_TILES` 只有约 3.4-6.6 us；
  writer 贴近 critical path 主要还是因为等 compute 输出，而不是 DRAM store 太慢。
- 单纯把 writer drain group 切得更细不能解决瓶颈。下一步应该改 compute 产出节奏或 per-core work 排布，
  而不是继续调 writer store。

为了验证“更小 chunk 更早产出”是否值得继续做成代码改动，又跑了两个快速 shape：

| shape | q | k | variant | avg ms | best ms | sync ms | 结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 2K full | 64 | 128 | `copied_sdpa` | 0.308 | 0.297 | 0.284 | 比 q128/k128 慢，不继续 |
| 16K chunked | 64 | 128 | `copied_chunked` | 0.332 | 0.316 | 0.293 | 比 q128/k128 慢，不继续 |
| 2K full | 128 | 64 | `copied_sdpa` | 0.232 | 0.222 | 0.203 | 比 q128/k128 慢，不继续 |
| 16K chunked | 128 | 64 | `copied_chunked` | 0.238 | 0.224 | 0.207 | 比 q128/k128 慢，不继续 |

因此当前最好的实验基线仍是 `q=128,k=128,pipeline=auto`。
下一轮不应该继续做简单 chunk 缩小；更有价值的是在 copied compute kernel 中把输出生产点继续细分，
确认最后的等待来自 QK/softmax/QKTV 哪个阶段，然后再做 compute-side partial handoff 或 work rebalance。

## 2026-05-29 grid work-balance 实验

这轮继续只改实验 binary：新增 `--grid X,Y`，把
`SDPAProgramConfig.compute_with_storage_grid_size` 作为可控变量。
它同时作用于官方 TTNN baseline 和 copied 变体，所以这个实验回答的是
program config / work-balance 是否能降低 latency，不是 copied kernel 是否已经超过官方。

correctness smoke：

| setting | full SDPA max abs diff | full passed | chunked max abs diff | chunked passed |
| --- | ---: | --- | ---: | --- |
| `--grid 8,8` | 0.000000 | true | 0.000000 | true |

copied q128/k128，`pipeline=auto`，`prepared_no_q_copy`，`warmup=1 iters=5`：

| shape | grid | avg ms | best ms | worst ms | call ms | sync ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2K full | `auto` | 0.180 | 0.172 | 0.204 | 0.019 | 0.159 |
| 2K full | `8x8` | 0.167 | 0.157 | 0.203 | 0.020 | 0.145 |
| 2K full | `10x8` | 0.175 | 0.161 | 0.205 | 0.025 | 0.148 |
| 2K full | `11x10` | 0.181 | 0.172 | 0.211 | 0.020 | 0.158 |
| 2K full | `8x6` | 0.266 | 0.256 | 0.299 | 0.019 | 0.244 |
| 16K chunked | `auto` | 0.188 | 0.178 | 0.223 | 0.023 | 0.163 |
| 16K chunked | `8x8` | 0.170 | 0.161 | 0.204 | 0.022 | 0.146 |
| 16K chunked | `10x8` | 0.182 | 0.170 | 0.223 | 0.029 | 0.150 |
| 16K chunked | `11x10` | 0.188 | 0.178 | 0.220 | 0.022 | 0.163 |
| 16K chunked | `8x6` | 0.272 | 0.261 | 0.308 | 0.023 | 0.246 |

同一 grid tuning 对官方 TTNN baseline 也有效：

| shape | variant | grid | avg ms | best ms | sync ms |
| --- | --- | --- | ---: | ---: | ---: |
| 2K full | `ttnn_sdpa_baseline` | `auto` | 0.183 | 0.172 | 0.158 |
| 2K full | `ttnn_sdpa_baseline` | `8x8` | 0.170 | 0.157 | 0.144 |
| 16K chunked | `ttnn_chunked_baseline` | `auto` | 0.189 | 0.177 | 0.163 |
| 16K chunked | `ttnn_chunked_baseline` | `8x8` | 0.171 | 0.160 | 0.147 |

device critical path，copied q128/k128，单次 `warmup=1 iters=1`：

| shape | grid | reader us | compute us | writer us | writer wait us | reader count | writer count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full | `auto` | 143.198 | 149.085 | 152.034 | 127.727 | 110 | 110 |
| 2K full | `8x8` | 122.142 | 136.755 | 139.853 | 117.091 | 64 | 64 |
| 16K chunked | `auto` | 148.710 | 154.338 | 157.251 | 130.038 | 110 | 110 |
| 16K chunked | `8x8` | 124.104 | 138.658 | 141.544 | 117.170 | 64 | 64 |

解释：

- 当前 shape 是 `B=1,H=8,q_num_chunks=16`。
- 默认 11x10/auto grid 会让 program factory 选择 `nh_parallel=8`、
  `q_parallel=13`，于是 `q_per_core=ceil(16/13)=2`，但 q split 不整齐。
- `8x8` 对应 `nh_parallel=8,q_parallel=8,q_per_core=2`，64 个 reader/writer core 都有规则工作。
- `8x6` 变慢，因为 `q_parallel=6,q_per_core=3`，每个 core 的 Q chunk 负担变大。

结论：

- 8x8 是本轮最有价值的方向：它把 copied full 的 `sync_ms` 从 0.159 降到 0.145，
  copied chunked-first-chunk 从 0.163 降到 0.146。
- device 上也同步下降：2K full 的 writer critical 从 152.034 us 降到 139.853 us，
  `FAP_WRITER_WAIT_OUTPUT` 从 127.727 us 降到 117.091 us。
- 但官方 TTNN baseline 使用同样 grid 时也得到几乎相同收益，所以这不是 copied kernel
  已经比官方快，而是说明默认 grid 选择在这个 common prefill shape 上不够理想。
- 下一步代码实验应该把这个手动 `--grid 8,8` 变成 copied op 的 shape-aware grid policy：
  优先选择能让 `q_parallel` 整除 `q_num_chunks` 的 grid，而不是简单使用全设备 grid。

## 2026-05-29 copied balanced-q grid policy 实验

这轮把上面的手动 `--grid 8,8` 变成 copied-only 实验策略：

- 新参数：`--grid-policy copied_balanced_q`。
- 显式 `--grid X,Y` 仍然最高优先级。
- 如果没有显式 `--grid`，只有 `copied_sdpa` / `copied_chunked` 可能会按 shape 自动选 grid。
- 官方 TTNN baseline 仍使用 `grid=auto`，所以这轮能测试 copied policy 是否能在同一 CLI 下超过官方默认。
- policy 不是按 benchmark 名硬编码。它根据 `B`、`H`、`q_num_chunks` 和设备 grid，
  选择让 `q_parallel` 尽量整除 `q_num_chunks` 的矩形 grid。
- policy 只在两个条件都满足时覆盖：
  当前 full-grid 的 `q_parallel` 不能整除 `q_num_chunks`；
  balanced grid 不增加 `q_per_core`。

correctness smoke：

| setting | full SDPA max abs diff | full passed | chunked max abs diff | chunked passed |
| --- | ---: | --- | ---: | --- |
| `--grid-policy copied_balanced_q` | 0.000000 | true | 0.000000 | true |

host 对比，`prepared_no_q_copy`，`warmup=2 iters=10`：

| shape | variant | grid policy | resolved grid | avg ms | best ms | worst ms | call ms | sync ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2K full q128/k128 | `ttnn_sdpa_baseline` | `copied_balanced_q` | `auto` | 0.200 | 0.194 | 0.212 | 0.039 | 0.157 |
| 2K full q128/k128 | `copied_sdpa` | `copied_balanced_q` | `8x8` | 0.184 | 0.175 | 0.200 | 0.036 | 0.144 |
| 16K chunked q128/k128 | `ttnn_chunked_baseline` | `copied_balanced_q` | `auto` | 0.209 | 0.206 | 0.219 | 0.043 | 0.162 |
| 16K chunked q128/k128 | `copied_chunked` | `copied_balanced_q` | `8x8` | 0.173 | 0.169 | 0.183 | 0.023 | 0.148 |

相对官方默认：

| shape | copied policy speedup | avg ms delta | sync ms delta |
| --- | ---: | ---: | ---: |
| 2K full q128/k128 | 1.09x | -0.016 | -0.013 |
| 16K chunked q128/k128 | 1.21x | -0.036 | -0.014 |

policy guard 验证：

| shape | q chunks | default q split | policy resolved grid | 结论 |
| --- | ---: | --- | --- | --- |
| q128/k128 full/chunked | 16 | `q_parallel=13,q_per_core=2` | `8x8` | q split 不整齐，覆盖有效 |
| q256/k128 full | 8 | `q_parallel=8,q_per_core=1` | `auto` | 默认已整齐，不覆盖 |
| q256/k256 full/chunked | 8 | `q_parallel=8,q_per_core=1` | `auto` | 默认已整齐，不覆盖 |
| q64/k128 full/chunked | 32 | `q_parallel=13,q_per_core=3` | `auto` | balanced 会增加 q_per_core，不覆盖 |

这轮结果说明：

- copied op 现在在 common q128/k128 prefill shape 上能通过自动 grid policy 跑过官方默认路径。
- 这不是新数学 kernel 的收益，而是 scheduling policy 收益；官方如果传同样 `--grid 8,8` 也能受益。
- 但作为 copied fork 的下一步优化入口，它是有效的：不需要用户手动传 `--grid 8,8`，
  copied 变体能自动规避 full-grid 的不规则 q split。
- q256 和 q64 的 guard 已经避免了本轮发现的无意义覆盖；下一步需要继续测不同 head 数和 batch。

## 2026-05-29 分块和 pipeline depth 对 TTNN tuned 的实验

这轮回答的问题是：在 `grid=8x8` 这种 TTNN 也能手动 tune 的配置下，
copied fork 是否已经在底层 device kernel 上超过 TTNN tuned baseline。

本轮新增两个实验旋钮：

- `--chunks Q,K`：覆盖 shape 的 q/k chunk size，避免为每个分块写一个 preset。
- `--pipeline-depth N`：只改变 copied streaming output CB 深度，默认 2。

correctness：

| setting | full max abs diff | full passed | chunked max abs diff | chunked passed |
| --- | ---: | --- | ---: | --- |
| `q=128,k=256,grid=8x8,pipeline_depth=4` | 0.000000 | true | 0.000000 | true |

host no-copy 对照，`grid=8x8`，`prepared_no_q_copy`。
这里 `sync_ms` 比 `avg_ms` 更接近 device completion，但真正的 kernel 判断仍看后面的
device profiler `critical_us`：

| shape | q | k | variant | depth | avg ms | best ms | call ms | sync ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2K full | 128 | 128 | `ttnn_sdpa_baseline` | 2 | 0.186 | 0.176 | 0.036 | 0.146 |
| 2K full | 128 | 128 | `copied_sdpa` | 2 | 0.175 | 0.172 | 0.027 | 0.145 |
| 2K full | 128 | 128 | `copied_sdpa` | 4 | 0.188 | 0.174 | 0.035 | 0.149 |
| 2K full | 128 | 256 | `ttnn_sdpa_baseline` | 2 | 0.175 | 0.169 | 0.031 | 0.140 |
| 2K full | 128 | 256 | `copied_sdpa` | 2 | 0.181 | 0.170 | 0.036 | 0.140 |
| 2K full | 128 | 256 | `copied_sdpa` | 4 | 0.190 | 0.178 | 0.044 | 0.140 |
| 16K chunked | 128 | 128 | `ttnn_chunked_baseline` | 2 | 0.181 | 0.174 | 0.030 | 0.147 |
| 16K chunked | 128 | 128 | `copied_chunked` | 2 | 0.163 | 0.160 | 0.014 | 0.147 |
| 16K chunked | 128 | 128 | `copied_chunked` | 4 | 0.183 | 0.178 | 0.033 | 0.147 |
| 16K chunked | 128 | 256 | `ttnn_chunked_baseline` | 2 | 0.206 | 0.185 | 0.086 | 0.113 |
| 16K chunked | 128 | 256 | `copied_chunked` | 2 | 0.163 | 0.159 | 0.017 | 0.144 |
| 16K chunked | 128 | 256 | `copied_chunked` | 4 | 0.183 | 0.172 | 0.038 | 0.141 |

分块 sweep，`grid=8x8,pipeline_depth=2`，只列 `sync_ms`：

| q | k | full TTNN | full copied | chunked TTNN | chunked copied | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 64 | 0.240 | 0.238 | 0.239 | 0.241 | Q 太小，core/loop 开销变大 |
| 64 | 128 | 0.233 | 0.234 | 0.236 | 0.236 | 仍慢 |
| 128 | 64 | 0.200 | 0.198 | 0.202 | 0.201 | K 太小，K loop 太多 |
| 128 | 128 | 0.146 | 0.145 | 0.147 | 0.147 | 稳定 baseline |
| 128 | 256 | 0.140 | 0.140 | 0.142 | 0.142 | 本轮最好的分块方向 |
| 128 | 512 | 0.155 | 0.158 | 0.159 | 0.158 | K 继续变大开始回退 |
| 128 | 1024 | 0.185 | 0.186 | 0.193 | 0.190 | 明显回退 |
| 256 | 128 | 0.237 | 0.247 | 0.240 | 0.242 | Q 太大，单 core 负担变重 |
| 256 | 256 | 0.185 | 0.185 | 0.187 | 0.186 | 比 q128/k256 慢 |
| 256 | 512 | 0.172 | 0.167 | 0.155 | 0.173 | 不稳定，且 full/chunked 不一致 |

copied pipeline depth sweep，`grid=8x8`，只列 `sync_ms`：

| shape | q | k | depth 1 | depth 2 | depth 3 | depth 4 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2K full | 128 | 128 | 0.143 | 0.144 | 0.141 | 0.143 | 波动级，复验 depth 4 无收益 |
| 16K chunked | 128 | 128 | 0.146 | 0.145 | 0.144 | 0.157 | depth 4 回退 |
| 2K full | 128 | 256 | 0.149 | 0.140 | 0.140 | 0.138 | host sync 看似略好，device critical 不支持 |
| 16K chunked | 128 | 256 | 0.143 | 0.140 | 0.142 | 0.143 | 基本无差异 |

device profiler，`critical_us`，`grid=8x8,pipeline_depth=2`：

| shape | q | k | reader us | compute us | writer us | writer wait us | store tiles us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full copied | 128 | 128 | 122.647 | 137.297 | 139.787 | 117.250 | 6.027 |
| 2K full copied | 128 | 256 | 125.957 | 133.717 | 134.727 | 106.841 | 3.024 |
| 16K chunked copied | 128 | 128 | 123.879 | 138.469 | 141.876 | 117.160 | 6.461 |
| 16K chunked copied | 128 | 256 | 128.178 | 136.130 | 137.443 | 107.146 | 4.544 |

pipeline depth 的 device profiler 对照，`q=128,k=256,grid=8x8`：

| shape | depth | reader us | compute us | writer us | writer wait us | store tiles us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full copied | 2 | 125.957 | 133.717 | 134.727 | 106.841 | 3.024 |
| 2K full copied | 4 | 126.633 | 134.842 | 135.713 | 108.338 | 2.116 |
| 16K chunked copied | 2 | 128.178 | 136.130 | 137.443 | 107.146 | 4.544 |
| 16K chunked copied | 4 | 128.836 | 136.527 | 137.763 | 107.877 | 3.196 |

官方 TTNN tuned 的同一 q/k/grid coarse kernel critical：

| shape | q | k | BRISC us | TRISC us | NCRISC us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K full TTNN | 128 | 256 | 134.956 | 133.859 | 126.239 |
| 16K chunked TTNN | 128 | 256 | 137.619 | 136.829 | 129.031 |

结论：

- 应该继续先测分块和流水线深度，但判定目标必须是 device `critical_us`
  比 TTNN tuned 低，而不是 host `avg_ms` 偶然低。
- `q=128,k=256` 是目前最有价值的分块：相比 copied q128/k128，
  full writer critical 从 139.787 us 降到 134.727 us，chunked 从
  141.876 us 降到 137.443 us；`FAP_WRITER_WAIT_OUTPUT` 也下降约 10 us。
- 但这还不是 copied kernel 超过 TTNN tuned。官方 TTNN 用同样
  `q=128,k=256,grid=8x8` 时 coarse critical path 基本一样。
- `pipeline_depth=4` 没有形成 device 侧收益，虽然 store tiles 稍低，
  但 writer wait 和 compute critical 没降；不应作为新 baseline。
- 现在的 tuned copied baseline 应更新为
  `q=128,k=256,grid=8x8,pipeline_depth=2`，下一步要改 compute 产出节奏，
  才有机会真正超过 TTNN tuned。

## 2026-05-29 pipeline v2 实验计划：compute-side handoff

这轮不改 TTNN official SDPA，也不新增第二套实验目录。
只在 `flash_attention_profile` 增加 copied-only pipeline 模式。
目标是回答：在 `q=128,k=256,grid=8x8,pipeline_depth=2` 这个 TTNN 也能手动 tuned 的配置下，
copied fork 能否通过 compute 内部产出节奏调整降低 device critical path。

当前数据说明 writer 的主要时间不是 store：

- 2K full copied tuned：`FAP_WRITER=134.727 us`，
  `FAP_WRITER_WAIT_OUTPUT=106.841 us`，`STORE_TILES=3.024 us`。
- 16K chunked copied tuned：`FAP_WRITER=137.443 us`，
  `FAP_WRITER_WAIT_OUTPUT=107.146 us`，`STORE_TILES=4.544 us`。

所以这轮不继续优先优化 writer store，而是测试两类 compute-side 修改：

1. `qktv_h1`
   - 现有 `stream_h1` 只限制 host subblock solver 的输出 group 高度；
     `streaming_qktv_h()` 仍可能把 V-matmul row group 合并回 2。
   - 新模式要真正强制 Phase-2 `QKTV` 的 matmul/drain row group 高度为 1，
     验证更细粒度的 output handoff 是否能减少 writer 等 compute 的尾部时间。
2. `salad_first`
   - 现有 Phase-2 main loop 是：先为 previous group 计算 `exp(max_old-max_new)`，
     然后做 current group 的 `QKTV` matmul，最后才把 previous group 做 SALAD correction、
     normalize/push 给 writer。
   - 新模式在最后一次 K chunk 上尝试把 previous group 的 SALAD+normalize+push 提前到 current
     group 的 `QKTV` matmul 前，让 writer 更早拿到已归一化输出。
   - 这个实验可能减少 writer wait，也可能拉长 compute critical path；只有 device profiler 能判断。

判定标准：

- correctness 必须通过，仍以官方 TTNN baseline 输出对比 copied 输出。
- 主要性能指标是 device profiler `critical_us`，host `avg_ms` 只作为补充。
- copied tuned 必须低于官方 TTNN tuned coarse critical，才算底层 kernel 真正跑过 TTNN tuned。
- 如果 `FAP_WRITER_WAIT_OUTPUT` 降低但 `FAP_COMPUTE` 增加更多，则该模式不作为下一轮 baseline。

## 2026-05-29 pipeline v2 实验结果

本轮先做了目录收敛：删除原来的双目录管理方式，只保留
`tt_metal/programming_examples/profiler/flash_attention_profile`。
`flash_attention_profile_experiments` 不再作为第二个 binary 或第二套源码入口存在。
官方 TTNN baseline 仍通过 `ttnn_*_baseline` 变体在同一 binary 中比较。

新增 copied-only pipeline：

- `qktv_h1`：强制 Phase-2 `QKTV` matmul/drain row group height 为 1。
- `salad_first`：最后 K chunk 上提前 previous group 的 SALAD+normalize+push。
- `qktv_h1_salad_first`：两者同时启用。

correctness：

| setting | full max abs diff | full passed | chunked max abs diff | chunked passed |
| --- | ---: | --- | ---: | --- |
| smoke `qktv_h1` | 0.000000 | true | 0.000000 | true |
| smoke `salad_first` | 0.000000 | true | 0.000000 | true |
| smoke `qktv_h1_salad_first` | 0.000000 | true | 0.000000 | true |
| 2K full/chunked `qktv_h1`, q128/k256/grid8x8 | 0.005859 | true | 0.005859 | true |
| 2K full/chunked `salad_first`, q128/k256/grid8x8 | 0.000000 | true | 0.000000 | true |

host no-copy 对比，`q=128,k=256,grid=8x8,pipeline_depth=2`：

| shape | variant | pipeline | avg ms | best ms | worst ms | sync ms | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2K full | `ttnn_sdpa_baseline` | `auto` | 0.155 | 0.154 | 0.158 | 0.143 | 官方 tuned |
| 2K full | `copied_sdpa` | `auto` | 0.161 | 0.155 | 0.176 | 0.142 | copied tuned baseline |
| 2K full | `copied_sdpa` | `qktv_h1` | 0.794 | 0.185 | 3.468 | 0.707 | 不稳定且明显回退 |
| 2K full | `copied_sdpa` | `salad_first` | 0.266 | 0.163 | 0.875 | 0.202 | 回退 |
| 2K full | `copied_sdpa` | `qktv_h1_salad_first` | timeout | - | - | - | tuned full shape 不可用 |
| 16K chunked | `ttnn_chunked_baseline` | `auto` | 0.159 | 0.158 | 0.165 | 0.146 | 官方 tuned |
| 16K chunked | `copied_chunked` | `auto` | 0.161 | 0.158 | 0.168 | 0.145 | copied tuned baseline |
| 16K chunked | `copied_chunked` | `qktv_h1` | 0.174 | 0.159 | 0.223 | 0.145 | host 基本持平，avg 回退 |
| 16K chunked | `copied_chunked` | `salad_first` | 0.188 | 0.161 | 0.235 | 0.145 | avg 回退 |
| 16K chunked | `copied_chunked` | `qktv_h1_salad_first` | 0.170 | 0.159 | 0.208 | 0.147 | 进程结束时报 bus error，不作为候选 |

device profiler，`critical_us`，2K full `copied_sdpa`：

| pipeline | reader us | compute us | writer us | writer wait us | store tiles us | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `auto` | 126.794 | 134.502 | 135.441 | 107.921 | 1.841 | baseline |
| `qktv_h1` | 127.388 | 136.593 | 137.281 | 106.944 | 0.750 | wait/store 降，但总 critical 回退 |
| `salad_first` | 126.329 | 134.877 | 136.208 | 106.639 | 1.830 | wait 降，但 writer/critical 回退 |

device profiler，`critical_us`，16K chunked `copied_chunked`：

| pipeline | reader us | compute us | writer us | writer wait us | store tiles us | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `auto` | 129.210 | 136.401 | 138.436 | 108.590 | 3.801 | baseline |
| `qktv_h1` | 128.465 | 137.053 | 137.695 | 107.105 | 0.805 | writer 低于 auto，但 compute 上升 |
| `salad_first` | 129.314 | 137.667 | 138.608 | 106.816 | 1.441 | wait 降，但总 critical 回退 |

本轮结论：

- 两个实验都验证了前面的判断：writer 的 store 不是主瓶颈，`STORE_TILES`
  可以降到 1 us 左右，但总 critical 不会因此明显下降。
- `qktv_h1` 把 row group 拆细后，writer wait 和 store tiles 确实下降；
  但 `QKTV` 事件数量从 1728 增到 5184，compute critical 上升，full path 明显回退。
- `salad_first` 能更早交付部分输出，writer wait 降约 1 us；
  但 SALAD/normalize 提前后没有形成足够重叠，writer critical 仍高于 auto。
- `qktv_h1_salad_first` 在 smoke 正确，但 tuned full shape 超时，16K chunked 结束时报 bus error，
  说明组合模式目前不是可用优化路径。
- 当前 tuned baseline 仍保持：
  `q=128,k=256,grid=8x8,pipeline_depth=2,pipeline=auto`。
  copied fork 仍没有在底层 device critical path 上明确超过 TTNN tuned。

下一步不应继续拆 writer row group，而应该转向减少 compute 侧重复工作或尾部不均衡：

1. 优先做 per-core work assignment / grid policy 扩展实验，覆盖更多 `H`、`B`、`q_num_chunks`，
   确认 `copied_balanced_q` 不只在 `H=8,q_chunks=16` 有收益。
2. 在 compute 内进一步拆 `QK_PHASE`：它的 max critical 约 15-17 us，
   比 `QKTV_MATMUL_PACK`、`ROW_NORM`、`SALAD_CORRECT` 都更值得优化。
3. 如果继续做 handoff，应该避免把 `QKTV` group 简单拆细；需要把 QK/softmax/QKTV 的 producer-consumer
   顺序重排，而不是增加更多小 matmul 调用。
4. 对 `qktv_h1_salad_first` 先做最小复现和 CB 生命周期检查，确认 bus error/timeout 的根因，
   在根因未清楚前不继续把它纳入性能对比。

## 当前瓶颈判断

当前 copied path 还没有比 TTNN official path 做新的数学/compute kernel 优化，所以 device critical
应当接近 TTNN tuned。当前最新结论仍是：
`q=128,k=256,grid=8x8,pipeline_depth=2,pipeline=auto` 可以复现 tuned 级别，
但 copied fork 还没有在底层 device critical path 上明确超过 TTNN tuned。

host 边界上，`prepared` 模式的主要开销不是 `call_avg_ms`，而是 Q runtime copy 和同步等待。
因此本目录保留 `prepared_no_q_copy` 作为 kernel/device 对比模式：

- 2K full SDPA：`copy_q_avg_ms` 约 0.33-0.35 ms，`sync_avg_ms` 约 0.24-0.25 ms。
- 16K chunked：`copy_q_avg_ms` 约 0.50-0.58 ms，`sync_avg_ms` 约 0.46-0.49 ms。
- 16K chunked trace 把 `call_avg_ms` 降到 0.003-0.005 ms，但总时间仍接近 0.95 ms，
  因为 Q copy 和 device completion wait 仍然存在。

device 边界上，2K full SDPA 和 16K chunked trace 都显示：

- `FAP_WRITER` 和 `FAP_COMPUTE` 的 critical time 几乎贴着总 kernel critical path。
- `FAP_READER` 比 compute/writer 早结束；在最新 q128/k256 tuned 里约早 8-10 us。
- `CBP_FW_LOCAL_CB_INIT` 只有约 0.45 us，不是本轮瓶颈。

因此本轮最可信的瓶颈结论是：

1. host/runtime 层：measured loop 仍被 Q runtime copy 和同步等待主导。trace 能消掉大部分 host call
   overhead，但不能自动消掉 Q copy 和设备执行时间。
2. device 层：reader 不是尾部瓶颈，chunked trace 里 `FAP_READER_CHUNK_START=2.641 us`、
   `FAP_READER_PAGE_TABLE=0.920 us`，都很小；2K/16K 两个 shape 里 reader 也都比 compute/writer 早
   收尾，说明它更像是先把输入搬完，然后等待后面的计算和写回。
3. compute/writer 层：真正贴近 critical path 的是 `FAP_COMPUTE` 和 `FAP_WRITER`，其中
   latest tuned 里 `FAP_WRITER_WAIT_OUTPUT` 仍约 `107-109 us`，明显大于
   `STORE_TILES` / `STORE_FLUSH` / `STORE_BARRIER`，所以 writer 的主要空闲时间不是在写，
   而是在等 compute 输出可用；compute 侧则主要由 `QK_PHASE` 和相关阶段撑起。

这里还不能严肃宣称“某个硬件在某个 cycle 精确空闲”，因为当前
`FAP_*` 是粗粒度 zone。
如果要回答精确 idle 原因，下一步需要加更细事件：

- compute kernel 内：QK、softmax、normalization、QKTV、mask、
  reduce 的 zone。
- dataflow kernel 内：DRAM read、page-table read、CB reserve/push、
  writer pack/store 的 zone。
- CB 等待：reader 等待空 CB、compute 等待输入 CB、
  writer 等待输出 CB 的计数或 cycle。
- per-core work assignment：确认尾部是否来自 head/batch work 不均衡，
  或最后一个 q/k chunk。

## 2026-05-29 QK phase 细分 profile

本轮只改 profiling instrumentation，不改调度和数学逻辑。在
`FAP_COMPUTE_QK_PHASE` 内新增子 zone：

- `FAP_COMPUTE_QK_SETUP`：Q 输入等待、format/pack 配置、QK matmul MOP 初始化。
- `FAP_COMPUTE_QK_MATMUL_PACK`：`blocked_matmul_and_pack` 的 QK matmul/pack 主体。
- `FAP_COMPUTE_QK_SOFTMAX_EXP_SUM`：上一组 row 的 `sub_exp_block_bcast_cols`，
  即减 max、exp、sum 的 softmax 前半段。
- `FAP_COMPUTE_QK_MASK`：轻量 causal/padding mask stamp。
- `FAP_COMPUTE_QK_MAX_REDUCE`：QK row max reduce。

命令：

```bash
rm -rf generated/profiler/.logs
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset llama_prefill_2k --chunks 128,256 \
  --variant copied_sdpa --mode prepared_no_q_copy \
  --pipeline auto --pipeline-depth 2 --grid 8,8 \
  --warmup 0 --iters 1

rm -rf generated/profiler/.logs
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset llama_prefill_16k_chunked --chunks 128,256 \
  --variant copied_chunked --mode prepared_no_q_copy \
  --pipeline auto --pipeline-depth 2 --grid 8,8 \
  --warmup 0 --iters 1
```

`warmup=0` 的 host `avg_ms/call_avg_ms` 会包含首次 program/JIT/cache 路径，
本节只使用 device profiler 的 `critical_us`。子 zone 的 `critical_us`
是各 zone 自己的 max duration，不是可以直接相加的父子拆账；这里用它判断哪一段的最长事件最重。

2K full，`llama_prefill_2k_q128_k256 copied_sdpa prepared_no_q_copy`：

| zone | count | avg cycles | max cycles | critical us | 相对 QK phase | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `FAP_COMPUTE_QK_PHASE` | 1536 | 9218 | 21893 | 16.217 | 100.0% | 父 zone |
| `FAP_COMPUTE_QK_MATMUL_PACK` | 5664 | 1515 | 15404 | 11.410 | 70.4% | 最大子项 |
| `FAP_COMPUTE_QK_SETUP` | 3008 | 266 | 4302 | 3.187 | 19.7% | 配置/等待有可见成本 |
| `FAP_COMPUTE_QK_SOFTMAX_EXP_SUM` | 2720 | 630 | 2765 | 2.048 | 12.6% | 次要但可优化 |
| `FAP_COMPUTE_QK_MASK` | 2880 | 196 | 1230 | 0.911 | 5.6% | 不是主瓶颈 |
| `FAP_COMPUTE_QK_MAX_REDUCE` | 2880 | 293 | 696 | 0.516 | 3.2% | 不是主瓶颈 |

16K chunked first chunk，`llama_prefill_16k_chunked_q128_k256 copied_chunked prepared_no_q_copy`：

| zone | count | avg cycles | max cycles | critical us | 相对 QK phase | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `FAP_COMPUTE_QK_PHASE` | 1536 | 9394 | 23395 | 17.330 | 100.0% | 父 zone |
| `FAP_COMPUTE_QK_MATMUL_PACK` | 5664 | 1560 | 17010 | 12.600 | 72.7% | 最大子项 |
| `FAP_COMPUTE_QK_SETUP` | 3008 | 269 | 4591 | 3.401 | 19.6% | 配置/等待有可见成本 |
| `FAP_COMPUTE_QK_SOFTMAX_EXP_SUM` | 2720 | 630 | 2848 | 2.110 | 12.2% | 次要但可优化 |
| `FAP_COMPUTE_QK_MASK` | 2880 | 197 | 1229 | 0.910 | 5.3% | 不是主瓶颈 |
| `FAP_COMPUTE_QK_MAX_REDUCE` | 2880 | 294 | 691 | 0.512 | 3.0% | 不是主瓶颈 |

结论：

- QK phase 的第一优化目标应当是 `QK_MATMUL_PACK`，full 和 chunked 都是最大子项，
  critical 约 11.4-12.6 us，占父 `QK_PHASE` 最长事件的约 70-73%。
- `QK_SETUP` 不是纯数学，但有 3.2-3.4 us 的最长事件，说明 pack/unpack reconfig、
  MOP init/reinit、Q 输入等待这条路径也值得继续拆。
- `QK_SOFTMAX_EXP_SUM` 约 2.0-2.1 us，有优化价值，但优先级低于 QK matmul。
- `QK_MASK` 和 `QK_MAX_REDUCE` 都低于 1 us，不是当前导致 writer 等 compute 的主要原因。
- 下一步不要先优化 mask/reduce；应继续把 `QK_MATMUL_PACK` 拆成
  data-format reconfig、MOP init/reinit、真实 `blocked_matmul_and_pack` 三段，
  并实验 QK subblock 宽度/高度或减少 reinit 次数是否能降低 worst-case matmul event。

## 2026-05-29 QK overlap / 串行化验证

为了验证“看起来像流水线，实际可能在同一个 TRISC 上顺序执行”的假设，
本轮继续细分：

- `FAP_COMPUTE_QK_WAIT_Q`：`cb_wait_front(cb_q_in, q_wait_tiles)`。
- `FAP_COMPUTE_QK_INIT`：初始 format/pack 配置和 `mm_no_mop_init_short`。
- `FAP_COMPUTE_QK_REINIT_AFTER_SOFTMAX`：softmax 后切回 QK matmul 的 pack/unpack
  reconfig 和 `mm_no_mop_reinit_short`。
- `FAP_COMPUTE_QK_MATMUL_RECONFIG`：matmul 前 data-format reconfig。
- `FAP_COMPUTE_QK_MATMUL_BODY`：真实 `blocked_matmul_and_pack`。

注意：这轮 instrumentation 更细，会增加 profiler 事件数；不同轮次之间的 `count`
不要横向比较。本节结论只使用两个信号：

- zone `critical_us` 判断最长事件是谁。
- raw `profile_log_device.csv` 的同一 `core_x/core_y/RISC` interval 判断是否 overlap。

2K full，`llama_prefill_2k_q128_k256 copied_sdpa prepared_no_q_copy`：

| zone | count | avg cycles | max cycles | critical us | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAP_COMPUTE_QK_PHASE` | 936 | 10771 | 22132 | 16.394 | 父 zone |
| `FAP_COMPUTE_QK_MATMUL_PACK` | 3056 | 1920 | 15723 | 11.647 | QK matmul 包装整体 |
| `FAP_COMPUTE_QK_MATMUL_BODY` | 3000 | 1824 | 15613 | 11.565 | 主要瓶颈就是 matmul body |
| `FAP_COMPUTE_QK_MATMUL_RECONFIG` | 3048 | 29 | 64 | 0.047 | reconfig 很小 |
| `FAP_COMPUTE_QK_SETUP` | 1656 | 431 | 4558 | 3.376 | 主要来自 Q 等待 |
| `FAP_COMPUTE_QK_WAIT_Q` | 1632 | 230 | 4336 | 3.212 | setup 内最大项 |
| `FAP_COMPUTE_QK_INIT` | 1632 | 128 | 323 | 0.239 | init 很小 |
| `FAP_COMPUTE_QK_SOFTMAX_EXP_SUM` | 1440 | 628 | 2424 | 1.796 | 次要 |
| `FAP_COMPUTE_QK_REINIT_AFTER_SOFTMAX` | 1440 | 86 | 158 | 0.117 | reinit 很小 |
| `FAP_COMPUTE_QK_MASK` | 1536 | 276 | 1241 | 0.919 | 非主瓶颈 |
| `FAP_COMPUTE_QK_MAX_REDUCE` | 1536 | 290 | 786 | 0.582 | 非主瓶颈 |

16K chunked first chunk，`llama_prefill_16k_chunked_q128_k256 copied_chunked prepared_no_q_copy`：

| zone | count | avg cycles | max cycles | critical us | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAP_COMPUTE_QK_PHASE` | 936 | 10969 | 23616 | 17.493 | 父 zone |
| `FAP_COMPUTE_QK_MATMUL_PACK` | 3056 | 1969 | 16820 | 12.459 | QK matmul 包装整体 |
| `FAP_COMPUTE_QK_MATMUL_BODY` | 3000 | 1875 | 16701 | 12.371 | 主要瓶颈就是 matmul body |
| `FAP_COMPUTE_QK_MATMUL_RECONFIG` | 3048 | 29 | 64 | 0.047 | reconfig 很小 |
| `FAP_COMPUTE_QK_SETUP` | 1656 | 432 | 4498 | 3.332 | 主要来自 Q 等待 |
| `FAP_COMPUTE_QK_WAIT_Q` | 1632 | 230 | 4275 | 3.167 | setup 内最大项 |
| `FAP_COMPUTE_QK_INIT` | 1632 | 128 | 336 | 0.249 | init 很小 |
| `FAP_COMPUTE_QK_SOFTMAX_EXP_SUM` | 1440 | 637 | 2760 | 2.044 | 次要 |
| `FAP_COMPUTE_QK_REINIT_AFTER_SOFTMAX` | 1440 | 86 | 158 | 0.117 | reinit 很小 |
| `FAP_COMPUTE_QK_MASK` | 1536 | 278 | 1245 | 0.922 | 非主瓶颈 |
| `FAP_COMPUTE_QK_MAX_REDUCE` | 1536 | 289 | 807 | 0.598 | 非主瓶颈 |

raw CSV overlap 检查，按同一 core 和同一 RISC 配对 interval：

| shape | interval pair | overlap pairs | overlap cycles | 顺序间隙 |
| --- | --- | ---: | ---: | --- |
| 2K full | `QK_SOFTMAX_EXP_SUM` vs `QK_MATMUL_BODY` | 0 | 0 | softmax -> reinit 平均 36.7 cycles，reinit -> matmul 平均 37.2 cycles |
| 2K full | `QK_REINIT_AFTER_SOFTMAX` vs `QK_MATMUL_BODY` | 0 | 0 | reinit 和 matmul body 也是顺序 |
| 2K full | `QK_MATMUL_RECONFIG` vs `QK_MATMUL_BODY` | 0 | 0 | reconfig -> body 平均 32.5 cycles |
| 16K chunked | `QK_SOFTMAX_EXP_SUM` vs `QK_MATMUL_BODY` | 0 | 0 | softmax -> reinit 平均 36.7 cycles，reinit -> matmul 平均 37.0 cycles |
| 16K chunked | `QK_REINIT_AFTER_SOFTMAX` vs `QK_MATMUL_BODY` | 0 | 0 | reinit 和 matmul body 也是顺序 |
| 16K chunked | `QK_MATMUL_RECONFIG` vs `QK_MATMUL_BODY` | 0 | 0 | reconfig -> body 平均 32.3 cycles |

这个结果说明：

- 当前 QK phase 内部没有形成“softmax(prev) 与 matmul(cur) 在同一 TRISC 上 overlap”的流水。
  代码顺序就是：上一组 softmax/sub-exp，随后 reinit，再做当前组 QK matmul body。
- 这不是单纯的 reconfig 问题。`QK_MATMUL_RECONFIG` 和
  `QK_REINIT_AFTER_SOFTMAX` 都只有约 0.05-0.12 us，真正重的是
  `QK_MATMUL_BODY`，full/chunked 分别约 11.6 / 12.4 us。
- `QK_WAIT_Q` 的 worst case 约 3.2 us，说明 reader/compute 在某些 row group
  还有输入等待，但它低于 matmul body，优先级第二。
- 因此，下一步如果要追 overlap，不能只把 reconfig 挪来挪去；需要改变 QK phase
  的 producer/consumer 排布，让 softmax 和下一段 matmul 不再竞争同一个 TRISC/DST/pack
  执行路径。更现实的近期实验是先做 `QK_MATMUL_BODY` 的 subblock 形状扫描，
  同时给 `QK_WAIT_Q` 加 reader/CB wait 对照，确认是否有输入预取不足。

## 2026-05-29 QK matmul body 细分 profile

为了判断是否是 buffer / pipeline wait 卡住，本轮把
`blocked_matmul_and_pack<true, KT_stride, KT_stride>` 内部继续拆开，只对 QK 调用启用：

- `FAP_COMPUTE_QK_BODY_TILE_REGS_ACQUIRE`
- `FAP_COMPUTE_QK_BODY_MATMUL_LOOP`
- `FAP_COMPUTE_QK_BODY_TILE_REGS_COMMIT`
- `FAP_COMPUTE_QK_BODY_TILE_REGS_WAIT`
- `FAP_COMPUTE_QK_BODY_PACK_CONFIG`
- `FAP_COMPUTE_QK_BODY_PACK_ROWS`
- `FAP_COMPUTE_QK_BODY_REDUCE_SIGNAL`
- `FAP_COMPUTE_QK_BODY_TILE_REGS_RELEASE`

2K full，`llama_prefill_2k_q128_k256 copied_sdpa prepared_no_q_copy`：

| zone | count | avg cycles | max cycles | critical us | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAP_COMPUTE_QK_MATMUL_BODY` | 1656 | 2634 | 15807 | 11.709 | 父 zone |
| `FAP_COMPUTE_QK_BODY_MATMUL_LOOP` | 1536 | 1330 | 15475 | 11.463 | math loop 长尾 |
| `FAP_COMPUTE_QK_BODY_PACK_ROWS` | 1536 | 938 | 15431 | 11.430 | pack rows 长尾 |
| `FAP_COMPUTE_QK_BODY_PACK_CONFIG` | 720 | 35 | 91 | 0.067 | 很小 |
| `FAP_COMPUTE_QK_BODY_TILE_REGS_RELEASE` | 1536 | 32 | 71 | 0.053 | 很小 |
| `FAP_COMPUTE_QK_BODY_TILE_REGS_COMMIT` | 1536 | 27 | 59 | 0.044 | 很小 |
| `FAP_COMPUTE_QK_BODY_TILE_REGS_WAIT` | 1536 | 21 | 37 | 0.027 | 不是等待瓶颈 |
| `FAP_COMPUTE_QK_BODY_TILE_REGS_ACQUIRE` | 1576 | 21 | 36 | 0.027 | 不是等待瓶颈 |
| `FAP_COMPUTE_QK_BODY_REDUCE_SIGNAL` | 720 | 20 | 28 | 0.021 | 很小 |

16K chunked first chunk，
`llama_prefill_16k_chunked_q128_k256 copied_chunked prepared_no_q_copy`：

| zone | count | avg cycles | max cycles | critical us | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAP_COMPUTE_QK_MATMUL_BODY` | 1656 | 2727 | 17782 | 13.172 | 父 zone |
| `FAP_COMPUTE_QK_BODY_PACK_ROWS` | 1536 | 987 | 17418 | 12.902 | pack rows 长尾 |
| `FAP_COMPUTE_QK_BODY_MATMUL_LOOP` | 1536 | 1373 | 17315 | 12.826 | math loop 长尾 |
| `FAP_COMPUTE_QK_BODY_PACK_CONFIG` | 720 | 35 | 83 | 0.061 | 很小 |
| `FAP_COMPUTE_QK_BODY_TILE_REGS_RELEASE` | 1536 | 32 | 66 | 0.049 | 很小 |
| `FAP_COMPUTE_QK_BODY_TILE_REGS_COMMIT` | 1536 | 27 | 59 | 0.044 | 很小 |
| `FAP_COMPUTE_QK_BODY_TILE_REGS_ACQUIRE` | 1576 | 21 | 40 | 0.030 | 不是等待瓶颈 |
| `FAP_COMPUTE_QK_BODY_TILE_REGS_WAIT` | 1536 | 21 | 36 | 0.027 | 不是等待瓶颈 |
| `FAP_COMPUTE_QK_BODY_REDUCE_SIGNAL` | 720 | 20 | 29 | 0.021 | 很小 |

raw interval 进一步显示，最长 `QK_MATMUL_BODY` 事件分布在不同 TRISC：

- 在 `TRISC_1` 上，长尾通常来自 `QK_BODY_MATMUL_LOOP`。
- 在 `TRISC_2` 上，长尾通常来自 `QK_BODY_PACK_ROWS`。
- `QK_BODY_MATMUL_LOOP` 和 `QK_BODY_PACK_ROWS` 在同一 core 的不同 TRISC 上已经有大量重叠；
  这说明 math/pack pipeline 并不是完全没启动，而是两条路径本身都很重。

buffer 判断：

- `TILE_REGS_WAIT`、`ACQUIRE`、`RELEASE` 都只有几十 cycles，不像 DST/tile-reg
  buffer 深度不足导致的等待。
- `FAP_WRITER_WAIT_OUTPUT` 仍然很大，说明 writer 在等 compute 产出；没有证据说明 compute
  被 writer/output CB 反压卡住。
- `QK_WAIT_Q` 有 3.1-3.7 us 的 worst case，说明输入侧可能有 Q 预取/CB ready 问题；
  如果要加 buffer，优先怀疑的是 Q/read-side buffering，而不是 QK body 内部再加 output buffer。
- 对 `QK_MATMUL_BODY` 本身，更直接的实验不是先加 buffer，而是扫 QK subblock 形状：
  降低单次 `matmul_block_no_mop` 和 `pack_contiguous_rows_nocfg` 的 worst-case event，
  同时观察 event 数量增加是否抵消收益。

## 2026-05-29 QK subblock / buffer / DST sync 实验

本轮新增 copied-only profile 旋钮：

- `--qk-subblock H,W`：覆盖 copied SDPA program factory 里的 QK
  `qk_out_subblock_h/w`，验证 DST subblock 形状。
- `--q-buffer-factor N`：覆盖 Q 输入 CB buffer factor，验证 Q/read-side buffering。
- `--dst-full-sync` / `--dst-half-sync`：覆盖 copied compute kernel 的 DST sync 模式，
  反向验证 half-sync/DST 双缓冲是否已经在起作用。

这些旋钮只作用于 `copied_sdpa` / `copied_chunked`，并进入 copied op 的 program hash。
本节只用 device profiler 的 `critical_us` 下结论；`call_ms` 在新编译参数首次运行时会包含
JIT/program-cache 路径，不能用于判断 kernel steady-state。

2K full，`llama_prefill_2k_q128_k256 copied_sdpa prepared_no_q_copy`：

| qk subblock | QK phase | Q wait | QK body | math loop | pack rows | regs wait | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto | 18.044 | 3.167 | 11.802 | 11.523 | 11.525 | 0.027 | 默认，等价接近 `2x4` |
| `1x8` | 18.189 | 2.529 | 11.415 | 11.164 | 9.997 | 0.030 | pack 变短，但 phase 没赢 |
| `2x4` | 17.767 | 3.227 | 11.594 | 11.362 | 11.320 | 0.025 | 本轮最好，保持默认方向 |
| `4x2` | 18.496 | 0.044 | 14.458 | 14.229 | 14.184 | 0.024 | body 长尾明显变差 |
| `1x4` | 20.767 | 2.555 | 10.612 | 10.369 | 10.242 | 0.033 | body 变短但 phase 变差 |
| `2x2` | 19.416 | 3.175 | 11.271 | 11.027 | 10.876 | 0.031 | event 数量/phase 成本抵消收益 |
| `4x1` | 19.904 | 0.039 | 14.030 | 13.752 | 13.648 | 0.027 | body 长尾明显变差 |

2K full 的 Q buffer / DST sync 对照：

| 变体 | QK phase | Q wait | QK body | math loop | pack rows | regs wait | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `q_buffer_factor=1` | 17.602 | 3.241 | 11.724 | 11.478 | 11.445 | 0.028 | 不比默认差很多，也没消掉 Q wait |
| `q_buffer_factor=2` | 17.801 | 3.308 | 11.789 | 11.541 | 11.509 | 0.030 | 默认策略 |
| `q_buffer_factor=3` | 17.521 | 3.153 | 11.568 | 11.320 | 11.287 | 0.030 | 只有轻微改善，未改变瓶颈 |
| `q_buffer_factor=4` | 17.887 | 3.427 | 12.010 | 11.730 | 11.733 | 0.027 | 更深 buffer 无收益 |
| `dst_half_sync` | 17.659 | 3.175 | 11.676 | 11.423 | 11.399 | 0.031 | 与默认一致，保持 half-sync |
| `dst_full_sync` | 18.799 | 3.412 | 12.116 | 11.891 | 11.832 | 0.024 | 更慢，说明 full-sync 不是方向 |

16K chunked first chunk 复核，`llama_prefill_16k_chunked_q128_k256 copied_chunked prepared_no_q_copy`：

| 变体 | QK phase | Q wait | QK body | math loop | pack rows | regs wait | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto | 18.310 | 3.316 | 12.613 | 12.291 | 12.348 | 0.026 | chunked baseline |
| `qk_subblock=1x4` | 22.155 | 2.642 | 11.553 | 11.281 | 11.239 | 0.030 | body 变短但 phase 明显变差 |
| `q_buffer_factor=1` | 19.143 | 3.350 | 12.819 | 12.493 | 12.549 | 0.030 | Q buffer 变浅无收益 |
| `q_buffer_factor=3` | 18.530 | 3.241 | 12.551 | 12.205 | 12.281 | 0.025 | 只有轻微波动 |
| `dst_full_sync` | 19.931 | 3.567 | 12.867 | 12.564 | 12.618 | 0.024 | 同样更慢 |

raw CSV 的 pack/math overlap 结果：

| 实验组 | 代表变体 | pack 总时间 | pack 与 math 异 RISC overlap | overlap 比例 | 同 RISC overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| 2K subblock | auto | 1068.5 us | 955.6 us | 89.4% | 0.0 us |
| 2K subblock | `1x4` | 640.6 us | 570.3 us | 89.0% | 0.0 us |
| 2K subblock | `2x4` | 1077.5 us | 964.8 us | 89.5% | 0.0 us |
| 2K buffer | `q_buffer_factor=3` | 1089.3 us | 977.4 us | 89.7% | 0.0 us |
| 2K DST | `dst_full_sync` | 1080.1 us | 942.7 us | 87.3% | 0.0 us |
| 16K chunked | auto | 1109.7 us | 998.1 us | 89.9% | 0.0 us |
| 16K chunked | `qk_subblock=1x4` | 702.7 us | 631.9 us | 89.9% | 0.0 us |

结论：

- pack 和 math 并不是完全没有 overlap。同一 core、不同 TRISC 上约 87%-90% 的
  `PACK_ROWS` 时间被 `MATMUL_LOOP` 覆盖；同一 RISC overlap 为 0，符合
  TRISC 分工和顺序发射模型。
- `TILE_REGS_WAIT` 始终只有约 0.024-0.033 us；没有看到“pack 等 math、math 等 pack”
  造成的 tile-register wait 长尾。
- 继续加 Q buffer 不能消掉 `QK_WAIT_Q`，说明当前 Q wait 更像 reader 发射/调度位置或
  Q subblock ready 时序问题，不是简单 CB 容量不够。
- `dst_full_sync` 在 full 和 chunked 上都更慢；当前 half-sync/DST 双缓冲已经是正确方向，
  不应该为了“多开寄存器 buffer”改成 full-sync。
- `1x4` 这类小 subblock 会缩短单个 body event，但会增加 phase 调度/事件数量成本，
  最终 QK phase 变差。当前应保留默认 `2x4` QK subblock。
- 下一步真正要优化的不是简单加 buffer，而是改变 QK phase 的排布：把 softmax/sub-exp、
  QK matmul、reduce handoff 的顺序依赖拆开，或者更细地定位 reader Q ready
  为什么在局部 row group 上仍有 3 us 级 wait。

功能验证：

- `smoke --variant all --mode prepared --qk-subblock 1,4 --q-buffer-factor 3 --dst-full-sync`
  correctness 通过；full 与 chunked copied 输出相对 TTNN baseline 的 `max_abs_diff=0`。

## 2026-05-29 LLK / softmax 细分 profile

当前 copied kernel 不是纯 high-level TTNN 调用。`compute_streaming.hpp` 里已经直接使用
TT-Metal compute API 和 LLK/SFPU/PACK/UNPACK 相关原语，例如
`matmul_block_no_mop`、`sub_tiles_bcast_cols_custom`、`exp_packthread_tile`、
`llk_pack_relu_config`、`pack_tile` 和 `PACK(TTI_STALLWAIT(...))`。所以这轮问题不是
“要不要改成 LLK”，而是：更细粒度控制能不能让 softmax/vector 路径与 QK matmul 在同一
core 上真正 overlap。

实现上新增 copied-only 旋钮：

```bash
--qk-softmax-profile none|wait_max|sub_math|wait_sub|exp_sfpu|pack
```

每次只打开一个 QK softmax 子阶段 zone。一次打开所有子阶段会让 compute kernel 从可运行状态变成
`Program size (74000) too large for kernel config buffer (70656)`，所以保留全量 zone
不是可用方案。

2K full，`llama_prefill_2k_q128_k256 copied_sdpa prepared_no_q_copy`，
`grid=8x8,pipeline_depth=2`：

| profile stage | softmax 子阶段 critical us | softmax 父 zone | QK body | QK phase | Q wait | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wait_max` | 0.167 | 1.653 | 11.829 | 18.208 | 3.227 | max CB wait 很小 |
| `sub_math` | 0.827 | 1.584 | 11.857 | 18.359 | 3.292 | softmax 中较重，但仍远小于 QK body |
| `wait_sub` | 0.026 | 1.720 | 11.740 | 17.846 | 3.196 | tile-reg wait 不是瓶颈 |
| `exp_sfpu` | 1.085 | 1.653 | 12.067 | 18.111 | 3.159 | SFPU exp 是 softmax 内最重段之一 |
| `pack` | 0.475 | 1.676 | 11.789 | 17.849 | 3.270 | pack/sum 有成本，但不是主 critical path |

2K full raw interval overlap，`softmax 子阶段` 对 `FAP_COMPUTE_QK_BODY_MATMUL_LOOP`：

| profile stage | 子阶段总时间 | 同 RISC overlap | 异 RISC union overlap | 下一个同 RISC matmul median gap |
| --- | ---: | ---: | ---: | ---: |
| `wait_max` | 20.041 | 0.000 (0.0%) | 2.097 (10.5%) | 0.429 |
| `sub_math` | 78.620 | 0.000 (0.0%) | 2.133 (2.7%) | 0.327 |
| `wait_sub` | 12.186 | 0.000 (0.0%) | 1.723 (14.1%) | 0.330 |
| `exp_sfpu` | 103.598 | 0.000 (0.0%) | 64.859 (62.6%) | 0.330 |
| `pack` | 115.955 | 0.000 (0.0%) | 84.971 (73.3%) | 0.313 |

16K chunked first chunk，
`llama_prefill_16k_chunked_q128_k256 copied_chunked prepared_no_q_copy`，
`grid=8x8,pipeline_depth=2`：

| profile stage | softmax 子阶段 critical us | softmax 父 zone | QK body | QK phase | Q wait | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wait_max` | 0.125 | 1.622 | 12.510 | 18.639 | 3.130 | max CB wait 很小 |
| `sub_math` | 1.000 | 1.800 | 12.238 | 18.772 | 2.981 | 仍小于 QK body |
| `wait_sub` | 0.030 | 1.793 | 12.567 | 18.773 | 3.233 | tile-reg wait 不是瓶颈 |
| `exp_sfpu` | 1.379 | 1.944 | 12.809 | 18.897 | 3.359 | chunked 上 exp 长尾略更明显 |
| `pack` | 0.478 | 2.083 | 12.477 | 18.599 | 3.222 | pack/sum 不是主 critical path |

16K chunked raw interval overlap：

| profile stage | 子阶段总时间 | 同 RISC overlap | 异 RISC union overlap | 下一个同 RISC matmul median gap |
| --- | ---: | ---: | ---: | ---: |
| `wait_max` | 19.108 | 0.000 (0.0%) | 2.109 (11.0%) | 0.427 |
| `sub_math` | 80.047 | 0.000 (0.0%) | 2.289 (2.9%) | 0.325 |
| `wait_sub` | 12.175 | 0.000 (0.0%) | 1.657 (13.6%) | 0.330 |
| `exp_sfpu` | 107.630 | 0.000 (0.0%) | 64.385 (59.8%) | 0.330 |
| `pack` | 115.343 | 0.000 (0.0%) | 83.026 (72.0%) | 0.310 |

结论：

- softmax 和后续 QK matmul 在同一 RISC 上没有 overlap；这证明当前代码顺序仍然让同一执行路径串行排队。
- `exp_sfpu` / `pack` 与 matmul 的异 RISC overlap 已经存在一部分，但这不是“同一路径并行”。
  仅把代码改写成更底层 LLK 调用，不会自动让同一 core 的 MATH/SFPU 路径同时执行两个计算段。
- softmax 父 zone 约 1.6-2.1 us，而 QK body 约 11.7-12.8 us，`QK_WAIT_Q` 仍有约
  3 us。对当前 `q=128,k=256,D=128` 形状，local critical path 不是单纯 softmax
  压倒 matmul，而是 QK matmul body 加上 Q ready 时序占主导。
- FA3/FA4 论文里强调 softmax 瓶颈并不与这里矛盾：论文讨论的是算法级流水、跨 tile
  producer/consumer 和低精度矩阵引擎利用率；当前 copied TT-Metal kernel 的 measured
  shape 已经把 softmax 控制在 2 us 左右，而 QK body 的单事件长尾更大。
- 下一步应该做排布实验，不是继续加同类 profiler zone：
  1. 保留当前 `--qk-softmax-profile` 作为复现实验工具。
  2. 新建 copied schedule 变体，把当前循环中 `softmax(prev q_subblock, kt_subblock) -> matmul(cur q_subblock, kt_subblock)`
     的顺序改成 matmul-first / delayed-softmax 对照，观察 QK phase 是否下降。
  3. 如果 matmul-first 只搬移等待、不降 QK phase，再做 Q reader 侧细分，因为
     `QK_WAIT_Q` 仍是 3 us 级别。

功能验证：

- `smoke --variant all --mode prepared --qk-softmax-profile exp_sfpu`
  correctness 通过；full 与 chunked copied 输出相对 TTNN baseline 的 `max_abs_diff=0`。

## 2026-05-29 QK softmax schedule 实验

基于上面的 raw overlap，新增 copied-only schedule 旋钮：

```bash
--qk-softmax-schedule before_matmul|after_matmul|after_matmul_except_final_kt
```

默认 `before_matmul` 保持原顺序：

```text
softmax(prev q_subblock, kt_subblock)
reinit QK matmul
matmul(cur q_subblock, kt_subblock)
```

实验 `after_matmul` 改成：

```text
matmul(cur q_subblock, kt_subblock)
softmax(prev q_subblock, kt_subblock)
next kt_subblock 开始前再 reinit QK matmul
```

实验 `after_matmul_except_final_kt` 改成：

```text
非 final kt: matmul(cur q_subblock, kt_subblock) -> softmax(prev q_subblock, kt_subblock)
final kt:    softmax(prev q_subblock, kt_subblock) -> reinit -> matmul(cur q_subblock, kt_subblock)
```

目的不是作为最终优化，而是验证一个更窄的假设：`after_matmul` 的主要副作用是否来自
final kt 上把 previous-row softmax 插到当前 row `matmul -> max_reduce` handoff 中。

这个实验不改官方 TTNN baseline，只作用于 copied kernel，并进入 copied op program hash。

功能验证：

- `smoke --variant all --mode prepared --qk-softmax-schedule after_matmul`
  correctness 通过；full 与 chunked copied 输出相对 TTNN baseline 的 `max_abs_diff=0`。

单次 device profiler 对照，`warmup=0,iters=1`：

| shape | schedule | QK phase | Q wait | softmax | reinit | QK body | matmul loop | pack rows | max reduce | writer wait |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full | `before_matmul` | 18.343 | 3.173 | 1.964 | 0.107 | 11.741 | 11.454 | 11.437 | 1.105 | 108.821 |
| 2K full | `after_matmul` | 18.159 | 3.247 | 0.959 | 0.105 | 11.876 | 11.589 | 11.544 | 1.387 | 107.979 |
| 16K chunked | `before_matmul` | 19.021 | 3.313 | 1.951 | 0.107 | 12.645 | 12.324 | 12.367 | 1.104 | 108.601 |
| 16K chunked | `after_matmul` | 18.955 | 3.321 | 0.961 | 0.103 | 12.697 | 12.406 | 12.402 | 1.381 | 108.840 |

2K full 三次复测，`warmup=1,iters=3`：

| schedule | QK phase | Q wait | softmax | reinit | QK body | matmul loop | pack rows | max reduce | writer wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `before_matmul` | 18.466 | 3.419 | 1.865 | 0.107 | 12.157 | 11.831 | 11.738 | 1.106 | 109.240 |
| `after_matmul` | 18.500 | 3.175 | 0.914 | 0.112 | 11.844 | 11.541 | 11.421 | 1.383 | 109.355 |

16K chunked 的 `warmup=1,iters=3` 复测没有完成：运行时设备被另一个
`podman` 任务占用，UMD 报 `Failed to allocate TLB window`。所以这里不把 16K
三次复测作为证据，只保留单次 device-profiler 交叉检查。

结论：

- `after_matmul` 确实把 `FAP_COMPUTE_QK_SOFTMAX_EXP_SUM` 的 critical event 从约
  1.9 us 降到约 0.9-1.0 us。
- 但是 QK phase 没有形成实质改善：2K 三次复测从 18.466 us 变成 18.500 us；
  16K 单次只从 19.021 us 变成 18.955 us，变化小于本轮噪声边界。
- 代价集中在 `FAP_COMPUTE_QK_MAX_REDUCE`：`after_matmul` 把 final kt 的 matmul
  和当前 row max reduce 之间插入了 previous row softmax，破坏了原先
  matmul pack/reduce-trigger 到 max-reduce 的近距离 handoff。max reduce 从约
  1.10 us 变成约 1.38 us。
- 因此简单 matmul-first 不够。下一步 schedule 变体应该只延后非 final kt 的 previous
  softmax，final kt 保持 `softmax -> matmul -> max_reduce` 的短 handoff，或者把 max
  reduce 进一步提前到 final kt matmul 后、delayed softmax 前。

## 2026-05-29 非 final kt delayed-softmax 实验

新增 copied-only schedule：

```bash
--qk-softmax-schedule after_matmul_except_final_kt
```

实现位置：

- `kernels/compute/compute_streaming.hpp`
  - `qk_softmax_schedule == 2`：
    非 final kt 执行 `matmul -> previous-row softmax`。
    final kt 执行 `previous-row softmax -> reinit -> matmul`，然后落回原来的 mask/push/max-reduce。
- `flash_attention_profile.cpp`
  - CLI 解析和结果标签新增 `after_matmul_except_final_kt`。
- `host/copied_sdpa/device/sdpa_program_factory.cpp`
  - copied op compile-time arg 38 的合法范围从 `[0,1]` 放宽到 `[0,2]`。

功能验证：

- `smoke --variant all --mode prepared --qk-softmax-schedule after_matmul_except_final_kt`
  correctness 通过。

| pair | elements | max abs diff | passed |
| --- | ---: | ---: | --- |
| full SDPA | 131072 | 0.000000 | true |
| chunked SDPA | 32768 | 0.000000 | true |

device profiler 配置：

- `q=128,k=256,grid=8x8,pipeline_depth=2,pipeline=auto`
- `mode=prepared_no_q_copy`
- `warmup=1,iters=3`
- 2K full：`--preset llama_prefill_2k --variant copied_sdpa`
- 16K chunked：`--preset llama_prefill_16k_chunked --variant copied_chunked`

device profiler 对照，单位 us：

| shape | schedule | QK phase | Q wait | softmax | reinit | QK body | matmul loop | pack rows | max reduce | writer wait |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full | `before_matmul` | 18.381 | 3.435 | 1.907 | 0.103 | 12.086 | 11.756 | 11.657 | 1.098 | 108.990 |
| 2K full | `after_matmul` | 18.210 | 3.223 | 0.911 | 0.113 | 11.739 | 11.430 | 11.290 | 1.392 | 108.834 |
| 2K full | `after_matmul_except_final_kt` | 19.166 | 3.395 | 2.013 | 0.104 | 12.450 | 12.132 | 11.794 | 1.104 | 109.653 |
| 16K chunked | `before_matmul` | 19.557 | 3.343 | 2.044 | 0.107 | 12.758 | 12.448 | 12.441 | 1.098 | 109.390 |
| 16K chunked | `after_matmul` | 19.453 | 3.417 | 0.975 | 0.109 | 12.817 | 12.498 | 12.472 | 1.388 | 110.494 |
| 16K chunked | `after_matmul_except_final_kt` | 19.551 | 3.380 | 1.934 | 0.108 | 12.892 | 12.574 | 12.568 | 1.099 | 110.186 |

不开 device profiler 的 host timing 辅助对照，单位 ms：

| shape | schedule | avg | best | worst | call | sync |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2K full | `before_matmul` | 0.166 | 0.158 | 0.212 | 0.021 | 0.142 |
| 2K full | `after_matmul` | 0.217 | 0.194 | 0.258 | 0.069 | 0.140 |
| 2K full | `after_matmul_except_final_kt` | 0.181 | 0.172 | 0.192 | 0.035 | 0.142 |
| 16K chunked | `before_matmul` | 0.220 | 0.204 | 0.290 | 0.069 | 0.142 |
| 16K chunked | `after_matmul` | 0.203 | 0.197 | 0.217 | 0.053 | 0.144 |
| 16K chunked | `after_matmul_except_final_kt` | 0.196 | 0.185 | 0.211 | 0.046 | 0.145 |

host timing 这里只作为辅助指标：`prepared_no_q_copy` 的 `sync` 基本都在 0.14 ms 左右，
差异主要来自 host call 侧和运行噪声。是否继续优化仍以 device profiler 的阶段表判断。

结论：

- `after_matmul_except_final_kt` 成功验证了 handoff 假设的一半：
  `FAP_COMPUTE_QK_MAX_REDUCE` 回到 baseline 水平。2K 从 `after_matmul` 的
  1.392 us 回到 1.104 us；16K 从 1.388 us 回到 1.099 us。
- 但是它没有降低 QK phase。2K 从 baseline 18.381 us 变成 19.166 us；
  16K 基本打平 baseline，19.557 us 对 19.551 us。
- 原因是这个排布在 final kt 边界形成了新的串行气泡：
  非 final kt 已经延后 previous-row softmax，到了 final kt 又必须先把 previous-row
  final-column softmax 做完，才能 reinit 并做 final kt matmul。这样确实修复了
  `matmul -> max_reduce` 的近距离 handoff，但把 softmax critical event 又推回
  1.9-2.0 us，并没有让它真正和 QK matmul 同路径重叠。
- 当前三种 schedule 里没有一个是可作为优化落点的稳定收益：
  `after_matmul` 降 softmax 但伤 max reduce；`after_matmul_except_final_kt`
  修 max reduce 但恢复 softmax 串行气泡。

下一步更合理的实验不是继续沿这个分支微调，而是做 `reduce-first delayed-softmax`：

```text
final kt matmul(cur row)
mask/push/max_reduce(cur row)
然后再补 previous-row delayed softmax
```

这个方向能同时保住 `matmul -> max_reduce` handoff，又避免 final kt 前连续 softmax。
实现时要特别注意 pack 配置：当前 `sub_exp_block_bcast_cols(..., skip_pack_configure=true)`
依赖 QK matmul 留下的 `actual_sbw` pack width；如果把 delayed softmax 放到 max-reduce
之后，max-reduce 会把 pack width 改成 1，因此补 softmax 前必须重新配置
`cb_qkt_im` 的 row pack width，或者让该次 sub-exp 自己配置 pack。

## 2026-05-30 QK threads / reader Q 细分 profile

本轮目的不是再猜“matmul 慢”还是“softmax 慢”，而是把之前的
`FAP_COMPUTE_QK_MATMUL_BODY` 继续拆到真实执行线程：

- `qk_threads`：
  - `FAP_COMPUTE_QK_BODY_UNPACK_AB`：只在 `TRISC_0` 记录。
  - `FAP_COMPUTE_QK_BODY_MATH_MATMUL`：只在 `TRISC_1` 记录。
  - `FAP_COMPUTE_QK_BODY_PACK_THREAD`：只在 `TRISC_2` 记录。
- `q_reader`：
  - `FAP_READER_Q_RESERVE`
  - `FAP_READER_Q_NOC_READS`
  - `FAP_READER_Q_READ_BARRIER`
  - `FAP_READER_Q_PUSH`
- host parser 新增 `--device-profiler-by-risc`，同一个 zone 可以看到
  `TRISC_0` / `TRISC_1` / `TRISC_2` / `NCRISC` / `BRISC` 各自的
  `critical_us`。

device profiler 配置：

- `q=128,k=256,grid=8x8,pipeline_depth=2,pipeline=auto`
- `mode=prepared_no_q_copy,warmup=1,iters=3`
- 2K full：`--preset llama_prefill_2k --variant copied_sdpa`
- 16K chunked：`--preset llama_prefill_16k_chunked --variant copied_chunked`
- raw logs：
  - `/wafer/gsh/tmp/fa_profile_qk_threads_2k.log`
  - `/wafer/gsh/tmp/fa_profile_q_reader_2k.log`
  - `/wafer/gsh/tmp/fa_profile_qk_threads_16k.log`
  - `/wafer/gsh/tmp/fa_profile_q_reader_16k.log`

QK body thread 细分，单位 us，取 device profiler `critical_us`：

| shape | QK phase | QK body | MATH TRISC_1 | PACK TRISC_2 | UNPACK TRISC_0 | Q wait TRISC_0 | softmax exp/sum | writer wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full | 18.716 | 11.878 | 10.869 | 11.470 | 0.407 | 3.316 | 1.941 | 108.410 |
| 16K chunked | 19.841 | 12.919 | 11.884 | 12.535 | 0.330 | 3.342 | 1.747 | 109.219 |

同一个 QK phase 按 TRISC 拆开，单位 us：

| shape | QK phase TRISC_0 | QK phase TRISC_1 | QK phase TRISC_2 | QK body TRISC_0 | QK body TRISC_1 | QK body TRISC_2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full | 10.267 | 18.587 | 18.716 | 1.239 | 11.878 | 11.830 |
| 16K chunked | 10.225 | 19.633 | 19.841 | 1.166 | 12.919 | 12.907 |

reader Q path 细分，单位 us：

| shape | reader Q total | Q NOC reads | reserve | read barrier | push | compute Q wait | reader total | compute total | writer total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full | 5.956 | 3.056 | 0.050 | 0.042 | 0.040 | 3.083 | 126.498 | 135.223 | 136.533 |
| 16K chunked | 5.761 | 3.024 | 0.048 | 0.041 | 0.040 | 3.052 | 129.179 | 138.121 | 139.059 |

raw interval 计算出的 MATH/PACK overlap。这里是所有事件的累计 interval，
不是单个 `critical_us`：

| shape | math total us | pack total us | overlap us | pack 被 math 覆盖 | math 被 pack 覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K full | 4302.225 | 4170.624 | 3422.909 | 82.1% | 79.6% |
| 16K chunked | 4460.467 | 4338.553 | 3594.184 | 82.8% | 80.6% |

结论：

- 当前“慢”的第一责任段是 QK body 中的 MATH/PACK 组合，不是 UNPACK。
  `UNPACK_AB` 只有约 0.3-0.4 us，而 `MATH_MATMUL` 和 `PACK_THREAD`
  都是 11-12.5 us 级。
- PACK 和 MATH 已经大量 overlap。之前“pack 和 math 没有 overlap”的猜想不成立；
  真正的问题是 QK body 的完成时间要等 `TRISC_1` math 和 `TRISC_2` pack
  都结束，二者都贴近 critical path。
- `QK_WAIT_Q` 只发生在 `TRISC_0` 上，`TRISC_1/2` 的同名 zone 只有约
  0.02 us；reader 侧对应的主要成本是 `FAP_READER_Q_NOC_READS`，
  reserve、read barrier、push 都只有约 0.04-0.05 us。
- 因此如果继续优化 Q wait，重点不是加大 CB 或优化 push/barrier，
  而是调整 reader 中 Q subblock 的发射位置，让 Q NOC read 更早被 K wait
  或后续 compute 覆盖。
- softmax exp/sum 仍是 1.6-1.9 us 级，不是当前最大 critical event；
  继续只围绕 softmax 微调 schedule，不会解决 writer 长时间等 compute 输出的问题。

## 2026-05-30 reader first Q before K 实验

本轮实验验证上一节的判断：`QK_WAIT_Q` 是否真的是 reader 侧 Q NOC read
没有被覆盖。做法是在 copied-only reader schedule 中新增
`--q-reader-schedule first_before_k`，只在 `use_q_subblock_push` 路径里把
第一个 Q subblock 提前到 K loop 之前读取：

- 代码位置：`kernels/dataflow/reader_interleaved.cpp`
  - `FAP_READER_Q_EARLY_FIRST`：K loop 之前读取第一个 Q subblock。
  - `FAP_READER_Q`：K loop 内继续读取剩余 Q subblocks。
- 对照配置：
  - `q=128,k=256,grid=8x8,pipeline_depth=2,pipeline=auto`
  - `mode=prepared_no_q_copy,warmup=1,iters=3`
  - `--qk-detail-profile q_reader --device-profiler-by-risc`
- raw logs：
  - `/wafer/gsh/tmp/fa_profile_q_reader_default2_2k.log`
  - `/wafer/gsh/tmp/fa_profile_q_reader_first_2k.log`
  - `/wafer/gsh/tmp/fa_profile_q_reader_default2_16k.log`
  - `/wafer/gsh/tmp/fa_profile_q_reader_first_16k.log`

host 侧统计，单位 ms：

| case | avg | call avg | sync avg |
| --- | ---: | ---: | ---: |
| 2K default | 0.214 | 0.071 | 0.138 |
| 2K first_before_k | 0.213 | 0.068 | 0.141 |
| 16K default | 0.217 | 0.076 | 0.137 |
| 16K first_before_k | 0.230 | 0.087 | 0.137 |

device critical path，单位 us：

| case | compute | writer | reader | QK phase | QK body | QK wait Q | QK setup | softmax exp/sum | reader Q | early first Q | Q NOC reads | K read | V read | writer wait output | writer store tiles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K default | 135.998 | 136.926 | 126.821 | 18.366 | 11.767 | 3.171 | 3.353 | 2.013 | 5.987 | 0.000 | 3.093 | 8.471 | 7.746 | 109.043 | 3.464 |
| 2K first_before_k | 136.287 | 137.389 | 127.564 | 19.240 | 12.645 | 0.144 | 0.292 | 1.312 | 2.730 | 5.282 | 5.070 | 8.374 | 7.350 | 109.141 | 3.003 |
| 16K default | 137.384 | 138.359 | 128.667 | 19.227 | 12.670 | 3.293 | 3.482 | 1.670 | 6.039 | 0.000 | 3.194 | 8.199 | 7.583 | 109.990 | 3.123 |
| 16K first_before_k | 138.104 | 139.089 | 129.084 | 19.196 | 12.725 | 0.319 | 0.460 | 1.424 | 3.034 | 3.412 | 3.207 | 8.336 | 7.767 | 109.524 | 4.737 |

按 RISC 看 Q wait 与 reader Q NOC read，单位 us：

| case | Q wait TRISC_0 | Q wait TRISC_1 | Q wait TRISC_2 | Q NOC reads NCRISC | early first Q NCRISC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K default | 3.171 | 0.026 | 0.024 | 3.093 | 0.000 |
| 2K first_before_k | 0.144 | 0.026 | 0.028 | 5.070 | 5.282 |
| 16K default | 3.293 | 0.026 | 0.024 | 3.194 | 0.000 |
| 16K first_before_k | 0.319 | 0.026 | 0.024 | 3.207 | 3.412 |

结论：

- `first_before_k` 证明 `QK_WAIT_Q` 的来源判断是对的：compute 侧
  `TRISC_0` 的等待从 3 us 级下降到 0.1-0.3 us，而 `TRISC_1/2`
  始终只有约 0.02 us。
- 这个改动不是收益路径。2K 的 writer critical path 从 136.926 us
  增到 137.389 us，16K 从 138.359 us 增到 139.089 us；
  host 16K avg 也从 0.217 ms 增到 0.230 ms。
- 原因不是 Q read 更慢本身，而是“把第一个 Q 放到所有 K 之前”过早：
  它消除了 compute 的 Q wait，但把 reader 时间线提前占用，扰动了 K/V
  与后续 QK body 的到达时机。2K 上 `QK body` 从 11.767 us 增到
  12.645 us；16K 上 `writer store tiles` 从 3.123 us 增到 4.737 us。
- 因此 `first_before_k` 只作为定位实验保留，不作为新的 tuned baseline。
  下一步不应继续“越早读 Q 越好”，而应做更窄的 prefetch 位置：
  在 K read request 已经发出、且不破坏 K forward/link 写约束的边界上，
  只提前发射 first Q subblock，让 Q NOC read 被 K wait 或后续 compute
  覆盖，但不阻塞第一段 K 的可用性。

## 2026-05-30 reader first Q during K final barrier 实验

本轮验证上一节的“更窄 prefetch 位置”：新增
`--q-reader-schedule first_during_k_read`，只影响 chunked/paged K path。
实现位置是 `kernels/dataflow/reader_interleaved.cpp`：

- `FAP_READER_K_READ_Q_DURING_FINAL_BARRIER`：第一段 K 的前面 tile
  仍按原 K read 顺序发出，只跳过最后一次 K barrier。
- `FAP_READER_Q_DURING_K_READ`：在第一段 K read 的最后 barrier
  之前发出第一个 Q subblock，然后由 Q subblock 的 read barrier
  同时等待 K 尾部和 Q。
- 后续 Q subblocks 仍在 `FAP_READER_Q` 中读取；非 chunked/full path
  不走这个实验分支。

correctness：

| case | baseline | candidate | elements | max abs diff | passed |
| --- | --- | --- | ---: | ---: | --- |
| smoke copied full | TTNN full SDPA | copied full | 131072 | 0.000000 | true |
| smoke copied chunked | TTNN chunked | copied chunked | 32768 | 0.000000 | true |
| 16K copied full | TTNN full SDPA | copied full | 16777216 | 0.000000 | true |
| 16K copied chunked | TTNN chunked | copied chunked | 2097152 | 0.000000 | true |

raw logs：

- `/wafer/gsh/tmp/fa_profile_q_reader_default_current_16k_grid8.log`
- `/wafer/gsh/tmp/fa_profile_q_reader_first_current_16k_grid8.log`
- `/wafer/gsh/tmp/fa_profile_q_reader_duringk_16k_grid8.log`
- `/wafer/gsh/tmp/fa_profile_host_ttnn_chunked_16k_grid8_rerun.log`
- `/wafer/gsh/tmp/fa_profile_host_copied_default_16k_grid8_rerun.log`
- `/wafer/gsh/tmp/fa_profile_host_copied_qduringk_16k_grid8.log`
- `/wafer/gsh/tmp/fa_profile_host_copied_first_current_16k_grid8.log`

device profile，同一版代码、`qk_detail_profile=q_reader`、`grid=8x8`，
单位 us：

| schedule | compute | writer | reader | wait K | QK phase | QK body | QK wait Q | reader Q | early/during Q | Q NOC reads | K read | K+Q final barrier | V read | writer wait output | writer store |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 138.040 | 138.967 | 129.005 | 6.252 | 19.151 | 12.627 | 3.198 | 5.886 | 0.000 | 3.116 | 8.230 | 0.000 | 7.671 | 110.360 | 2.861 |
| first_before_k | 137.424 | 138.470 | 128.567 | 9.432 | 19.493 | 12.751 | 0.149 | 2.853 | 3.507 | 3.296 | 7.833 | 0.000 | 7.433 | 110.359 | 5.026 |
| first_during_k_read | 137.590 | 138.822 | 128.584 | 9.259 | 19.147 | 12.556 | 0.412 | 3.029 | 3.332 | 3.113 | 9.869 | 9.821 | 7.747 | 109.830 | 3.819 |

按 RISC 看等待迁移，单位 us：

| schedule | wait K TRISC_0 | wait K TRISC_1 | wait K TRISC_2 | Q wait TRISC_0 | Q wait TRISC_1 | Q wait TRISC_2 | Q NOC reads NCRISC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 6.252 | 0.021 | 0.028 | 3.198 | 0.026 | 0.024 | 3.116 |
| first_before_k | 9.432 | 0.020 | 0.029 | 0.149 | 0.026 | 0.028 | 3.296 |
| first_during_k_read | 9.259 | 0.020 | 0.030 | 0.412 | 0.026 | 0.024 | 3.113 |

不开 device profiler 的 host no-copy 对照，单位 ms：

| variant / schedule | avg | best | worst | call | sync |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTNN chunked baseline | 0.158 | 0.157 | 0.160 | 0.010 | 0.146 |
| copied default | 0.159 | 0.157 | 0.163 | 0.011 | 0.146 |
| copied `first_during_k_read` | 0.161 | 0.158 | 0.176 | 0.012 | 0.146 |
| copied `first_before_k` | 0.167 | 0.158 | 0.272 | 0.021 | 0.144 |

结论：

- `first_during_k_read` 证明 Q NOC read 可以被移动：`QK_WAIT_Q`
  从 default 的 3.198 us 降到 0.412 us。
- 但它没有提升端到端性能，因为等待转移到了 `FAP_COMPUTE_WAIT_K`：
  default 是 6.252 us，`first_during_k_read` 是 9.259 us。
  当前实现为了共享 K 尾部和 Q 的 barrier，会在第一个 Q subblock
  完成后才 `cb_push_back(cb_k_in)`，所以 compute 先等 K 时看到的是
  “K 尾部 + Q first subblock”合并后的到达时间。
- `first_before_k` 更极端：Q wait 降到 0.149 us，但 wait K 升到
  9.432 us，writer store 也升到 5.026 us。它验证定位有效，
  但不是收益路径。
- 这个实验回答了“是否让 load KV 与 Q load overlap”：当前 K 和 V
  不是一起 load。reader 对每个 `k_chunk` 的顺序是 K read/push、
  Q subblock push、V read/push；V 读可以和 compute 的 QK phase
  overlap，但 K 和 V 本身是分开的 CB、分开的 read 段。
- 下一步如果继续 reader 方向，不能再简单提前 Q。需要把 K 的可用性
  保住，例如把第一段 K 的 `cb_push_back` 放回 K barrier 之后，
  再用单独的 Q helper 发射 first Q；否则只是把 `QK_WAIT_Q`
  转移成 `WAIT_K`。

## 2026-05-30 tech report 与当前 compute 排布判定

`third_party/tt-metal/tech_reports/FlashAttention/FlashAttention.md`
描述的是：TT-Metal FlashAttention 使用 FA2 的在线 softmax / KV chunk
算法，并吸收 FA3 的异步 data movement 与 compute pipeline。文档的
future work 还明确把“在不同 compute units 上 pipeline matmul 和 softmax”
列为后续优化，因此不能把当前实现理解成已经完整实现 FA3 的 matmul/softmax
并行流水。

对照当前 copied kernel：

- `kernels/compute/compute_streaming.hpp` 的 Phase 1 是
  `Q@KT -> in-place sub_exp(prev row) -> max_reduce`。默认顺序仍是
  previous row softmax 与当前 QK matmul 在同一个 compute path 中串行交替，
  不是独立指令流上的并行执行。
- Phase 2 才进入 `QKT@V`：先 drain 最后一行的 sub_exp，再做
  `QKT@V`，并在后续 row group 中把 SALAD(prev) 与当前 V matmul
  交错。也就是说代码有 split-drain/SALAD 与 V matmul 的局部交叠，
  但主循环不是“上一段 QK 与下一段 PV 放在一起，然后再做 softmax”的完整 FA3
  排布。
- reader 侧可以把 V load 放在 compute QK phase 期间覆盖一部分数据搬运；
  但 compute 侧的主阶段仍是先完成当前 K chunk 的 QK/softmax/max
  产物，再进入该 K chunk 的 QKT@V/PV drain。

因此当前代码不是简单的三 kernel `QK -> softmax -> PV` DRAM baseline；
它已经是 fused streaming SDPA，L1 中做在线 softmax、split-drain
和 SALAD。但它也不是 tech report future-work 里说的“FA3-style
matmul/softmax 不同 compute units 并行流水”的完成形态。我们后续要优化的
重点正是这个 compute-stage 排布，而不是再微调已证明会转移等待的 reader
Q 提前位置。

## 2026-05-30 QK body 首段阻塞细分

上一节仍然只说明了 reader Q wait 的来源，没有回答“`QK_MATMUL_BODY`
内部到底是谁阻塞”。本节从 raw device CSV 重建 interval，不只看每个 zone
的 `critical_us`，而是按 `(run host ID, core)` 排序，比较每个 core/run
的第一个 QK body 和后续 QK body。

使用的 raw profile：

- `/wafer/gsh/tmp/fa_profile_prof_qk_threads_2k/.logs/profile_log_device.csv`
- `/wafer/gsh/tmp/fa_profile_prof_qk_threads_16k/.logs/profile_log_device.csv`

2K full，单位 us：

| zone / RISC | first avg | first p90 | first max | rest avg | rest p90 | rest max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `QK_BODY_MATMUL_LOOP` / TRISC_1 | 9.613 | 10.975 | 11.558 | 1.413 | 4.137 | 6.390 |
| `QK_BODY_MATH_MATMUL` / TRISC_1 | 8.921 | 10.282 | 10.869 | 0.303 | 0.187 | 5.710 |
| `QK_BODY_PACK_ROWS` / TRISC_2 | 9.559 | 10.928 | 11.517 | 1.013 | 3.659 | 6.234 |
| `QK_BODY_PACK_THREAD` / TRISC_2 | 9.514 | 10.885 | 11.470 | 0.986 | 3.639 | 6.195 |
| `QK_WAIT_Q` / TRISC_0 | 2.077 | 2.846 | 3.316 | 0.280 | 1.407 | 2.784 |
| `QK_SETUP` / TRISC_0 | 2.248 | 3.019 | 3.499 | 0.435 | 1.558 | 2.942 |

16K chunked，单位 us：

| zone / RISC | first avg | first p90 | first max | rest avg | rest p90 | rest max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `QK_BODY_MATMUL_LOOP` / TRISC_1 | 10.577 | 11.868 | 12.596 | 1.353 | 3.752 | 6.081 |
| `QK_BODY_MATH_MATMUL` / TRISC_1 | 9.878 | 11.169 | 11.884 | 0.290 | 0.193 | 5.399 |
| `QK_BODY_PACK_ROWS` / TRISC_2 | 10.578 | 11.872 | 12.582 | 0.962 | 3.353 | 5.919 |
| `QK_BODY_PACK_THREAD` / TRISC_2 | 10.532 | 11.829 | 12.535 | 0.933 | 3.322 | 5.884 |
| `QK_WAIT_Q` / TRISC_0 | 2.040 | 2.751 | 3.342 | 0.293 | 1.603 | 2.231 |
| `QK_SETUP` / TRISC_0 | 2.218 | 2.938 | 3.527 | 0.448 | 1.759 | 2.386 |

QK phase 也呈现同样的 first/rest 分裂，单位 us：

| shape | phase TRISC_1 first avg | phase TRISC_1 rest avg | phase TRISC_2 first avg | phase TRISC_2 rest avg | body TRISC_1 first avg | body TRISC_1 rest avg | body TRISC_2 first avg | body TRISC_2 rest avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full | 15.448 | 9.507 | 16.207 | 9.975 | 9.925 | 1.775 | 9.865 | 1.354 |
| 16K chunked | 16.452 | 9.212 | 17.295 | 9.684 | 10.889 | 1.717 | 10.885 | 1.302 |

额外分布检查：

- `QK_BODY_TILE_REGS_WAIT` 不是瓶颈。2K/16K 上 TRISC_1 和 TRISC_2
  的 max 都只有约 0.025-0.030 us。
- `UNPACK_AB` 不是瓶颈。2K max 0.407 us，16K max 0.330 us。
- 慢事件集中在每个 `(run host ID, core)` 的第 0 个 QK body：
  16K 上 `QK_BODY_MATMUL_LOOP` 的 256 个 core/run 第 0 个事件全部
  >= 9 us；`QK_BODY_PACK_ROWS` 同样 256/256 个第 0 个事件 >= 9 us。
  后续事件平均只有 1 us 左右。
- `QK_WAIT_Q` 和 `QK_SETUP` 也在 first body 前更大，但 first avg
  只有约 2.0-2.2 us，解释不了 `MATH_MATMUL` / `PACK_THREAD`
  9-10.5 us 的主体成本。

定位结论：

- 当前最明确的阻塞不是“所有 matmul 都慢”，而是每个 kernel launch / core
  的第一个 QK body 在 `TRISC_1` math 和 `TRISC_2` pack 上同时出现
  9-10.5 us 级首段填充成本。后续 QK body 已经进入比较好的流水状态。
- 这个阻塞发生在实际 LLK 调用内部：
  `llk_math_matmul_no_mop` 所在的 `FAP_COMPUTE_QK_BODY_MATH_MATMUL`
  和 `pack_contiguous_rows_nocfg` 所在的 `FAP_COMPUTE_QK_BODY_PACK_THREAD`
  / `PACK_ROWS`，不是 `tile_regs_wait`、`pack_config`、`reduce_signal`
  或 UNPACK。
- writer 侧 `FAP_WRITER_WAIT_OUTPUT` 仍是结果等待，不是写回本身：
  同轮 profile 里 `writer wait output` max 约 108-109 us，而
  `writer store tiles` 只有约 2.7-4.4 us。
- 因此下一步应把优化焦点从“多加 Q buffer / pack register buffer”转到
  QK body 首段流水填充：尝试在正式第一个 QK body 前做更轻量的 warmup
  或调整第一个 `q_subblock=0, kt_subblock=0` 的排布，让 math/pack 的首段
  成本被 reader wait 或其他准备工作覆盖。继续扫普通后续 subblock 的
  tile_regs buffer 意义不大。

本节之后补了新的 compute wait 测点：

- `FAP_COMPUTE_WAIT_K`：包住 QK phase 前的 `cb_wait_front(cb_kt_in, ...)`。
- `FAP_COMPUTE_QKTV_WAIT_QKT`
- `FAP_COMPUTE_QKTV_WAIT_V`

这些测点已编译通过，但本轮尝试重跑时设备被其他用户进程持有
`CHIP_IN_USE_0_PCIe` 锁，因此没有把它们计入上面的实测表。后续设备空闲后，
需要用同一配置补跑 2K/16K 的 `qk_threads`，排除 K/V wait 是否在 QK body
首段之外继续贡献 critical path。

## 2026-05-30 compute wait / 首段 warmup 实验

本轮补上上一节缺失的 compute-side wait 测点，并验证“首段 QK body 是否可以通过
warmup/提前配置隐藏”的假设。新增 copied-only 测点：

- `FAP_COMPUTE_WAIT_K`：QK phase 前等待 `cb_kt_in`。
- `FAP_COMPUTE_QKTV_WAIT_QKT`：QKTV 前等待 `cb_qkt_im`。
- `FAP_COMPUTE_QKTV_WAIT_V`：QKTV 前等待 `cb_v_in`。
- `FAP_COMPUTE_QK_FIRST_BODY_WARMUP`：`tiny_matmul` warmup。
- `FAP_COMPUTE_QK_FIRST_BODY_SAME_CONFIG_INIT`：`same_config_init` 提前配置。

对照配置：

- `llama_prefill_16k_chunked --chunks 128,256`
- `variant=copied_chunked,mode=prepared_no_q_copy,grid=8x8`
- `pipeline=auto,pipeline_depth=2`
- device profiler：`warmup=1,iters=3`
- host no-profiler：`warmup=5,iters=20`

`qk_threads` profile，单位 us/ms：

| case | host avg ms | sync ms | compute | writer | reader | wait K | QK phase | QK setup | QK init | QK body | MATH | PACK | Q wait | QKTV wait V | QKV phase | writer wait | store |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.213 | 0.138 | 136.703 | 138.331 | 128.688 | 6.252 | 19.414 | 3.530 | 0.234 | 12.901 | 11.852 | 12.534 | 3.351 | 4.269 | 5.754 | 109.447 | 3.424 |
| `same_config_init` | 0.228 | 0.153 | 137.645 | 138.593 | 128.624 | 6.027 | 19.150 | 3.384 | 0.269 | 12.559 | 11.527 | 12.034 | 3.275 | 4.087 | 5.861 | 110.296 | 4.327 |

coarse profile（`qk_detail_profile=none`），单位 us/ms：

| case | host avg ms | sync ms | compute | writer | reader | wait K | QK phase | QK setup | QK init | QK body | Q wait | QKTV wait V | QKV phase | writer wait | store |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.228 | 0.137 | 137.611 | 138.892 | 129.210 | 6.278 | 19.353 | 3.599 | 0.242 | 12.908 | 3.425 | 4.131 | 5.976 | 110.350 | 4.498 |
| `same_config_init` | 0.213 | 0.138 | 136.915 | 138.361 | 129.256 | 6.094 | 19.239 | 3.341 | 0.269 | 12.652 | 3.240 | 4.215 | 5.930 | 109.591 | 4.277 |

不开 device profiler 的 host no-copy 对照，单位 ms：

| variant / config | avg | best | worst | call | sync | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TTNN chunked baseline | 0.158 | 0.157 | 0.164 | 0.011 | 0.145 | 官方 tuned 边界 |
| copied default | 0.158 | 0.157 | 0.161 | 0.010 | 0.146 | 与 baseline 持平 |
| copied `same_config_init` | 0.159 | 0.158 | 0.166 | 0.012 | 0.145 | 没有实际 host 收益 |

`qk_init` 单独 profile 的默认路径，单位 us：

| zone | critical us | 判断 |
| --- | ---: | --- |
| `FAP_COMPUTE_QK_INIT` | 0.328 | 初始化本身很小 |
| `FAP_COMPUTE_QK_INIT_MM_INIT` | 0.139 | 不是首段 10us 级瓶颈 |
| `FAP_COMPUTE_QK_INIT_PACK_DF` | 0.128 | 很小 |
| `FAP_COMPUTE_QK_INIT_UNPACK_DF` | 0.061 | 很小 |
| `FAP_COMPUTE_QK_INIT_PACK_WIDTH` | 0.065 | 很小 |

correctness：

| candidate | baseline | elements | max abs diff | passed |
| --- | --- | ---: | ---: | --- |
| copied full SDPA + `same_config_init` | TTNN full SDPA | 16777216 | 0.000000 | true |
| copied chunked + `same_config_init` | TTNN chunked | 2097152 | 0.000000 | true |

结论：

- `same_config_init` 能让 QK 局部阶段略短：coarse profile 里 `QK body`
  从 12.908 us 到 12.652 us，`QK phase` 从 19.353 us 到 19.239 us。
- 这个收益没有稳定转化成端到端。关闭 device profiler 后，copied default 与
  TTNN chunked baseline 都是 0.158 ms，`same_config_init` 是 0.159 ms。
- `qk_init` 细分说明初始化只有 0.3 us 级，不是首段 MATH/PACK 10 us 级成本的根因。
  `tiny_matmul` / `same_config_init` 的价值是证明首段可以被扰动，但不是新的 tuned baseline。
- `qk_init` 与 `same_config_init` 同时打开会超过 TENSIX kernel config buffer：
  `Program size (72976) too large for kernel config buffer (70656)`。因此
  `qk_init` 只作为单独 profile 档位，不和 warmup/thread 细分混编。

同时尝试过 `reduce_first_delayed_softmax` 排布：

```text
final kt matmul(cur row)
mask/push/max_reduce(cur row)
再补 previous-row delayed softmax
```

这个实验在 smoke correctness 失败，copied full SDPA 相对 TTNN baseline 的
`max_abs_diff` 直接发散到约 `1.8875e38`，说明它破坏了 QK in-place
softmax / max-reduce / pack 配置之间的时序。该 broken schedule 没有保留成
CLI 选项；后续如果继续做 reduce-first，必须先把相关 CB/pack/semaphore 时序做成
更小的 correctness repro，而不是直接进入性能 profile。

## 2026-05-30 phase timeline / partial handoff v1 实验

本轮实现两个 copied-only 能力：

- `--qk-detail-profile phase_timeline`：只打开 phase boundary / row / kt-step
  等时间线测点，不再同时打开 QK body thread 细分。否则
  `partial_handoff_v1 + phase_timeline` 会超过 TENSIX kernel config buffer。
- `--compute-pipeline-schedule partial_handoff_v1`：当 `q_subblock > 0` 时，
  把上一 row group 的 `QKT@V` 提前到 Phase 1 内执行，而不是等所有 QK row
  都完成后再统一进入 Phase 2。

正确性：

| case | baseline | candidate | elements | max abs diff | passed |
| --- | --- | --- | ---: | ---: | --- |
| 16K full SDPA + `partial_handoff_v1` | TTNN full SDPA | copied full | 16777216 | 0.005859 | true |
| 16K chunked + `partial_handoff_v1` | TTNN chunked | copied chunked | 2097152 | 0.005859 | true |

host no-profiler 对照，`llama_prefill_16k_chunked --chunks 128,256`，
`grid=8x8,prepared_no_q_copy,warmup=5,iters=20`，单位 ms：

| variant / schedule | avg | best | worst | call | sync | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TTNN chunked baseline | 0.157 | 0.157 | 0.162 | 0.010 | 0.145 | 官方 tuned 边界 |
| copied default | 0.163 | 0.158 | 0.214 | 0.015 | 0.145 | copied 默认 |
| copied `partial_handoff_v1` | 0.161 | 0.159 | 0.169 | 0.011 | 0.148 | avg 噪声级改善，sync 回退 |

轻量 device profiler（不开 `phase_timeline`）对照，单位 us：

| variant / schedule | reader | compute | writer | wait K | QK phase | QK body | softmax | QKV phase | QKTV matmul | QKTV wait V | writer wait | store |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTNN chunked baseline | 128.708 NCRISC | 135.865 TRISC | 137.081 BRISC | - | - | - | - | - | - | - | - | - |
| copied default | 129.393 | 138.454 | 139.271 | 6.262 | 17.936 | 12.430 | 2.325 | 6.755 | 2.048 | 4.251 | 110.537 | 4.403 |
| copied `partial_handoff_v1` | 128.250 | 139.175 | 140.480 | 6.215 | 22.504 | 12.767 | 2.121 | 2.131 | 6.866 | 5.203 | 110.090 | 2.584 |

`phase_timeline` 细分对照，单位 us：

| zone | default | `partial_handoff_v1` | 解释 |
| --- | ---: | ---: | --- |
| `FAP_COMPUTE` | 136.576 | 140.031 | partial 总 compute critical 回退 |
| `FAP_WRITER` | 138.417 | 141.045 | writer 仍等 compute，整体回退 |
| `FAP_COMPUTE_QK_PHASE` | 18.424 | 22.521 | 提前 `QKT@V` 被放进 QK phase，拉长 critical path |
| `FAP_TIMELINE_QK_ROW` | 14.476 | 14.445 | 单个 QK row 本体没有明显变快 |
| `FAP_TIMELINE_QK_KT_STEP` | 12.907 | 12.778 | KT step 本体基本持平 |
| `FAP_COMPUTE_QK_WAIT_Q` | 3.306 | 3.350 | Q ready 等待没有改善 |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` | 2.054 | 6.290 | QKTV matmul 被拆进 Phase 1 后变成更重的 critical event |
| `FAP_COMPUTE_QKV_PHASE` | 6.464 | 2.216 | Phase 2 drain 变短，但只是把工作前移了 |
| `FAP_TIMELINE_QKTV_DRAIN_LAST_ROW` | 6.429 | 2.170 | 最后一行 drain 变短 |
| `FAP_TIMELINE_PARTIAL_HANDOFF` | - | 6.439 | 新增 handoff 本身进入 critical path |
| `FAP_COMPUTE_QKTV_WAIT_V` | 4.670 | 4.675 | V 等待没有改善 |
| `FAP_WRITER_WAIT_OUTPUT` | 109.024 | 110.408 | writer 等输出没有减少 |
| `FAP_WRITER_STORE_TILES` | 3.865 | 2.284 | store 降低，但不是主瓶颈 |

按 RISC 看，`partial_handoff_v1` 的回退集中在 compute path：

| zone | RISC | default | `partial_handoff_v1` |
| --- | --- | ---: | ---: |
| `FAP_COMPUTE` | `TRISC_1` | 136.557 | 139.967 |
| `FAP_COMPUTE` | `TRISC_2` | 136.576 | 140.031 |
| `FAP_COMPUTE_QK_PHASE` | `TRISC_1` | 17.914 | 22.381 |
| `FAP_COMPUTE_QK_PHASE` | `TRISC_2` | 18.424 | 22.521 |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` | `TRISC_1` | 2.054 | 6.290 |
| `FAP_TIMELINE_PARTIAL_HANDOFF` | `TRISC_1` | - | 6.439 |
| `FAP_WRITER_WAIT_OUTPUT` | `BRISC` | 109.024 | 110.408 |

softmax / LLK 子阶段复核，`qk_softmax_profile` 每次只开一个子阶段，
`grid=8x8,warmup=1,iters=3`，单位 us：

| profile stage | 子阶段 | softmax 父 zone | QK body | QK phase | Q wait | writer wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wait_max` | 0.264 | 2.132 | 12.646 | 18.047 | 3.424 | 108.701 |
| `sub_math` | 1.376 | 2.062 | 12.672 | 18.052 | 3.389 | 108.544 |
| `wait_sub` | 0.031 | 1.974 | 12.511 | 17.821 | 3.354 | 109.107 |
| `exp_sfpu` | 1.451 | 2.010 | 12.681 | 18.211 | 3.369 | 109.179 |
| `pack` | 0.478 | 2.098 | 12.667 | 17.711 | 3.428 | 108.569 |

结论：

- `partial_handoff_v1` 的方向验证失败。它确实缩短了 Phase 2 的 `QKV` drain，
  但不是通过隐藏工作，而是把 `QKT@V` 工作前移进 QK phase；结果
  `QK_PHASE` 从 18.424 us 增到 22.521 us，writer critical 也从
  138.417 us 增到 141.045 us。
- writer store 继续不是主瓶颈。`partial_handoff_v1` 把 `STORE_TILES`
  降到 2.284 us，但 `WRITER_WAIT_OUTPUT` 没有下降，说明输出更早交付没有形成。
- V 已经在 L1 后，compute 仍主要卡在 QK/提前 QKTV 的串行路径上。
  `QKTV_WAIT_V` 基本不变，说明这轮不是 V load 问题。
- softmax 子阶段仍低于 QK body。`sub_math` / `exp_sfpu` 是 softmax 内较重段，
  但只有 1.3-1.5 us；QK body 仍是 12.5-12.7 us 级。
- 当前硬件上确实有 `TRISC_1` / `TRISC_2` 等不同执行线程，但这个单 compute
  kernel 的指令排布仍让 QK matmul、softmax、QKTV handoff 在同一条 producer/consumer
  路径上串行推进。简单把后续 `QKT@V` 塞进 Phase 1，只会占用 math/pack critical path。

下一步不要把 `partial_handoff_v1` 作为 baseline，也不要继续简单提前整段
`QKT@V`。更合理的后续实验是：

1. 做更小粒度的 handoff：只提前不会占用大块 math/pack 的准备动作，
   或只处理已经完成 softmax 的极小行组，避免把完整 `QKT@V` matmul 放到 QK critical path。
2. 如果要追 FA3-style overlap，需要拆出真正独立的 compute producer/consumer，
   例如不同 core/不同 kernel 的 row-group pipeline；在当前单 compute kernel 内，
   仅重排语句还没有证明能让 softmax 与 QK matmul 同时前进。
3. 继续保留 `phase_timeline` 用于定位 phase boundary，但不要和 `qk_threads`
   同时打开；需要 thread-level 证据时单独跑 `qk_threads`。

## 2026-05-30 QKTV detail / compute 拆分 profile

本轮继续把 compute 拆细，新增 `--qk-detail-profile qktv_detail`。第一次把
reserve/reconfig/mm-init/pack-width 等小片段也全部打开时，device program 超过
TENSIX kernel config buffer：`Program size (72992) too large for kernel config buffer (70656)`。
这些小片段不是当前判断流水的关键，所以最终只保留会影响 Phase 2 / handoff 的测点：

- `FAP_COMPUTE_QKTV_PACK_UNPACK_BARRIER`
- `FAP_COMPUTE_QKTV_DRAIN_SUB_EXP`
- `FAP_COMPUTE_QKTV_DRAIN_MATMUL_KT`
- `FAP_COMPUTE_QKTV_MATMUL_PACK`
- `FAP_COMPUTE_QKTV_V_SUBBLOCK`
- `FAP_COMPUTE_QKTV_WAIT_QKT`
- `FAP_COMPUTE_QKTV_WAIT_V`

正确性 smoke：

| case | baseline | candidate | max abs diff | passed |
| --- | --- | --- | ---: | --- |
| smoke default | TTNN full/chunked | copied full/chunked | 0.000000 | true |
| smoke `partial_handoff_v1` | TTNN full/chunked | copied full/chunked | 0.000000 | true |

profile 配置：`llama_prefill_16k_chunked --chunks 128,256`，
`variant=copied_chunked,mode=prepared_no_q_copy,warmup=1,iters=3,grid=8x8,
qk_detail_profile=qktv_detail`。下面 host 表只用于同一 instrumentation 下的
default/partial 对比，不与 no-profiler 最快表直接混用，单位 ms：

| metric | default | `partial_handoff_v1` | delta |
| --- | ---: | ---: | ---: |
| avg | 0.211 | 0.214 | +0.003 |
| best | 0.161 | 0.164 | +0.003 |
| worst | 0.301 | 0.306 | +0.005 |
| call | 0.070 | 0.072 | +0.002 |
| sync | 0.137 | 0.138 | +0.001 |

device stage critical path，单位 us：

| zone | default | `partial_handoff_v1` | delta | count default | count partial |
| --- | ---: | ---: | ---: | ---: | ---: |
| `FAP_COMPUTE` | 137.175 | 139.767 | +2.592 | 576 | 576 |
| `FAP_WRITER` | 138.337 | 141.028 | +2.691 | 192 | 192 |
| `FAP_READER` | 128.347 | 128.547 | +0.200 | 192 | 192 |
| `FAP_COMPUTE_QK_PHASE` | 18.293 | 22.670 | +4.377 | 1944 | 2232 |
| `FAP_COMPUTE_QKV_PHASE` | 6.555 | 2.255 | -4.300 | 1728 | 1728 |
| `FAP_COMPUTE_QK_MATMUL_PACK` | 12.629 | 12.739 | +0.110 | 6864 | 6912 |
| `FAP_COMPUTE_QK_SOFTMAX_EXP_SUM` | 2.053 | 2.222 | +0.169 | 3240 | 3240 |
| `FAP_COMPUTE_QK_MAX_REDUCE` | 0.895 | 0.922 | +0.027 | 3456 | 3456 |
| `FAP_COMPUTE_QKTV_WAIT_V` | 4.615 | 4.991 | +0.376 | 1728 | 1728 |
| `FAP_COMPUTE_QKTV_WAIT_QKT` | 0.044 | 0.037 | -0.007 | 1728 | 1728 |
| `FAP_COMPUTE_QKTV_PACK_UNPACK_BARRIER` | 6.207 | 7.513 | +1.306 | 1728 | 3456 |
| `FAP_COMPUTE_QKTV_DRAIN_SUB_EXP` | 1.370 | 1.129 | -0.241 | 3240 | 3240 |
| `FAP_COMPUTE_QKTV_DRAIN_MATMUL_KT` | 5.555 | - | - | 3240 | - |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` | 2.224 | 6.704 | +4.480 | 1728 | 3456 |
| `FAP_COMPUTE_QKTV_V_SUBBLOCK` | 5.490 | 6.630 | +1.140 | 4920 | 3456 |
| `FAP_COMPUTE_ROW_NORM` | 1.463 | 1.482 | +0.019 | 744 | 864 |
| `FAP_COMPUTE_SALAD_CORRECT` | 0.616 | 0.615 | -0.001 | 1512 | 1728 |
| `FAP_WRITER_WAIT_OUTPUT` | 109.513 | 110.522 | +1.009 | 768 | 768 |
| `FAP_WRITER_STORE_TILES` | 4.664 | 2.828 | -1.836 | 768 | 768 |
| `TRISC-KERNEL` | 137.210 | 139.803 | +2.593 | 576 | 576 |
| `BRISC-KERNEL` | 138.371 | 141.064 | +2.693 | 192 | 192 |
| `NCRISC-KERNEL` | 128.384 | 128.582 | +0.198 | 192 | 192 |

QKTV 子阶段按 RISC 的 critical path，单位 us：

| zone / RISC | default | `partial_handoff_v1` | delta |
| --- | ---: | ---: | ---: |
| `FAP_COMPUTE_QKTV_PACK_UNPACK_BARRIER` / `TRISC_0` | 6.207 | 7.513 | +1.306 |
| `FAP_COMPUTE_QKTV_PACK_UNPACK_BARRIER` / `TRISC_1` | 5.963 | 7.390 | +1.427 |
| `FAP_COMPUTE_QKTV_PACK_UNPACK_BARRIER` / `TRISC_2` | 5.446 | 5.444 | -0.002 |
| `FAP_COMPUTE_QKTV_DRAIN_SUB_EXP` / `TRISC_2` | 1.370 | 1.129 | -0.241 |
| `FAP_COMPUTE_QKTV_DRAIN_MATMUL_KT` / `TRISC_1` | 5.555 | - | - |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` / `TRISC_1` | 2.224 | 6.704 | +4.480 |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` / `TRISC_2` | 1.794 | 6.125 | +4.331 |
| `FAP_COMPUTE_QKTV_V_SUBBLOCK` / `TRISC_1` | 5.490 | 6.630 | +1.140 |
| `FAP_COMPUTE_QKTV_V_SUBBLOCK` / `TRISC_2` | 4.659 | 6.040 | +1.381 |
| `FAP_COMPUTE_QKTV_WAIT_V` / `TRISC_0` | 4.615 | 4.991 | +0.376 |
| `FAP_COMPUTE_QKV_PHASE` / `TRISC_1` | 6.555 | 0.426 | -6.129 |
| `FAP_COMPUTE_QKV_PHASE` / `TRISC_2` | 6.258 | 2.255 | -4.003 |

本轮结论：

- `partial_handoff_v1` 的 Phase 2 变短是搬移，不是 overlap。
  `QKV_PHASE` 从 6.555 us 降到 2.255 us，但 `QK_PHASE`
  从 18.293 us 增到 22.670 us，总 compute 和 writer critical path 分别回退
  2.592 us / 2.691 us。
- QK matmul 本体没有因为 partial 变慢。`QK_MATMUL_PACK` 只从 12.629 us
  到 12.739 us；真正新增的是提前 `QKT@V` 的 `QKTV_MATMUL_PACK`
  和 pack/unpack barrier。
- V 数据不是这轮主阻塞。`QKTV_WAIT_V` 只增加 0.376 us，`QKTV_WAIT_QKT`
  约等于 0；说明数据已经基本 ready，compute 仍卡在同一条 pack/unpack/math
  producer-consumer 路径。
- `QKTV_PACK_UNPACK_BARRIER` 是新的关键证据。默认路径已经有 6.207 us，
  partial 后变成 7.513 us 且 count 翻倍。它对应代码里 sub-exp in-place
  写 `cb_qkt_im` 后，`QKT@V` 立即从同一 CB 做 UNPACK 读取前的显式同步。
  这说明当前排布不是“等 V”，而是 pack 写完 / unpack 可读之间的同步把流水切断。

## 2026-05-30 barrier split / drain-all schedule 实验

上一节的 `qktv_detail` 把 `FAP_COMPUTE_QKTV_PACK_UNPACK_BARRIER` 量到
6-7 us，但这个 zone 包住了周围的 compute path，不能直接当成 semaphore
本身的代价。本轮继续做两步：

1. 新增 `--qk-detail-profile qktv_barrier_split`，只量显式
   `PACK_DONE` semaphore handoff 的 `FAP_QKTV_BARRIER_TOTAL`。
2. 新增 `--compute-pipeline-schedule qktv_drain_all_then_matmul`，把 Phase 2
   第一行 group 改成“先 drain 全部 kt 的 `sub_exp`，再做完整 inner-dim
   `QKT@V`”，验证 split-drain 分段 matmul/accumulation 是否才是问题。

没有实现 `qktv_pingpong_cb`。这个方向原本依赖“显式 barrier 很大”的假设；
本轮 `qktv_barrier_split` 已经证明 semaphore handoff 约 0.06 us，不值得先为它增加
第二套 CB 和 L1 占用。代码中也没有保留这个 CLI 入口，避免误跑一个没有语义的实验。

正确性：

| case | baseline | candidate | max abs diff | passed |
| --- | --- | --- | ---: | --- |
| smoke + `qktv_barrier_split` | TTNN full/chunked | copied full/chunked | 0.000000 | true |
| smoke + `qktv_drain_all_then_matmul` | TTNN full/chunked | copied full/chunked | 0.000000 | true |
| 16K chunked semantic check + `qktv_drain_all_then_matmul` | TTNN full/chunked | copied full/chunked | 0.005859 | true |

profile 配置：`llama_prefill_16k_chunked --chunks 128,256`，
`variant=copied_chunked,mode=prepared_no_q_copy,warmup=1,iters=3,grid=8x8,
qk_detail_profile=qktv_barrier_split`。下面 host 表只用于同一 instrumentation
下的 default/drain-all 对比，不与 no-profiler 最快表直接混用，单位 ms：

| metric | default | `qktv_drain_all_then_matmul` | delta |
| --- | ---: | ---: | ---: |
| avg | 0.249 | 0.217 | -0.032 |
| best | 0.175 | 0.163 | -0.012 |
| worst | 0.376 | 0.313 | -0.063 |
| call | 0.107 | 0.074 | -0.033 |
| sync | 0.135 | 0.139 | +0.004 |

device stage critical path，单位 us：

| zone | default | `qktv_drain_all_then_matmul` | delta |
| --- | ---: | ---: | ---: |
| `FAP_COMPUTE` | 137.484 | 140.277 | +2.793 |
| `FAP_READER` | 129.019 | 128.790 | -0.229 |
| `FAP_WRITER` | 138.360 | 141.260 | +2.900 |
| `FAP_COMPUTE_QK_PHASE` | 18.336 | 18.248 | -0.088 |
| `FAP_COMPUTE_QKV_PHASE` | 6.661 | 6.459 | -0.202 |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` | 2.050 | 5.219 | +3.169 |
| `FAP_COMPUTE_QKTV_WAIT_V` | 4.279 | 4.225 | -0.054 |
| `FAP_COMPUTE_QKTV_WAIT_QKT` | 0.045 | 0.040 | -0.005 |
| `FAP_QKTV_BARRIER_TOTAL` | 0.061 | 0.064 | +0.003 |
| `FAP_WRITER_WAIT_OUTPUT` | 109.576 | 111.399 | +1.823 |
| `FAP_WRITER_STORE_TILES` | 4.510 | 1.841 | -2.669 |
| `TRISC-KERNEL` | 137.555 | 140.313 | +2.758 |
| `BRISC-KERNEL` | 138.394 | 141.298 | +2.904 |
| `NCRISC-KERNEL` | 129.057 | 128.826 | -0.231 |

QKTV 关键 zone 按 RISC 的 critical path，单位 us：

| zone / RISC | default | `qktv_drain_all_then_matmul` | delta |
| --- | ---: | ---: | ---: |
| `FAP_QKTV_BARRIER_TOTAL` / `TRISC_0` | 0.061 | 0.064 | +0.003 |
| `FAP_QKTV_BARRIER_TOTAL` / `TRISC_1` | 0.017 | 0.024 | +0.007 |
| `FAP_QKTV_BARRIER_TOTAL` / `TRISC_2` | 0.059 | 0.054 | -0.005 |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` / `TRISC_1` | 2.050 | 5.219 | +3.169 |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` / `TRISC_2` | 1.727 | 4.381 | +2.654 |

本轮结论：

- 显式 pack/unpack semaphore handoff 不是瓶颈。`FAP_QKTV_BARRIER_TOTAL`
  只有约 0.06 us；上一节 `FAP_COMPUTE_QKTV_PACK_UNPACK_BARRIER` 的 6-7 us
  主要是被周围 compute path 包进来了，不能当成纯 barrier 时间。
- `qktv_drain_all_then_matmul` correctness 通过，但不是性能优化。它让
  `QKV_PHASE` 略降 0.202 us，却把 `QKTV_MATMUL_PACK` 从 2.050 us 放大到
  5.219 us，最终 `FAP_COMPUTE` 回退 2.793 us，writer critical path 回退
  2.900 us。
- host avg 看起来改善主要来自 `call_avg_ms` 的 profiler/instrumentation 噪声；
  `sync_avg_ms` 和 device critical path 都回退，所以不能把它当作 kernel win。
- 由于 Step 1/2 的门槛都没有支持 CB ping-pong，`qktv_pingpong_cb` 不继续做。
  现在更像是 `QKT@V` matmul work placement / row group 排布 / writer 等 compute
  输出的问题，而不是显式 semaphore 或 V 数据等待问题。

## 2026-05-30 QKTV matmul body 细分 profile

上一轮已经证明显式 barrier 不是瓶颈，但 `QKTV_MATMUL_PACK` 在
`qktv_drain_all_then_matmul` 下明显放大。本轮继续把它拆到更细粒度：

- `qktv_matmul_detail`：线程级拆分，记录 `UNPACK` / `MATH` / `PACK`
  三条 TRISC 路径。
- `qktv_math_detail`：只拆 math 侧 enclosure，包括 tile regs
  acquire/commit/wait/release、matmul loop、UNPACK/MATH。
- `qktv_pack_detail`：只拆 pack rows / pack thread。

最开始尝试把 tile regs、reinit、pack config、thread 细分一次性全开，
smoke 时失败：

```text
Program size (72288) too large for kernel config buffer (70656)
```

因此最终采用互斥 profile 档位。另一次更细的 pack row/tile 测点会生成空
zone 名，已删除；最终结果只保留可解释、可复跑的 zone。

correctness：

| case | baseline | candidate | max abs diff | passed |
| --- | --- | --- | ---: | --- |
| smoke + `qktv_math_detail` | TTNN full/chunked | copied full/chunked | 0.000000 | true |
| smoke + `qktv_pack_detail` | TTNN full/chunked | copied full/chunked | 0.000000 | true |

profile 配置：`llama_prefill_16k_chunked --chunks 128,256`，
`variant=copied_chunked,mode=prepared_no_q_copy,warmup=1,iters=3,grid=8x8`。
host 表只用于确认 run 条件，不作为 kernel 胜负依据，单位 ms：

| profile | schedule | avg | best | worst | call | sync |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qktv_matmul_detail` | default | 0.214 | 0.161 | 0.311 | 0.071 | 0.139 |
| `qktv_matmul_detail` | drain-all | 0.225 | 0.167 | 0.317 | 0.081 | 0.139 |
| `qktv_math_detail` | default | 0.229 | 0.171 | 0.322 | 0.089 | 0.135 |
| `qktv_math_detail` | drain-all | 0.273 | 0.230 | 0.351 | 0.094 | 0.170 |
| `qktv_pack_detail` | default | 0.250 | 0.197 | 0.353 | 0.105 | 0.137 |
| `qktv_pack_detail` | drain-all | 0.210 | 0.164 | 0.293 | 0.065 | 0.140 |

device critical path，单位 us：

| zone | matmul default | matmul drain | math default | math drain | pack default | pack drain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FAP_COMPUTE` | 137.647 | 139.612 | 137.850 | 139.772 | 138.030 | 139.920 |
| `FAP_WRITER` | 138.428 | 140.595 | 138.632 | 140.624 | 138.804 | 140.885 |
| `FAP_READER` | 128.716 | 128.319 | 128.633 | 128.201 | 129.111 | 128.500 |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` | 2.258 | 5.465 | 2.381 | 5.584 | 2.291 | 5.500 |
| `FAP_COMPUTE_QKTV_V_SUBBLOCK` | 5.205 | 5.390 | 5.531 | 5.512 | 5.507 | 5.433 |
| `FAP_COMPUTE_QKTV_WAIT_V` | 4.303 | 4.434 | 4.416 | 4.327 | 4.609 | 4.510 |
| `FAP_COMPUTE_QKTV_WAIT_QKT` | 0.042 | 0.041 | 0.045 | 0.041 | 0.044 | 0.041 |
| `FAP_COMPUTE_QKV_PHASE` | 6.906 | 6.596 | 7.246 | 6.875 | 7.010 | 6.794 |
| `FAP_WRITER_WAIT_OUTPUT` | 108.788 | 111.516 | 109.993 | 111.511 | 109.133 | 112.039 |
| `TRISC-KERNEL` | 137.721 | 139.647 | 137.916 | 139.835 | 138.096 | 139.956 |

QKTV body 细分，单位 us：

| zone | default | drain-all | profile |
| --- | ---: | ---: | --- |
| `FAP_COMPUTE_QKTV_BODY_UNPACK_AB` | 0.591 | 1.720 | `qktv_matmul_detail` |
| `FAP_COMPUTE_QKTV_BODY_MATH_MATMUL` | 4.483 | 4.681 | `qktv_matmul_detail` |
| `FAP_COMPUTE_QKTV_BODY_PACK_THREAD` | 4.330 | 4.524 | `qktv_matmul_detail` |
| `FAP_COMPUTE_QKTV_BODY_MATMUL_LOOP` | 5.303 | 5.260 | `qktv_math_detail` |
| `FAP_COMPUTE_QKTV_MATH_DETAIL_MATMUL` | 4.593 | 4.559 | `qktv_math_detail` |
| `FAP_COMPUTE_QKTV_MATH_DETAIL_UNPACK_AB` | 0.436 | 1.672 | `qktv_math_detail` |
| `FAP_COMPUTE_QKTV_BODY_TILE_REGS_ACQUIRE` | 0.026 | 0.024 | `qktv_math_detail` |
| `FAP_COMPUTE_QKTV_BODY_TILE_REGS_COMMIT` | 0.042 | 0.042 | `qktv_math_detail` |
| `FAP_COMPUTE_QKTV_BODY_TILE_REGS_WAIT` | 0.032 | 0.030 | `qktv_math_detail` |
| `FAP_COMPUTE_QKTV_BODY_TILE_REGS_RELEASE` | 0.049 | 0.051 | `qktv_math_detail` |
| `FAP_COMPUTE_QKTV_BODY_PACK_ROWS` | 4.609 | 4.583 | `qktv_pack_detail` |

按 RISC 看关键路径，单位 us：

| zone / RISC | default | drain-all | profile |
| --- | ---: | ---: | --- |
| `QKTV_BODY_UNPACK_AB` / `TRISC_0` | 0.591 | 1.720 | `qktv_matmul_detail` |
| `QKTV_BODY_MATH_MATMUL` / `TRISC_1` | 4.483 | 4.681 | `qktv_matmul_detail` |
| `QKTV_BODY_PACK_THREAD` / `TRISC_2` | 4.330 | 4.524 | `qktv_matmul_detail` |
| `QKTV_BODY_MATMUL_LOOP` / `TRISC_1` | 5.303 | 5.260 | `qktv_math_detail` |
| `QKTV_MATH_DETAIL_MATMUL` / `TRISC_1` | 4.593 | 4.559 | `qktv_math_detail` |
| `QKTV_MATH_DETAIL_UNPACK_AB` / `TRISC_0` | 0.436 | 1.672 | `qktv_math_detail` |
| `QKTV_BODY_PACK_ROWS` / `TRISC_2` | 4.609 | 4.583 | `qktv_pack_detail` |
| `QKTV_MATMUL_PACK` / `TRISC_1` | 2.258 | 5.465 | `qktv_matmul_detail` |
| `QKTV_MATMUL_PACK` / `TRISC_2` | 1.819 | 4.667 | `qktv_matmul_detail` |

本轮结论：

- `tile_regs_wait` 不是瓶颈。`QKTV` body 中 acquire/commit/wait/release
  都只有约 0.02-0.05 us，因此“多开寄存器 buffer 掩盖 wait”的方向没有数据支撑。
- `QKTV_WAIT_QKT` 基本为 0.04 us；QKT 已经 ready，不是卡点。
- `QKTV_WAIT_V` 是 4.3-4.6 us 级，但 drain-all 并没有降低它；说明这轮回退
  不是简单“等 V 数据”，而是 compute work placement 变差。
- `MATH` 和 `PACK` 都贴近 critical path：默认下 math 约 4.48-4.59 us，
  pack 约 4.33-4.61 us；drain-all 下二者仍约 4.5 us。
- drain-all 真正放大的是 enclosure：`QKTV_MATMUL_PACK` 从约 2.3 us
  变成约 5.5 us，同时 writer wait 从约 109 us 增到 111-112 us。
  这说明原 split-drain 的局部 partial matmul/accumulation 虽然看起来碎，
  但能把部分成本藏进 sub-exp / V wait / writer 时间线；改成“全 drain 后
  一次完整 inner-dim matmul”反而让 matmul enclosure 集中进入 critical path。
- 下一步不应继续扫 subblock buffer 或寄存器 buffer。更值得做的是
  `QKT@V` work shape / handoff 粒度实验：保留 split-drain 的局部性，
  但调整 row group 和 V subblock 的组合，尝试让 writer 更早看到可 push 的
  output，同时不把完整 `QKT@V` 提前塞进 QK phase。

## 2026-05-30 QKTV pipeline / writer row-group 细分 profile

上一节已经把 `QKT@V` body 拆开，但仍有两个大框没有回答清楚：

- Phase 2 内部到底是 group0 drain、main-group `QKT@V`、SALAD/norm 哪一段撑起时间线。
- `FAP_WRITER_WAIT_OUTPUT` 是每个 row group 都在等，还是首个输出迟到导致 writer
  后续整体被串行拖住。

本轮新增两个互斥 profile 档位：

- `qktv_pipeline_detail`：只记录 Phase 2 结构段：
  `DRAIN_GROUP0`、`GROUP0_SUB_EXP`、`GROUP0_MATMUL_KT` 或
  `GROUP0_MATMUL`、`MAIN_GROUP`、`MAIN_MATMUL`、`EXP_MAX_DIFF`、
  `FINAL_DRAIN`。第一次把重复 wait/barrier/row norm/SALAD 测点也放进去时，
  drain-all 编译失败：

```text
Program size (71680) too large for kernel config buffer (70656)
```

因此最终删掉重复测点，只保留 Phase 2 独有结构段。

- `writer_pipeline_detail`：把 streaming writer row-group drain 拆成
  首个 group 和后续 group：
  `FIRST_GROUP_WAIT_OUTPUT`、`NEXT_GROUP_WAIT_OUTPUT`、
  `FIRST_GROUP_STORE_TILES`、`NEXT_GROUP_STORE_TILES`、
  `FIRST_GROUP_STORE_FLUSH`、`NEXT_GROUP_STORE_FLUSH`。

correctness：

| case | baseline | candidate | max abs diff | passed |
| --- | --- | --- | ---: | --- |
| smoke + `qktv_pipeline_detail` | TTNN full/chunked | copied full/chunked | 0.000000 | true |
| smoke + `writer_pipeline_detail` | TTNN full/chunked | copied full/chunked | 0.000000 | true |

profile 配置：`llama_prefill_16k_chunked --chunks 128,256`，
`variant=copied_chunked,mode=prepared_no_q_copy,warmup=1,iters=3,grid=8x8`。
raw logs：

- `/wafer/gsh/tmp/fa_profile_qktv_pipeline_default_16k_v2.log`
- `/wafer/gsh/tmp/fa_profile_qktv_pipeline_drainall_16k_v2.log`
- `/wafer/gsh/tmp/fa_profile_writer_pipeline_default_16k.log`
- `/wafer/gsh/tmp/fa_profile_writer_pipeline_drainall_16k.log`

host 表只用于确认 run 条件，不作为 kernel 胜负依据，单位 ms：

| profile | schedule | avg | best | worst | call | sync |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qktv_pipeline_detail` | default | 0.226 | 0.163 | 0.337 | 0.084 | 0.135 |
| `qktv_pipeline_detail` | drain-all | 0.217 | 0.164 | 0.310 | 0.073 | 0.140 |
| `writer_pipeline_detail` | default | 0.217 | 0.163 | 0.314 | 0.073 | 0.140 |
| `writer_pipeline_detail` | drain-all | 0.245 | 0.190 | 0.338 | 0.101 | 0.137 |

device critical path，单位 us：

| zone | qktv default | qktv drain-all | writer default | writer drain-all |
| --- | ---: | ---: | ---: | ---: |
| `FAP_COMPUTE` | 137.574 | 139.813 | 137.915 | 139.346 |
| `FAP_WRITER` | 138.684 | 140.796 | 139.021 | 140.420 |
| `FAP_READER` | 128.801 | 128.111 | 129.253 | 128.104 |
| `FAP_COMPUTE_QKV_PHASE` | 6.854 | 6.754 | 6.549 | 6.412 |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` | 2.081 | 5.302 | 2.046 | 5.192 |
| `FAP_COMPUTE_QKTV_WAIT_V` | 4.359 | 4.294 | - | - |
| `FAP_COMPUTE_QKTV_WAIT_QKT` | 0.047 | 0.045 | - | - |
| `FAP_COMPUTE_ROW_NORM` | 1.483 | 1.480 | 1.442 | 1.445 |
| `FAP_COMPUTE_SALAD_CORRECT` | 0.621 | 0.634 | 0.613 | 0.610 |
| `FAP_WRITER_WAIT_OUTPUT` | 109.427 | 111.845 | 110.059 | 111.004 |
| `FAP_WRITER_STORE_TILES` | 3.278 | 2.591 | 3.878 | 2.059 |

Phase 2 结构段细分，单位 us：

| zone | default | drain-all | 解释 |
| --- | ---: | ---: | --- |
| `FAP_QKTV_PIPE_DRAIN_GROUP0` | 6.804 | 6.710 | 第一组 drain 整体几乎不变 |
| `FAP_QKTV_PIPE_GROUP0_SUB_EXP` | 1.341 | 1.320 | group0 softmax/sub-exp 不是回退来源 |
| `FAP_QKTV_PIPE_GROUP0_MATMUL_KT` | 5.249 | - | default split-drain 的分段 matmul |
| `FAP_QKTV_PIPE_GROUP0_MATMUL` | - | 5.354 | drain-all 的完整 inner-dim matmul |
| `FAP_QKTV_PIPE_MAIN_GROUP` | 6.070 | 5.733 | 主循环 group 本身没有变慢 |
| `FAP_QKTV_PIPE_MAIN_MATMUL` | 2.121 | 1.790 | 主循环 matmul 不是回退来源 |
| `FAP_QKTV_PIPE_EXP_MAX_DIFF` | 0.536 | 0.513 | SALAD 前 exp max diff 很小 |
| `FAP_QKTV_PIPE_FINAL_DRAIN` | 1.044 | 1.032 | final drain 很小 |

writer row-group 细分，单位 us：

| zone | default | drain-all | 解释 |
| --- | ---: | ---: | --- |
| `FAP_WRITER_WAIT_OUTPUT` | 110.059 | 111.004 | 大等待来自 output 首次可见时间 |
| `FAP_WRITER_FIRST_GROUP_WAIT_OUTPUT` | 110.027 | 110.972 | 几乎等于总 wait |
| `FAP_WRITER_NEXT_GROUP_WAIT_OUTPUT` | 0.543 | 0.543 | 后续 group 基本不等 |
| `FAP_WRITER_FIRST_GROUP_STORE_TILES` | 3.378 | 1.891 | store 不是主瓶颈 |
| `FAP_WRITER_NEXT_GROUP_STORE_TILES` | 3.841 | 2.022 | store 不是主瓶颈 |
| `FAP_WRITER_FIRST_GROUP_STORE_FLUSH` | 0.037 | 0.033 | flush 不是主瓶颈 |
| `FAP_WRITER_NEXT_GROUP_STORE_FLUSH` | 0.040 | 0.030 | flush 不是主瓶颈 |

本轮结论：

- `FAP_WRITER_WAIT_OUTPUT` 的 109-111 us 不是每个 row group 都慢，而是
  `FIRST_GROUP_WAIT_OUTPUT` 几乎独占总等待。后续 row group 等待只有约
  0.54 us。writer 真正空闲在“首个可写输出迟到”这一处。
- Phase 2 内部没有发现 SALAD/norm 撑起 critical path：row norm 约
  1.45 us，SALAD correct 约 0.61 us，`EXP_MAX_DIFF` 约 0.5 us。
- drain-all 回退仍然集中在 `QKTV_MATMUL_PACK` enclosure：
  默认约 2.1 us，drain-all 约 5.2-5.3 us。它没有让首个 writer wait
  变短，反而把 `FIRST_GROUP_WAIT_OUTPUT` 从约 110.0 us 增到约 111.0 us。
- 所以当前瓶颈不是 writer store、writer flush、QKT readiness，也不是 SALAD/norm。
  问题是首个 normalized output 太晚被 push 到 `cb_out`，writer 在这之前一直空闲。
- 下一步的优化应该直接围绕“首个输出更早可见”做新 schedule，而不是继续加 buffer：
  尝试在 last K chunk 上对 group0 完成 `QKT@V` 后立刻对 group0 做
  normalize+push，再处理后续 main groups；或者把 group0 的
  `QKT@V -> normalize -> push` 做成更早 handoff。这个实验必须只改 copied
  schedule，且需要先过 correctness，因为当前代码默认要在 main loop 中统一处理
  SALAD/current sum 的交替。

## 2026-05-30 first-output handoff / group0 early-push 实验

上一轮已经证明 writer 大等待主要来自首个 row group 输出迟到。本轮继续把
“首个输出可见”链路拆开，并新建一个更窄的 schedule 实验：

- `first_output_handoff`：只 profile 首个 output handoff 的 compute 片段。
- `group0_early_push_v1`：只在 last K chunk 上，把 group0 的
  SALAD+normalize+push 提到 next-group `QKT@V` 之前。

correctness：

| case | baseline | candidate | max abs diff | passed |
| --- | --- | --- | ---: | --- |
| smoke + `first_output_handoff` | TTNN full/chunked | copied full/chunked | 0.000000 | true |
| smoke + `group0_early_push_v1` | TTNN full/chunked | copied full/chunked | 0.000000 | true |
| 16K q128/k256 + `group0_early_push_v1` | TTNN full | copied full | 0.000000 | true |
| 16K q128/k256 + `group0_early_push_v1` | TTNN chunked | copied chunked | 0.000000 | true |

本轮 raw logs：

- `/wafer/gsh/tmp/fa_profile_first_output_handoff_16k.log`
- `/wafer/gsh/tmp/fa_profile_group0_early_push_first_output_16k.log`
- `/wafer/gsh/tmp/fa_profile_group0_early_push_writer_16k.log`
- `/wafer/gsh/tmp/fa_profile_group0_early_push_none_16k.log`
- `/wafer/gsh/tmp/fa_profile_group0_early_push_depth4_writer_16k.log`
- `/wafer/gsh/tmp/fa_profile_default_none_16k_noprof.log`
- `/wafer/gsh/tmp/fa_profile_group0_early_push_none_16k_noprof.log`

`first_output_handoff` 首个输出链路，单位 us：

| zone | default | `group0_early_push_v1` | 解释 |
| --- | ---: | ---: | --- |
| `FAP_FIRST_OUT_GROUP0_DRAIN_MATMUL` | 6.878 | 6.723 | group0 drain+`QKT@V` 本身没有变慢 |
| `FAP_FIRST_OUT_BARRIER_AFTER_GROUP0` | 0.022 | 0.020 | barrier 不是瓶颈 |
| `FAP_FIRST_OUT_NEXT_GROUP_MATMUL` | 2.053 | 2.046 | default 会在首个 push 前先做下一组 `QKT@V` |
| `FAP_FIRST_OUT_ROW_NORM` | 1.504 | 1.347 | 首个 normalize 很小 |
| `FAP_FIRST_OUT_EXP_MAX_DIFF` | 0.507 | 0.519 | 很小 |
| `FAP_FIRST_OUT_SALAD_CORRECT` | 0.573 | 1.319 | early-push 把首个 SALAD 暴露到关键路径上 |
| `FAP_WRITER_WAIT_OUTPUT` | 109.638 | 106.986 | 首个 output 更早可见，writer wait 确实下降 |

writer row-group 复核，单位 us：

| zone | default | `group0_early_push_v1` | 解释 |
| --- | ---: | ---: | --- |
| `FAP_WRITER_WAIT_OUTPUT` | 110.059 | 107.018 | 总 wait 降约 3.0 us |
| `FAP_WRITER_FIRST_GROUP_WAIT_OUTPUT` | 110.027 | 106.985 | 下降来自首个 row group |
| `FAP_WRITER_NEXT_GROUP_WAIT_OUTPUT` | 0.543 | 3.022 | 后续 row group 等待变长，说明等待被部分后移 |
| `FAP_WRITER_FIRST_GROUP_STORE_TILES` | 3.378 | 1.647 | store 不是主因 |
| `FAP_WRITER_NEXT_GROUP_STORE_TILES` | 3.841 | 2.147 | store 不是主因 |

为了排除“只是 streaming output CB slot 不够”的可能，又补测了
`group0_early_push_v1 + pipeline_depth=4`：

| zone / metric | depth=2 | depth=4 | 结论 |
| --- | ---: | ---: | --- |
| host `sync_ms` | 0.154 | 0.170 | depth=4 更慢 |
| `BRISC-KERNEL` | 138.825 | 139.104 | 总 critical 不降 |
| `FAP_WRITER_WAIT_OUTPUT` | 107.018 | 106.558 | 只小幅下降 |
| `FAP_WRITER_FIRST_GROUP_WAIT_OUTPUT` | 106.985 | 106.525 | 只小幅下降 |
| `FAP_WRITER_NEXT_GROUP_WAIT_OUTPUT` | 3.022 | 3.019 | 后续等待没有被 slot 深度解决 |

无细分 device profile 对比，单位 us：

| zone | default | `group0_early_push_v1` | delta | 结论 |
| --- | ---: | ---: | ---: | --- |
| `BRISC-KERNEL` | 139.055 | 139.122 | +0.067 | writer/BRISC critical 没改善 |
| `TRISC-KERNEL` | 137.610 | 138.189 | +0.579 | compute critical 轻微回退 |
| `NCRISC-KERNEL` | 128.837 | 128.494 | -0.343 | reader 不在瓶颈上 |
| `FAP_COMPUTE` | 137.915 | 138.153 | +0.238 | compute 没赢 |
| `FAP_WRITER` | 139.021 | 139.088 | +0.067 | writer 总体没赢 |
| `FAP_WRITER_WAIT_OUTPUT` | 110.059 | 107.784 | -2.275 | 局部 wait 降了 |
| `FAP_COMPUTE_SALAD_CORRECT` | 0.613 | 1.319 | +0.706 | early push 把 SALAD 成本提前并放大 |

host no-profiler 对照，单位 ms：

| schedule | avg | best | worst | call | sync | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default | 0.164 | 0.157 | 0.183 | 0.017 | 0.145 | 当前 tuned 对照 |
| `group0_early_push_v1` | 0.159 | 0.157 | 0.164 | 0.011 | 0.146 | avg 有噪声级改善，但 sync 没改善 |

本轮结论：

- 当前代码确实存在首个输出 handoff 延迟：group0 `QKT@V` 后，default 会先做
  next-group `QKT@V`，首个 writer-visible output 才出现。
- `group0_early_push_v1` 验证了这个判断：首个 writer wait 从约 110 us
  降到约 107 us。
- 但这个 schedule 没有降低总 critical path。原因是它把首个 SALAD/normalize
  暴露到更早的关键路径，同时让后续 row group 的 writer wait 从约 0.54 us
  增到约 3.02 us。换句话说，它减少了首个空洞，但破坏了一部分原本被 main-loop
  `QKT@V` 隐藏的流水。
- `pipeline_depth=4` 不能修复这个后续等待，说明当前问题不是简单 `cb_out`
  slot 不够，而是 compute 侧生产顺序把后续 row group 的产出节奏打散。
- 因此 `group0_early_push_v1` 不升级为 current fastest。真正下一步不是简单
  “越早 push 越好”，而是做双组交错：group0 完成后准备首个 output，但不要让
  SALAD 完全挡住 next-group `QKT@V`；需要把 SALAD/normalize 拆得更细，尝试
  `exp_max_diff -> next-group QKT@V -> group0 SALAD+norm/push` 或者按 V subblock
  做 partial normalize/handoff。

## 2026-05-30 FA3-style PV/softmax pipeline 实验

本轮目标是验证用户提出的 FA3-style 方向：尽量不要让 FPU/MATH 空下来，
尝试把 previous row group 的 `PV` 提前塞到当前 QK/softmax 之间。实现保持
copied-only，不修改 TTNN baseline：

- `fa3_pipe_detail`：新增 FA3-style 细分测点。
- `fa3_pv_softmax_v1`：在 QK phase 内，某个 row group 的 softmax kt 分片完成后，
  立即做该 row group 对应分片的 partial `P@V`；Phase 2 跳过已经提前完成的 group。
- 这个 schedule 只允许 streaming compute、默认 SALAD handoff，并要求
  `qktv_h == qkt_subblock_h`。不满足条件时直接报错。

实验过程中先发现一个重要问题：第一版触发条件写成 `q_subblock > 1`，但主线
16K q128/k256 形状只有两个 QK row group，因此提前 `PV` 分支没有真正执行。
已修正为 `ready_group = q_subblock - 1`，并在 softmax PACK 写入到 PV UNPACK
读取之间加入显式 pack/unpack barrier。下表只记录修正后“真触发”的结果。

correctness：

| case | baseline | candidate | elements | max abs diff | passed |
| --- | --- | --- | ---: | ---: | --- |
| smoke full + `fa3_pv_softmax_v1` | TTNN full SDPA | copied full | 131072 | 0.000000 | true |
| smoke chunked + `fa3_pv_softmax_v1` | TTNN chunked | copied chunked | 32768 | 0.000000 | true |
| 16K full + `fa3_pv_softmax_v1` | TTNN full SDPA | copied full | 16777216 | 0.000000 | true |
| 16K chunked + `fa3_pv_softmax_v1` | TTNN chunked | copied chunked | 2097152 | 0.000000 | true |

本轮 raw logs：

- `/wafer/gsh/tmp/fa_profile_fa3_v1_triggered_smoke_correctness.log`
- `/wafer/gsh/tmp/fa_profile_fa3_v1_triggered_16k_correctness.log`
- `/wafer/gsh/tmp/fa_profile_fa3_default_host_16k.log`
- `/wafer/gsh/tmp/fa_profile_fa3_v1_triggered_host_16k.log`
- `/wafer/gsh/tmp/fa_profile_fa3_default_detail_risc_16k.log`
- `/wafer/gsh/tmp/fa_profile_fa3_v1_triggered_detail_risc_16k.log`

16K q128/k256 `prepared_no_q_copy` host no-profiler 对照，单位 ms：

| schedule | avg | best | worst | call | sync | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default | 0.158 | 0.157 | 0.161 | 0.010 | 0.145 | 当前 copied tuned 对照 |
| `fa3_pv_softmax_v1` | 0.170 | 0.169 | 0.177 | 0.010 | 0.158 | 真触发后回退，sync 多约 0.013 ms |

同一机器、同一 shape、device profiler + by-RISC，单位 us：

| zone | default | `fa3_pv_softmax_v1` | delta | 解释 |
| --- | ---: | ---: | ---: | --- |
| `BRISC-KERNEL` | 138.643 | 152.065 | +13.422 | writer/overall critical 回退 |
| `TRISC-KERNEL` | 137.314 | 150.807 | +13.493 | compute critical 明显回退 |
| `NCRISC-KERNEL` | 128.574 | 128.006 | -0.568 | reader 不是本轮瓶颈 |
| `FAP_COMPUTE` | 137.279 | 150.770 | +13.491 | 回退集中在 compute path |
| `FAP_WRITER` | 138.603 | 152.031 | +13.428 | writer 等 output 被 compute 拖长 |
| `FAP_WRITER_WAIT_OUTPUT` | 109.316 | 119.859 | +10.543 | output 反而更晚可见 |
| `FAP_COMPUTE_QK_PHASE` | 18.421 | 26.141 | +7.720 | 提前 `PV` 被塞进 QK phase |
| `FAP_COMPUTE_QKV_PHASE` | 6.776 | 2.415 | -4.361 | Phase 2 变短是 work 搬移，不是 overlap |
| `FAP_COMPUTE_QKTV_MATMUL_PACK` | 2.106 | 8.119 | +6.013 | `PV` matmul/pack 成本进入 critical path |
| `FAP_COMPUTE_QK_MATMUL_PACK` | 12.544 | 12.503 | -0.041 | QK matmul 本体没有变快 |
| `FAP_COMPUTE_QK_MATMUL_BODY` | 12.453 | 12.404 | -0.049 | QK math 本体没有变快 |
| `FAP_COMPUTE_QK_SOFTMAX_EXP_SUM` | 2.076 | 1.893 | -0.183 | softmax 小幅下降，但不足以抵消 PV |
| `FAP_FA3_SOFTMAX_PV_OVERLAP_CANDIDATE` | - | 8.179 | +8.179 | 真触发的新路径本身成为大块成本 |
| `FAP_FA3_PV_PREV_MATMUL` | 5.139 | 8.075 | +2.936 | previous `PV` 更重且更早暴露 |
| `FAP_FA3_NEXT_PV_MATMUL` | 2.147 | 3.511 | +1.364 | 后续 `PV` 也变慢 |

按 TRISC 看，回退不是某个 reader/writer 等待，而是同一 compute 指令流上的
MATH/PACK 路径被更长的 `PV` work 占住：

| zone / RISC | default | `fa3_pv_softmax_v1` | 解释 |
| --- | ---: | ---: | --- |
| `TRISC-KERNEL` / `TRISC_0` | 136.849 | 150.333 | 总 compute path 回退 |
| `TRISC-KERNEL` / `TRISC_1` | 137.302 | 150.792 | 总 compute path 回退 |
| `TRISC-KERNEL` / `TRISC_2` | 137.314 | 150.807 | 总 compute path 回退 |
| `FAP_COMPUTE_QK_MATMUL_PACK` / `TRISC_1` | 12.477 | 12.470 | QK pack/math 主体没有收益 |
| `FAP_COMPUTE_QK_MATMUL_PACK` / `TRISC_2` | 12.544 | 12.503 | QK pack/math 主体没有收益 |
| `FAP_FA3_SOFTMAX_PV_OVERLAP_CANDIDATE` / `TRISC_1` | - | 8.179 | 新插入 partial `PV` 占用 compute |
| `FAP_FA3_SOFTMAX_PV_OVERLAP_CANDIDATE` / `TRISC_2` | - | 7.359 | 新插入 partial `PV` 占用 compute |
| `FAP_FA3_SOFTMAX_CUR` / `TRISC_2` | 1.356 | 1.359 | softmax 当前分片没有被隐藏成“免费” |

本轮结论：

- `fa3_pv_softmax_v1` correctness 通过，但性能失败；它不升级为 current fastest。
- 失败原因不是 reader 慢，也不是 host 计时误差。`NCRISC-KERNEL` 没变慢，
  `TRISC-KERNEL` 和 `FAP_COMPUTE` 回退约 13.5 us。
- 它确实把 Phase 2 从 6.776 us 缩到 2.415 us，但这是把 `PV` work 搬到
  QK phase，导致 QK phase 从 18.421 us 拉长到 26.141 us。对主线形状而言，
  粗粒度 previous `PV` 没有和 softmax 形成有效并行，而是在同一 TRISC
  MATH/PACK 路径上串行排队。
- 当前 TT-Metal 单个 compute kernel 的三条 TRISC 线程可以分别跑
  UNPACK/MATH/PACK，但不是两个独立可同时调度的 matmul/softmax 指令流。
  把整段 `PV` 提前放进 QK phase，会抢占后续 QK/softmax/SALAD 所需的
  DST、PACK 和 MATH 路径。

下一步不要继续把完整 `PV` 提前塞进 QK phase。更合理的 FA3-style 方向是：

1. 做更小粒度的 handoff，只提前不占用大块 MATH/PACK 的准备工作，例如
   `exp_max_diff`、状态指针、CB readiness 或很小的 V subblock。
2. 如果继续做 previous `PV`，必须把粒度降到不会把 `FAP_COMPUTE_QK_PHASE`
   拉长的单位，并用 by-RISC 验证 `TRISC_1/2` critical path 是否下降。
3. 更激进的方向需要拆成两个 compute kernel 或更低层 LLK/firmware 级别的
   双执行流实验；在当前单 compute kernel 指令流里，整段 `PV` 与 softmax/QK
   不能自然并行。

## 2026-05-30 两条新实验分支

本轮把下一步实验从 “继续改 copied baseline kernel” 拆成两个独立 host/device
结构。这样后续可以分别推进两个方向，而不会把所有尝试混在一份 device kernel
里：

| experiment | host/device 路径 | kernel 路径 | 当前 v1 问题 |
| --- | --- | --- | --- |
| `copied` | `host/copied_sdpa/device/` | `kernels/compute/`, `kernels/dataflow/` | 当前 copied/tuned 对照 |
| `split_compute_v1` | `host/split_compute_sdpa/device/` | `kernels/split_compute_sdpa/` | 验证两个 compute kernel 的 program/profile 边界 |
| `llk_microflow_v1` | `host/llk_microflow_sdpa/device/` | `kernels/llk_microflow_sdpa/` | 拆更细 LLK 片段时间线 |

`split_compute_v1` 的 v1 不是最终 producer-consumer FA。它做了两件事：

- producer SDPA compute kernel 继续跑完整 SDPA，但自动把 producer grid
  优先缩到 `device_grid.x, device_grid.y - 1`，给 device grid 末尾保留
  probe core；如果设备只有一行，才退到 `device_grid.x - 1, device_grid.y`。
- 保留 core 上挂第二个 compute kernel
  `sdpa_consumer_probe.cpp`，只打 `FAP_SPLIT_COMPUTE_V1_CONSUMER_PROBE`
  zone，不读写输出。这样可以先验证 TT-Metal program、program cache、
  profiler CSV 是否能稳定表达“两条 compute kernel 边界”。如果这个边界稳定，
  下一步才把 QK/softmax state 通过 L1/CB/信号交给 consumer 做 `PV`。

`llk_microflow_v1` 保持数值路径不变，只在独立 kernel 里增加更细粒度测点：

- `FAP_LLK_MICROFLOW_EXP_MAX_DIFF`
- `FAP_LLK_MICROFLOW_PV_GROUP_MATMUL`
- `FAP_LLK_MICROFLOW_PREV_GROUP_SALAD`
- `FAP_LLK_MICROFLOW_DRAIN_EXP_MAX_DIFF`
- `FAP_LLK_MICROFLOW_DRAIN_SALAD`

这些 zone 只在 `--qk-detail-profile fa3_pipe_detail` 时打开。目标是判断
previous-group SALAD、drain SALAD、`exp_max_diff`、`PV` group matmul
谁真正挤占 TRISC critical path，再决定移动哪一个小片段。它不是兼容层，也不是
另一个 baseline；它是后续 LLK/micro-schedule 实验的独立 fork。

本轮验证命令全部通过：

```bash
cmake --build build_Release --target flash_attention_profile --parallel $(nproc)

./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset smoke --variant all --mode prepared --warmup 1 --iters 1 \
  --check-correctness --no-device-profiler-read \
  --experiment llk_microflow_v1

./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset smoke --variant all --mode prepared --warmup 1 --iters 1 \
  --check-correctness --no-device-profiler-read \
  --experiment split_compute_v1
```

smoke device profiler 只用于确认新 zone 能被稳定采集，不用于判断 16K 性能：

| experiment | zone | count | critical us | 解释 |
| --- | --- | ---: | ---: | --- |
| `split_compute_v1` | `FAP_SPLIT_COMPUTE_V1_PRODUCER` | 297 | 72.493 | producer 仍跑完整 SDPA |
| `split_compute_v1` | `FAP_SPLIT_COMPUTE_V1_CONSUMER_PROBE` | 3 | 0.015 | 第二 compute kernel 边界可见 |
| `llk_microflow_v1` | `FAP_LLK_MICROFLOW_V1_COMPUTE` | 330 | 73.855 | 独立 compute fork 可见 |
| `llk_microflow_v1` | `FAP_LLK_MICROFLOW_PV_GROUP_MATMUL` | 63 | 1.061 | `PV` group matmul 小片段 |
| `llk_microflow_v1` | `FAP_LLK_MICROFLOW_EXP_MAX_DIFF` | 39 | 0.541 | softmax state 小片段 |
| `llk_microflow_v1` | `FAP_LLK_MICROFLOW_PREV_GROUP_SALAD` | 39 | 0.573 | previous-group SALAD 小片段 |
| `llk_microflow_v1` | `FAP_LLK_MICROFLOW_DRAIN_SALAD` | 39 | 0.699 | drain SALAD 小片段 |

16K chunked q128/k256 的 host no-profiler 对照如下，命令统一使用
`--variant copied_chunked --mode prepared_no_q_copy --warmup 2 --iters 5`：

| experiment | grid | avg ms | best ms | worst ms | call ms | sync ms | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `copied` | `auto` | 0.227 | 0.188 | 0.255 | 0.044 | 0.179 | 本轮 host avg 有扰动，best 接近其他 fork |
| `llk_microflow_v1` | `auto` | 0.186 | 0.184 | 0.191 | 0.014 | 0.170 | 数值路径不变，主要证明独立 LLK 测点 fork 可跑 |
| `split_compute_v1` | `11x9` | 0.186 | 0.184 | 0.189 | 0.013 | 0.171 | producer grid 少一行，probe 不做真实消费 |

raw logs：

- `/wafer/gsh/tmp/fa_profile_split_compute_v1_smoke_profile.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_v1_smoke_profile.log`
- `/wafer/gsh/tmp/fa_profile_copied_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_v1_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_split_compute_v1_16k_q128_k256_noprof.log`

这张表不能解读为两个新 fork 已经优化成功：`llk_microflow_v1` 目前只是
profile-only fork；`split_compute_v1` 甚至减少了 producer core grid，并且 consumer
只是 no-op probe。它们的价值是把后续两条实验路线分开管理：

1. `split_compute_v1` 下一步要加真实 L1/CB/信号 handoff，让 consumer 消费已经
   ready 的上一段 probability/state 做 `PV` 或 normalize，而不是只挂 probe。
2. `llk_microflow_v1` 下一步要在 by-RISC profile 下移动更小片段，例如
   `exp_max_diff`、SALAD 或很小的 V subblock，确认是否能降低 TRISC critical path。

## 2026-05-30 `llk_microflow_v1` 小片段排布实验

本轮按上一节的方向，在 `llk_microflow_v1` 独立 fork 中新增 4 个
micro schedule。它们不是 copied baseline 的兼容模式，也不作用于官方 TTNN
baseline：

| schedule | 实验点 | 预期验证 |
| --- | --- | --- |
| `llk_drain_exp_before_pv_v1` | group0 drain 的 `sub_exp` 全部放到 `PV` 前 | split-drain 交错是否必要 |
| `llk_prev_exp_after_pv_v1` | previous-group `exp_max_diff` 延后到当前 `PV` 后 | softmax state 是否能避开 `PV` 前路径 |
| `llk_salad_before_pv_v1` | previous-group SALAD 放到当前 `PV` 前 | SALAD 是否应该更早完成 |
| `llk_v_ready_prefetch_v1` | QK phase 内更早等待 V ready | V wait 是否能被隐藏 |

验证边界：

- build：`cmake --build build_Release --target flash_attention_profile --parallel $(nproc)`
- smoke correctness：4 个 schedule 的 full/chunked 都通过，`max_abs_diff=0`。
- 16K q128/k256 correctness：4 个 schedule 的 full/chunked 都通过；
  `llk_drain_exp_before_pv_v1` 有 `max_abs_diff=0.005859`，仍低于
  `tolerance=0.125000`，其余 schedule 为 `0.000000`。
- 原 `fa3_pipe_detail` 在 16K q128/k256 下会超过 TENSIX kernel config buffer：
  `Program size (72704) too large for kernel config buffer (70656)`。
  因此本轮 device profiler 使用新的 `--qk-detail-profile llk_microflow_detail`。

16K chunked q128/k256 host no-profiler，对照命令统一使用
`--variant copied_chunked --mode prepared_no_q_copy --warmup 2 --iters 5`
和 `--experiment llk_microflow_v1`：

| schedule | avg ms | best ms | worst ms | call ms | sync ms | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `default` | 0.238 | 0.216 | 0.297 | 0.069 | 0.165 | 本轮 default host call 抖动较大 |
| `llk_drain_exp_before_pv_v1` | 0.194 | 0.186 | 0.221 | 0.023 | 0.169 | sync 略慢，不是收益 |
| `llk_prev_exp_after_pv_v1` | 0.186 | 0.184 | 0.190 | 0.013 | 0.171 | host 稳定，但 sync 慢于 default |
| `llk_salad_before_pv_v1` | 0.186 | 0.185 | 0.189 | 0.013 | 0.171 | host 稳定，但 sync 慢于 default |
| `llk_v_ready_prefetch_v1` | 0.189 | 0.187 | 0.196 | 0.013 | 0.174 | V wait 提前后 sync 更慢 |

同一 shape 的 device profiler，单位 us，取 `critical_us`：

| schedule | FAP_COMPUTE | QK phase | QKV phase | WAIT_V | PV group | exp/state | prev SALAD | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `default` | 160.616 | 21.090 | 6.998 | 4.909 | 2.080 | 0.517 | 0.582 | 基线 |
| `llk_drain_exp_before_pv_v1` | 160.924 | 21.454 | 6.628 | 4.437 | 1.816 | 1.333 / 0.459 | 0.583 | QKV 变短但 QK/overall 回退 |
| `llk_prev_exp_after_pv_v1` | 160.784 | 21.156 | 6.756 | 4.830 | 2.081 | 0.525 | 0.929 | 延后 exp 后 SALAD 变重 |
| `llk_salad_before_pv_v1` | 161.391 | 20.934 | 6.926 | 4.636 | 2.063 | 0.504 | 2.435 | SALAD 前移明显变重 |
| `llk_v_ready_prefetch_v1` | 165.606 | 28.180 | 3.162 | 8.361 | 2.085 | 0.520 | 0.584 | QKV 变短但 QK critical path 大幅变长 |

by-RISC 结果显示，整体 critical path 仍由 `TRISC_1/TRISC_2` 维持：

| schedule | FAP_COMPUTE TRISC_0 | TRISC_1 | TRISC_2 | QK phase TRISC_1 | QK phase TRISC_2 | QKV phase TRISC_1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `default` | 160.116 | 160.530 | 160.616 | 20.799 | 21.090 | 6.998 |
| `llk_drain_exp_before_pv_v1` | 160.489 | 160.870 | 160.924 | 21.211 | 21.454 | 6.628 |
| `llk_prev_exp_after_pv_v1` | 160.349 | 160.731 | 160.784 | 20.937 | 21.156 | 6.756 |
| `llk_salad_before_pv_v1` | 160.933 | 161.381 | 161.391 | 20.747 | 20.934 | 6.926 |
| `llk_v_ready_prefetch_v1` | 165.119 | 165.537 | 165.606 | 27.644 | 28.180 | 3.162 |

本轮结论：

- 四个 schedule 都没有让 `FAP_COMPUTE` 稳定下降 2% 以上，因此不升级为
  current fastest。
- `llk_v_ready_prefetch_v1` 是明确负例：虽然 `QKV phase` 从 6.998 us 降到
  3.162 us，但 `QK phase` 从 21.090 us 拉到 28.180 us，说明过早等待 V
  只是把等待塞进 QK critical path。
- `llk_drain_exp_before_pv_v1` 也是负例：split-drain 的局部 `sub_exp/PV`
  交错是有价值的，把 drain `sub_exp` 全部提前会让 group0 的 `PV` 变重。
- `llk_prev_exp_after_pv_v1` 和 `llk_salad_before_pv_v1` 都只改变了小片段位置，
  但 overall critical path 没有改善；SALAD 前移尤其会把 SALAD 自身变成更大的
  critical event。

下一步不要继续在单 compute kernel 里移动这些小片段。更有价值的方向是回到
`split_compute_v1`：先做 signal-only handoff，再做 metadata/state handoff。
原因是本轮已经证明在同一个 compute kernel 指令流中，重新排序 `exp/SALAD/V wait`
只是在 `QK phase` 和 `QKV phase` 之间搬移等待或工作量，没有真正产生第二条可并行
执行路径。

raw logs：

- `/wafer/gsh/tmp/fa_profile_llk_microflow_default_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_llk_drain_exp_before_pv_v1_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_llk_prev_exp_after_pv_v1_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_llk_salad_before_pv_v1_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_llk_v_ready_prefetch_v1_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_default_16k_q128_k256_profile.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_llk_drain_exp_before_pv_v1_16k_q128_k256_profile.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_llk_prev_exp_after_pv_v1_16k_q128_k256_profile.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_llk_salad_before_pv_v1_16k_q128_k256_profile.log`
- `/wafer/gsh/tmp/fa_profile_llk_microflow_llk_v_ready_prefetch_v1_16k_q128_k256_profile.log`

建议验证命令：

```bash
./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset smoke --variant all --mode prepared --warmup 1 --iters 1 \
  --check-correctness --no-device-profiler-read \
  --experiment llk_microflow_v1

./build_Release/programming_examples/profiler/flash_attention_profile \
  --preset smoke --variant all --mode prepared --warmup 1 --iters 1 \
  --check-correctness --no-device-profiler-read \
  --experiment split_compute_v1
```

## 2026-05-30 split handoff 继续实验

上一轮 `split_signal_only_v1` / `split_output_stream_signal_v1` 只证明跨 kernel
信号链路很便宜，但 token 是从 writer 路径发出的。它回答不了一个关键问题：
downstream consumer 到底能不能比 writer 写完 output 更早启动。

本轮按数据继续做了两步更早 handoff：

1. `split_l1_ready_signal_v1`
   - writer 在 `cb_out` row group 已经可读、但还没写 DRAM 前发 token。
   - 目标：测 final normalized output 的 L1 可见性是否明显早于 writer-after-store。
2. `split_state_ready_signal_v1`
   - producer compute 在 final K chunk 的 QK/softmax state ready 后向本地 writer
     CB 发 token，writer 只负责转发远端 token。
   - 目标：测 downstream consumer 如果接在 softmax state 后，而不是 final output 后，
     到底能提前多少。

correctness：

- `split_l1_ready_signal_v1` smoke full/chunked 通过，`max_abs_diff=0`。
- `split_state_ready_signal_v1` smoke full/chunked 通过，`max_abs_diff=0`。

16K chunked q128/k256 host no-profiler 对照，命令统一使用
`--variant copied_chunked --mode prepared_no_q_copy --warmup 2 --iters 5
--experiment split_compute_v1`：

| schedule | avg ms | best ms | worst ms | call ms | sync ms | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `default` | 0.189 | 0.185 | 0.195 | 0.018 | 0.169 | split fork 默认对照 |
| `split_output_stream_signal_v1` | 0.188 | 0.185 | 0.192 | 0.015 | 0.171 | writer-after-store token，基本不改变性能 |
| `split_l1_ready_signal_v1` | 0.187 | 0.185 | 0.193 | 0.015 | 0.171 | L1-ready token 不产生可见收益 |
| `split_state_ready_signal_v1` | 0.186 | 0.185 | 0.190 | 0.015 | 0.170 | state-ready token 也只是噪声级变化，不是最终优化 |

16K chunked q128/k256 device profiler 关键 zone，单位 us：

| schedule | producer compute | writer wait output | state/local wait | consumer token wait | token send | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `split_output_stream_signal_v1` | 160.125 | 128.473 | n/a | 128.299 | 0.164 | token 从 writer 写完 output 后发，consumer 几乎等完整 writer/output 路径 |
| `split_l1_ready_signal_v1` | 161.564 | 129.057 | n/a | 126.546 | 0.178 | L1 可读只比 writer-after-store 早约 2 us，final output handoff 太晚 |
| `split_state_ready_signal_v1` | 160.901 | 8.746 | 121.383 | 127.792 | 0.164 | 等待从 `cb_out` 转移到 state-ready CB，说明 final output 前还有约 8-9 us 的 PV/norm/store 尾部窗口 |

补充解释：

- `split_l1_ready_signal_v1` 的 token 粒度是 row group，所以 profiler 中
  `FAP_SPLIT_SIGNAL_L1_READY_SEND` count=4；它只让 consumer critical wait 从
  128.299 us 降到 126.546 us，收益太小。
- `split_state_ready_signal_v1` 中 `FAP_WRITER_WAIT_OUTPUT` 从 128.473 us 降到
  8.746 us，不是因为 output 更快，而是 writer 先在
  `FAP_SPLIT_STATE_READY_LOCAL_WAIT=121.383 us` 等 compute 发来的 softmax-state
  token。这个实验把真实边界分开了：大头仍是 QK/softmax state 产生，state 之后到
  final output 还有约 8-9 us 可被拆给 consumer 尝试 overlap。
- 这说明下一步不应该再围绕 final output token 做文章。真正值得做的是把 consumer
  接到 state-ready token 后面，让第二个 compute kernel 开始做 `PV` 形状的工作，
  然后再补真实 state / `P` / `V` 数据搬运。

raw logs：

- `/wafer/gsh/tmp/fa_profile_split_l1_ready_signal_v1_smoke_correctness.log`
- `/wafer/gsh/tmp/fa_profile_split_state_ready_signal_v1_smoke_correctness.log`
- `/wafer/gsh/tmp/fa_profile_split_compute_default_16k_q128_k256_noprof_rerun.log`
- `/wafer/gsh/tmp/fa_profile_split_output_stream_signal_v1_16k_q128_k256_noprof_rerun.log`
- `/wafer/gsh/tmp/fa_profile_split_l1_ready_signal_v1_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_split_state_ready_signal_v1_16k_q128_k256_noprof.log`
- `/wafer/gsh/tmp/fa_profile_split_output_stream_signal_v1_16k_q128_k256_profile_rerun.log`
- `/wafer/gsh/tmp/fa_profile_split_l1_ready_signal_v1_16k_q128_k256_profile.log`
- `/wafer/gsh/tmp/fa_profile_split_state_ready_signal_v1_16k_q128_k256_profile.log`

下一步实验：

1. 新建 `split_state_consumer_probe_v1`，consumer compute 在 state-ready token 后执行
   一个 `PV` 形状的 dummy/micro matmul 或最小 LLK 循环，测第二 compute kernel 是否能把
   这 8-9 us 尾部窗口真正吃掉。
2. 如果 consumer probe 能和 producer 的后续路径 overlap，再加真实数据路径：
   producer 通过本地 CB/DRAM scratch 导出 QK/softmax state 或 `P` tile，consumer
   读取 `V` 并做 partial `P@V`。
3. 如果 probe 不能 overlap，说明单 program 内第二 compute kernel 的调度/资源隔离还不够，
   下一步应转向更底层的 compute/datamovement 拆分，而不是继续搬 token。

## 2026-05-30 split consumer probe 与 shape sweep

上一节已经证明 final-output token 太晚，state-ready token 更早。本轮继续做
`split_state_consumer_probe_v1`：consumer 收到 state-ready token 后执行一个
`PV` 形状的 dummy matmul probe。输入由 consumer dataflow 预先填零，因此它不测
真实 `P/V` 搬运成本，只测第二 compute kernel 是否能把一部分 `PV` 形状工作藏进
producer 还在运行的窗口里。

新增 stress schedule：

| schedule | consumer probe 工作量 | 目的 |
| --- | --- | --- |
| `split_state_consumer_probe_v1` | 1x `PV` subblock groups | 最小可行 consumer work |
| `split_state_consumer_probe_x4_v1` | 4x groups | 测隐藏窗口开始露出的边界 |
| `split_state_consumer_probe_x8_v1` | 8x groups | 测更重 consumer work 是否能被大 shape 隐藏 |
| `split_state_consumer_vprefetch_x8_v1` | 真实 `V` 预取 + 8x groups | 测 V 可提前预取时是否仍能隐藏 |
| `split_state_consumer_vafter_state_x8_v1` | state-ready 后真实读 `V` + 8x groups | 测最保守 V 读是否进入 critical path |

correctness：

- `split_state_consumer_probe_v1` smoke full/chunked 通过，`max_abs_diff=0`。
- `split_state_consumer_probe_x4_v1` smoke full/chunked 通过，`max_abs_diff=0`。
- `split_state_consumer_probe_x8_v1` smoke full/chunked 通过，`max_abs_diff=0`。
- `split_state_consumer_vprefetch_x8_v1` smoke full/chunked 通过，`max_abs_diff=0`。
- `split_state_consumer_vafter_state_x8_v1` smoke full/chunked 通过，`max_abs_diff=0`。

### 先在 16K q128/k256 上找容量边界

16K chunked q128/k256 no-profiler，统一使用
`--variant copied_chunked --mode prepared_no_q_copy --warmup 2 --iters 5`：

| schedule | avg ms | best ms | worst ms | call ms | sync ms | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `split_state_ready_signal_v1` | 0.186 | 0.185 | 0.190 | 0.015 | 0.170 | state-ready 信号对照 |
| `split_state_consumer_probe_v1` | 0.186 | 0.184 | 0.190 | 0.013 | 0.171 | 1x 基本可隐藏 |
| `split_state_consumer_probe_x4_v1` | 0.188 | 0.186 | 0.193 | 0.013 | 0.173 | x4 开始轻微露出 |
| `split_state_consumer_probe_x8_v1` | 0.201 | 0.200 | 0.205 | 0.013 | 0.186 | x8 明显超出 16K 窗口 |

同 shape device CSV 解析结果，单位 us：

| schedule | producer `FAP_COMPUTE` | consumer total | writer wait output | consumer token wait | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| x1 | 160.959 | 153.248 | 8.810 | 128.444 | consumer 仍小于 producer，可隐藏 |
| x4 | 160.947 | 163.501 | 8.684 | 128.359 | consumer 略超过 producer，开始露出 |
| x8 | 160.350 | 176.626 | 9.123 | 128.941 | consumer 超出约 16 us，对应 sync 回退 |

这里不能把 x1/x4/x8 解读为最终性能优化，因为 consumer 还没有把真实 partial output
接回主路径。它们只是容量探针：16K q128/k256 大约能藏住 1x，小幅接近 x4，藏不住
x8。

### 输入序列长度 sweep

随后固定 `copied_chunked + prepared_no_q_copy + warmup2/iters5`，扫不同 S/q/k。
表中 `state-ready` 是不做 consumer work 的对照：

| shape | schedule | avg ms | best ms | sync ms | 观察 |
| --- | --- | ---: | ---: | ---: | --- |
| 2K q128/k256 | state-ready | 0.188 | 0.185 | 0.172 | 短序列对照 |
| 2K q128/k256 | x1 | 0.187 | 0.185 | 0.170 | 1x 可隐藏 |
| 2K q128/k256 | x4 | 0.202 | 0.189 | 0.171 | avg 受 call 抖动，sync 未明显变差 |
| 2K q128/k256 | x8 | 0.203 | 0.199 | 0.186 | x8 已经露出 |
| 2K q256/k256 | state-ready | 0.249 | 0.228 | 0.201 | call 抖动较大 |
| 2K q256/k256 | x8 | 0.221 | 0.218 | 0.204 | sync 基本持平，不应按 avg 断言加速 |
| 16K q128/k256 | state-ready | 0.186 | 0.184 | 0.169 | 当前主线对照 |
| 16K q128/k256 | x8 | 0.215 | 0.206 | 0.187 | x8 不可隐藏 |
| 16K q256/k256 | state-ready | 0.220 | 0.219 | 0.203 | 更大 q chunk |
| 16K q256/k256 | x8 | 0.234 | 0.219 | 0.203 | sync 持平，avg 主要是 call 侧变化 |
| 32K q128/k256 | state-ready | 0.618 | 0.594 | 0.559 | 长序列对照 |
| 32K q128/k256 | x1 | 0.601 | 0.587 | 0.570 | sync 只小幅增加 |
| 32K q128/k256 | x4 | 0.599 | 0.581 | 0.565 | 仍基本隐藏 |
| 32K q128/k256 | x8 | 0.589 | 0.578 | 0.564 | x8 也基本隐藏，avg 受 host call 降低影响 |

32K q128/k256 的 device profile 解释了原因：

| schedule | producer `FAP_COMPUTE` | consumer total | state/local wait | consumer token wait | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| state-ready | 554.817 | 314.584 | 271.983 | 268.831 | no-op consumer 小于 producer |
| x8 | 554.857 | 349.742 | 270.431 | 269.407 | x8 consumer 仍明显小于 producer，被长序列窗口隐藏 |

所以“大输入序列”确实更适合 split consumer 方向：不是因为 token 更便宜，而是 producer
每个 program 的 critical path 变长，第二 compute kernel 的工作更容易被隐藏。

### 权重/头维度更大的 shape

这里把“权重 shape 更大”落实为更大的 head dim `D`，以及更多 query heads `H`。
同样只比较 state-ready 与 x8 stress：

| shape | state avg/sync ms | x8 avg/sync ms | 结论 |
| --- | ---: | ---: | --- |
| 16K, H=8, D=256, q128/k256 | 0.354 / 0.325 | 0.388 / 0.356 | D 变大但 S 仍是 16K，x8 露出约 31 us |
| 32K, H=8, D=256, q128/k256 | 1.102 / 1.079 | 1.106 / 1.081 | D=256 且 S=32K 时 x8 基本隐藏 |
| 16K, H=16, D=128, q128/k256 | 0.371 / 0.324 | 0.355 / 0.328 | sync 基本持平，avg 有 host call 抖动 |
| 32K, H=16, D=128, q128/k256 | 0.937 / 0.905 | 0.954 / 0.904 | sync 持平，x8 没有 device-side 回退 |

D=256 的 device profile：

| shape | producer `FAP_COMPUTE` | x8 consumer total | state/local wait | consumer token wait | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 16K, D=256 | 310.604 | 340.687 | 240.355 | 254.222 | consumer 超过 producer，x8 露出 |
| 32K, D=256 | 1073.520 | 670.383 | 529.691 | 523.220 | producer 窗口足够大，x8 隐藏 |

本轮结论：

- 只增大 `H` 不一定增加单个 producer core 的 critical window，因为 head 维度更多是增加
  并行 work 和 program 总 work；从 sync 看 H16 的 x8 没有明显回退，但这不等于真实
  split FA 已经更快。
- 增大 `D` 会同时放大 producer QK/PV 和 consumer PV-shaped probe。16K+D256 下
  consumer x8 仍会露出，32K+D256 下才被 producer 长窗口盖住。
- 因此下一步真实优化应该优先在长序列、大 D 的 prefill shape 上做真实 data handoff；
  短序列/小 D 上，split compute 的额外同步和 consumer work 更容易直接出现在 critical path。
- 当前 x1/x4/x8 仍是 probe，不产生最终 output。真正的下一步是把 `P`/softmax state 与
  `V` 的真实数据路径接到 consumer，并优先选 `S=32K, D=128/256` 作为实验主线。

raw logs：

- `/wafer/gsh/tmp/fa_profile_shape_sweep_split_consumer_20260530.tsv`
- `/wafer/gsh/tmp/fa_profile_shape_sweep_weight_20260530.tsv`
- `/wafer/gsh/tmp/fa_profile_split_state_consumer_probe_v1_16k_q128_k256_profile_device.csv`
- `/wafer/gsh/tmp/fa_profile_split_state_consumer_probe_x4_v1_16k_q128_k256_profile_device.csv`
- `/wafer/gsh/tmp/fa_profile_split_state_consumer_probe_x8_v1_16k_q128_k256_profile_device.csv`
- `/wafer/gsh/tmp/fa_profile_32k_q128_k256_split_state_ready_signal_v1_profile_device.csv`
- `/wafer/gsh/tmp/fa_profile_32k_q128_k256_split_state_consumer_probe_x8_v1_profile_device.csv`
- `/wafer/gsh/tmp/fa_profile_16k_D256_q128_k256_split_state_consumer_probe_x8_v1_profile_device.csv`
- `/wafer/gsh/tmp/fa_profile_32k_D256_q128_k256_split_state_consumer_probe_x8_v1_profile_device.csv`

### 真实 V 读 probe

上一组 x8 仍然使用 zero-filled dummy `V`。本轮继续加两个 real-V probe：

1. `split_state_consumer_vprefetch_x8_v1`：consumer 收到 start token 后就读取真实
   `V` tiles，之后等 state-ready token 才执行 x8 `PV` probe。
2. `split_state_consumer_vafter_state_x8_v1`：consumer 等 state-ready token 后才读取
   真实 `V` tiles，再执行 x8 `PV` probe。

注意：这里的真实 `V` 读是从 `input_tensor_v` buffer 读取实际 tile 数据，用来测
consumer 侧 NoC/DRAM 读成本；`P`/softmax state 仍然是 dummy，最终 output 还没有接回
主路径。

no-profiler 对照，统一使用
`--variant copied_chunked --mode prepared_no_q_copy --warmup 2 --iters 5`：

| shape | schedule | avg ms | best ms | sync ms | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 32K, D=128 | state-ready | 0.592 | 0.580 | 0.566 | 对照 |
| 32K, D=128 | dummy x8 | 0.580 | 0.578 | 0.563 | x8 可隐藏 |
| 32K, D=128 | V prefetch x8 | 0.594 | 0.591 | 0.566 | 真实 V 预取没有增加 sync |
| 32K, D=128 | V after-state x8 | 0.591 | 0.586 | 0.564 | 保守 V 读也基本隐藏 |
| 16K, D=256 | state-ready | 0.371 | 0.362 | 0.320 | 对照 |
| 16K, D=256 | dummy x8 | 0.384 | 0.372 | 0.352 | x8 已经露出 |
| 16K, D=256 | V prefetch x8 | 0.387 | 0.382 | 0.350 | V 读不是主要新增瓶颈 |
| 16K, D=256 | V after-state x8 | 0.387 | 0.376 | 0.350 | 保守 V 读也没有额外拉开 |
| 32K, D=256 | state-ready | 1.113 | 1.100 | 1.083 | 对照 |
| 32K, D=256 | dummy x8 | 1.125 | 1.108 | 1.078 | sync 持平，avg 有 call 抖动 |
| 32K, D=256 | V prefetch x8 | 1.111 | 1.107 | 1.077 | 真实 V 预取仍可隐藏 |
| 32K, D=256 | V after-state x8 | 1.122 | 1.107 | 1.077 | state 后读 V 也没有 device-side 回退 |

代表性 device profile，单位 us：

| shape / schedule | producer `FAP_COMPUTE` | consumer total | V read zone | `PV` probe group | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 16K D256, V after-state x8 | 310.466 | 341.593 | 1.478 | 201.720 | consumer 总量超过 producer，16K 藏不住 |
| 32K D256, V prefetch x8 | 1071.939 | 671.047 | 1.330 | 471.902 | V 读和 x8 probe 都在 producer 窗口内 |
| 32K D256, V after-state x8 | 1068.147 | 670.349 | 1.371 | 471.027 | 即使 state 后读 V，也仍小于 producer 窗口 |

本轮结论：

- 真实 `V` 读本身不是当前 split 方向的主要风险；代表性 D256 shape 下只有约
  `1.3-1.5 us`。
- 16K D256 慢，是因为 x8 consumer 总 compute 已经超过 producer 窗口，不是因为 V
  搬运太慢。
- 32K D256 下，即使最保守地等 state-ready 后才读 V，consumer 总量仍约
  `670 us`，小于 producer `1068 us`，因此大 shape 上继续做真实 handoff 是合理的。
- 下一步应该把 dummy `P` 换成真实 `P`/softmax-state handoff。优先做
  “producer 导出一个 row-group 的 softmaxed `P` 到 scratch/CB，consumer 读真实 `P`
  + 真实 `V` 做 partial `P@V`”，先不做完整输出合并。

raw logs：

- `/wafer/gsh/tmp/fa_profile_real_v_probe_20260530.tsv`
- `/wafer/gsh/tmp/fa_profile_split_state_consumer_vprefetch_x8_v1_smoke_correctness.log`
- `/wafer/gsh/tmp/fa_profile_split_state_consumer_vafter_state_x8_v1_smoke_correctness.log`
- `/wafer/gsh/tmp/fa_profile_16k_D256_q128_k256_split_state_consumer_vafter_state_x8_v1_profile_device.csv`
- `/wafer/gsh/tmp/fa_profile_32k_D256_q128_k256_split_state_consumer_vprefetch_x8_v1_profile_device.csv`
- `/wafer/gsh/tmp/fa_profile_32k_D256_q128_k256_split_state_consumer_vafter_state_x8_v1_profile_device.csv`

## 下一步优化重点

1. 先保持 common full SDPA 为主线：优化 `copied_sdpa`，不要先跳到
   `sdpa_decode` 或 ring/joint 变体。
2. 降低 host/runtime 占比：如果 Q 来自上游 device op，优先做 device-resident Q；
   如果 Q 必须从 host 来，就把 runtime update 从完整 Q window 缩到更小的块。
3. 先把 `q=128,k=256,grid=8x8,pipeline_depth=2,pipeline=auto,
   compute_pipeline_schedule=default` 作为当前实验 fork 的 tuned baseline。
   `stream_h1`、`pipeline_depth=4`、`qktv_h1`、`salad_first`、
   `partial_handoff_v1` 和简单 `q=64` / `k=64` 都没有形成可推广收益。
4. 下一步先扩大 grid policy 的 shape sweep，确认不同 head/batch 没有明显回退；
   q256 和 q64 在本轮已经通过 guard 避免无意义覆盖。
5. grid policy 稳定后，再做 compute-side 优化；当前数据已经说明简单拆 writer/QKTV
   row group、把完整 `QKT@V` 提前塞进 QK phase、或者把 Phase 2 第一行改成
   drain-all 再 matmul 都不是正确方向。
6. QK subblock、Q buffer factor、DST sync 已经完成对照；继续保持默认 `2x4`
   和 half-sync，不在这个方向继续堆 buffer。
7. `--qk-softmax-profile` 已证明 softmax 子阶段与下一段 QK matmul 在同一 RISC 上没有 overlap；
   但 softmax 本身不是当前最大 critical event。
8. `--qk-softmax-schedule after_matmul` 与 `after_matmul_except_final_kt` 都已验证：
   简单 matmul-first 会降低 softmax event 但伤 max reduce；非 final kt delayed-softmax
   能修复 max reduce handoff，但恢复 softmax 串行气泡。`reduce-first delayed-softmax`
   已尝试但 correctness 失败并撤回；下一次若重启这个方向，先做最小 correctness
   repro，不再直接作为 performance schedule。
9. `QK_WAIT_Q` 对应 reader path 已细分：主要是 Q NOC read，不是 reserve/push/barrier。
   `first_before_k` 已证明等待可以消掉，但过早发 Q 会恶化整体 critical path。
   下一步只值得做更窄的 reader prefetch：K read 已发出之后、K forward/link
   写安全边界之前或之后的精确插入点，而不是把 Q 整体搬到 K 之前。
10. QK body 现在要按 first/rest 分开优化：首个 body 是 9-10.5 us 级
   pipeline fill，后续 body 平均只有约 1 us。下一步优先做首段 warmup /
   首段排布实验，但 `tiny_matmul` 和 `same_config_init` 都没有形成实际端到端收益；
   后续要么把 warmup 完全隐藏进 reader/K wait，要么转向更窄的 reader Q
   prefetch 插入点。
11. 再做 kernel 改动：只有当细粒度数据证明某段阻塞后，才替换那一段，
   不要按变体名字写分叉优化。
12. `QKT@V` body 已经拆完：寄存器等待、QKT 等待、显式 barrier 都不是主因。
   `writer_pipeline_detail` 进一步确认 writer 大等待几乎全部来自首个 row group
   输出迟到。`group0_early_push_v1` 已验证首个 wait 能降，但会把等待后移到
   next group 并暴露 SALAD critical path。下一轮不要继续做“更早整组 push”，
   而是做更细的双组交错 schedule：保持 split-drain 局部性，尝试把
   `exp_max_diff`、next-group `QKT@V`、group0 SALAD/norm/push 重新排序，
   目标是既降低 first-group wait，又不增加 next-group wait。
13. `fa3_pv_softmax_v1` 已验证“previous row 的整段 partial `PV` 提前插入
    QK phase”不是正确粒度。它让 `FAP_FA3_SOFTMAX_PV_OVERLAP_CANDIDATE`
    真正出现，但把 `TRISC-KERNEL` 从 137.314 us 拉到 150.807 us，
    `FAP_COMPUTE_QK_PHASE` 从 18.421 us 拉到 26.141 us。下一轮 FA3-style
    只能做更小的 handoff 单元，或者改成两个 compute kernel / 更低层执行流
    实验；不要再把完整 `PV` matmul 塞进 QK phase。
14. split owner-output 方向已经证明 ACK/backpressure 不是瓶颈：
    `split_pv_owner_output_no_ack_v1` 去掉 ACK 后，host avg 变好但 device critical
    path 不降。下一步应做 direct compute-to-consumer state-ready：producer compute
    在某个 row group softmax state ready 后直接唤醒 consumer，而不是等 writer
    转发；随后做 consumer-owned online state 版本，让 consumer 跨 K chunk 维护
    max/sum/partial output。先保持一个 row group 的 correctness-capable 路径，不再
    重启 all-groups ownership 或 mailbox ring。
