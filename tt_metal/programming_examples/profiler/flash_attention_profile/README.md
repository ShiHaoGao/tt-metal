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
qk_softmax_schedule=before_matmul
```

刷新状态：2026-05-29 23:43 尝试重跑同条件 host 对照时，设备 TLB window 被外部
`ttmetal_matmul_ttnn_mcast_1d_m512_k1024_n1024_tuned_standalone`
进程占用而失败；下表使用本 README 中最后一次完成的同条件数据。

相对官方 TTNN tuned baseline，主要看 device critical path：

| shape | metric | TTNN tuned baseline | copied fastest | speedup | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 2K full q128/k256 | device critical us | 134.956 | 134.727 | 1.002x | 基本持平，不能宣称明确超越 |
| 16K chunked q128/k256 | device critical us | 137.619 | 137.443 | 1.001x | 基本持平，差异低于噪声边界 |

同一 tuned 配置的 host no-copy 辅助指标：

| shape | metric | TTNN tuned baseline | copied fastest | speedup | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| 2K full q128/k256 | avg ms | 0.175 | 0.181 | 0.967x | copied avg 略慢 |
| 2K full q128/k256 | sync ms | 0.140 | 0.140 | 1.000x | device completion 基本相同 |
| 16K chunked q128/k256 | avg ms | 0.206 | 0.163 | 1.264x | avg 受 host call 差异影响较大 |
| 16K chunked q128/k256 | sync ms | 0.113 | 0.144 | 0.785x | 与 avg 方向相反，不作为 kernel 胜负依据 |

相对官方默认 baseline，`copied_balanced_q` 在 q128/k128 shape 上有明确 host 收益：

| shape | official default avg ms | copied policy avg ms | avg speedup | official default sync ms | copied policy sync ms | sync speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K full q128/k128 | 0.200 | 0.184 | 1.09x | 0.157 | 0.144 | 1.09x |
| 16K chunked q128/k128 | 0.209 | 0.173 | 1.21x | 0.162 | 0.148 | 1.09x |

解释：

- 对默认 baseline 的收益主要来自 grid scheduling policy，不是新数学 kernel。
  官方 TTNN 如果手动传同样 `grid=8x8` 也能获得类似收益。
- 对 tuned baseline 的 device critical path 目前只是 parity，2K/16K 都只有约
  0.1-0.2% 的名义优势，不能作为“已经比 TTNN tuned 快”的结论。
- 后续只有当 copied fastest 的 device critical path 稳定低于 TTNN tuned baseline
  2% 以上，才把本节结论改成明确性能胜出。

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

## 下一步优化重点

1. 先保持 common full SDPA 为主线：优化 `copied_sdpa`，不要先跳到
   `sdpa_decode` 或 ring/joint 变体。
2. 降低 host/runtime 占比：如果 Q 来自上游 device op，优先做 device-resident Q；
   如果 Q 必须从 host 来，就把 runtime update 从完整 Q window 缩到更小的块。
3. 先把 `q=128,k=256,grid=8x8,pipeline_depth=2,pipeline=auto` 作为当前实验 fork 的 tuned baseline。
   `stream_h1`、`pipeline_depth=4`、`qktv_h1`、`salad_first` 和简单 `q=64` / `k=64`
   都没有形成可推广收益。
4. 下一步先扩大 grid policy 的 shape sweep，确认不同 head/batch 没有明显回退；
   q256 和 q64 在本轮已经通过 guard 避免无意义覆盖。
5. grid policy 稳定后，再做 compute-side 优化；当前数据已经说明简单拆 writer/QKTV
   row group 不是正确方向。
6. QK subblock、Q buffer factor、DST sync 已经完成对照；继续保持默认 `2x4`
   和 half-sync，不在这个方向继续堆 buffer。
7. `--qk-softmax-profile` 已证明 softmax 子阶段与下一段 QK matmul 在同一 RISC 上没有 overlap；
   但 softmax 本身不是当前最大 critical event。
8. `--qk-softmax-schedule after_matmul` 与 `after_matmul_except_final_kt` 都已验证：
   简单 matmul-first 会降低 softmax event 但伤 max reduce；非 final kt delayed-softmax
   能修复 max reduce handoff，但恢复 softmax 串行气泡。下一轮只值得做
   `reduce-first delayed-softmax`，不要继续微调这两个已失败排布。
9. 如果 `reduce-first delayed-softmax` 仍不降 QK phase，再把 `QK_WAIT_Q` 对应 reader path 继续细分：区分 Q DRAM/NOC read、
   `cb_reserve_back`、`cb_push_back`、reader 侧 Q subblock 间隔，以及 compute
   侧 `cb_wait_front` 的对应关系。
10. 在 writer 中加 storeback/CB wait zone，判断 writer 接近 critical path
   是真实写回瓶颈，还是等待 compute 输出。
11. 再做 kernel 改动：只有当细粒度数据证明某段阻塞后，才替换那一段，
   不要按变体名字写分叉优化。
