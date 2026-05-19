# Compiler-Managed L1 Dataflow 实验入口

这个目录是 TTNN static-protocol / compiler-managed L1 dataflow ABI 实验的统一入口。

当前结论不是“用户手动管理 TT-Metal `LocalCBInterface`”，而是：**CB/L1Queue 只作为 compiler-time scheduling abstraction 存在**。编译器侧表达显式 L1 allocation、view、queue/sync、compute operand 和 static schedule；lowering 后应直接生成地址计算、同步和 pack/unpack 所需代码。TT-Metal CB 可以作为兼容 backend 或 baseline，但不是 Level C generated code 中的 runtime object。

参照 CuTe 的 `Tensor = Engine + Layout` 和 MLIR `memref` 的分层方式，需要区分 compiler IR 和 lowered metadata。Level C 的 canonical tensor IR 是 `AxeTensor{name, element_type, storage, layout}`：`AxeStorage` 是 CuTe-style storage engine，可以表达 owned device allocation、owned mesh allocation、borrowed host storage、borrowed device address、view-of-storage 和 register fragment；`AxeLayout` 同时表达 domain 和 coordinate mapping，因此不再在 `AxeTensor` 里维护另一份 `logical_shape`。真实生成的 Level C kernel 不应该依赖 `ShapeDescriptor` 这个 C++ 类型；如果 shape/layout 完全静态，它们应该被折叠成常量地址表达式和 loop bounds。当前 `ShapeDescriptor{rank, dims[4]}` 只是 lowering descriptor-table bring-up 的 runtime-dynamic metadata carrier，用来承载 lowering 后仍需要运行时传递的信息。

TT-Metal 里已存在的 `TensorLayout`、`PageConfig`、`Tile`、`ShardSpec`、`NdShardSpec`、`MeshBuffer`、`TensorAccessor`、`BufferShardingArgs`、`Interleaved/Sharded` 这些 layout 相关语义，可以被统一抽象成 Axe 风格的 layout algebra：`D(shard)`、`R(replica)`、`O(offset)`，再加上 tile/face/block/atom 作为嵌套布局层。进一步地，`Buffer` / `MeshBuffer` / `Tensor` 中的 allocation、ownership、lifetime、device visibility 和 view/root 语义可以进入 `AxeStorage` 这层 storage engine。也就是说，**`AxeLayout` 单独不替代资源对象；`AxeTensor + AxeStorage + AxeLayout` 可以作为统一 compiler IR 替代这些对象的语义职责**。当前实验不直接删除 TTNN/TT-Metal public API 类型，而是先证明统一表达能力和 lowering 边界。

当前 `AxeLayout` 已按 Axe 论文 Figure 1 收敛：`AxeAxis{name, kind}` 是独立 compiler IR 对象，`AxeIter` 是显式 IR 类型，内容为 `(extent, stride, axis)`；layout 顶层由 `D(list<AxeIter>)`、`R(set<AxeIter>)` 和单个 `O(fixed coordinate offset)` 组成。因为这是 compiler IR，不是 device ABI descriptor，所以 `AxeLayout` 直接用三个 vector 表达：`d_iters`、`r_iters`、`o_entries`。其中 `D` 的 extent 是 tensor domain 的唯一事实源，`o_entries` 不是多个 offset，而是单个 coordinate `O` 的稀疏 axis-value entries；未出现的 axis 默认 offset 为 0。CuTe/CUTLASS 的 `Shape + Stride` 只作为理解单个 iter 的背景：它解释 extent/stride 的机械含义，但不是 Level C layout IR 的顶层结构。`tile_height`、`face_height`、`block_size_bytes`、`atom_size_bytes`、`buffer_layout` 这类 TT-Metal 特化或可推导字段不再进入统一 layout 类型。

因此，TT-Metal 里所有与 layout 相关的规则并不是“都可以原封不动保留”，而是“都应该被重构到同一套 layout algebra 上，再由不同 backend 去 lower”：row-major、tile/face、interleaved、sharded、ND sharding、mesh replication、TensorAccessor 地址表达，都应该从统一 layout 中导出，而不是分别维护一套互相打架的规则。

独立的 pipeline warmup 实验不在这里，路径是：

```text
tt_metal/programming_examples/pipeline_warmup_experiments
```

## 目录结构

- `docs/`：当前研究问题、阶段状态、关键结论和下一步实验顺序。
- `include/`：Level C 头文件分两层：`level_c_ir.hpp` 定义纯 compiler IR；`level_c_lowering_descriptors.hpp` 定义后续 lowering / ABI descriptor。这里的 `Descriptor` 指跨 host/device ABI 或 runtime-dynamic metadata carrier，不是所有 IR 类型都必须叫 descriptor。
- `level_c/`：Level C bring-up / sanity targets；当前包含 descriptor schema ABI sanity、Axe layout coverage sanity、TT-Metal layout adapter coverage sanity、lowering adapter sanity、tile-add codegen sanity、firmware skip-CB-init contract sanity 和 TT target dialect sanity。
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

单独构建并运行 Level C sanity：

```bash
conda run -n tt cmake --build build_Release \
  --target level_c_descriptor_schema_sanity \
           level_c_axe_layout_coverage_sanity \
           level_c_tt_metal_layout_adapter_sanity \
           level_c_lowering_adapter_sanity \
           level_c_tile_add_codegen_sanity \
           level_c_fw_skip_cb_init_contract_sanity \
           level_c_tt_target_dialect_sanity -j8

build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_descriptor_schema_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_axe_layout_coverage_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_tt_metal_layout_adapter_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_lowering_adapter_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_tile_add_codegen_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_fw_skip_cb_init_contract_sanity
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_tt_target_dialect_sanity
```

可执行文件输出在：

```text
build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/
```

## 推荐入口

先看主结论文档：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/docs/ttnn_compiler_managed_l1_dataflow_abi_2026_05_18.md
```

TTNN 算子横向总结、收益归因、firmware/descriptor 链路和 Level C / Level 3 实验计划已经合并到这个主文档中；`docs/` 下不再维护第二份结论文档。

TT 底层 dialect v0 设计入口：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/docs/tt_target_dialect_design_2026_05_19.md
```

该文档把 `TensorAccessor` / CB API 明确归为 legacy fact source，并把 `tt.target` 的 canonical bottom IR 固定为 NoC、stream register、L1 address、tile register、unpack/math/pack 等硬件动作。

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

当前 Level B compute-path smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rtadd_cbregs_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=static-streamreg-cbregs --tiles=4 --num-pages=2 --repeats=1 --device-id=0
```

当前 Level C first-proof smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rtadd_level_c_generated_static_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=level-c-generated-static --tiles=4 --num-pages=2 --repeats=1 --device-id=0
```

当前 Level C first-proof device-profiler 对比：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/level_c_tile_add_first_proof_2026_05_19 \
  --tiles 256 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb static-streamreg-cbregs level-c-generated-static
```

当前 Level C LLK direct-address smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rtadd_level_c_llk_direct_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=level-c-llk-direct --tiles=4 --num-pages=2 --repeats=1 --device-id=0

TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rmp_level_c_llk_direct_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=level-c-llk-direct --M=64 --N=64 --K=64 --num-pages=2 --repeats=1 --device-id=0
```

当前 Level C firmware / launch descriptor gate smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rtadd_level_c_fw_skip_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=level-c-llk-direct-fw-skip-cb-init --tiles=4 --num-pages=2 --repeats=1 --device-id=0

TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rmp_level_c_fw_skip_smoke \
  conda run -n tt \
  build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=level-c-llk-direct-fw-skip-cb-init --M=64 --N=64 --K=64 --num-pages=2 --repeats=1 --device-id=0
```

当前 Level C LLK direct-address device-profiler 对比：

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

当前 Level C firmware / launch descriptor gate device-profiler 对比：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/level_c_tile_add_fw_skip_cb_init_proof_2026_05_19_rerun \
  --tiles 256 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb static-streamreg-cbregs level-c-llk-direct level-c-llk-direct-fw-skip-cb-init

conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py \
  --out-dir /tmp/level_c_matmul_fw_skip_cb_init_proof_2026_05_19_rerun \
  --dims 64 \
  --Ks 64 \
  --num-pages 2 \
  --repeats 1 \
  --modes profiled-cb static-input-output-cbregs level-c-llk-direct level-c-llk-direct-fw-skip-cb-init
```

## 当前结论

- stream-register 方向已经锁定为 per-CB ABI：每个 logical CB 使用自己的 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`。
- dataflow-only copy 可以用 `static-streamreg-scratch` 做 ablation，但不能把它推广成 compute-path ABI。
- Level B 的正式 compute-path mode 是 `static-streamreg-cbregs`：per-CB stream-register counter，per-core L1 地址和 work partition 通过 runtime args 传入。
- `static-streamreg-cbregs-compiletime` 只作为 compile-time config ablation / upper bound，不作为 Level B 多核工程路径。
- 大收益来自 memory-bound / simple elementwise 路径上的 static ring/schedule 替代 CB FIFO 动态管理。
- `static-streamreg-cbregs` 验证 ABI 边界；在当前 tile add 上相对 `static-runtime` 几乎没有新的 steady-state 收益，说明 runtime args/config 不是主要瓶颈。
- 真实 TTNN fork 应优先复制原始 C++ program factory 和 device kernels，再做最小协议替换；`ttnn_paged_update_cache_protocol` 已按这个路线验证。
- `ttnn_binary_ng_no_bcast` 复跑仍为正，`static-runtime` 约 `19.8-23.4 cycles/local-tile`，median speedup 约 `1.035x`。
- `ttnn_bcast_to_protocol` row-broadcast 已通过 device profiler；critical stage 是 writer。1024/4096/16384 tiles 上，`static-runtime` 约 `+5.03/+1.04/+0.27 cycles/tile`，`static-streamreg-cbregs` 约 `+11.44/+10.85/+2.74 cycles/tile`。这是 broadcast direct fork 的第一条证据，但还不能推广到所有 broadcast/SFPU-heavy 算子。
- `paged_update_cache` 的 8-user decode-like 真实形状上，static protocol 有稳定 device critical-path 收益，本轮复跑约 `313-535 cycles`、median speedup 约 `1.032x`；32-user static path 当前仍是 fork 的 scalability 待修项。
- matmul 仍然 mixed，不能 broad promote。
- 下一步已经落到真实 TTNN binary broadcast/SFPU-heavy、RMSNorm、Softmax decode、Paged KV read / embedding lookup、layout movement fork；baseline 只用于选择 direct fork，不用于直接宣称 static protocol speedup。
- Layout 方向上，Axe-like 统一建模可覆盖 TT-Metal 当前大部分 layout/address mapping 语义；资源方向上，`AxeStorage` 把 `Buffer` / `MeshBuffer` / `Tensor` 的 owned allocation、borrowed storage、view/root、device/mesh scope 和 register fragment 语义放进同一套 storage engine。
- Level C 已新增 `AxeTensor` / `AxeStorage` / `AxeLayout`，并用 `level_c_axe_layout_coverage_sanity` 与 `level_c_tt_metal_layout_adapter_sanity` 做两层证明。前者证明 Axe layout algebra 自身覆盖 row-major、tile/face、interleaved bank placement、height/width/block/ND sharding、mesh replicated/sharded、offset view 和 TensorAccessor-style address expression；后者对照 TT-Metal tech report / 源码类型，证明 `TensorLayout`、`PageConfig`、`Tile`、`MemoryConfig`、`ShardSpec`、`ShardSpecBuffer`、`NdShardSpec`、`MeshBuffer` 的 layout/storage/address mapping 语义，以及 `Buffer`/`MeshBuffer`/`Tensor` 风格资源语义，可以落到 `AxeTensor + AxeStorage + AxeLayout`。
- Level C 还新增 `level_c_lowering_adapter_sanity`，证明现有 CB payload、queue sync、operand view、TensorAccessor DRAM 地址和 view offset 可以先进入 `AxeTensor`，再 lowering 成 `TensorLayoutDescriptor` / `L1QueueDescriptor` / `QueueSyncDescriptor` / `OperandViewDescriptor` 和等价地址表达式。
- `real_tile_add_protocol --mode=level-c-generated-static` 已作为真实 device first-proof hook 通过 smoke，`max_abs_error=0`。device profiler first proof 中 `tiles=256,num_pages=2` 的 critical stage 仍为 writer，CB critical `283809 cycles`，Level C hook critical `150432 cycles`，节省 `521.00 cycles/tile`，speedup `1.887x`。边界必须写清楚：该 hook 仍复用 TT-Metal CB descriptor 与当前 LLK CB operand metadata，只证明 compiler-lowered static schedule 和真实 device path 已打通，不证明完整 CB-less / firmware-less Level C。
- `level-c-llk-direct` 已扩展到两个 proof path：tile-add 和 real_matmul_protocol。tile-add 证明简单 elementwise 的 unpack/math/pack 可以绕过 `add_tiles()` / `pack_tile()` wrapper 直接喂 raw L1 地址；matmul 参考 `programming_examples/matmul` 的真实 large-block kernel 结构，在 `real_matmul_protocol` 中新增 `K=64,num_blocks=1` 的 raw-address `_llk_unpack_AB_matmul_` / `_llk_math_matmul_` / `_llk_pack_` 路径。它们仍复用 host `CircularBufferConfig` 和现有 launch/metadata 初始化，只回答“LLK 能否不依赖 CB wrapper 接收 compiler-lowered L1 地址”，不回答 firmware/launch descriptor 是否已替换。当前 device-profiler proof：tile-add `tiles=256,num_pages=2` 中 `level-c-llk-direct` writer critical `149072 cycles`，相对 CB writer `283532 cycles` 节省 `525.23 cycles/tile`；matmul `M=N=K=64,num_pages=2` 中 `level-c-llk-direct` writer critical `2472 cycles`，相对 CB writer `2703 cycles` 节省 `231 cycles`，与 Level B `static-input-output-cbregs` 的 `2461 cycles` 基本持平。
- `level-c-llk-direct-fw-skip-cb-init` 是本轮新增的 Level C firmware / launch descriptor gate。这个模式不在 host path 调用 `CreateCircularBuffer` / `CircularBufferConfig`，因此 launch descriptor 中 `local_cb_mask=0`、remote CB range 为空；firmware 在 mask/range 为空时不进入 `setup_local_cb_read_write_interfaces()` / `setup_remote_cb_interfaces()`，device profiler 中也不出现 `CBP_FW_LOCAL_CB_INIT` / `CBP_FW_REMOTE_CB_INIT`。kernel 所需 L1 base、page size、format/tile words、tile shape 和 sync binding 由 compile-time descriptor / runtime args / 显式 stream-register binding 提供，不再把 `CBInterface[]` 当作 operand/queue metadata 的事实来源。本轮没有新增全局 `DISPATCH_MODE_COMPILER_L1` enum，而是先用实验 executable 的 descriptor-empty/sentinel gate 做最小可验证 proof；后续产品化再把它提升为显式 launch mode。
- 本轮 fw-skip proof 结果：tile-add `tiles=256,num_pages=2` 中 CB writer `284846 cycles`，Level B `static-streamreg-cbregs` writer `159337 cycles`，旧 `level-c-llk-direct` writer `149240 cycles`，`level-c-llk-direct-fw-skip-cb-init` writer `160302 cycles`；matmul `M=N=K=64,num_pages=2` 中 CB writer `2660 cycles`，Level B writer `2559 cycles`，旧 LLK-direct writer `2555 cycles`，fw-skip writer `2575 cycles`。结论是 firmware/launch ownership gate 已跨过，但 steady-state critical stage 没有比 Level B / raw-LLK direct 增加有意义收益。这里还要注意：fw-skip 当前把 L1 base/page/sync metadata 作为 runtime args 运输，而旧 `level-c-llk-direct` 仍是 compile-time protocol args upper-bound；两者差异不能解释为跳过 firmware CB init 的性能回退。CB init 属于 launch 前后一次性 firmware path，不在 steady-state writer critical loop 上。
- “一网打尽”的新结论边界是：`AxeLayout` 一网打尽 layout mapping；`AxeStorage` 一网打尽 storage/resource engine 语义；`AxeTensor` 统一承载 element type、storage engine 和 layout。当前只是 proof 和实验 IR，不代表已经迁移或删除 TTNN public API。
