# Real Tile Add Protocol Profiler

`real_tile_add_protocol` 比较普通 CB-based add pipeline 和 static L1 ring / semaphore protocol。它是当前最强的正例，用来验证 memory-bound/simple elementwise 路径是否能从 static schedule 中受益。

## 模式

- `cb`：标准 `cb_wait_front` / `cb_reserve_back` / `cb_pop_front` / `cb_push_back`。
- `static-runtime`：显式 L1 ring 和 semaphore，通过 runtime args 传入。
- `static-compiletime`：同样的 static protocol，但地址和 semaphore id baked into kernel defines。
- `static-streamreg-cbregs`：payload movement、L1 ring、CB descriptor、pack/unpack setup、per-tile schedule 与 `static-runtime` 对齐；per-CB queue state 使用 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`，launch rendezvous 使用 idle stream register。
- `static-streamreg-cbregs-compiletime`：compile-time config ablation / upper bound，在 `static-streamreg-cbregs` 上继续把 ring/layout/config baked into kernel defines。
- `level-c-generated-static`：Level C first-proof hook。它把 ring/layout/work partition 视作 compiler-lowered static schedule，使用 compile-time constants 和独立 profiler zone 验证真实 device path；当前仍复用 TT-Metal CB descriptor 与 LLK CB-derived operand metadata，因此不是完整 CB-less / firmware-less 实现。
- `level-c-llk-direct`：Level C LLK direct-address proof。它复用同一套 static L1 ring / sync，但 compute kernel 不再通过 `add_tiles()` / `pack_tile()` 的 CB wrapper 推导 operand 地址；unpack、math、pack 三段分别手工展开到 raw-address LLK 内部入口。它仍保留 host `CircularBufferConfig` 用于现有 kernel launch / metadata 初始化，不是 firmware/launch descriptor 改造。
- `level-c-llk-direct-fw-skip-cb-init`：Level C firmware / launch descriptor gate proof。host 不调用 `CreateCircularBuffer` / `CircularBufferConfig`；launch descriptor 不设置 local CB mask；firmware 因 `local_cb_mask=0` 且 remote CB range 为空而跳过 local/remote CB init；kernel 从 compile-time descriptor / runtime args / 显式 stream-register binding 获得 L1 base、page size、format/tile words、tile shape 和 sync binding。

旧 `static-streamreg` compute 模式已禁用。它把一个 idle stream 当成多个 logical CB 的共享 scratch register，不是有效 compute-path 对比。

## 构建和运行

```bash
cmake --build build_Release --target real_tile_add_protocol -j8
```

```bash
TT_METAL_DEVICE_PROFILER=1 \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=all --tiles=1024 --num-pages=2 --repeats=5
```

稳定 per-case sweep：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/real_tile_add_protocol_cases \
  --tiles 256 1024 4096 \
  --num-pages 2 4 \
  --repeats 3
```

2x2 multi-core sweep：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/real_tile_add_protocol_multicore \
  --tiles 1024 4096 \
  --num-pages 2 \
  --repeats 3 \
  --modes cb static-runtime static-streamreg-cbregs \
  --core-grid-x 2 \
  --core-grid-y 2
```

Level B smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/real_tile_add_level_b_smoke \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=static-streamreg-cbregs --tiles=4 --num-pages=2 --repeats=1
```

Level C first-proof 对比：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/level_c_tile_add_first_proof_2026_05_19 \
  --tiles 256 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb static-streamreg-cbregs level-c-generated-static
```

LLK direct-address smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rtadd_level_c_llk_direct_smoke \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_tile_add_protocol \
  --mode=level-c-llk-direct --tiles=4 --num-pages=2 --repeats=1
```

LLK direct-address profiler proof：

```bash
conda run -n tt python \
  tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/level_c_llk_direct_proof_2026_05_19 \
  --tiles 256 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb level-c-generated-static level-c-llk-direct
```

Firmware / launch descriptor gate profiler proof：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/run_real_tile_add_protocol_cases.py \
  --out-dir /tmp/level_c_tile_add_fw_skip_cb_init_proof_2026_05_19_rerun \
  --tiles 256 \
  --num-pages 2 \
  --repeats 1 \
  --modes cb static-streamreg-cbregs level-c-llk-direct level-c-llk-direct-fw-skip-cb-init
```

## 当前结论

- `static-runtime` 相对 CB 有稳定大收益，约 `435-525 cycles/tile`，device critical path speedup 约 `1.66x-1.89x`。
- `static-streamreg-cbregs` 保留这部分收益，但相对 `static-runtime` 基本是 near parity；差异通常只有几个 cycles/tile。
- `static-streamreg-cbregs-compiletime` 用来隔离 per-CB counter backend 后剩余 runtime/config 成本；当前只作为 single-core upper-bound ablation。
- 因此 cbregs 的意义主要是验证 per-CB ABI 边界和 launch/control placement，不是新的 steady-state 大优化来源。
- `level-c-generated-static` 当前 first-proof 结果为正：`tiles=256,num_pages=2` 时 writer 仍是 critical stage，CB critical 为 `283809 cycles`，Level C hook critical 为 `150432 cycles`，节省 `521.00 cycles/tile`，speedup `1.887x`。这个数字只证明 compiler-lowered static schedule hook 可在真实 device path 上正确运行并保持正向，不证明已移除 firmware `CBInterface` 或 LLK CB operand dependency。
- `level-c-llk-direct` 是当前 Level C direct-address proof hook：目标是先证明 LLK 的 unpack/math/pack 可以被 compiler-lowered static schedule 直接喂 raw L1 地址，而不是先改 firmware。raw LLK 地址必须按 Tensix 约定使用 `(l1_addr >> 4) - 1`，tile size 使用 16B word count。当前 smoke 已通过，`max_abs_error=0`；device-profiler proof 中 `tiles=256,num_pages=2` 的 writer critical 为 `149072 cycles`，相对 CB writer `283532 cycles` 节省 `525.23 cycles/tile`，与 `level-c-generated-static` 的 `149239 cycles` 基本持平。只有这个路径继续稳定后，才进入 firmware / launch descriptor gate。
- `level-c-llk-direct-fw-skip-cb-init` 已进入 firmware / launch descriptor gate：`tiles=4` smoke 通过，`max_abs_error=0`；`tiles=256,num_pages=2` device-profiler proof 中 CB writer critical `284846 cycles`，Level B `static-streamreg-cbregs` writer `159337 cycles`，旧 `level-c-llk-direct` writer `149240 cycles`，fw-skip writer `160302 cycles`。对应 profiler CSV 中 `cb`、`static-streamreg-cbregs`、旧 `level-c-llk-direct` 都出现 `CBP_FW_LOCAL_CB_INIT`，而 fw-skip case 不出现 `CBP_FW_LOCAL_CB_INIT` / `CBP_FW_REMOTE_CB_INIT`。这证明 `CBInterface[]` 不再是该 experimental path 的 operand/queue metadata 事实来源；同时也说明跳过 firmware CB init 不会给 steady-state writer loop 带来额外显著收益。当前 fw-skip 用 runtime args 运输 L1 base/page/sync metadata，而旧 `level-c-llk-direct` 仍是 compile-time protocol args upper-bound，所以二者 cycle 差异不能归因成 firmware skip 的回退。

Profiler CSV：

```text
generated/profiler/.logs/profile_log_device.csv
```
