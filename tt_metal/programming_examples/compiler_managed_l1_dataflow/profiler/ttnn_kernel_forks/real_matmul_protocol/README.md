# real_matmul_protocol

这个 benchmark 测量真实 `matmul_multicore_reuse` 路径，在 stock CB、L1 semaphore static、per-CB stream-register static 之间做对比。

## 模式

- `profiled-cb`：stock CB FIFO 管理。
- `static-input-only`：只替换 input CB `c_0` / `c_1` 的 static L1 ring protocol。
- `static-output-only`：只替换 output CB `c_16` 的 static L1 ring protocol。
- `static-input-output`：同时替换 `c_0`、`c_1`、`c_16`。
- `static-input-only-cbregs`：input-only 的 per-CB stream-register counter 版本。
- `static-output-only-cbregs`：output-only 的 per-CB stream-register counter 版本。
- `static-input-output-cbregs`：input/output 的 per-CB stream-register counter 版本。
- `static-input-only-cbregs-compiletime` / `static-output-only-cbregs-compiletime` / `static-input-output-cbregs-compiletime`：matmul 专属 compile-time ablation，在对应 input/output 替换范围内把 protocol args 静态化。
- `level-c-llk-direct`：Level C matmul LLK direct-address proof。它复用 static input/output L1 ring 和 per-CB stream-register counter，但 compute kernel 不再通过 `matmul_tiles()` / `pack_tile()` 的 CB wrapper 推导 operand 地址，而是手工展开 `_llk_unpack_AB_matmul_`、`_llk_math_matmul_`、`_llk_pack_`。当前仍保留 host `CircularBufferConfig` 做 launch / metadata bring-up，不是 firmware-less / launch-descriptor 替换。
- `level-c-llk-direct-fw-skip-cb-init`：Level C firmware / launch descriptor gate proof。host 不注册 input/output CB，launch descriptor 的 local CB mask 为空，firmware 不进入 local/remote CB init；reader/writer/compute 从 compile-time descriptor / runtime args / 显式 stream-register binding 获取 L1 ring、tile/page 和同步元数据。

`*-cbregs` 模式与现有 static 模式一一对应，是 matmul 的 Level B 标准对照；`*-cbregs-compiletime` 只作为 single-core low-K/GEMV-like 上界消融。

## 当前范围

- `B=1`
- static modes 使用 `num-pages=2`
- 当前主要关注 low-K / exposed shapes，不把 large prefill GEMM 当 proof point。
- `level-c-llk-direct` 第一版只支持 `K=64` / `num_blocks=1` / single-core active shape。这样先隔离主 matmul unpack/math/pack direct-address 路径；`K>64` 的 partial-sum spill/reload 需要单独展开 `copy_tile` 等价 LLK 路径后再纳入。

## 构建和运行

```bash
cmake --build build_Release --target real_matmul_protocol -j8
```

```bash
TT_METAL_CACHE=/tmp/real_matmul_protocol_smoke \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=static-input-output-cbregs --M=64 --N=64 --K=64 --repeats=1 --num-pages=2
```

Level C matmul LLK direct-address smoke：

```bash
TT_METAL_DEVICE_PROFILER=0 TT_METAL_CACHE=/tmp/rmp_level_c_llk_direct_smoke \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=level-c-llk-direct --M=64 --N=64 --K=64 --repeats=1 --num-pages=2
```

Level C matmul device-profiler 对比：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py \
  --out-dir /tmp/level_c_matmul_llk_direct_proof_2026_05_19 \
  --dims 64 \
  --Ks 64 \
  --num-pages 2 \
  --repeats 1 \
  --modes profiled-cb static-input-output-cbregs level-c-llk-direct
```

Level C matmul firmware / launch descriptor gate 对比：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py \
  --out-dir /tmp/level_c_matmul_fw_skip_cb_init_proof_2026_05_19_rerun \
  --dims 64 \
  --Ks 64 \
  --num-pages 2 \
  --repeats 1 \
  --modes profiled-cb static-input-output-cbregs level-c-llk-direct level-c-llk-direct-fw-skip-cb-init
```

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py \
  --out-dir /tmp/real_matmul_protocol_cases
```

## 当前结论

- matmul 结果仍然 mixed：有些 output-static shape 有收益，input-static 可能回退。
- delta 相对总 critical-stage cost 很小，而且随 shape 和模式变号。
- 当前证据不能 broad promote matmul；下一步只应追 low-K、GEMV-like、multicast、decode-like exposed shapes。
- `level-c-llk-direct` 的实验目标不是宣称 matmul 变快，而是证明比 add 更复杂的真实 TTNN-style matmul compute path 也可以把 unpack/math/pack 拆到 raw-address LLK 入口。当前 smoke 已通过：`M=N=K=64` 时 `pcc=0.999049`，`max_abs_error=0.015625`。device-profiler proof 中 CB writer critical `2703 cycles`，`static-input-output-cbregs` writer `2461 cycles`，`level-c-llk-direct` writer `2472 cycles`；因此这个 mode 证明 wrapper dependency 可拆，但没有比 Level B cbregs 产生额外收益。
- `level-c-llk-direct-fw-skip-cb-init` 已通过 `M=N=K=64,num_pages=2` smoke，`pcc=0.999049`，`max_abs_error=0.015625`。device-profiler proof 中 CB writer `2660 cycles`，Level B `static-input-output-cbregs` writer `2559 cycles`，旧 `level-c-llk-direct` writer `2555 cycles`，fw-skip writer `2575 cycles`。CB / Level B / 旧 LLK-direct 的 CSV 都有 `CBP_FW_LOCAL_CB_INIT`，fw-skip CSV 没有 `CBP_FW_LOCAL_CB_INIT` / `CBP_FW_REMOTE_CB_INIT`。结论是 matmul 也跨过了 firmware/launch ownership gate，但这个 gate 不是 matmul steady-state 性能瓶颈。
