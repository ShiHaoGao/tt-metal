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

`*-cbregs` 模式与现有 static 模式一一对应，用来区分 L1 semaphore cost 和 matmul reuse/compute 行为。

## 当前范围

- `B=1`
- static modes 使用 `num-pages=2`
- 当前主要关注 low-K / exposed shapes，不把 large prefill GEMM 当 proof point。

## 构建和运行

```bash
cmake --build build_Release --target real_matmul_protocol -j8
```

```bash
TT_METAL_CACHE=/tmp/real_matmul_protocol_smoke \
  ./build_Release/programming_examples/compiler_managed_l1_dataflow/profiler/real_matmul_protocol \
  --mode=all --M=512 --N=512 --K=64 --repeats=1 --num-pages=2
```

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/run_real_matmul_protocol_cases.py \
  --out-dir /tmp/real_matmul_protocol_cases
```

## 当前结论

- matmul 结果仍然 mixed：有些 output-static shape 有收益，input-static 可能回退。
- delta 相对总 critical-stage cost 很小，而且随 shape 和模式变号。
- 当前证据不能 broad promote matmul；下一步只应追 low-K、GEMV-like、multicast、decode-like exposed shapes。
