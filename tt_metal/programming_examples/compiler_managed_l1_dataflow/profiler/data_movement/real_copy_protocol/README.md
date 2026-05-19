# Real Copy Protocol Profiler

`real_copy_protocol` 是 compiler-managed L1 dataflow ABI 研究里的 dataflow-only 直接 fork，用来回答：**纯 reader/writer copy 路径是否能从 CB FIFO 替换为 compiler-owned L1 queue state 中受益**。

## 模式

- `cb`：reader 和 writer 之间使用 TT-Metal circular-buffer FIFO。
- `static-runtime`：compiler-managed L1 ring + local semaphore protocol，地址通过 runtime args 传入。
- `static-compiletime`：同样的 static protocol，但地址和 shape 常量 baked into kernel defines。
- `static-streamreg-scratch`：payload movement、ring layout、tile partition、profiler zones 都和 `static-runtime` 一致，只把单个 producer/consumer queue counter 从 L1 semaphore storage 换到 idle stream overlay scratch registers。
- `static-streamreg-scratch-compiletime`：保留 `static-streamreg-scratch` 的 ready/consumed/start 同步语义，但把 ring 地址、page size、page count、start value 和迭代数 baked into kernel defines，用来隔离 runtime 参数和动态 layout 配置成本。

这个 benchmark 从 DRAM 读 tiled BF16 到 L1 staging area，再写回 DRAM；不使用 compute engine。因此它只能作为 dataflow-only evidence，不能作为 compute-path stream-register ABI 结论。

## 运行

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/data_movement/real_copy_protocol/run_real_copy_protocol_cases.py \
  --out-dir /tmp/real_copy_protocol_cases \
  --tiles 256 1024 4096 \
  --num-pages 2 \
  --repeats 3 \
  --modes cb static-runtime static-compiletime static-streamreg-scratch static-streamreg-scratch-compiletime
```

## 当前结论

- L1 semaphore 版本的 `static-runtime` 是负例，本轮 1x1 sweep 中比 CB 慢约 `109-127 cycles/local tile`，median `-124.38`。
- `static-compiletime` 能减少一部分损失，但仍然慢于 CB，median `-64.68 cycles/local tile`；这说明 runtime 地址/形状加载不是唯一成本。
- `static-streamreg-scratch` 保留 runtime args 后仍比 CB 慢约 `5-13 cycles/local tile`，median `-11.66`；这说明 CB baseline 本身已经使用 stream-register backed counter，scratch register 只移除了 L1 semaphore storage 的大头成本。
- `static-streamreg-scratch-compiletime` 保留 ready/consumed 同步，但移除 runtime 参数和动态 layout 配置后转为稳定正向，本轮 256/1024/4096/16384 tiles 上为 `+21.57/+20.17/+20.29/+20.41 cycles/local tile`，median `+20.35`，speedup 约 `1.037x-1.040x`。
- 结论是：correctness synchronization 不能省；但 dataflow-only copy 中确实有一部分 runtime/config 成本可以通过 compile-time queue/layout 规划去掉。
- 因为这里只有一个 logical queue，它不能推广为 compute-path stream-register 方案。reader/compute/writer pipeline 必须使用 `static-streamreg-cbregs`，并让每个 logical CB 独立使用 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`。

最新数据来自：

```text
/tmp/real_copy_protocol_compiletime_ablation_2026_05_19
```
