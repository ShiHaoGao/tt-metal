# Real Copy Protocol Profiler

`real_copy_protocol` 是 compiler-managed L1 dataflow ABI 研究里的 dataflow-only 直接 fork，用来回答：**纯 reader/writer copy 路径是否能从 CB FIFO 替换为 compiler-owned L1 queue state 中受益**。

## 模式

- `cb`：reader 和 writer 之间使用 TT-Metal circular-buffer FIFO。
- `static-runtime`：compiler-managed L1 ring + local semaphore protocol，地址通过 runtime args 传入。
- `static-compiletime`：同样的 static protocol，但地址和 shape 常量 baked into kernel defines。
- `static-streamreg-scratch`：payload movement、ring layout、tile partition、profiler zones 都和 `static-runtime` 一致，只把单个 producer/consumer queue counter 从 L1 semaphore storage 换到 idle stream overlay scratch registers。

这个 benchmark 从 DRAM 读 tiled BF16 到 L1 staging area，再写回 DRAM；不使用 compute engine。因此它只能作为 dataflow-only evidence，不能作为 compute-path stream-register ABI 结论。

## 运行

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/data_movement/real_copy_protocol/run_real_copy_protocol_cases.py \
  --out-dir /tmp/real_copy_protocol_cases \
  --tiles 256 1024 4096 \
  --num-pages 2 \
  --repeats 3 \
  --modes cb static-runtime static-compiletime static-streamreg-scratch
```

## 当前结论

- L1 semaphore 版本的 `static-runtime` 是负例，通常比 CB 慢约 `112-122 cycles/local tile`。
- `static-compiletime` 能减少一部分损失，但仍然慢于 CB。
- `static-streamreg-scratch` 几乎追平 CB，说明 copy 负例的主要问题不是 payload 搬运，而是同步状态存储/协议成本。
- 因为这里只有一个 logical queue，它不能推广为 compute-path stream-register 方案。reader/compute/writer pipeline 必须使用 `static-streamreg-cbregs`，并让每个 logical CB 独立使用 `get_cb_tiles_received_ptr(cbid)` / `get_cb_tiles_acked_ptr(cbid)`。
