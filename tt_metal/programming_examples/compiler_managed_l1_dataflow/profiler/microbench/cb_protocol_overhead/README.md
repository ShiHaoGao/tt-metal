# CB Protocol Overhead Profiler 示例

这个 microbenchmark 用来比较 TT-Metal circular-buffer API 和简单 static L1 scratchpad / explicit-ring protocol 的开销。默认 payload 较小，目的是突出 metadata、pointer、counter 和 synchronization protocol cost，而不是 DRAM / NoC bandwidth。

在当前研究计划中，它承担两个角色：

- Phase 0：CB/runtime/init overhead baseline。
- Phase 1：通过 `--mode=dram` 验证 raw L1 / NoC / semaphore data-movement baseline。

## 构建

```bash
./build_metal.sh --build-programming-examples
```

可执行文件：

```bash
build/programming_examples/compiler_managed_l1_dataflow/profiler/cb_protocol_overhead
```

## 运行

基础模式：

```bash
TT_METAL_DEVICE_PROFILER=1 \
  ./build/programming_examples/compiler_managed_l1_dataflow/profiler/cb_protocol_overhead \
  --mode=all --iterations=10000 --page-size=64 --num-pages=8 --repeats=5
```

system-level CB replacement：

```bash
TT_METAL_DEVICE_PROFILER=1 \
  ./build/programming_examples/compiler_managed_l1_dataflow/profiler/cb_protocol_overhead \
  --mode=system --iterations=10000 --page-size=64 --num-pages=8 \
  --num-cbs=8 --num-rv=2 --repeats=5
```

DRAM traffic：

```bash
TT_METAL_DEVICE_PROFILER=1 \
  ./build/programming_examples/compiler_managed_l1_dataflow/profiler/cb_protocol_overhead \
  --mode=dram --iterations=10000 --page-size=2048 --num-pages=8 \
  --num-cbs=1 --num-rv=2 --repeats=5
```

device profiler 输出：

```text
generated/profiler/.logs/profile_log_device.csv
```

不要同时打开 `TT_METAL_DPRINT_CORES`、`TT_METAL_WATCHER` 和 `TT_METAL_DEVICE_PROFILER`；它们会竞争 device L1 storage。

## 模式

- `empty`：loop baseline，用于扣除 loop overhead。
- `cb-get-write-ptr` / `cb-get-read-write-ptr` / `cb-get-tile-size`：本地 CB metadata API 开销。
- `cb-api-roundtrip`：同一个 RISC-V 上 reserve/write/push/wait/read/pop 一页。
- `static-ring`：显式 L1 buffer 上的手写 ring pointer。
- `static-counter`：手写 ring + volatile produced/consumed counters。
- `cross-empty`：producer / consumer RISC-V 同时跑 empty loop。
- `cross-cb`：RISCV_0 通过正常 CB 生产，RISCV_1 消费。
- `cross-static`：RISCV_0 / RISCV_1 使用显式 L1 payload ring + 两个 L1 semaphores。
- `cb-system`：创建 `--num-cbs` 个 CB，用来暴露 firmware/dispatch setup cost。
- `static-runtime-addr`：显式 L1 ring + semaphores，地址和 size 通过 runtime args。
- `static-compiletime-addr`：同样 protocol，但地址和 size baked into kernel defines。
- `static-no-sync`：单 RISC 显式 L1 ring lower bound，不是合法 cross-RISC sync baseline。
- `dram-cb`：RISCV_0 从 DRAM 读入 CB，RISCV_1 从 CB 写回 DRAM。
- `dram-static-runtime-addr` / `dram-static-compiletime-addr`：同样 DRAM traffic，但走显式 L1 ring + semaphores。
- `dram-single-nosync`：单 RISC 串行读写 DRAM reference path，不是 two-RISC pipeline lower bound。

通常看这个数字：

```text
(mode_cycles - empty_cycles) / iterations
```

cross-RISC 模式要比较 producer / consumer 的 pair max，并减去 `cross-empty` pair max。

CSV 汇总：

```bash
python3 tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/microbench/cb_protocol_overhead/analyze_profile_csv.py \
  generated/profiler/.logs/profile_log_device.csv --iterations=10000
```

## 当前结果

在 Blackhole 系统上，`--mode=all --iterations=10000 --page-size=64 --num-pages=8 --repeats=5` 的本地 metadata API 开销较小：

```text
cb-get-tile-size       net ~= 1.00 cycles/iteration
cb-get-write-ptr       net ~= 3.00 cycles/iteration
cb-get-read-write-ptr  net ~= 4.00 cycles/iteration
```

同 RISC 的 full CB roundtrip 更大：

```text
cb-api-roundtrip       net ~= 77.50 cycles/iteration
```

更相关的是 cross-RISC producer/consumer：

```text
cross-empty pair max      ~=  4.01 cycles/iteration
cross-cb pair max         ~= 43.76 cycles/iteration, net ~= 39.75
cross-static pair max     ~= 35.30 cycles/iteration, net ~= 31.29
cross-cb - cross-static   ~=  8.46 cycles/iteration
```

解释：

- 小页 cross-RISC microbenchmark 中，CB abstraction 相对 explicit L1 ring + TT-Metal L1 semaphore baseline 多约 `8.5 cycles/page`。
- 大部分 cross-RISC 总成本仍来自同步和 ordering，而不是 metadata lookup。
- 这不是完整 TTNN op speedup 结论；它只说明在保留 launch、L1 allocation、可靠 semaphore sync 后，剩余 CB protocol overhead 在该配置下是 single-digit cycles/page。

system-level 结果：

```text
system-cb pair max                  ~= 43.7688 cycles/iteration, net ~= 39.7646
system-static-runtime pair max      ~= 35.3049 cycles/iteration, net ~= 31.3007
system-static-compiletime pair max  ~= 34.2411 cycles/iteration, net ~= 30.2369

cb - static-runtime                 ~=  8.4639 cycles/iteration
static-runtime - static-compiletime ~=  1.0638 cycles/iteration
cb - static-compiletime             ~=  9.5277 cycles/iteration
```

DRAM traffic 结果：

```text
page_size  dram-cb   static-runtime  static-compiletime  cb-runtime delta  cb-compiletime delta
64         494.47    476.12          468.37              18.36             26.11
1024       526.94    511.20          506.81              15.74             20.13
2048       546.54    532.05          520.45              14.49             26.09
```

解释：

- DRAM traffic 下 CB-vs-static delta 仍可见，但 payload 越大，百分比越小。
- 2048-byte page 上 CB-to-compiletime-static delta 约为 `26.09 / 546.54 = 4.8%`。
- 该 benchmark 仍不是 end-to-end model speedup：它只有一个 producer core、一个 consumer core，没有 compute kernel 消费 tile，TT-Metal launch/buffer allocation 仍在路径上。
