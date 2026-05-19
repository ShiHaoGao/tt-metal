# TT target dialect v0 设计

本文档定义 Level C 后续要使用的 TT 底层 dialect 方向。结论是：可以基于当前从 tile-add 与 real matmul proof 中反推出的新底层 IR 来封装 dialect，但必须把它定义成 **硬件动作 dialect**，而不是把 TT-Metal 现有 `TensorAccessor` / CB API 换个名字搬进去。

## 分层边界

Level C 编译链路应分三层：

1. `AxeTensor / AxeLayout / AxeStorage`
   - 负责 tensor、layout、storage engine 语义。
   - 替代 `TensorAccessor`、`TensorLayout`、`Buffer` / `MeshBuffer` / `Tensor` 在 compiler IR 中作为事实来源的职责。

2. schedule / lowering IR
   - 负责 per-core work partition、L1 lifetime、ring slot 复用、stage 顺序、producer/consumer 依赖。
   - `L1Queue` 只能在这层作为 compiler-time scheduling abstraction 出现，不能变成 device runtime object。

3. `tt.target` dialect
   - 负责真实硬件动作：NoC、stream register、L1 地址、tile register、unpack/math/pack。
   - 第一阶段 lowering 到现有 `noc_async_*`、`NOC_STREAM_*`、`_llk_*` 和少量 Tensix macro；后续再把选定 op 继续 lowering 到 raw Tensix / MOP。

因此，`TensorAccessor`、`CircularBuffer`、`CircularBufferConfig`、`CreateCircularBuffer`、`cb_*`、`get_local_cb_interface`、`CBInterface[]` 都不是 `tt.target` op/type。它们只允许作为 legacy adapter 的输入或 baseline 对照出现。

## 当前源码证据

当前两个 Level C proof 已经覆盖 `tt.target` v0 需要表达的核心动作。

### tile-add

代表源码：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/kernels/dataflow/reader_binary_tiles.cpp
tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/kernels/dataflow/writer_tiles.cpp
tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/kernels/compute/add_tiles.cpp
```

需要表达的动作：

- reader：global tile id 到 DRAM page address，`noc_async_read` 到 L1 ring slot，read barrier，publish input ready。
- compute：等待 input ready，`tile_regs_acquire`，raw-address `_llk_unpack_AB_`，`_llk_math_eltwise_binary_`，等待 output slot 可复用，`_llk_pack_` 到 L1 ring slot，publish input consumed / output ready。
- writer：等待 output ready，`noc_async_write` L1 ring slot 到 DRAM，write barrier，publish output consumed。

### real matmul

代表源码：

```text
tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/kernels/dataflow/reader_bmm_tile_layout_protocol.cpp
tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/kernels/dataflow/writer_bmm_tile_layout_protocol.cpp
tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/real_matmul_protocol/kernels/compute/bmm_large_block_zm_protocol.cpp
```

需要表达的动作：

- reader：block/tile loop 生成 input tile id，`noc_async_read_tile` 到 L1 input slot，read barrier，publish input ready。
- compute：等待 input ready，计算 input/output ring slot，`acquire_dst`，raw-address `_llk_unpack_AB_matmul_`，`_llk_math_matmul_`，`_llk_pack_` output subblock，publish input consumed / output ready。
- writer：等待 output ready，按 output subblock 生成 tile id，`noc_async_write_tile`，write barrier，publish output consumed。

matmul 比 add 多 block/subblock/tile-index algebra，但底层动作仍落在同一组 `tt.target` op family 上。

## `tt.target` v0 op set

v0 只覆盖已验证的 add / matmul proof，不尝试一次性覆盖所有 TTNN kernel。

| op family | v0 op | 语义 |
|---|---|---|
| 参数 | `tt.target.get_arg` | 读取 runtime arg。 |
| 参数 | `tt.target.get_ct_arg` | 读取 compile-time arg。 |
| 地址 | `tt.target.addr.add` | base + offset 地址算术。 |
| 地址 | `tt.target.addr.ring_slot` | `iteration % slot_count`。 |
| 地址 | `tt.target.addr.to_llk_addr` | L1 byte address 到 LLK address token。 |
| NoC | `tt.target.noc.read` | byte-range NoC read。 |
| NoC | `tt.target.noc.write` | byte-range NoC write。 |
| NoC | `tt.target.noc.read_tile` | tensor tile id 到 L1 的 tile read。 |
| NoC | `tt.target.noc.write_tile` | L1 到 tensor tile id 的 tile write。 |
| NoC | `tt.target.noc.barrier` | read/write barrier。 |
| sync | `tt.target.stream_reg.addr` | stream id + register index 到 register address。 |
| sync | `tt.target.reg.read` | 读取 register / stream register。 |
| sync | `tt.target.reg.write` | 写 register / stream register。 |
| sync | `tt.target.wait.eq` | 等待 register 等于目标值。 |
| sync | `tt.target.wait.ge` | 等待 register 大于等于目标值。 |
| tile regs | `tt.target.tile.acquire` | 获取 DST / tile register 资源。 |
| tile regs | `tt.target.tile.commit` | 提交 compute work。 |
| tile regs | `tt.target.tile.wait` | 等待 tile register work 完成。 |
| tile regs | `tt.target.tile.release` | 释放 tile register 资源。 |
| unpack | `tt.target.unpack.ab` | 两输入 elementwise unpack。 |
| unpack | `tt.target.unpack.ab_matmul` | matmul unpack。 |
| math | `tt.target.math.add` | tile add math。 |
| math | `tt.target.math.matmul` | matmul math。 |
| pack | `tt.target.pack` | pack DST tile 到 L1 地址。 |
| escape hatch | `tt.target.tensix.intrinsic` | 临时承载尚未结构化的 `_llk_*` / `TTI_*` / `TT_*` macro。 |

`tt.target.tensix.intrinsic` 只能作为 bring-up 逃生口，不能长期替代结构化 op。每新增一个 intrinsic，都应该在文档里说明它对应哪个尚未建模的硬件动作。

## legacy 替换关系

| legacy fact source | compiler-managed 替换 |
|---|---|
| `TensorAccessorArgs` / `TensorAccessor` | `AxeTensor + AxeLayout` lowering 出来的地址表达式。 |
| `InterleavedAddrGen*` / `get_noc_addr` | layout/address lowering 生成的 NoC address expression。 |
| `CircularBufferConfig` / `CreateCircularBuffer` | compiler-owned L1 allocation + schedule metadata；firmware 不 materialize CBInterface 作为事实来源。 |
| `cb_reserve_back` / `cb_wait_front` / `cb_push_back` / `cb_pop_front` | `tt.target.wait.*` + compiler-generated slot address + `tt.target.reg.write`。 |
| `get_read_ptr` / `get_write_ptr` | `tt.target.addr.ring_slot` + `tt.target.addr.add`。 |
| `get_local_cb_interface(cbid)` | compiler-owned operand binding，最终 lowering 到 raw-address unpack/pack setup。 |
| `get_cb_tiles_received_ptr` / `get_cb_tiles_acked_ptr` | explicit stream-register binding。 |

这个替换不是说硬件动作消失。NoC、stream register、tile register、unpack/math/pack 都必须保留；消失的是 TT-Metal CB/TensorAccessor 作为 compiler IR 和 kernel 事实来源的地位。

## pseudo-MLIR proof: tile-add

```mlir
func.func @tile_add(%src0: !axe.tensor, %src1: !axe.tensor, %dst: !axe.tensor) {
  %q0 = tt.l1.alloc "src0_ring" {base = 0x10000, slots = 2, page_bytes = 2048}
  %q1 = tt.l1.alloc "src1_ring" {base = 0x12000, slots = 2, page_bytes = 2048}
  %qo = tt.l1.alloc "dst_ring"  {base = 0x14000, slots = 2, page_bytes = 2048}

  tt.schedule.stage @reader {
    %slot = tt.target.addr.ring_slot %i, 2
    %a0 = tt.target.addr.add %q0.base, %slot * 2048
    %a1 = tt.target.addr.add %q1.base, %slot * 2048
    %g0 = axe.lower_addr %src0[%i]
    %g1 = axe.lower_addr %src1[%i]
    tt.target.wait.ge %input_consumed0, %i_minus_slots
    tt.target.noc.read %g0, %a0, 2048
    tt.target.noc.read %g1, %a1, 2048
    tt.target.noc.barrier read
    tt.dep.produce input_ready0, %i_plus_1
    tt.dep.produce input_ready1, %i_plus_1
  }

  tt.schedule.stage @compute {
    %slot = tt.target.addr.ring_slot %i, 2
    %a0 = tt.target.addr.add %q0.base, %slot * 2048
    %a1 = tt.target.addr.add %q1.base, %slot * 2048
    %ao = tt.target.addr.add %qo.base, %slot * 2048
    tt.dep.consume input_ready0, %i_plus_1
    tt.dep.consume input_ready1, %i_plus_1
    tt.target.tile.acquire
    tt.target.unpack.ab %a0, %a1
    tt.target.math.add
    tt.target.tile.commit
    tt.dep.consume output_consumed, %i_minus_slots
    tt.target.tile.wait
    tt.target.pack %ao
    tt.target.tile.release
    tt.dep.produce input_consumed0, %i_plus_1
    tt.dep.produce input_consumed1, %i_plus_1
    tt.dep.produce output_ready, %i_plus_1
  }

  tt.schedule.stage @writer {
    %slot = tt.target.addr.ring_slot %i, 2
    %ao = tt.target.addr.add %qo.base, %slot * 2048
    %go = axe.lower_addr %dst[%i]
    tt.dep.consume output_ready, %i_plus_1
    tt.target.noc.write %ao, %go, 2048
    tt.target.noc.barrier write
    tt.dep.produce output_consumed, %i_plus_1
  }
}
```

## pseudo-MLIR proof: real matmul

```mlir
func.func @matmul_reuse(%a: !axe.tensor, %b: !axe.tensor, %c: !axe.tensor) {
  %qa = tt.l1.alloc "a_ring" {base = 0x20000, slots = 2, page_bytes = 8192}
  %qb = tt.l1.alloc "b_ring" {base = 0x24000, slots = 2, page_bytes = 8192}
  %qo = tt.l1.alloc "out_ring" {base = 0x28000, slots = 2, page_bytes = 8192}

  tt.schedule.stage @reader {
    %gen = tt.target.addr.add %block, 1
    %slot = tt.target.addr.ring_slot %block, 2
    %a_l1 = tt.target.addr.add %qa.base, %slot * 8192
    %b_l1 = tt.target.addr.add %qb.base, %slot * 8192
    tt.target.wait.ge %input_consumed0, %gen_minus_slots
    tt.target.noc.read_tile %a_tile_id, %a, %a_l1
    tt.target.noc.read_tile %b_tile_id, %b, %b_l1
    tt.target.noc.barrier read
    tt.dep.produce input_ready0, %gen
    tt.dep.produce input_ready1, %gen
  }

  tt.schedule.stage @compute {
    %gen = tt.target.addr.add %block, 1
    %slot = tt.target.addr.ring_slot %block, 2
    %a_l1 = tt.target.addr.add %qa.base, %slot * 8192
    %b_l1 = tt.target.addr.add %qb.base, %slot * 8192
    tt.dep.consume input_ready0, %gen
    tt.dep.consume input_ready1, %gen
    tt.target.tile.acquire
    scf.for %inner = 0 to %k_tiles {
      tt.target.unpack.ab_matmul %a_l1, %b_l1, %a_tile_index, %b_tile_index
      tt.target.math.matmul %dst_index
    }
    tt.target.tile.commit
    tt.dep.consume output_consumed, %output_gen_minus_slots
    tt.target.tile.wait
    tt.target.pack %out_l1
    tt.target.tile.release
    tt.dep.produce input_consumed0, %gen
    tt.dep.produce input_consumed1, %gen
    tt.dep.produce output_ready, %output_gen
  }

  tt.schedule.stage @writer {
    tt.dep.consume output_ready, %output_gen
    tt.target.noc.write_tile %out_tile_id, %c, %out_l1
    tt.target.noc.barrier write
    tt.dep.produce output_consumed, %output_gen
  }
}
```

## sanity checker

`level_c_tt_target_dialect_sanity` 是当前 v0 checker。它不实现 MLIR parser，而是用固定 pseudo IR 和显式规则验证 dialect 边界：

- pseudo IR 不包含 legacy token：`TensorAccessor`、`CircularBuffer`、`CreateCircularBuffer`、`cb_*`、`CBInterface`、`get_local_cb_interface`。
- add / matmul 都包含 reader、compute、writer 三类 stage。
- add 包含 NoC read/write、stream-register wait/write、tile register lifecycle、unpack/add/pack。
- matmul 包含 tile NoC read/write、stream-register wait/write、tile register lifecycle、matmul unpack/math/pack。
- producer/consumer token 成对出现。
- ring slot address evaluator 对 add 和 matmul 的 slot wrap 语义正确。

运行方式：

```bash
conda run -n tt cmake --build build_Release \
  --target level_c_tt_target_dialect_sanity -j8

build_Release/programming_examples/compiler_managed_l1_dataflow/level_c/level_c_tt_target_dialect_sanity
```

## 下一步

1. 把文档中的 pseudo IR 固化成真正的 TableGen dialect spec。
2. 先只接入 `tt.target` op，不把 TTNN high-level op、CB、TensorAccessor 放进 dialect。
3. 做一个 C++ emitter，把 `tt.target` v0 发射成现有 `_llk_*` / `noc_async_*` / `NOC_STREAM_*` kernel code。
4. 用 tile-add 和 real matmul 的 `level-c-llk-direct-fw-skip-cb-init` 作为第一批 codegen 等价性对照。
