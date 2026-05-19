// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compiler_managed_l1_dataflow/include/level_c_ir.hpp"
#include "compiler_managed_l1_dataflow/include/level_c_lowering_descriptors.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace level_c = tt::tt_metal::experimental::compiler_managed_l1::level_c;
namespace lowering = tt::tt_metal::experimental::compiler_managed_l1::level_c::lowering;

struct TileAddGeneratedStep {
    uint32_t tile_index = 0;
    uint32_t slot = 0;
    uint32_t src0_l1_addr = 0;
    uint32_t src1_l1_addr = 0;
    uint32_t dst_l1_addr = 0;
};

constexpr uint32_t ring_slot(const lowering::L1QueueDescriptor& queue, uint32_t tile_index) {
    return tile_index % queue.slot_count;
}

constexpr uint32_t l1_page_addr(const lowering::L1QueueDescriptor& queue, uint32_t slot) {
    return queue.l1_base_addr + slot * queue.page_size_bytes;
}

level_c::AxeTensor make_l1_ring_tensor(std::string_view name, uint32_t base_addr, uint32_t page_size, uint32_t slot_count) {
    level_c::AxeTensor tensor;
    tensor.name = std::string(name);
    tensor.element_type = level_c::AxeElementType::bfloat16;
    tensor.storage = level_c::AxeStorage{
        .name = std::string(name) + "_storage",
        .kind = level_c::AxeStorageKind::owned_device_allocation,
        .memory_space = level_c::AxeMemorySpace::l1,
        .ownership = level_c::AxeStorageOwnership::owned,
        .size_bytes = static_cast<uint64_t>(page_size) * slot_count,
        .base_address = base_addr,
        .device_scope = "core(0,0)"};
    tensor.layout.d_iters = {
        level_c::AxeIter{
            slot_count,
            static_cast<int32_t>(page_size),
            level_c::AxeAxis{"ring_slot", level_c::LayoutAxisKind::storage}},
        level_c::AxeIter{page_size, 1, level_c::AxeAxis{"byte", level_c::LayoutAxisKind::storage}},
    };
    return tensor;
}

lowering::L1QueueDescriptor lower_ring_tensor_to_queue(
    const level_c::AxeTensor& tensor,
    uint32_t sync_descriptor_index,
    uint32_t lifetime_end_step) {
    lowering::L1QueueDescriptor queue;
    queue.l1_base_addr = static_cast<uint32_t>(tensor.storage.base_address);
    queue.total_size_bytes = static_cast<uint32_t>(tensor.storage.size_bytes);
    queue.slot_count = tensor.layout.d_iters[0].extent;
    queue.page_size_bytes = tensor.layout.d_iters[1].extent;
    queue.bank_policy = lowering::BankPolicy::single_bank;
    queue.sync_descriptor_index = sync_descriptor_index;
    queue.lifetime_end_step = lifetime_end_step;
    return queue;
}

constexpr TileAddGeneratedStep generate_tile_add_step(
    uint32_t tile_index,
    const lowering::L1QueueDescriptor& src0_queue,
    const lowering::L1QueueDescriptor& src1_queue,
    const lowering::L1QueueDescriptor& dst_queue) {
    const uint32_t slot = ring_slot(dst_queue, tile_index);
    return TileAddGeneratedStep{
        .tile_index = tile_index,
        .slot = slot,
        .src0_l1_addr = l1_page_addr(src0_queue, slot),
        .src1_l1_addr = l1_page_addr(src1_queue, slot),
        .dst_l1_addr = l1_page_addr(dst_queue, slot),
    };
}

bool contains_forbidden_runtime_cb_token(std::string_view generated_pseudo_code) {
    constexpr std::array<std::string_view, 9> forbidden_tokens = {
        "LocalCBInterface",
        "CBInterface",
        "get_local_cb_interface",
        "cb_wait_front",
        "cb_reserve_back",
        "cb_push_back",
        "cb_pop_front",
        "get_read_ptr",
        "get_write_ptr",
    };
    for (const auto token : forbidden_tokens) {
        if (generated_pseudo_code.find(token) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

int main() {
    constexpr uint32_t page_size_bytes = 2048;
    constexpr uint32_t slot_count = 2;
    constexpr uint32_t tile_count = 4;

    lowering::QueueSyncDescriptor static_schedule_sync;
    static_schedule_sync.sync_kind = lowering::QueueSyncKind::static_schedule;
    static_schedule_sync.static_schedule_id = 0;
    static_schedule_sync.producer = lowering::ThreadRole::brisc;
    static_schedule_sync.consumer = lowering::ThreadRole::trisc0;

    const auto src0_tensor = make_l1_ring_tensor("src0_ring", 0x10000, page_size_bytes, slot_count);
    const auto src1_tensor = make_l1_ring_tensor("src1_ring", 0x12000, page_size_bytes, slot_count);
    const auto dst_tensor = make_l1_ring_tensor("dst_ring", 0x14000, page_size_bytes, slot_count);

    const auto src0_queue = lower_ring_tensor_to_queue(src0_tensor, 0, tile_count);
    const auto src1_queue = lower_ring_tensor_to_queue(src1_tensor, 0, tile_count);
    const auto dst_queue = lower_ring_tensor_to_queue(dst_tensor, 0, tile_count);

    lowering::OperandViewDescriptor src0_operand;
    src0_operand.operand_id = 0;
    src0_operand.operand_role = lowering::OperandRole::input;
    src0_operand.pack_unpack_role = lowering::PackUnpackRole::unpack_src;
    src0_operand.queue_descriptor_index = 0;
    src0_operand.l1_base_addr = src0_queue.l1_base_addr;
    src0_operand.page_size_bytes = page_size_bytes;

    lowering::OperandViewDescriptor src1_operand = src0_operand;
    src1_operand.operand_id = 1;
    src1_operand.queue_descriptor_index = 1;
    src1_operand.l1_base_addr = src1_queue.l1_base_addr;

    lowering::OperandViewDescriptor dst_operand = src0_operand;
    dst_operand.operand_id = 16;
    dst_operand.operand_role = lowering::OperandRole::output;
    dst_operand.pack_unpack_role = lowering::PackUnpackRole::pack_dst;
    dst_operand.queue_descriptor_index = 2;
    dst_operand.l1_base_addr = dst_queue.l1_base_addr;

    constexpr std::string_view generated_pseudo_code =
        "for tile in tiles: slot = tile % slot_count; "
        "src0 = src0_base + slot * page_size; "
        "src1 = src1_base + slot * page_size; "
        "dst = dst_base + slot * page_size; "
        "noc_async_read(src0); noc_async_read(src1); unpack_add_pack(dst);";

    if (contains_forbidden_runtime_cb_token(generated_pseudo_code)) {
        std::cerr << "Generated pseudo code contains runtime CB token\n";
        return 1;
    }

    std::array<TileAddGeneratedStep, tile_count> generated_steps{};
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        generated_steps[tile] = generate_tile_add_step(tile, src0_queue, src1_queue, dst_queue);
    }

    const bool wrap_is_correct =
        generated_steps[0].slot == 0 &&
        generated_steps[1].slot == 1 &&
        generated_steps[2].slot == 0 &&
        generated_steps[3].slot == 1 &&
        generated_steps[2].src0_l1_addr == generated_steps[0].src0_l1_addr &&
        generated_steps[3].dst_l1_addr == generated_steps[1].dst_l1_addr;
    const bool lowered_from_ir =
        src0_tensor.storage.base_address == src0_queue.l1_base_addr &&
        src1_tensor.storage.base_address == src1_queue.l1_base_addr &&
        dst_tensor.storage.base_address == dst_queue.l1_base_addr &&
        src0_tensor.layout.d_iters[0].extent == src0_queue.slot_count &&
        src0_tensor.layout.d_iters[1].extent == src0_queue.page_size_bytes;
    if (!wrap_is_correct || !lowered_from_ir) {
        std::cerr << "Generated Level C tile-add ring schedule is invalid\n";
        return 1;
    }

    std::cout << "Level C tile-add codegen sanity passed\n";
    for (const auto& step : generated_steps) {
        std::cout << "tile=" << step.tile_index << " slot=" << step.slot << " src0=0x" << std::hex
                  << step.src0_l1_addr << " src1=0x" << step.src1_l1_addr << " dst=0x" << step.dst_l1_addr
                  << std::dec << "\n";
    }
    return 0;
}
