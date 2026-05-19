// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compiler_managed_l1_dataflow/include/level_c_lowering_descriptors.hpp"
#include "tt-metalium/circular_buffer_constants.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace lowering = tt::tt_metal::experimental::compiler_managed_l1::level_c::lowering;

struct ExperimentalLaunchContract {
    uint64_t local_cb_mask = 0;
    uint32_t min_remote_cb_start_index = NUM_CIRCULAR_BUFFERS;
    bool host_registered_circular_buffers = false;
    lowering::L1QueueDescriptor src0_queue{};
    lowering::L1QueueDescriptor src1_queue{};
    lowering::L1QueueDescriptor dst_queue{};
    lowering::QueueSyncDescriptor src0_sync{};
    lowering::QueueSyncDescriptor src1_sync{};
    lowering::QueueSyncDescriptor dst_sync{};
    lowering::OperandViewDescriptor src0_operand{};
    lowering::OperandViewDescriptor src1_operand{};
    lowering::OperandViewDescriptor dst_operand{};
};

constexpr bool firmware_local_cb_init_would_run(const ExperimentalLaunchContract& contract) {
    return contract.local_cb_mask != 0;
}

constexpr bool firmware_remote_cb_init_would_run(const ExperimentalLaunchContract& contract) {
    return contract.min_remote_cb_start_index < NUM_CIRCULAR_BUFFERS;
}

bool contains_forbidden_fw_skip_token(std::string_view generated_kernel) {
    constexpr std::array<std::string_view, 13> forbidden_tokens = {
        "CreateCircularBuffer",
        "CircularBufferConfig",
        "local_cb_mask",
        "setup_local_cb_read_write_interfaces",
        "setup_remote_cb_interfaces",
        "LocalCBInterface",
        "CBInterface",
        "get_local_cb_interface",
        "get_cb_tiles_received_ptr",
        "get_cb_tiles_acked_ptr",
        "cb_wait_front",
        "cb_reserve_back",
        "pack_tile(",
    };
    for (const auto token : forbidden_tokens) {
        if (generated_kernel.find(token) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

ExperimentalLaunchContract make_add_fw_skip_contract() {
    ExperimentalLaunchContract contract;
    contract.local_cb_mask = 0;
    contract.min_remote_cb_start_index = NUM_CIRCULAR_BUFFERS;
    contract.host_registered_circular_buffers = false;

    contract.src0_queue.l1_base_addr = 0x10000;
    contract.src0_queue.total_size_bytes = 4096;
    contract.src0_queue.slot_count = 2;
    contract.src0_queue.page_size_bytes = 2048;
    contract.src0_queue.sync_descriptor_index = 0;
    contract.src0_queue.bank_policy = lowering::BankPolicy::single_bank;

    contract.src1_queue = contract.src0_queue;
    contract.src1_queue.l1_base_addr = 0x12000;
    contract.src1_queue.sync_descriptor_index = 1;

    contract.dst_queue = contract.src0_queue;
    contract.dst_queue.l1_base_addr = 0x14000;
    contract.dst_queue.sync_descriptor_index = 2;

    contract.src0_sync.sync_kind = lowering::QueueSyncKind::stream_register_pair;
    contract.src0_sync.producer = lowering::ThreadRole::brisc;
    contract.src0_sync.consumer = lowering::ThreadRole::trisc0;
    contract.src0_sync.produced_stream_reg = 0;
    contract.src0_sync.consumed_stream_reg = 1;

    contract.src1_sync = contract.src0_sync;
    contract.src1_sync.produced_stream_reg = 2;
    contract.src1_sync.consumed_stream_reg = 3;

    contract.dst_sync.sync_kind = lowering::QueueSyncKind::stream_register_pair;
    contract.dst_sync.producer = lowering::ThreadRole::trisc2;
    contract.dst_sync.consumer = lowering::ThreadRole::ncrisc;
    contract.dst_sync.produced_stream_reg = 4;
    contract.dst_sync.consumed_stream_reg = 5;

    contract.src0_operand.operand_id = 0;
    contract.src0_operand.operand_role = lowering::OperandRole::input;
    contract.src0_operand.pack_unpack_role = lowering::PackUnpackRole::unpack_src;
    contract.src0_operand.queue_descriptor_index = 0;
    contract.src0_operand.l1_base_addr = contract.src0_queue.l1_base_addr;
    contract.src0_operand.page_size_bytes = contract.src0_queue.page_size_bytes;

    contract.src1_operand = contract.src0_operand;
    contract.src1_operand.operand_id = 1;
    contract.src1_operand.queue_descriptor_index = 1;
    contract.src1_operand.l1_base_addr = contract.src1_queue.l1_base_addr;

    contract.dst_operand = contract.src0_operand;
    contract.dst_operand.operand_id = 16;
    contract.dst_operand.operand_role = lowering::OperandRole::output;
    contract.dst_operand.pack_unpack_role = lowering::PackUnpackRole::pack_dst;
    contract.dst_operand.queue_descriptor_index = 2;
    contract.dst_operand.l1_base_addr = contract.dst_queue.l1_base_addr;
    return contract;
}

bool contract_has_descriptor_source_of_truth(const ExperimentalLaunchContract& contract) {
    return !contract.host_registered_circular_buffers &&
           !firmware_local_cb_init_would_run(contract) &&
           !firmware_remote_cb_init_would_run(contract) &&
           contract.src0_sync.sync_kind == lowering::QueueSyncKind::stream_register_pair &&
           contract.src1_sync.sync_kind == lowering::QueueSyncKind::stream_register_pair &&
           contract.dst_sync.sync_kind == lowering::QueueSyncKind::stream_register_pair &&
           contract.src0_operand.l1_base_addr == contract.src0_queue.l1_base_addr &&
           contract.src1_operand.l1_base_addr == contract.src1_queue.l1_base_addr &&
           contract.dst_operand.l1_base_addr == contract.dst_queue.l1_base_addr &&
           contract.src0_operand.page_size_bytes == contract.src0_queue.page_size_bytes &&
           contract.src1_operand.page_size_bytes == contract.src1_queue.page_size_bytes &&
           contract.dst_operand.page_size_bytes == contract.dst_queue.page_size_bytes;
}

int main() {
    const auto contract = make_add_fw_skip_contract();
    constexpr std::string_view generated_kernel =
        "src0 = src0_operand.l1_base_addr + slot * src0_operand.page_size_bytes; "
        "src1 = src1_operand.l1_base_addr + slot * src1_operand.page_size_bytes; "
        "dst = dst_operand.l1_base_addr + slot * dst_operand.page_size_bytes; "
        "wait(stream_register_pair.src0_ready); "
        "_llk_unpack_AB_(src0, src1); _llk_math_eltwise_binary_(); _llk_pack_(dst); "
        "publish(stream_register_pair.dst_ready);";

    if (!contract_has_descriptor_source_of_truth(contract)) {
        std::cerr << "Level C fw-skip contract still depends on CB launch/materialization state\n";
        return 1;
    }
    if (contains_forbidden_fw_skip_token(generated_kernel)) {
        std::cerr << "Level C fw-skip generated kernel contains forbidden CB token\n";
        return 1;
    }

    std::cout << "Level C fw-skip CB-init contract sanity passed\n";
    std::cout << "local_cb_mask=" << contract.local_cb_mask
              << " min_remote_cb_start_index=" << contract.min_remote_cb_start_index
              << " src0_l1=0x" << std::hex << contract.src0_operand.l1_base_addr
              << " dst_l1=0x" << contract.dst_operand.l1_base_addr << std::dec << "\n";
    return 0;
}
