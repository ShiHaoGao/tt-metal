// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compiler_managed_l1_dataflow/include/level_c_ir.hpp"
#include "compiler_managed_l1_dataflow/include/level_c_lowering_descriptors.hpp"

#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>

namespace level_c = tt::tt_metal::experimental::compiler_managed_l1::level_c;
namespace lowering = tt::tt_metal::experimental::compiler_managed_l1::level_c::lowering;

template <typename Descriptor>
constexpr bool is_descriptor_abi_safe() {
    return std::is_standard_layout_v<Descriptor> && std::is_trivially_copyable_v<Descriptor>;
}

int main() {
    static_assert(is_descriptor_abi_safe<lowering::TensorLayoutDescriptor>());
    static_assert(is_descriptor_abi_safe<lowering::QueueSyncDescriptor>());
    static_assert(is_descriptor_abi_safe<lowering::L1QueueDescriptor>());
    static_assert(is_descriptor_abi_safe<lowering::OperandViewDescriptor>());
    static_assert(is_descriptor_abi_safe<lowering::DescriptorTableHeader>());
    static_assert(!std::is_trivially_copyable_v<level_c::AxeLayout>);
    static_assert(!std::is_trivially_copyable_v<level_c::AxeStorage>);
    static_assert(!std::is_trivially_copyable_v<level_c::AxeTensor>);

    lowering::TensorLayoutDescriptor tensor_layout;
    tensor_layout.memory_space = lowering::MemorySpace::l1;
    tensor_layout.address_policy = lowering::AddressPolicy::buffer_base_plus_page;
    tensor_layout.base_addr = 0x10000;
    tensor_layout.page_size_bytes = 2048;
    tensor_layout.page_count = 4;
    tensor_layout.logical_shape = lowering::make_shape_descriptor(2, 1, 4);
    tensor_layout.physical_shape = tensor_layout.logical_shape;

    level_c::AxeLayout axe_layout;
    axe_layout.d_iters.push_back(level_c::AxeIter{1, 4, level_c::AxeAxis{"row", level_c::LayoutAxisKind::row}});
    axe_layout.d_iters.push_back(level_c::AxeIter{4, 1, level_c::AxeAxis{"col", level_c::LayoutAxisKind::col}});

    level_c::AxeTensor tensor_ir;
    tensor_ir.name = "input_tiles";
    tensor_ir.element_type = level_c::AxeElementType::bfloat16;
    tensor_ir.storage = level_c::AxeStorage{
        .name = "input_l1",
        .kind = level_c::AxeStorageKind::owned_device_allocation,
        .memory_space = level_c::AxeMemorySpace::l1,
        .ownership = level_c::AxeStorageOwnership::owned,
        .size_bytes = 8192,
        .base_address = 0x10000,
        .device_scope = "single_core"};
    tensor_ir.layout = axe_layout;

    lowering::QueueSyncDescriptor queue_sync;
    queue_sync.sync_kind = lowering::QueueSyncKind::stream_register_pair;
    queue_sync.producer = lowering::ThreadRole::brisc;
    queue_sync.consumer = lowering::ThreadRole::trisc0;
    queue_sync.produced_stream_reg = 0;
    queue_sync.consumed_stream_reg = 1;

    lowering::L1QueueDescriptor queue;
    queue.l1_base_addr = 0x10000;
    queue.total_size_bytes = 8192;
    queue.slot_count = 4;
    queue.page_size_bytes = 2048;
    queue.bank_policy = lowering::BankPolicy::single_bank;
    queue.sync_descriptor_index = 0;
    queue.lifetime_end_step = 1;

    lowering::OperandViewDescriptor operand;
    operand.operand_id = 0;
    operand.operand_role = lowering::OperandRole::input;
    operand.pack_unpack_role = lowering::PackUnpackRole::unpack_src;
    operand.queue_descriptor_index = 0;
    operand.tensor_layout_index = 0;
    operand.l1_base_addr = queue.l1_base_addr;
    operand.page_size_bytes = queue.page_size_bytes;

    lowering::DescriptorTableHeader header;
    header.tensor_layout_count = 1;
    header.queue_sync_count = 1;
    header.l1_queue_count = 1;
    header.operand_view_count = 1;
    header.tensor_layout_offset_bytes = sizeof(lowering::DescriptorTableHeader);
    header.queue_sync_offset_bytes = header.tensor_layout_offset_bytes + sizeof(lowering::TensorLayoutDescriptor);
    header.l1_queue_offset_bytes = header.queue_sync_offset_bytes + sizeof(lowering::QueueSyncDescriptor);
    header.operand_view_offset_bytes = header.l1_queue_offset_bytes + sizeof(lowering::L1QueueDescriptor);
    header.total_size_bytes = header.operand_view_offset_bytes + sizeof(lowering::OperandViewDescriptor);

    const bool valid =
        tensor_layout.schema_version == lowering::kDescriptorSchemaVersion &&
        axe_layout.d_iters.size() == 2 &&
        axe_layout.r_iters.empty() &&
        axe_layout.o_entries.empty() &&
        axe_layout.d_iters[0].axis.name == "row" &&
        axe_layout.d_iters[1].axis.name == "col" &&
        axe_layout.d_iters[0].axis.kind == level_c::LayoutAxisKind::row &&
        axe_layout.d_iters[1].axis.kind == level_c::LayoutAxisKind::col &&
        tensor_ir.storage.name == "input_l1" &&
        tensor_ir.storage.kind == level_c::AxeStorageKind::owned_device_allocation &&
        tensor_ir.storage.memory_space == level_c::AxeMemorySpace::l1 &&
        tensor_ir.storage.ownership == level_c::AxeStorageOwnership::owned &&
        tensor_ir.storage.size_bytes == 8192 &&
        tensor_ir.layout.d_iters.size() == axe_layout.d_iters.size() &&
        queue_sync.schema_version == lowering::kDescriptorSchemaVersion &&
        queue.schema_version == lowering::kDescriptorSchemaVersion &&
        operand.schema_version == lowering::kDescriptorSchemaVersion &&
        header.total_size_bytes > header.operand_view_offset_bytes;
    const bool shape_is_valid =
        lowering::is_valid_shape_descriptor(tensor_layout.logical_shape) &&
        tensor_layout.logical_shape.rank == 2 &&
        tensor_layout.logical_shape.dims[0] == 1 &&
        tensor_layout.logical_shape.dims[1] == 4;
    if (!valid || !shape_is_valid) {
        std::cerr << "Level C descriptor schema sanity failed\n";
        return 1;
    }

    std::cout << "Level C descriptor schema sanity passed\n";
    std::cout << "AxeTensor IR storage=" << tensor_ir.storage.name << " d_iters=" << tensor_ir.layout.d_iters.size()
              << "\n";
    std::cout << "TensorLayoutDescriptor=" << sizeof(lowering::TensorLayoutDescriptor) << " bytes\n";
    std::cout << "QueueSyncDescriptor=" << sizeof(lowering::QueueSyncDescriptor) << " bytes\n";
    std::cout << "L1QueueDescriptor=" << sizeof(lowering::L1QueueDescriptor) << " bytes\n";
    std::cout << "OperandViewDescriptor=" << sizeof(lowering::OperandViewDescriptor) << " bytes\n";
    std::cout << "DescriptorTableHeader=" << sizeof(lowering::DescriptorTableHeader) << " bytes\n";
    std::cout << "Example descriptor table=" << header.total_size_bytes << " bytes\n";
    return 0;
}
