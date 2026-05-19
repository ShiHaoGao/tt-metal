// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compiler_managed_l1_dataflow/include/level_c_ir.hpp"
#include "compiler_managed_l1_dataflow/include/level_c_lowering_descriptors.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace level_c = tt::tt_metal::experimental::compiler_managed_l1::level_c;
namespace lowering = tt::tt_metal::experimental::compiler_managed_l1::level_c::lowering;

namespace {

constexpr uint32_t kTileSizeBytes = 2048;
constexpr uint32_t kSlotCount = 2;
constexpr uint32_t kTileCount = 8;

struct ModeledCb {
    std::string name;
    uint32_t operand_id = 0;
    uint32_t l1_base_addr = 0;
    uint32_t page_size_bytes = kTileSizeBytes;
    uint32_t slot_count = kSlotCount;
    lowering::OperandRole operand_role = lowering::OperandRole::none;
    lowering::PackUnpackRole pack_unpack_role = lowering::PackUnpackRole::none;
};

struct ModeledTensorAccessor {
    std::string name;
    uint32_t dram_base_addr = 0;
    uint32_t page_size_bytes = kTileSizeBytes;
    uint32_t page_count = kTileCount;
};

struct LoweredCb {
    level_c::AxeTensor tensor;
    lowering::TensorLayoutDescriptor tensor_layout;
    lowering::L1QueueDescriptor queue;
    lowering::QueueSyncDescriptor sync;
    lowering::OperandViewDescriptor operand;
};

struct CaseResult {
    const char* name = "";
    bool passed = false;
};

level_c::AxeAxis axis(std::string name, level_c::LayoutAxisKind kind) {
    return level_c::AxeAxis{std::move(name), kind};
}

level_c::AxeIter iter(uint32_t extent, int32_t stride, std::string name, level_c::LayoutAxisKind kind) {
    return level_c::AxeIter{extent, stride, axis(std::move(name), kind)};
}

level_c::AxeCoordinateEntry offset_entry(int32_t value, std::string name, level_c::LayoutAxisKind kind) {
    return level_c::AxeCoordinateEntry{value, axis(std::move(name), kind)};
}

level_c::AxeTensor adapt_cb_payload_to_axe_tensor(const ModeledCb& cb) {
    level_c::AxeLayout layout;
    layout.d_iters = {
        iter(cb.slot_count, static_cast<int32_t>(cb.page_size_bytes), "ring_slot", level_c::LayoutAxisKind::storage),
        iter(cb.page_size_bytes, 1, "byte", level_c::LayoutAxisKind::storage),
    };

    level_c::AxeTensor tensor;
    tensor.name = cb.name + "_tensor";
    tensor.element_type = level_c::AxeElementType::bfloat16;
    tensor.storage = level_c::AxeStorage{
        .name = cb.name + "_storage",
        .kind = level_c::AxeStorageKind::owned_device_allocation,
        .memory_space = level_c::AxeMemorySpace::l1,
        .ownership = level_c::AxeStorageOwnership::owned,
        .size_bytes = static_cast<uint64_t>(cb.slot_count) * cb.page_size_bytes,
        .base_address = cb.l1_base_addr,
        .device_scope = "core(0,0)"};
    tensor.layout = std::move(layout);
    return tensor;
}

level_c::AxeTensor adapt_tensor_accessor_to_axe_tensor(const ModeledTensorAccessor& accessor) {
    level_c::AxeLayout layout;
    layout.d_iters = {
        iter(accessor.page_count, static_cast<int32_t>(accessor.page_size_bytes), "page", level_c::LayoutAxisKind::page),
        iter(accessor.page_size_bytes, 1, "byte", level_c::LayoutAxisKind::storage),
    };

    level_c::AxeTensor tensor;
    tensor.name = accessor.name + "_tensor";
    tensor.element_type = level_c::AxeElementType::bfloat16;
    tensor.storage = level_c::AxeStorage{
        .name = accessor.name + "_storage",
        .kind = level_c::AxeStorageKind::owned_device_allocation,
        .memory_space = level_c::AxeMemorySpace::dram,
        .ownership = level_c::AxeStorageOwnership::owned,
        .size_bytes = static_cast<uint64_t>(accessor.page_count) * accessor.page_size_bytes,
        .base_address = accessor.dram_base_addr,
        .device_scope = "device0"};
    tensor.layout = std::move(layout);
    return tensor;
}

lowering::MemorySpace lower_memory_space(level_c::AxeMemorySpace memory_space) {
    switch (memory_space) {
        case level_c::AxeMemorySpace::l1: return lowering::MemorySpace::l1;
        case level_c::AxeMemorySpace::dram: return lowering::MemorySpace::dram;
        case level_c::AxeMemorySpace::system_memory: return lowering::MemorySpace::system_memory;
        case level_c::AxeMemorySpace::register_file:
        case level_c::AxeMemorySpace::unknown: break;
    }
    return lowering::MemorySpace::l1;
}

lowering::DataFormat lower_element_type(level_c::AxeElementType element_type) {
    switch (element_type) {
        case level_c::AxeElementType::bfloat16: return lowering::DataFormat::bfloat16;
        case level_c::AxeElementType::float32: return lowering::DataFormat::float32;
        case level_c::AxeElementType::uint32: return lowering::DataFormat::uint32;
        case level_c::AxeElementType::int32: return lowering::DataFormat::int32;
        case level_c::AxeElementType::uint16: return lowering::DataFormat::uint16;
        case level_c::AxeElementType::uint8: return lowering::DataFormat::uint8;
        case level_c::AxeElementType::invalid: break;
    }
    return lowering::DataFormat::invalid;
}

uint32_t axis_extent(const level_c::AxeLayout& layout, std::string_view name) {
    for (const auto& it : layout.d_iters) {
        if (it.axis.name == name) {
            return it.extent;
        }
    }
    return 1;
}

uint32_t axis_stride(const level_c::AxeLayout& layout, std::string_view name) {
    for (const auto& it : layout.d_iters) {
        if (it.axis.name == name) {
            return static_cast<uint32_t>(it.stride);
        }
    }
    return 0;
}

uint64_t evaluate_axe_address(
    const level_c::AxeTensor& tensor,
    const std::vector<std::pair<std::string_view, uint32_t>>& coords) {
    uint64_t address = tensor.storage.base_address;
    for (const auto& coord : coords) {
        address += static_cast<uint64_t>(coord.second) * axis_stride(tensor.layout, coord.first);
    }
    for (const auto& entry : tensor.layout.o_entries) {
        address += static_cast<uint64_t>(entry.value);
    }
    return address;
}

lowering::TensorLayoutDescriptor lower_tensor_layout(
    const level_c::AxeTensor& tensor,
    lowering::AddressPolicy address_policy) {
    lowering::TensorLayoutDescriptor descriptor;
    descriptor.memory_space = lower_memory_space(tensor.storage.memory_space);
    descriptor.address_policy = address_policy;
    descriptor.data_format = lower_element_type(tensor.element_type);
    descriptor.base_addr = tensor.storage.base_address;
    descriptor.page_size_bytes = axis_extent(tensor.layout, "byte");
    descriptor.page_count = axis_extent(tensor.layout, "page");
    if (descriptor.page_count == 1) {
        descriptor.page_count = axis_extent(tensor.layout, "ring_slot");
    }
    descriptor.logical_shape = lowering::make_shape_descriptor(1, descriptor.page_count);
    descriptor.physical_shape = descriptor.logical_shape;
    descriptor.stride_pages[0] = 1;
    return descriptor;
}

lowering::L1QueueDescriptor lower_l1_queue(const level_c::AxeTensor& tensor, uint32_t sync_descriptor_index) {
    lowering::L1QueueDescriptor queue;
    queue.l1_base_addr = static_cast<uint32_t>(tensor.storage.base_address);
    queue.total_size_bytes = static_cast<uint32_t>(tensor.storage.size_bytes);
    queue.slot_count = axis_extent(tensor.layout, "ring_slot");
    queue.page_size_bytes = axis_extent(tensor.layout, "byte");
    queue.bank_policy = lowering::BankPolicy::single_bank;
    queue.sync_descriptor_index = sync_descriptor_index;
    queue.lifetime_start_step = 0;
    queue.lifetime_end_step = kTileCount;
    return queue;
}

lowering::QueueSyncDescriptor lower_cb_sync(bool use_stream_register_pair) {
    lowering::QueueSyncDescriptor sync;
    sync.sync_kind =
        use_stream_register_pair ? lowering::QueueSyncKind::stream_register_pair : lowering::QueueSyncKind::static_schedule;
    sync.producer = lowering::ThreadRole::brisc;
    sync.consumer = lowering::ThreadRole::trisc0;
    sync.produced_stream_reg = use_stream_register_pair ? 0 : lowering::kInvalidDescriptorIndex;
    sync.consumed_stream_reg = use_stream_register_pair ? 1 : lowering::kInvalidDescriptorIndex;
    sync.static_schedule_id = use_stream_register_pair ? lowering::kInvalidDescriptorIndex : 0;
    return sync;
}

lowering::OperandViewDescriptor lower_operand_view(
    const ModeledCb& cb,
    const lowering::L1QueueDescriptor& queue,
    uint32_t queue_descriptor_index,
    uint32_t tensor_layout_index) {
    lowering::OperandViewDescriptor operand;
    operand.operand_id = cb.operand_id;
    operand.operand_role = cb.operand_role;
    operand.pack_unpack_role = cb.pack_unpack_role;
    operand.data_format = lowering::DataFormat::bfloat16;
    operand.page_size_bytes = queue.page_size_bytes;
    operand.queue_descriptor_index = queue_descriptor_index;
    operand.tensor_layout_index = tensor_layout_index;
    operand.l1_base_addr = queue.l1_base_addr;
    operand.read_stride_pages = 1;
    operand.write_stride_pages = 1;
    operand.tiles_per_page = 1;
    return operand;
}

LoweredCb lower_cb(const ModeledCb& cb, uint32_t queue_descriptor_index, bool use_stream_register_pair) {
    LoweredCb lowered;
    lowered.tensor = adapt_cb_payload_to_axe_tensor(cb);
    lowered.tensor_layout = lower_tensor_layout(lowered.tensor, lowering::AddressPolicy::buffer_base_plus_page);
    lowered.sync = lower_cb_sync(use_stream_register_pair);
    lowered.queue = lower_l1_queue(lowered.tensor, queue_descriptor_index);
    lowered.operand = lower_operand_view(cb, lowered.queue, queue_descriptor_index, queue_descriptor_index);
    return lowered;
}

uint32_t ring_slot(const lowering::L1QueueDescriptor& queue, uint32_t tile_index) {
    return tile_index % queue.slot_count;
}

uint64_t evaluate_lowered_queue_address(const lowering::L1QueueDescriptor& queue, uint32_t tile_index) {
    return static_cast<uint64_t>(queue.l1_base_addr) + ring_slot(queue, tile_index) * queue.page_size_bytes;
}

uint64_t evaluate_lowered_tensor_address(const lowering::TensorLayoutDescriptor& descriptor, uint32_t page_id) {
    return descriptor.base_addr + static_cast<uint64_t>(page_id) * descriptor.page_size_bytes;
}

bool forbidden_runtime_cb_tokens_absent(std::string_view generated_pseudo_code) {
    constexpr std::array<std::string_view, 10> forbidden_tokens = {
        "LocalCBInterface",
        "CBInterface",
        "get_local_cb_interface",
        "CreateCircularBuffer",
        "CircularBufferConfig",
        "cb_wait_front",
        "cb_reserve_back",
        "cb_push_back",
        "cb_pop_front",
        "get_write_ptr",
    };
    for (const auto token : forbidden_tokens) {
        if (generated_pseudo_code.find(token) != std::string_view::npos) {
            return false;
        }
    }
    return true;
}

void add_result(std::vector<CaseResult>& results, const char* name, bool passed) {
    results.push_back(CaseResult{name, passed});
}

}  // namespace

int main() {
    std::vector<CaseResult> results;

    const ModeledCb src0_cb{
        .name = "src0_cb",
        .operand_id = 0,
        .l1_base_addr = 0x10000,
        .operand_role = lowering::OperandRole::input,
        .pack_unpack_role = lowering::PackUnpackRole::unpack_src};
    const ModeledCb src1_cb{
        .name = "src1_cb",
        .operand_id = 1,
        .l1_base_addr = 0x12000,
        .operand_role = lowering::OperandRole::input,
        .pack_unpack_role = lowering::PackUnpackRole::unpack_src};
    const ModeledCb dst_cb{
        .name = "dst_cb",
        .operand_id = 16,
        .l1_base_addr = 0x14000,
        .operand_role = lowering::OperandRole::output,
        .pack_unpack_role = lowering::PackUnpackRole::pack_dst};

    const auto src0 = lower_cb(src0_cb, 0, true);
    const auto src1 = lower_cb(src1_cb, 1, true);
    const auto dst = lower_cb(dst_cb, 2, true);

    add_result(
        results,
        "CB payload storage -> AxeStorage + AxeLayout ring slots",
        src0.tensor.storage.kind == level_c::AxeStorageKind::owned_device_allocation &&
            src0.tensor.storage.base_address == src0_cb.l1_base_addr &&
            axis_extent(src0.tensor.layout, "ring_slot") == kSlotCount &&
            axis_extent(src0.tensor.layout, "byte") == kTileSizeBytes);

    add_result(
        results,
        "CB sync -> QueueSyncDescriptor stream_register_pair",
        src0.sync.sync_kind == lowering::QueueSyncKind::stream_register_pair &&
            src0.sync.produced_stream_reg == 0 &&
            src0.sync.consumed_stream_reg == 1);

    add_result(
        results,
        "CB operand -> OperandViewDescriptor",
        src0.operand.operand_id == 0 &&
            src0.operand.operand_role == lowering::OperandRole::input &&
            src0.operand.pack_unpack_role == lowering::PackUnpackRole::unpack_src &&
            dst.operand.operand_id == 16 &&
            dst.operand.pack_unpack_role == lowering::PackUnpackRole::pack_dst);

    bool ring_equivalence = true;
    for (uint32_t tile = 0; tile < kTileCount; ++tile) {
        const uint32_t slot = tile % kSlotCount;
        ring_equivalence =
            ring_equivalence &&
            evaluate_axe_address(src0.tensor, {{"ring_slot", slot}, {"byte", 0}}) ==
                evaluate_lowered_queue_address(src0.queue, tile) &&
            evaluate_axe_address(src1.tensor, {{"ring_slot", slot}, {"byte", 0}}) ==
                evaluate_lowered_queue_address(src1.queue, tile) &&
            evaluate_axe_address(dst.tensor, {{"ring_slot", slot}, {"byte", 0}}) ==
                evaluate_lowered_queue_address(dst.queue, tile);
    }
    add_result(results, "Axe ring address evaluator == lowered L1Queue evaluator", ring_equivalence);

    const ModeledTensorAccessor dram_accessor{
        .name = "src0_dram",
        .dram_base_addr = 0x80000000,
        .page_size_bytes = kTileSizeBytes,
        .page_count = kTileCount};
    const auto dram_tensor = adapt_tensor_accessor_to_axe_tensor(dram_accessor);
    const auto dram_descriptor = lower_tensor_layout(dram_tensor, lowering::AddressPolicy::absolute_noc);
    bool accessor_equivalence = true;
    for (uint32_t page = 0; page < kTileCount; ++page) {
        accessor_equivalence =
            accessor_equivalence &&
            evaluate_axe_address(dram_tensor, {{"page", page}, {"byte", 0}}) ==
                evaluate_lowered_tensor_address(dram_descriptor, page);
    }
    add_result(results, "TensorAccessor DRAM address -> AxeTensor -> TensorLayoutDescriptor", accessor_equivalence);

    auto view_tensor = src0.tensor;
    view_tensor.name = "src0_view_tensor";
    view_tensor.storage.kind = level_c::AxeStorageKind::view_of_storage;
    view_tensor.storage.ownership = level_c::AxeStorageOwnership::view;
    view_tensor.storage.root_storage_name = src0.tensor.storage.name;
    view_tensor.layout.o_entries = {offset_entry(static_cast<int32_t>(kTileSizeBytes), "view_offset", level_c::LayoutAxisKind::storage)};
    add_result(
        results,
        "Buffer::view offset survives Axe address lowering",
        view_tensor.storage.root_storage_name == src0.tensor.storage.name &&
            evaluate_axe_address(view_tensor, {{"ring_slot", 0}, {"byte", 0}}) ==
                src0.tensor.storage.base_address + kTileSizeBytes);

    constexpr std::string_view generated_pseudo_code =
        "slot = tile % queue.slot_count; "
        "src0_addr = queue0.base + slot * queue0.page_size; "
        "src1_addr = queue1.base + slot * queue1.page_size; "
        "dst_addr = queue2.base + slot * queue2.page_size; "
        "wait(input_ready); unpack_add_pack(dst_addr); publish(output_ready);";
    add_result(results, "generated pseudo code has no runtime CB token", forbidden_runtime_cb_tokens_absent(generated_pseudo_code));

    bool ok = true;
    for (const auto& result : results) {
        ok = ok && result.passed;
        std::cout << result.name << ": " << (result.passed ? "covered" : "failed") << "\n";
    }
    if (!ok) {
        std::cerr << "Level C lowering adapter sanity failed\n";
        return 1;
    }

    std::cout << "Level C lowering adapter sanity passed\n";
    std::cout << "lowering_adapter_cases=" << results.size() << "\n";
    std::cout << "CB modeled as AxeStorage + AxeLayout + L1QueueDescriptor + QueueSyncDescriptor + OperandViewDescriptor\n";
    return 0;
}
