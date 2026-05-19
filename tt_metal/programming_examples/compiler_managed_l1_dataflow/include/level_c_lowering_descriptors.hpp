// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tt::tt_metal::experimental::compiler_managed_l1::level_c::lowering {

inline constexpr uint32_t kDescriptorSchemaVersion = 1;
inline constexpr uint32_t kInvalidDescriptorIndex = 0xFFFFFFFFu;
inline constexpr uint32_t kInvalidAddress = 0xFFFFFFFFu;
inline constexpr uint32_t kMaxTensorRank = 4;

enum class MemorySpace : uint32_t {
    l1 = 0,
    dram = 1,
    system_memory = 2,
};

enum class DataFormat : uint32_t {
    invalid = 0,
    bfloat16 = 1,
    float32 = 2,
    uint32 = 3,
    int32 = 4,
    uint16 = 5,
    uint8 = 6,
};

enum class BankPolicy : uint32_t {
    unspecified = 0,
    single_bank = 1,
    round_robin = 2,
    explicit_map = 3,
};

enum class QueueSyncKind : uint32_t {
    none = 0,
    static_schedule = 1,
    l1_counter_pair = 2,
    stream_register_pair = 3,
    legacy_cb_fallback = 4,
};

enum class ThreadRole : uint32_t {
    none = 0,
    brisc = 1,
    ncrisc = 2,
    trisc0 = 3,
    trisc1 = 4,
    trisc2 = 5,
    erisc = 6,
};

enum class OperandRole : uint32_t {
    none = 0,
    input = 1,
    output = 2,
    intermediate = 3,
};

enum class PackUnpackRole : uint32_t {
    none = 0,
    unpack_src = 1,
    pack_dst = 2,
    direct_dataflow = 3,
};

enum class AddressPolicy : uint32_t {
    absolute_noc = 0,
    buffer_base_plus_page = 1,
    sharded_l1 = 2,
    generated_by_kernel = 3,
};

struct ShapeDescriptor {
    uint32_t rank = 0;
    uint32_t dims[kMaxTensorRank] = {1, 1, 1, 1};
};

constexpr ShapeDescriptor make_shape_descriptor(
    uint32_t rank,
    uint32_t dim0 = 1,
    uint32_t dim1 = 1,
    uint32_t dim2 = 1,
    uint32_t dim3 = 1) {
    return ShapeDescriptor{rank, {dim0, dim1, dim2, dim3}};
}

constexpr bool is_valid_shape_descriptor(const ShapeDescriptor& shape) {
    return shape.rank <= kMaxTensorRank;
}

struct TileShape {
    uint16_t height = 32;
    uint16_t width = 32;
};

// Lowered metadata for mapping a tensor view to physical storage. This is a
// POD descriptor-table carrier, not the canonical compiler IR tensor type.
struct TensorLayoutDescriptor {
    uint32_t schema_version = kDescriptorSchemaVersion;
    MemorySpace memory_space = MemorySpace::l1;
    AddressPolicy address_policy = AddressPolicy::buffer_base_plus_page;
    DataFormat data_format = DataFormat::bfloat16;
    uint64_t base_addr = 0;
    uint32_t page_size_bytes = 0;
    uint32_t page_count = 0;
    ShapeDescriptor logical_shape{};
    ShapeDescriptor physical_shape{};
    uint32_t stride_pages[kMaxTensorRank] = {0, 0, 0, 1};
    uint32_t shard_map_index = kInvalidDescriptorIndex;
    uint32_t bank_map_index = kInvalidDescriptorIndex;
};

struct QueueSyncDescriptor {
    uint32_t schema_version = kDescriptorSchemaVersion;
    QueueSyncKind sync_kind = QueueSyncKind::none;
    ThreadRole producer = ThreadRole::none;
    ThreadRole consumer = ThreadRole::none;
    uint32_t produced_addr = kInvalidAddress;
    uint32_t consumed_addr = kInvalidAddress;
    uint32_t produced_stream_reg = kInvalidDescriptorIndex;
    uint32_t consumed_stream_reg = kInvalidDescriptorIndex;
    uint32_t static_schedule_id = kInvalidDescriptorIndex;
    uint32_t initial_produced = 0;
    uint32_t initial_consumed = 0;
};

struct L1QueueDescriptor {
    uint32_t schema_version = kDescriptorSchemaVersion;
    uint32_t l1_base_addr = 0;
    uint32_t total_size_bytes = 0;
    uint32_t slot_count = 0;
    uint32_t page_size_bytes = 0;
    BankPolicy bank_policy = BankPolicy::unspecified;
    uint32_t bank_map_index = kInvalidDescriptorIndex;
    uint32_t sync_descriptor_index = kInvalidDescriptorIndex;
    uint32_t lifetime_start_step = 0;
    uint32_t lifetime_end_step = 0;
    uint32_t flags = 0;
};

struct OperandViewDescriptor {
    uint32_t schema_version = kDescriptorSchemaVersion;
    uint32_t operand_id = 0;
    OperandRole operand_role = OperandRole::none;
    PackUnpackRole pack_unpack_role = PackUnpackRole::none;
    DataFormat data_format = DataFormat::bfloat16;
    TileShape tile_shape{};
    uint32_t page_size_bytes = 0;
    uint32_t queue_descriptor_index = kInvalidDescriptorIndex;
    uint32_t tensor_layout_index = kInvalidDescriptorIndex;
    uint32_t l1_base_addr = 0;
    uint32_t read_stride_pages = 1;
    uint32_t write_stride_pages = 1;
    uint32_t tiles_per_page = 1;
    uint32_t flags = 0;
};

struct DescriptorTableHeader {
    uint32_t schema_version = kDescriptorSchemaVersion;
    uint32_t tensor_layout_count = 0;
    uint32_t queue_sync_count = 0;
    uint32_t l1_queue_count = 0;
    uint32_t operand_view_count = 0;
    uint32_t tensor_layout_offset_bytes = 0;
    uint32_t queue_sync_offset_bytes = 0;
    uint32_t l1_queue_offset_bytes = 0;
    uint32_t operand_view_offset_bytes = 0;
    uint32_t total_size_bytes = 0;
};

static_assert(std::is_trivially_copyable_v<ShapeDescriptor>);
static_assert(std::is_trivially_copyable_v<TileShape>);
static_assert(std::is_trivially_copyable_v<TensorLayoutDescriptor>);
static_assert(std::is_trivially_copyable_v<QueueSyncDescriptor>);
static_assert(std::is_trivially_copyable_v<L1QueueDescriptor>);
static_assert(std::is_trivially_copyable_v<OperandViewDescriptor>);
static_assert(std::is_trivially_copyable_v<DescriptorTableHeader>);

static_assert(alignof(TensorLayoutDescriptor) <= alignof(uint64_t));
static_assert(alignof(QueueSyncDescriptor) <= alignof(uint64_t));
static_assert(alignof(L1QueueDescriptor) <= alignof(uint64_t));
static_assert(alignof(OperandViewDescriptor) <= alignof(uint64_t));
static_assert(alignof(DescriptorTableHeader) <= alignof(uint64_t));

static_assert(offsetof(TensorLayoutDescriptor, schema_version) == 0);
static_assert(offsetof(QueueSyncDescriptor, schema_version) == 0);
static_assert(offsetof(L1QueueDescriptor, schema_version) == 0);
static_assert(offsetof(OperandViewDescriptor, schema_version) == 0);
static_assert(offsetof(DescriptorTableHeader, schema_version) == 0);

}  // namespace tt::tt_metal::experimental::compiler_managed_l1::level_c::lowering
