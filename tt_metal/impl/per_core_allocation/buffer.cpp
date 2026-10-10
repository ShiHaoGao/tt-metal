// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include "impl/buffers/buffer_sharding_args_impl.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

bool is_per_core_allocation(const Buffer& buffer) { return buffer.impl().per_core_allocation_; }

DeviceAddr get_per_core_address(const Buffer& buffer, CoreCoord core) {
    TT_FATAL(
        buffer.impl().per_core_allocation_,
        "get_per_core_address() called on buffer without per-core allocation enabled");
    auto it = buffer.impl().per_core_addresses_.find(core);
    TT_FATAL(it != buffer.impl().per_core_addresses_.end(), "No per-core address for core ({}, {})", core.x, core.y);
    return it->second;
}

const std::unordered_map<CoreCoord, DeviceAddr>& get_per_core_addresses(const Buffer& buffer) {
    return buffer.impl().per_core_addresses_;
}

DeviceAddr get_shard_base_address(const Buffer& buffer, CoreCoord core) {
    if (is_per_core_allocation(buffer)) {
        return get_per_core_address(buffer, core);
    }
    return buffer.address();
}

void copy_per_core_addresses(Buffer& dst, const Buffer& src) {
    TT_FATAL(
        dst.impl().per_core_allocation_ && src.impl().per_core_allocation_,
        "copy_per_core_addresses requires both buffers to use per-core allocation");
    TT_FATAL(dst.impl().per_core_address_mode_ == src.impl().per_core_address_mode_,
             "copy_per_core_addresses requires the same per-core address policy");
    dst.impl().per_core_addresses_ = src.impl().per_core_addresses_;
}

BufferShardingArgs& set_per_core_allocation(BufferShardingArgs& args, bool enable) {
    if (enable) {
        // The reverse of the check in set_range_lockstep_allocation. Without it the two flags can
        // both end up set by calling the setters the other way round, and allocate_buffer takes the
        // per-core branch first, so range lockstep is silently ignored rather than reported.
        TT_FATAL(
            !range_lockstep_allocation::is_range_lockstep_allocation(args),
            "per_core_allocation and range_lockstep_allocation are mutually exclusive ownership policies");
    }
    args.impl().per_core_allocation_ = enable;
    args.impl().per_core_address_mode_ = AddressMode::Independent;
    return args;
}

bool is_per_core_allocation(const BufferShardingArgs& args) { return args.impl().per_core_allocation_; }

AddressMode get_address_mode(const Buffer& buffer) { return buffer.impl().per_core_address_mode_; }

AddressMode get_address_mode(const BufferShardingArgs& args) { return args.impl().per_core_address_mode_; }

BufferShardingArgs& set_per_core_allocation(BufferShardingArgs& args, AddressMode mode) {
    TT_FATAL(mode == AddressMode::Independent || mode == AddressMode::Common, "Unknown per-core address mode");
    if (mode == AddressMode::Common) {
        TT_FATAL(args.shard_spec().has_value(), "Common per-core allocation requires an explicit shard grid");
        TT_FATAL(!args.buffer_distribution_spec().has_value(),
                 "Common per-core allocation does not accept a second distribution scope");
    }
    set_per_core_allocation(args, true);
    args.impl().per_core_address_mode_ = mode;
    return args;
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
