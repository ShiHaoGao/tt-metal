// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>

namespace common_core_allocation_tests {
using namespace tt::tt_metal;
namespace per_core = tt::tt_metal::experimental::per_core_allocation;
namespace range_lockstep = tt::tt_metal::experimental::range_lockstep_allocation;

BufferShardingArgs common_sharding_args() {
    return BufferShardingArgs(
        ShardSpecBuffer(CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0))),
                        {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {2, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
}

TEST(CommonCoreAllocation, TypedModeSurvivesCopyAndOldBooleanResetsIndependent) {
    auto args = common_sharding_args();
    per_core::set_per_core_allocation(args, per_core::AddressMode::Common);
    ASSERT_TRUE(per_core::is_per_core_allocation(args));
    EXPECT_EQ(per_core::get_address_mode(args), per_core::AddressMode::Common);
    auto copied = args;
    EXPECT_EQ(per_core::get_address_mode(copied), per_core::AddressMode::Common);
    per_core::set_per_core_allocation(copied, true);
    EXPECT_EQ(per_core::get_address_mode(copied), per_core::AddressMode::Independent);
    per_core::set_per_core_allocation(args, false);
    EXPECT_FALSE(per_core::is_per_core_allocation(args));
    EXPECT_EQ(per_core::get_address_mode(args), per_core::AddressMode::Independent);
}

TEST(CommonCoreAllocation, TypedCommonModeRejectsLockstepConflictAndMissingGrid) {
    auto args = common_sharding_args();
    range_lockstep::set_range_lockstep_allocation(args, true);
    EXPECT_ANY_THROW(per_core::set_per_core_allocation(args, per_core::AddressMode::Common));
    auto common = common_sharding_args();
    per_core::set_per_core_allocation(common, per_core::AddressMode::Common);
    EXPECT_ANY_THROW(range_lockstep::set_range_lockstep_allocation(common, true));
    BufferShardingArgs interleaved;
    EXPECT_ANY_THROW(per_core::set_per_core_allocation(interleaved, per_core::AddressMode::Common));
    EXPECT_ANY_THROW(per_core::set_per_core_allocation(common, static_cast<per_core::AddressMode>(7)));
    EXPECT_EQ(per_core::get_address_mode(common), per_core::AddressMode::Common);
}
} // namespace common_core_allocation_tests
