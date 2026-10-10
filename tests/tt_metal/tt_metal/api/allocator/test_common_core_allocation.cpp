// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include "tt_metal/impl/allocator/bank_manager.hpp"

namespace common_core_allocation_tests {
using namespace tt::tt_metal;
using AllocatorID = BankManager::AllocatorDependencies::AllocatorID;
using Regions = std::vector<std::pair<DeviceAddr, DeviceAddr>>;

constexpr DeviceAddr kAlignment = 64;
constexpr DeviceAddr kBase = 0x10000;
constexpr DeviceAddr kBankBytes = 4096;
constexpr AllocatorID kLockstep{0};
constexpr AllocatorID kBank0{1};
constexpr AllocatorID kBank1{2};
constexpr AllocatorID kBank2{3};
constexpr AllocatorID kBank3{4};

BankManager::AllocatorDependencies dependencies(uint32_t banks) {
    std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>> graph;
    ttsl::SmallVector<AllocatorID> lockstep;
    for (uint32_t id = 1; id <= banks; ++id) {
        lockstep.push_back(AllocatorID{id});
        graph[AllocatorID{id}] = {kLockstep};
    }
    graph[kLockstep] = std::move(lockstep);
    return BankManager::AllocatorDependencies{graph};
}

struct CommonBanks {
    uint32_t bankCount;
    CoreRangeSet grid;
    BankManager manager;

    explicit CommonBanks(uint32_t count = 4, DeviceAddr bytes = kBankBytes) :
        bankCount(count),
        grid(CoreRange(CoreCoord(0, 0), CoreCoord(count - 1, 0))),
        manager(BufferType::L1, std::vector<int64_t>(count, 0), bytes,
                kAlignment, kAlignment, kBase, false, dependencies(count)) {}

    DeviceAddr one(AllocatorID bank, DeviceAddr bytes, bool bottomUp = true) {
        return manager.allocate_buffer(bytes, bytes, bottomUp, grid, 1, bank);
    }

    DeviceAddr common(const std::vector<AllocatorID>& banks, DeviceAddr bytes,
                      bool bottomUp = true, const Regions& extra = {}) {
        return manager.allocate_buffer_on_banks(bytes, bytes, bottomUp, grid,
                                                banks, extra);
    }

    Regions regions(AllocatorID bank) const {
        return manager.extract_state(bank).allocated_regions;
    }

    std::vector<Regions> snapshot() const {
        std::vector<Regions> result;
        for (uint32_t id = 0; id <= bankCount; ++id)
            result.push_back(regions(AllocatorID{id}));
        return result;
    }

    void release(const std::vector<AllocatorID>& banks, DeviceAddr address) {
        for (auto bank : banks)
            manager.deallocate_buffer(address, bank);
    }
};

TEST(CommonCoreAllocation, DisjointGroupsReuseOneAddressWithoutGlobalReservation) {
    CommonBanks f;
    const auto first = f.common({kBank0, kBank1}, 512);
    const auto second = f.common({kBank2, kBank3}, 512);
    EXPECT_EQ(first, kBase);
    EXPECT_EQ(second, first);
    EXPECT_TRUE(f.regions(kLockstep).empty());
    for (auto bank : {kBank0, kBank1, kBank2, kBank3})
        EXPECT_EQ(f.regions(bank), (Regions{{kBase, kBase + 512}}));
}

TEST(CommonCoreAllocation, OverlappingGroupsReserveSeparateSpans) {
    CommonBanks f;
    const auto first = f.common({kBank0, kBank1}, 512);
    const auto second = f.common({kBank1, kBank2}, 512);
    EXPECT_EQ(first, kBase);
    EXPECT_EQ(second, kBase + 512);
    EXPECT_EQ(f.regions(kBank0), (Regions{{kBase, kBase + 512}}));
    EXPECT_EQ(f.regions(kBank1), (Regions{{kBase, kBase + 1024}}));
    EXPECT_EQ(f.regions(kBank2), (Regions{{kBase + 512, kBase + 1024}}));
    EXPECT_TRUE(f.regions(kBank3).empty());
    EXPECT_TRUE(f.regions(kLockstep).empty());
}

TEST(CommonCoreAllocation, FullNonmemberAndHoleyMembershipRemainIndependent) {
    CommonBanks f;
    ASSERT_EQ(f.one(kBank1, kBankBytes), kBase);
    EXPECT_EQ(f.common({kBank0, kBank2}, 512), kBase);
    EXPECT_EQ(f.regions(kBank1), (Regions{{kBase, kBase + kBankBytes}}));
    EXPECT_TRUE(f.regions(kBank3).empty());
    EXPECT_TRUE(f.regions(kLockstep).empty());
}

TEST(CommonCoreAllocation, SelectedBanksExistingReservationsConstrainBothMembers) {
    CommonBanks f;
    ASSERT_EQ(f.one(kBank0, 512), kBase);
    ASSERT_EQ(f.one(kBank2, 1024), kBase);
    EXPECT_EQ(f.common({kBank0, kBank2}, 512), kBase + 1024);
    EXPECT_EQ(f.regions(kBank0),
              (Regions{{kBase, kBase + 512}, {kBase + 1024, kBase + 1536}}));
    EXPECT_EQ(f.regions(kBank2), (Regions{{kBase, kBase + 1536}}));
}

TEST(CommonCoreAllocation, ExistingAndLaterLockstepAllocationsAvoidTheGroup) {
    CommonBanks f;
    ASSERT_EQ(f.one(kLockstep, 512), kBase);
    EXPECT_EQ(f.common({kBank0, kBank2}, 512), kBase + 512);
    EXPECT_EQ(f.one(kLockstep, 512), kBase + 1024);
    EXPECT_EQ(f.regions(kLockstep),
              (Regions{{kBase, kBase + 512}, {kBase + 1024, kBase + 1536}}));
    EXPECT_TRUE(f.regions(kBank1).empty());
    EXPECT_TRUE(f.regions(kBank3).empty());
}

TEST(CommonCoreAllocation, AdditionalPersistentRangesConstrainTheCommonAddress) {
    CommonBanks f;
    // Unsorted overlapping external reservations must still exclude their union.
    const Regions extra{{kBase + 256, kBase + 768}, {kBase, kBase + 512}};
    EXPECT_EQ(f.common({kBank0, kBank2}, 512, true, extra), kBase + 768);
    // These external ranges are placement constraints, not newly owned blocks.
    EXPECT_EQ(f.regions(kBank0), (Regions{{kBase + 768, kBase + 1280}}));
    EXPECT_EQ(f.regions(kBank2), f.regions(kBank0));
    EXPECT_TRUE(f.regions(kLockstep).empty());
}

TEST(CommonCoreAllocation, TopDownPlacementUsesOneAlignedIntersection) {
    CommonBanks f;
    ASSERT_EQ(f.one(kBank2, 1024, false), kBase + kBankBytes - 1024);
    const auto address = f.common({kBank0, kBank2}, 512, false);
    EXPECT_EQ(address, kBase + kBankBytes - 1536);
    EXPECT_EQ(address % kAlignment, 0u);
    EXPECT_EQ(f.regions(kBank0), (Regions{{address, address + 512}}));
}

TEST(CommonCoreAllocation, UnalignedExternalBoundaryDoesNotProduceUnalignedBase) {
    CommonBanks f;
    EXPECT_EQ(f.common({kBank0, kBank1}, kAlignment, true,
                       {{kBase, kBase + kAlignment + 1}}),
              kBase + 2 * kAlignment);
}

TEST(CommonCoreAllocation, DifferentShrunkBankAperturesUseTheirActualIntersection) {
    CommonBanks f;
    // FreeListOpt currently supports shrinking the low end only.
    f.manager.shrink_size(1024, true, kBank2);
    EXPECT_EQ(f.common({kBank0, kBank2}, 512), kBase + 1024);
    EXPECT_EQ(f.regions(kBank0), (Regions{{kBase + 1024, kBase + 1536}}));
    EXPECT_EQ(f.regions(kBank2), f.regions(kBank0));
}

TEST(CommonCoreAllocation, NoCommonWindowFailsWithoutChangingAnyBank) {
    CommonBanks f;
    // Each member has 2 KiB free, but the free halves do not overlap.
    ASSERT_EQ(f.one(kBank0, 2048), kBase);
    ASSERT_EQ(f.one(kBank1, 2048, false), kBase + 2048);
    const auto before = f.snapshot();
    EXPECT_ANY_THROW(f.common({kBank0, kBank1}, 512));
    EXPECT_EQ(f.snapshot(), before);
    // Independent allocations still succeed on both banks after failed preflight.
    EXPECT_EQ(f.one(kBank0, 512), kBase + 2048);
    EXPECT_EQ(f.one(kBank1, 512), kBase);
}

TEST(CommonCoreAllocation, SmallLaterBankFailsWithoutPartialReservations) {
    CommonBanks f;
    f.manager.shrink_size(kBankBytes - kAlignment, true, kBank2);
    const auto before = f.snapshot();
    EXPECT_ANY_THROW(f.common({kBank0, kBank2}, 512));
    EXPECT_EQ(f.snapshot(), before);
    EXPECT_TRUE(f.regions(kBank0).empty());
}

TEST(CommonCoreAllocation, InvalidBankSelectionsAndGeometryFailBeforeMutation) {
    CommonBanks f;
    ASSERT_EQ(f.one(kBank0, 512), kBase);
    const std::vector<std::vector<AllocatorID>> invalid{
        {}, {kLockstep}, {kBank0, kLockstep}, {kBank0, kBank0},
        {kBank0, AllocatorID{5}}};
    for (const auto& banks : invalid) {
        const auto before = f.snapshot();
        EXPECT_ANY_THROW(f.common(banks, 512));
        EXPECT_EQ(f.snapshot(), before);
    }
    const DeviceAddr maximum = std::numeric_limits<DeviceAddr>::max();
    const std::vector<std::pair<DeviceAddr, DeviceAddr>> invalidGeometry{
        {0, 1}, {1, 0}, {193, 128}, {maximum, maximum}, {maximum / kAlignment + 1, 1}};
    for (const auto& [bytes, page] : invalidGeometry) {
        const auto before = f.snapshot();
        EXPECT_ANY_THROW(f.manager.allocate_buffer_on_banks(
            bytes, page, true, f.grid, {kBank0, kBank1}));
        EXPECT_EQ(f.snapshot(), before);
    }
}

TEST(CommonCoreAllocation, ReleasingOneGroupPreservesAnotherGroupsBanks) {
    CommonBanks f;
    const auto first = f.common({kBank0, kBank1}, 512);
    const auto second = f.common({kBank2, kBank3}, 512);
    ASSERT_EQ(first, second);
    f.release({kBank0, kBank1}, first);
    EXPECT_TRUE(f.regions(kBank0).empty());
    EXPECT_TRUE(f.regions(kBank1).empty());
    EXPECT_EQ(f.regions(kBank2), (Regions{{second, second + 512}}));
    EXPECT_EQ(f.regions(kBank3), f.regions(kBank2));
    EXPECT_EQ(f.common({kBank0, kBank1}, 512), first);
    // Global lockstep still sees the surviving per-bank reservations.
    EXPECT_EQ(f.one(kLockstep, 512), first + 512);
}

TEST(CommonCoreAllocation, EightDisjoint256KiBRowsDoNotStackAcrossAllBanks) {
    constexpr uint32_t banks = 16;
    constexpr DeviceAddr bankBytes = 1024 * 1024;
    constexpr DeviceAddr rowBytes = 256 * 1024;
    CommonBanks f(banks, bankBytes);
    for (uint32_t row = 0; row != 8; ++row) {
        const AllocatorID left{row * 2 + 1};
        const AllocatorID right{row * 2 + 2};
        EXPECT_EQ(f.common({left, right}, rowBytes), kBase);
        EXPECT_EQ(f.regions(left), (Regions{{kBase, kBase + rowBytes}}));
        EXPECT_EQ(f.regions(right), f.regions(left));
    }
    EXPECT_TRUE(f.regions(kLockstep).empty());
    // Every bank retains the other 3/4 of its capacity after all eight rows.
    for (uint32_t bank = 1; bank <= banks; ++bank)
        EXPECT_EQ(f.one(AllocatorID{bank}, bankBytes - rowBytes), kBase + rowBytes);
}

} // namespace common_core_allocation_tests
