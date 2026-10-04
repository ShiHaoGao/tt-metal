// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <limits>
#include "impl/dispatch/system_memory_queue_plan.hpp"
namespace tt::tt_metal {
namespace {
SystemMemoryQueueInputs host_inputs() {
    return {HostQueueBacking{0, 0x60000, false, false, std::nullopt}, 2, 0x1000, 64, 0};
}
TEST(SystemMemoryQueuePlanTest, PreservesNonDefaultTwoQueueSplitAndAuxiliaryTail) {
    const auto plan = plan_system_memory_queues(host_inputs());
    ASSERT_EQ(plan.queues.size(), 2u);
    EXPECT_EQ(plan.cq_size, 0x2e000u);
    EXPECT_EQ(plan.auxiliary_offset, 0x5c000u);
    EXPECT_EQ(plan.auxiliary_size, 0x4000u);
    EXPECT_EQ(plan.queues[0].command_issue_region_size, 0x22000u);
    EXPECT_EQ(plan.queues[0].command_completion_region_size, 0xb000u);
    EXPECT_EQ(plan.queues[1].device_offset, 0x2e000u);
}
TEST(SystemMemoryQueuePlanTest, SeparatesHostViewOffsetFromDeviceChannelEncoding) {
    auto input = host_inputs();
    std::get<HostQueueBacking>(input.backing).channel = 5;
    const auto plan = plan_system_memory_queues(input);
    EXPECT_EQ(plan.host_view_offset, 0x10000000u);
    EXPECT_EQ(plan.channel_offset, 0x50000000u);
    EXPECT_EQ(plan.queues[1].device_offset, 0x5002e000u);
    EXPECT_EQ(plan.auxiliary_offset, 0x5c000u);
}
TEST(SystemMemoryQueuePlanTest, AppliesGalaxyPartitionBeforeReservationAndHonorsExplicitOverride) {
    auto input = host_inputs();
    auto& host = std::get<HostQueueBacking>(input.backing);
    host.channel_size = 0x40000000;
    host.galaxy = true;
    EXPECT_EQ(plan_system_memory_queues(input).cq_size, 0x7ffe000u);
    host.cq_size_override = 0x60000;
    EXPECT_EQ(plan_system_memory_queues(input).cq_size, 0x5e000u);
}
TEST(SystemMemoryQueuePlanTest, ReservesD2HTailAcrossAllQueuesWithPageRounding) {
    auto input = host_inputs();
    input.num_hw_cqs = 3;
    auto& host = std::get<HostQueueBacking>(input.backing);
    host.channel_size = 0x1800000;
    host.d2h_hugepage_fallback = true;
    const auto plan = plan_system_memory_queues(input);
    EXPECT_EQ(plan.cq_size, 0x753000u);
    EXPECT_EQ(plan.auxiliary_size, 0x207000u);
    EXPECT_EQ(plan.auxiliary_offset + plan.auxiliary_size, 0x1800000u);
}
TEST(SystemMemoryQueuePlanTest, PreservesDramDeviceBaseAndHostMirrorOffsets) {
    auto input = host_inputs();
    input.backing = DramQueueBacking{0x60000, 0x800000};
    const auto plan = plan_system_memory_queues(input);
    EXPECT_EQ(plan.cq_size, 0x2e000u);
    EXPECT_EQ(plan.host_view_offset, 0u);
    EXPECT_EQ(plan.channel_offset, 0u);
    EXPECT_EQ(plan.queues[1].device_offset, 0x82e000u);
    EXPECT_EQ(plan.auxiliary_offset, 0x5c000u);
}
TEST(SystemMemoryQueuePlanTest, InitializesFreshMutableCursorsFromPlannedDeviceAddresses) {
    auto plan = plan_system_memory_queues(host_inputs());
    SystemMemoryCQInterface queue(plan.queues[1]);
    EXPECT_EQ(queue.offset, 0x2e000u);
    EXPECT_EQ(queue.issue_fifo_wr_ptr, 0x2f00u);
    EXPECT_EQ(queue.issue_fifo_limit, 0x5100u);
    EXPECT_EQ(queue.completion_fifo_rd_ptr, 0x5100u);
    EXPECT_EQ(queue.completion_fifo_limit, 0x5c00u);
    EXPECT_FALSE(queue.issue_fifo_wr_toggle);
    EXPECT_FALSE(queue.completion_fifo_rd_toggle);
}
TEST(SystemMemoryQueuePlanTest, RejectsInsufficientIssueCapacityAndInvalidBackingBeforeUse) {
    auto input = host_inputs();
    input.minimum_issue_size = 0x22001;
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
    input = host_inputs(); input.num_hw_cqs = 0;
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
    input = host_inputs(); input.alignment = 0;
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
    input = host_inputs(); input.backing = DramQueueBacking{0x60001, 0x800000};
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
    input = host_inputs(); std::get<HostQueueBacking>(input.backing).cq_size_override = 0x2000;
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
    input = host_inputs(); std::get<HostQueueBacking>(input.backing).channel = 65535;
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
}
TEST(SystemMemoryQueuePlanTest, RejectsSingleQueueUnderflowMisalignmentAndDeviceAddressOverflow) {
    EXPECT_THROW(plan_system_memory_cq(0, 0, 0x1000, 0x2000, 64), std::invalid_argument);
    EXPECT_THROW(plan_system_memory_cq(0, 0, 0x60001, 0x1000, 64), std::invalid_argument);
    EXPECT_THROW(plan_system_memory_cq(0, 1, 0x60000, 0x1000, 64, 0xffff0000), std::invalid_argument);
}
TEST(SystemMemoryQueuePlanTest, RejectsModifiedLayoutBeforeInitializingQueueCursors) {
    auto layout = plan_system_memory_cq(0, 0, 0x60000, 0x1000, 64);
    layout.device_offset = 0xffff0000;
    EXPECT_THROW(SystemMemoryCQInterface{layout}, std::invalid_argument);
    layout = plan_system_memory_cq(0, 0, 0x60000, 0x1000, 64);
    layout.command_issue_region_size = 3;
    EXPECT_THROW(SystemMemoryCQInterface{layout}, std::invalid_argument);
}
TEST(SystemMemoryQueuePlanTest, RejectsDramAuxiliaryTailBeyondDeviceAddressSpace) {
    auto input = host_inputs();
    input.backing = DramQueueBacking{0x60000, 0xfffa2000};
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
}
TEST(SystemMemoryQueuePlanTest, AcceptsBackingEndingExactlyAtFourGiB) {
    auto input = host_inputs();
    auto& host = std::get<HostQueueBacking>(input.backing);
    host.channel = 3;
    host.channel_size = 0x40000000;
    const auto plan = plan_system_memory_queues(input);
    EXPECT_EQ(plan.channel_offset, 0xc0000000u);
    EXPECT_EQ(uint64_t(plan.channel_offset) + plan.auxiliary_offset + plan.auxiliary_size, uint64_t{1} << 32);
}
TEST(SystemMemoryQueuePlanTest, RejectsOverrideBeyondActualHostBackingBeforeMapping) {
    auto input = host_inputs();
    auto& host = std::get<HostQueueBacking>(input.backing);
    host.cq_size_override = 0x40000;
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
    host.galaxy = true;
    host.cq_size_override = 0x10000;
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
    // A valid device address does not establish that any host mapping exists.
    host.channel_size = 0;
    EXPECT_THROW(plan_system_memory_queues(input), std::invalid_argument);
}
TEST(SystemMemoryQueuePlanTest, RejectsEmptyCompletionQueueBeforeInitializingCursors) {
    EXPECT_THROW(plan_system_memory_cq(0, 0, 0x2000, 0x1000, 64), std::invalid_argument);
    auto layout = plan_system_memory_cq(0, 0, 0x60000, 0x1000, 64);
    layout.command_completion_region_size = 0;
    EXPECT_THROW(SystemMemoryCQInterface{layout}, std::invalid_argument);
}
} // namespace
} // namespace tt::tt_metal
