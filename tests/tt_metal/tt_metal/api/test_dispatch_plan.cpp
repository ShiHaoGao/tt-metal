// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <reflect>
#include "impl/experimental/published_deployment/dispatch_plan.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/system_memory_cq_interface.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::experimental {
namespace {
template <class T>
void initialize_unused(T& value) {
    if constexpr (requires { value.has_value(); }) {
        value = typename T::value_type{};
    } else {
        reflect::for_each([&](auto i) { initialize_unused(reflect::get<i>(value)); }, value);
    }
}
class DispatchPlanTest : public ::testing::Test {
protected:
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = "/tmp"}};
    Hal hal{tt::ARCH::BLACKHOLE, false, true, 0, false, false, true, true};
    DispatchMemMap memory{CoreType::WORKER, 2, hal, false, {false, 1}, options};
    // Explicit non-default deployment inputs, processed by the real SDK CQ owner.
    SystemMemoryCQInterface queue{
        0,
        1,
        0x60000,
        memory.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED),
        hal.get_alignment(HalMemType::HOST)};
    DispatchKernelConfiguration node() {
        DispatchResolvedConfiguration resolved{};
        resolved.command_queue_size = 0x60000;
        resolved.prefetch_q_entry_bits = memory.prefetch_q_entry_size_bytes() * 8;
        resolved.my_noc = {7, 9};
        resolved.upstream_noc = {10, 12};
        resolved.downstream_noc = {13, 15};
        resolved.subordinate_noc = {16, 18};
        resolved.worker_count = 120;
        resolved.worker_multicast = 0x12131415;
        resolved.physical_eth_cores = 8;
        resolved.virtual_eth_cores = 10;
        return {
            .node_id = 7,
            .device_id = 0,
            .servicing_device_id = 0,
            .cq_id = 1,
            .processor = {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0},
            .logical_core = {1, 2},
            .nocs = {NOC_0, NOC_1, NOC_0},
            .resolved = resolved};
    }
    DispatchKernelConfiguration prefetch() {
        auto n = node();
        PrefetchConfiguration p;
        initialize_unused(p.static_config);
        initialize_unused(p.dependent_config);
        p.static_config.is_h_variant = true;
        p.static_config.is_d_variant = true;
        p.static_config.pcie_base = queue.offset + queue.cq_start;
        p.static_config.pcie_size = queue.command_issue_region_size;
        p.static_config.prefetch_q_base =
            memory.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED, 1);
        p.static_config.prefetch_q_size = memory.prefetch_q_size();
        p.static_config.my_downstream_cb_sem_id = 3;
        p.dependent_config.downstream_cb_sem_id = 5;
        p.dependent_config.downstream_cb_base = memory.dispatch_buffer_base(1);
        n.kernel = p;
        return n;
    }
    DispatchKernelConfiguration dispatcher() {
        auto n = node();
        DispatcherConfiguration p;
        initialize_unused(p.static_config);
        initialize_unused(p.dependent_config);
        p.static_config.is_h_variant = true;
        p.static_config.is_d_variant = true;
        p.static_config.command_queue_base_addr = queue.offset;
        p.static_config.completion_queue_base_addr = queue.offset + queue.cq_start + queue.command_issue_region_size;
        p.static_config.completion_queue_size = queue.command_completion_region_size;
        n.kernel = p;
        return n;
    }
    DispatchKernelConfiguration subordinate() {
        auto n = node();
        SubordinateConfiguration p;
        initialize_unused(p.static_config);
        initialize_unused(p.dependent_config);
        p.static_config.cb_size = memory.dispatch_s_buffer_size();
        p.static_config.first_stream_used = memory.get_dispatch_stream_index(0);
        p.static_config.max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES;
        n.processor.processor_type = 1;
        n.kernel = p;
        return n;
    }
};
TEST_F(DispatchPlanTest, PreservesActualPrefetchQueueAndPeerInputs) {
    ASSERT_FALSE(MetalContext::instance_exists());
    auto plan = plan_dispatch_kernel(tt::ARCH::BLACKHOLE, prefetch());
    ASSERT_EQ(plan.kernels.size(), 1u);
    const auto& defines = plan.kernels.front().defines;
    EXPECT_EQ(defines.at("PCIE_BASE"), std::to_string(queue.offset + queue.cq_start));
    EXPECT_EQ(defines.at("PCIE_SIZE"), std::to_string(queue.command_issue_region_size));
    EXPECT_NE(defines.at("PCIE_SIZE"), "65536");
    EXPECT_EQ(defines.at("PREFETCH_Q_ENTRY_BITS"), std::to_string(memory.prefetch_q_entry_size_bytes() * 8));
    EXPECT_EQ(defines.at("MY_DOWNSTREAM_CB_SEM_ID"), "3");
    EXPECT_EQ(defines.at("DOWNSTREAM_CB_SEM_ID"), "5");
    EXPECT_EQ(defines.at("UPSTREAM_NOC_X"), "10");
    EXPECT_EQ(defines.at("DOWNSTREAM_SUBORDINATE_NOC_Y"), "18");
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST_F(DispatchPlanTest, PreservesCompletionQueueAndEthernetVirtualization) {
    const auto plan = plan_dispatch_kernel(tt::ARCH::BLACKHOLE, dispatcher());
    const auto& defines = plan.kernels.front().defines;
    EXPECT_EQ(defines.at("COMMAND_QUEUE_BASE_ADDR"), std::to_string(queue.offset));
    EXPECT_EQ(defines.at("COMPLETION_QUEUE_SIZE"), std::to_string(queue.command_completion_region_size));
    EXPECT_EQ(defines.at("NUM_PHYSICAL_UNICAST_CORES"), "8");
    EXPECT_EQ(defines.at("NUM_VIRTUAL_UNICAST_CORES"), "10");
    EXPECT_EQ(defines.at("VIRTUALIZE_UNICAST_CORES"), "1");
    EXPECT_EQ(defines.at("NUM_WORKER_CORES_TO_MCAST"), "120");
}
TEST_F(DispatchPlanTest, RequiresAllThreeSubordinateComputeRolesOnWorker) {
    const auto plan = plan_dispatch_kernel(tt::ARCH::BLACKHOLE, subordinate());
    ASSERT_EQ(plan.kernels.size(), 2u);
    EXPECT_EQ(plan.kernels[0].processors.size(), 1u);
    ASSERT_EQ(plan.kernels[1].processors.size(), 3u);
    for (int processor = 0; processor != 3; ++processor) {
        EXPECT_EQ(
            plan.kernels[1].processors[processor],
            (HalProcessorIdentifier{HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, processor}));
    }
    EXPECT_EQ(plan.kernels[1].defines.at("NUM_WORKER_CORES"), "120");
    EXPECT_EQ(plan.kernels[1].opt_level, KernelBuildOptLevel::O3);
}
TEST_F(DispatchPlanTest, EthernetSubordinateHasNoComputeCompanion) {
    auto n = subordinate();
    n.processor = {HalProgrammableCoreType::IDLE_ETH, HalProcessorClassType::DM, 0};
    const auto plan = plan_dispatch_kernel(tt::ARCH::BLACKHOLE, n);
    ASSERT_EQ(plan.kernels.size(), 1u);
    EXPECT_EQ(plan.kernels.front().opt_level, KernelBuildOptLevel::Os);
}
TEST_F(DispatchPlanTest, QuasarSubordinateKeepsSourcePathWithoutComputeCompanion) {
    const auto plan = plan_dispatch_kernel(tt::ARCH::QUASAR, subordinate());
    ASSERT_EQ(plan.kernels.size(), 1u);
    EXPECT_EQ(plan.kernels.front().kind, DispatchSourceKind::Subordinate);
}
TEST_F(DispatchPlanTest, MissingActiveFieldRejectsWithoutDefaultSubstitution) {
    auto n = prefetch();
    std::get<PrefetchConfiguration>(n.kernel).static_config.pcie_size.reset();
    EXPECT_THROW(plan_dispatch_kernel(tt::ARCH::BLACKHOLE, n), std::invalid_argument);
}
TEST_F(DispatchPlanTest, IndependentFabricSourceRendersButCannotPublish) {
    auto n = prefetch();
    std::get<PrefetchConfiguration>(n.kernel).static_config.is_d_variant = false;
    auto plan = plan_dispatch_kernel(tt::ARCH::BLACKHOLE, n);
    EXPECT_EQ(plan.kernels.front().defines.at("FABRIC_RELAY"), "1");
    EXPECT_THROW(validate_published_dispatch_configuration(plan.configuration), std::invalid_argument);
}
TEST_F(DispatchPlanTest, OwnsConfigurationAndRendersResolvedCommonOptions) {
    auto n = prefetch();
    n.resolved->force_watcher_no_inline = true;
    n.resolved->watcher_dispatch_disabled = true;
    n.resolved->reads_dispatch_cores = false;
    auto plan = plan_dispatch_kernel(tt::ARCH::BLACKHOLE, n);
    const auto& defines = plan.kernels.front().defines;
    EXPECT_EQ(defines.at("WATCHER_NOINLINE"), "1");
    EXPECT_EQ(defines.at("FORCE_WATCHER_OFF"), "1");
    EXPECT_EQ(defines.at("FORCE_DPRINT_OFF"), "1");
    n.resolved->command_queue_size = 0x10000;
    EXPECT_EQ(plan.configuration.resolved->command_queue_size, 0x60000u);
}
}  // namespace
}  // namespace tt::tt_metal::experimental
