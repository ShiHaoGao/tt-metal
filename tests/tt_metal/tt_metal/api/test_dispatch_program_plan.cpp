// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "impl/experimental/published_deployment/dispatch_program_plan.hpp"
#include "impl/experimental/published_deployment/dispatch_program_adapter.hpp"
#include "impl/dispatch/dispatch_settings.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"
#include <hostdevcommon/dispatch_telemetry_types.hpp>
#include "tools/native_firmware_bundle/DeploymentCatalog.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <limits>
namespace tt::tt_metal::experimental {
namespace {
DispatchCorePlacement placement(uint32_t x, uint32_t y) {
    return {{x, y}, {x + 16, y + 16}, {{{x + 16, y + 16}, {31 - x, 31 - y}}}};
}
DispatchProgramInputs inputs() {
    DispatchProgramInputs i{};
    auto& t = i.topology;
    t.arch = tt::ARCH::BLACKHOLE;
    t.device = 7;
    t.deployment_devices = {7}; t.serviced_devices = {7}; t.mmio = true;
    t.core_type = CoreType::WORKER; t.num_hw_cqs = 1; t.subordinate_enabled = true;
    t.nodes = {
        {10, 7, 7, 0, PREFETCH_HD, {}, {20, 30}, {NOC_0, NOC_0, NOC_0}, placement(1, 1)},
        {20, 7, 7, 0, DISPATCH_HD, {10}, {30}, {NOC_0, NOC_1, NOC_0}, placement(2, 1)},
        {30, 7, 7, 0, DISPATCH_S, {10}, {20}, {NOC_1, NOC_1, NOC_1}, placement(2, 1)}};
    t.completion_writer = placement(2, 1); t.unused_core = placement(0, 0);
    i.workers = {120, {0x10111213, 0x20212223}, 3, 4};
    i.queue.inputs = {HostQueueBacking{5, 0x40000000, false, false, std::nullopt}, 1, 0x1000, 32, 0x1000};
    i.queue.plan = plan_system_memory_queues(i.queue.inputs);
    i.hal = {16, 0x200000, 0x100, 0x200, {0, 4, 8, 12, 16, 32, 48, 64, 80}};
    i.memory = {0x10000, 0x1000, 4, 0x4000, 0x4004, 0x20000, 0x10000,
        0x30000, 0x10000, 0x1000, 32, 0x40000, 64, 0x1000, 16,
        0x4010, 0x4014, 0x4018, 0x401c, 0x10, 0x4020, 0x5000, {},
        0, 0, 0x6000, 0x7000, 0x8000, 48, 0};
    for (uint32_t n = 0; n < DispatchSettings::DISPATCH_MESSAGE_ENTRIES; ++n)
        i.memory.sync_offsets.push_back(i.hal.l1_alignment * n);
    return i;
}
uint32_t word(const DispatchInitialization& init) {
    EXPECT_EQ(init.bytes.size(), sizeof(uint32_t));
    uint32_t n = 0; std::memcpy(&n, init.bytes.data(), std::min(init.bytes.size(), sizeof(n))); return n;
}
const DispatchInitialization& action(const DispatchProgramPlan& p, DispatchInitializationKind kind) {
    auto at = std::ranges::find(p.initialization(), kind, &DispatchInitialization::kind);
    if (at == p.initialization().end()) throw std::logic_error("missing action");
    return *at;
}
void rejects(const DispatchProgramInputs& i, DispatchProgramReject reason) {
    try { (void)plan_dispatch_program(i); FAIL() << "accepted invalid input"; }
    catch (const DispatchProgramError& e) { EXPECT_EQ(e.reason(), reason) << e.what(); }
}
TEST(DispatchProgramPlan, PreservesActualGraphPlacementAndEveryProcessor) {
    ASSERT_FALSE(MetalContext::instance_exists());
    auto i = inputs(); auto p = plan_dispatch_program(i);
    ASSERT_EQ(p.dispatch_nodes().size(), 3u);
    ASSERT_EQ(p.kernel_plans().size(), 3u);
    for (const auto& n : p.dispatch_nodes()) {
        EXPECT_EQ(n.device_id, 7u); EXPECT_EQ(n.resolved->worker_count, 120u);
        EXPECT_NO_THROW(validate_published_dispatch_configuration(n));
    }
    const auto& subordinate = p.kernel_plans()[2];
    ASSERT_EQ(subordinate.kernels.size(), 2u);
    EXPECT_EQ(subordinate.kernels[1].processors.size(), 3u);
    EXPECT_EQ(subordinate.configuration.logical_core, CoreCoord(2, 1));
    EXPECT_EQ(p.core_registrations().size(), 2u);
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST(DispatchProgramPlan, AllocatesOneSharedCoreSemaphoreNamespaceAndPreservesTraversal) {
    auto i = inputs(); auto p = plan_dispatch_program(i);
    ASSERT_EQ(p.semaphores().size(), 6u);
    std::vector<uint32_t> ids; for (auto& s : p.semaphores()) ids.push_back(s.id);
    EXPECT_EQ(ids, (std::vector<uint32_t>{0, 1, 2, 0, 1, 2}));
    EXPECT_EQ(p.semaphores()[0].initial_value, 64u);
    EXPECT_EQ(p.semaphores()[2].initial_value, 16u);
    auto addresses = p.semaphore_addresses(0x1000, 0x80);
    EXPECT_EQ(addresses[5].address, 0x10a0u);
    std::swap(i.topology.nodes[1], i.topology.nodes[2]);
    auto reordered = plan_dispatch_program(i);
    const auto& dispatcher = std::get<DispatcherConfiguration>(reordered.dispatch_nodes()[2].kernel);
    EXPECT_EQ(dispatcher.static_config.my_dispatch_cb_sem_id, 2u);
    EXPECT_EQ(dispatcher.dependent_config.dispatch_d_shutdown_sem_id, 1u);
}
TEST(DispatchProgramPlan, UsesActualCqPlanForHostAndDramAndInitializesRuntimeWords) {
    for (bool dram : {false, true}) {
        auto i = inputs();
        if (dram) { i.queue.inputs.backing = DramQueueBacking{0x100000, 0x8000000}; i.queue.dram_bank = 3; }
        i.queue.plan = plan_system_memory_queues(i.queue.inputs);
        auto p = plan_dispatch_program(i); auto q = i.queue.plan.queues.front();
        EXPECT_EQ(word(action(p, DispatchInitializationKind::CompletionRead)),
            (q.device_offset + q.cq_start + q.command_issue_region_size) >> 4);
        EXPECT_EQ(word(action(p, DispatchInitializationKind::PrefetchPcieRead)), q.device_offset + q.cq_start);
        EXPECT_EQ(action(p, DispatchInitializationKind::PrefetchQueue).bytes.size(), i.memory.prefetch_q_size);
        EXPECT_EQ(p.runtime_arguments().size(), 2u);
        for (const auto& r : p.runtime_arguments()) EXPECT_EQ(r.words, (std::vector<uint32_t>{0, 0, 0}));
        EXPECT_EQ(p.dispatch_nodes().front().resolved->dram_backed_cq, dram);
        EXPECT_EQ(p.dispatch_nodes().front().resolved->dram_bank, dram ? 3u : 0u);
    }
}
TEST(DispatchProgramPlan, PreservesNineOrderedProfilerWritesWithoutClearingOtherFields) {
    auto i = inputs(); auto p = plan_dispatch_program(i); std::vector<uint32_t> addresses;
    for (const auto& a : p.initialization()) if (a.kind == DispatchInitializationKind::RealtimeProfiler) {
        addresses.push_back(a.address); EXPECT_EQ(word(a), 0u);
    }
    ASSERT_EQ(addresses.size(), 9u);
    for (size_t j = 0; j != addresses.size(); ++j) EXPECT_EQ(addresses[j], 0x8000 + i.hal.realtime_profiler_reset_offsets[j]);
}
TEST(DispatchProgramPlan, OwnsInputAndIncludesInitRuntimeAndSemanticsInCanonicalRecord) {
    auto i = inputs(); auto p = plan_dispatch_program(i);
    auto record = std::vector<std::byte>(p.canonical_record().begin(), p.canonical_record().end());
    i.memory.realtime_profiler += 0x100;
    auto changed = plan_dispatch_program(i);
    EXPECT_FALSE(std::ranges::equal(record, changed.canonical_record()));
    EXPECT_TRUE(std::ranges::equal(record, p.canonical_record()));
    EXPECT_EQ(action(p, DispatchInitializationKind::RealtimeProfiler).address, 0x8000u);
    i = inputs(); i.hal.l1_alignment = 32;
    for (size_t j = 0; j != i.memory.sync_offsets.size(); ++j) i.memory.sync_offsets[j] = j * 32;
    EXPECT_FALSE(std::ranges::equal(record, plan_dispatch_program(i).canonical_record()));
    i = inputs(); i.observation.telemetry_disabled = true;
    EXPECT_FALSE(std::ranges::equal(record, plan_dispatch_program(i).canonical_record()));
}
TEST(DispatchProgramPlan, RejectsUnsupportedFamiliesAndInvalidGraphs) {
    auto i = inputs(); i.topology.arch = tt::ARCH::WORMHOLE_B0; rejects(i, DispatchProgramReject::Architecture);
    i = inputs(); i.topology.num_hw_cqs = 2; rejects(i, DispatchProgramReject::Topology);
    i = inputs(); i.topology.fabric = true; rejects(i, DispatchProgramReject::Topology);
    i = inputs(); i.topology.nodes.pop_back(); rejects(i, DispatchProgramReject::Topology);
    i = inputs(); i.topology.nodes[1].downstream = {10}; rejects(i, DispatchProgramReject::Topology);
    i = inputs(); i.topology.nodes[2].core = placement(3, 1); rejects(i, DispatchProgramReject::Placement);
    i = inputs(); i.topology.nodes[0].core = i.topology.nodes[1].core; rejects(i, DispatchProgramReject::Placement);
    i = inputs(); i.observation.device_print_enabled = true; rejects(i, DispatchProgramReject::Observation);
    i = inputs(); i.workers.count = 0; rejects(i, DispatchProgramReject::Placement);
    i = inputs(); i.queue.plan.queues.front().device_offset += 16; rejects(i, DispatchProgramReject::Queue);
    i = inputs(); i.memory.prefetch_q_base = std::numeric_limits<uint32_t>::max(); rejects(i, DispatchProgramReject::Overflow);
    i = inputs(); i.topology.completion_writer = placement(6, 6); rejects(i, DispatchProgramReject::Placement);
    i = inputs(); i.topology.nodes[0].nocs.upstream_noc = NOC_1; rejects(i, DispatchProgramReject::Placement);
    i = inputs(); i.topology.nodes[0].tunnel_index = 0; rejects(i, DispatchProgramReject::Topology);
    i = inputs(); i.topology.serviced_devices.push_back(8); rejects(i, DispatchProgramReject::Topology);
    i = inputs(); i.topology.mock = true; rejects(i, DispatchProgramReject::Topology);
    i = inputs(); i.topology.nodes[2].role = FABRIC_MUX; rejects(i, DispatchProgramReject::Topology);
    i = inputs(); i.queue.dram_bank = 0; rejects(i, DispatchProgramReject::Queue);
    i = inputs(); i.memory.sync_offsets[1] += 4; rejects(i, DispatchProgramReject::MemoryMap);
    i = inputs(); i.memory.sync_sem_base = i.hal.l1_size; rejects(i, DispatchProgramReject::MemoryMap);
    i = inputs(); i.hal.realtime_profiler_reset_offsets[8] = 0; rejects(i, DispatchProgramReject::MemoryMap);
}
TEST(DispatchProgramPlan, ConsumesActualHalAndDispatchMemMapWithoutContextOrDevice) {
    ASSERT_FALSE(MetalContext::instance_exists());
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = "/tmp"}};
    Hal hal{tt::ARCH::BLACKHOLE, false, true, 0, false, false, true, true};
    DispatchMemMap memory{CoreType::WORKER, 1, hal, false, {false, 1}, options};
    auto i = inputs();
    i.hal = capture_dispatch_hal(hal);
    i.memory = capture_dispatch_memory(memory, 0);
    i.queue.inputs.cq_start = memory.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
    i.queue.inputs.alignment = hal.get_alignment(HalMemType::HOST);
    i.queue.plan = plan_system_memory_queues(i.queue.inputs);
    auto p = plan_dispatch_program(i);
    const auto& pref = std::get<PrefetchConfiguration>(p.dispatch_nodes()[0].kernel);
    EXPECT_EQ(pref.static_config.prefetch_q_base,
        memory.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED, 0));
    const auto& sub = std::get<SubordinateConfiguration>(p.dispatch_nodes()[2].kernel);
    EXPECT_EQ(sub.static_config.cb_base, memory.dispatch_buffer_base(0) +
        (memory.dispatch_buffer_pages() << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE));
    auto telemetry = action(p, DispatchInitializationKind::PrefetchTelemetry);
    dispatch_telemetry_types::PrefetchCoreTelemetry expected{};
    ASSERT_EQ(telemetry.bytes.size(), sizeof(expected));
    EXPECT_EQ(std::memcmp(telemetry.bytes.data(), &expected, sizeof(expected)), 0);
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST(DispatchProgramPlan, RejectsOverflowingOrMisalignedFinalSemaphoreAddress) {
    auto p = plan_dispatch_program(inputs());
    EXPECT_THROW(p.semaphore_addresses(0xffffffff, 0), DispatchProgramError);
    EXPECT_THROW(p.semaphore_addresses(0x1000, 4), DispatchProgramError);
    EXPECT_THROW(p.semaphore_addresses(0x200000, 0), DispatchProgramError);
}
TEST(DispatchProgramPlan, OfficialCatalogProducesExplicitOfflineCandidateWithoutLiveContext) {
    ASSERT_FALSE(MetalContext::instance_exists());
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = TT_NATIVE_FIRMWARE_SDK_ROOT}};
    options.set_enable_2_erisc_mode(true);
    Hal hal{tt::ARCH::BLACKHOLE, false, true, 0, false, false, true, true};
    catalog::Input input{
        TT_NATIVE_FIRMWARE_SDK_ROOT,
        {.noc_translation_enabled = true, .harvesting_masks = {.tensix_harvesting_mask = 1, .eth_harvesting_mask = 3, .pcie_harvesting_mask = 2}, .board_type = BoardType::P150},
        0, HostQueueBacking{0, DispatchSettings::MAX_HUGEPAGE_SIZE, false, false, std::nullopt}, {}};
    const auto output = catalog::make_input(input, hal, options);
    EXPECT_EQ(output.device.num_l1_banks, 120u);
    EXPECT_EQ(output.device.dispatch_core_axis, DispatchCoreAxis::COL);
    EXPECT_EQ(output.program.topology.nodes[0].core.logical, CoreCoord(12, 0));
    EXPECT_EQ(output.program.topology.nodes[1].core.logical, CoreCoord(12, 1));
    EXPECT_EQ(output.program.workers.count, 120u);
    EXPECT_GT(output.program.queue.plan.cq_size, 65536u);
    EXPECT_NO_THROW(DeploymentConfiguration::from_sdk(output.device, options,
        plan_dispatch_program(output.program)));
    EXPECT_FALSE(MetalContext::instance_exists());
}
} // namespace
} // namespace tt::tt_metal::experimental
