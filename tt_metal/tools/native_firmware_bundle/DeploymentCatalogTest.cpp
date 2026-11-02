#include "DeploymentCatalog.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_settings.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include <gtest/gtest.h>
#ifndef TT_NATIVE_FIRMWARE_SDK_ROOT
#error "Catalog tests require an explicit SDK source root"
#endif
namespace tt::tt_metal::experimental::catalog {
namespace {
Input input() {
    return {
        TT_NATIVE_FIRMWARE_SDK_ROOT,
        tt::ARCH::BLACKHOLE,
        {.noc_translation_enabled = true,
         .harvesting_masks = {.tensix_harvesting_mask = 1, .eth_harvesting_mask = 3, .pcie_harvesting_mask = 2},
         .board_type = BoardType::P150},
        0,
        HostQueueBacking{0, DispatchSettings::MAX_HUGEPAGE_SIZE, false, false, std::nullopt},
        {}};
}
void receives(DeviceProfilerMode mode, std::optional<uint32_t> count = std::nullopt) {
    ASSERT_FALSE(MetalContext::instance_exists());
    llrt::RunTimeOptions options{
        llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = TT_NATIVE_FIRMWARE_SDK_ROOT, .profiler_mode = mode}};
    options.set_enable_2_erisc_mode(true);
    if (count) {
        options.set_profiler_program_support_count(*count);
    }
    const uint32_t bytes = get_profiler_dram_bank_size_for_hal_allocation(options);
    Hal hal{tt::ARCH::BLACKHOLE, false, true, bytes, false, false, true, true};
    const auto actual = make_input(input(), hal, options);
    EXPECT_EQ(actual.device.profiler_dram_bank_size_per_risc_bytes, bytes);
    const auto plan = plan_dispatch_program(actual.program);
    EXPECT_NO_THROW({
        const auto deployment = DeploymentConfiguration::from_sdk(actual.device, options, plan);
        EXPECT_EQ(deployment.profiler_mode(), mode);
        EXPECT_EQ(deployment.device().profiler_dram_bank_size_per_risc_bytes, bytes);
    });
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST(DeploymentCatalog, DisabledModeReceivesActualHalAndCompleteProgram) { receives(DeviceProfilerMode::Disabled); }
TEST(DeploymentCatalog, ProgramModeReceivesActualProfilerAllocation) { receives(DeviceProfilerMode::Program); }
TEST(DeploymentCatalog, ProgramModePreservesExplicitProfilerCapacity) { receives(DeviceProfilerMode::Program, 4096); }
TEST(DeploymentCatalog, ReceivesWormholeN300ActualSocHalAndSingleMmioProgram) {
    ASSERT_FALSE(MetalContext::instance_exists());
    for (const auto mode : {DeviceProfilerMode::Disabled, DeviceProfilerMode::Program}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        llrt::RunTimeOptions options{
            llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = TT_NATIVE_FIRMWARE_SDK_ROOT, .profiler_mode = mode}};
        options.set_enable_2_erisc_mode(false);
        const auto bytes = get_profiler_dram_bank_size_for_hal_allocation(options);
        Hal hal{tt::ARCH::WORMHOLE_B0, true, false, bytes, false, false, false, false};
        auto facts = input();
        facts.architecture = tt::ARCH::WORMHOLE_B0;
        facts.device_id = 7;
        facts.chip = {
            .noc_translation_enabled = true,
            .harvesting_masks = {.tensix_harvesting_mask = 0x201},
            .board_type = BoardType::N300};
        const auto arch = std::make_shared<tt::umd::SocArchDescriptor>(
            (facts.sdk_root / "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml").string());
        const tt::umd::SocDescriptor soc(arch, facts.chip);
        const metal_SocDescriptor metal(soc, facts.chip.board_type);
        Output actual{};
        ASSERT_NO_THROW(actual = make_input(facts, hal, options));
        EXPECT_EQ(actual.device.arch, tt::ARCH::WORMHOLE_B0);
        EXPECT_EQ(actual.device.dispatch_core_axis, DispatchCoreAxis::ROW);
        EXPECT_EQ(actual.device.num_hw_cqs, 1u);
        EXPECT_TRUE(actual.device.routing_fw_enabled);
        EXPECT_EQ(actual.device.harvesting_mask, soc.harvesting_masks.tensix_harvesting_mask);
        EXPECT_EQ(actual.device.num_dram_banks, metal.get_num_dram_views());
        EXPECT_EQ(actual.device.profiler_dram_bank_size_per_risc_bytes, bytes);
        const auto& topology = actual.program.topology;
        EXPECT_EQ(topology.arch, tt::ARCH::WORMHOLE_B0);
        EXPECT_EQ(topology.device, facts.device_id);
        EXPECT_EQ(topology.deployment_devices, std::vector<ChipId>{facts.device_id});
        EXPECT_EQ(topology.serviced_devices, std::vector<ChipId>{facts.device_id});
        EXPECT_TRUE(topology.mmio);
        const auto plan = plan_dispatch_program(actual.program);
        const auto deployment = DeploymentConfiguration::from_sdk(actual.device, options, plan);
        EXPECT_EQ(deployment.profiler_mode(), mode);
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST(DeploymentCatalog, PreservesWormholeCoordinatesAcrossTranslationAndHarvestingStates) {
    ASSERT_FALSE(MetalContext::instance_exists());
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = TT_NATIVE_FIRMWARE_SDK_ROOT, .profiler_mode = DeviceProfilerMode::Disabled}};
    options.set_enable_2_erisc_mode(false);
    Hal hal{tt::ARCH::WORMHOLE_B0, true, false, 0, false, false, false, true};
    for (bool translated : {false, true}) {
        SCOPED_TRACE(translated);
        for (uint32_t mask : {0u, 1u, 0x201u}) {
            SCOPED_TRACE(mask);
            auto facts = input();
            facts.architecture = tt::ARCH::WORMHOLE_B0;
            facts.chip = {
                .noc_translation_enabled = translated,
                .harvesting_masks = {.tensix_harvesting_mask = mask},
                .board_type = BoardType::N300};
            const auto arch = std::make_shared<tt::umd::SocArchDescriptor>(
                (facts.sdk_root / "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml").string());
            const tt::umd::SocDescriptor soc(arch, facts.chip);
            Output actual{};
            ASSERT_NO_THROW(actual = make_input(facts, hal, options));
            const auto grid = soc.get_grid_size(CoreType::TENSIX);
            EXPECT_EQ(actual.program.workers.count, grid.x * (grid.y - 1));
            EXPECT_EQ(actual.device.num_l1_banks, actual.program.workers.count);
            ASSERT_EQ(actual.program.topology.nodes.size(), 3u);
            EXPECT_EQ(actual.program.topology.nodes[0].core.logical, CoreCoord(0, grid.y - 1));
            EXPECT_EQ(actual.program.topology.nodes[1].core.logical, CoreCoord(1, grid.y - 1));
            for (const auto& node : actual.program.topology.nodes) {
                const auto expected = soc.translate_coord_to(
                    {node.core.logical.x, node.core.logical.y, CoreType::TENSIX, CoordSystem::LOGICAL},
                    CoordSystem::TRANSLATED);
                const CoreCoord noc0{expected.x, expected.y};
                const CoreCoord noc1 = translated ? noc0
                                                  : CoreCoord{
                                                        hal.noc_coordinate(NOC_1, soc.grid_size.x, noc0.x),
                                                        hal.noc_coordinate(NOC_1, soc.grid_size.y, noc0.y)};
                EXPECT_EQ(node.core.virtual_core, noc0);
                EXPECT_EQ(node.core.noc[0], noc0);
                EXPECT_EQ(node.core.noc[1], noc1);
            }
            const auto lo =
                soc.translate_coord_to({0, 0, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED);
            const auto hi = soc.translate_coord_to(
                {grid.x - 1, grid.y - 2, CoreType::TENSIX, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED);
            EXPECT_EQ(actual.program.workers.multicast[0], hal.noc_multicast_encoding(lo.x, lo.y, hi.x, hi.y));
            const auto end_x = translated ? hi.x : hal.noc_coordinate(NOC_1, soc.grid_size.x, hi.x);
            const auto end_y = translated ? hi.y : hal.noc_coordinate(NOC_1, soc.grid_size.y, hi.y);
            const auto start_x = translated ? lo.x : hal.noc_coordinate(NOC_1, soc.grid_size.x, lo.x);
            const auto start_y = translated ? lo.y : hal.noc_coordinate(NOC_1, soc.grid_size.y, lo.y);
            EXPECT_EQ(actual.program.workers.multicast[1], hal.noc_multicast_encoding(end_x, end_y, start_x, start_y));
        }
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST(DeploymentCatalog, RejectsMismatchedArchitectureBoardAndEriscTuples) {
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = TT_NATIVE_FIRMWARE_SDK_ROOT, .profiler_mode = DeviceProfilerMode::Disabled}};
    options.set_enable_2_erisc_mode(false);
    Hal hal{tt::ARCH::WORMHOLE_B0, true, false, 0, false, false, false, true};
    const auto rejects = [&](const Input& facts, const char* diagnostic) {
        try {
            (void)make_input(facts, hal, options);
            FAIL() << "catalog accepted an invalid architecture tuple";
        } catch (const std::invalid_argument& error) {
            EXPECT_STREQ(error.what(), diagnostic);
        }
    };
    auto facts = input();
    rejects(facts, "catalog helper requires a matching supported architecture and host CQ");
    facts.architecture = tt::ARCH::WORMHOLE_B0;
    rejects(facts, "catalog requires official Wormhole N300 chip facts");
    facts.chip = {
        .noc_translation_enabled = true,
        .harvesting_masks = {.tensix_harvesting_mask = 1},
        .board_type = BoardType::N300};
    options.set_enable_2_erisc_mode(true);
    rejects(facts, "catalog ERISC mode differs from official architecture policy");
    options.set_enable_2_erisc_mode(false);
    facts.chip.harvesting_masks.tensix_harvesting_mask = 0x400;
    rejects(facts, "catalog requires official Wormhole N300 chip facts");
    facts.chip.harvesting_masks.tensix_harvesting_mask = 7;
    rejects(facts, "unsupported official core catalog harvesting count");
    facts.chip.harvesting_masks.tensix_harvesting_mask = 1;
    facts.chip.harvesting_masks.dram_harvesting_mask = 1;
    rejects(facts, "catalog requires official Wormhole N300 chip facts");
}
TEST(DeploymentCatalog, PreservesActualChipHarvestingAndTranslatedPcie) {
    ASSERT_FALSE(MetalContext::instance_exists());
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = TT_NATIVE_FIRMWARE_SDK_ROOT, .profiler_mode = DeviceProfilerMode::Disabled}};
    options.set_enable_2_erisc_mode(true);
    Hal hal{tt::ARCH::BLACKHOLE, false, true, 0, false, false, true, true};
    for (uint32_t mask : {1u, 2u}) {
        SCOPED_TRACE(mask);
        auto facts = input();
        facts.chip.harvesting_masks.tensix_harvesting_mask = mask;
        const auto arch = std::make_shared<tt::umd::SocArchDescriptor>(
            (facts.sdk_root / "tt_metal/soc_descriptors/blackhole_140_arch.yaml").string());
        const tt::umd::SocDescriptor soc(arch, facts.chip);
        const metal_SocDescriptor metal(soc, facts.chip.board_type);
        const auto pcie = metal.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
        ASSERT_FALSE(pcie.empty());
        const CoreCoord expected_pcie{pcie.front().x, pcie.front().y};
        const auto actual = make_input(facts, hal, options);
        EXPECT_EQ(actual.device.harvesting_mask, soc.harvesting_masks.tensix_harvesting_mask);
        EXPECT_EQ(actual.device.pcie_core, expected_pcie);
        EXPECT_EQ(actual.device.num_dram_banks, metal.get_num_dram_views());
        auto expected = actual.device;
        expected.harvesting_mask = soc.harvesting_masks.tensix_harvesting_mask;
        expected.pcie_core = expected_pcie;
        expected.num_dram_banks = metal.get_num_dram_views();
        const auto plan = plan_dispatch_program(actual.program);
        const auto stored = DeploymentConfiguration::from_sdk(actual.device, options, plan);
        EXPECT_TRUE(stored.matches(DeploymentConfiguration::from_sdk(expected, options, plan)));
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}
}  // namespace
}  // namespace tt::tt_metal::experimental::catalog
