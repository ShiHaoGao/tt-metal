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
    return {TT_NATIVE_FIRMWARE_SDK_ROOT,
        {.noc_translation_enabled = true,
         .harvesting_masks = {.tensix_harvesting_mask = 1, .eth_harvesting_mask = 3, .pcie_harvesting_mask = 2},
         .board_type = BoardType::P150},
        0, HostQueueBacking{0, DispatchSettings::MAX_HUGEPAGE_SIZE, false, false, std::nullopt}, {}};
}
void receives(DeviceProfilerMode mode, std::optional<uint32_t> count = std::nullopt) {
    ASSERT_FALSE(MetalContext::instance_exists());
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = TT_NATIVE_FIRMWARE_SDK_ROOT, .profiler_mode = mode}};
    options.set_enable_2_erisc_mode(true);
    if (count) options.set_profiler_program_support_count(*count);
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
TEST(DeploymentCatalog, DisabledModeReceivesActualHalAndCompleteProgram) {
    receives(DeviceProfilerMode::Disabled);
}
TEST(DeploymentCatalog, ProgramModeReceivesActualProfilerAllocation) {
    receives(DeviceProfilerMode::Program);
}
TEST(DeploymentCatalog, ProgramModePreservesExplicitProfilerCapacity) {
    receives(DeviceProfilerMode::Program, 4096);
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
}
}
