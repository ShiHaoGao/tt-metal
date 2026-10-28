// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

#include "impl/profiler/profiler_state_manager.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_elffile.hpp"
#include "tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h"

namespace tt::tt_metal {
namespace {

struct WormholeFirmwarePartition {
    HalProcessorClassType processor_class;
    uint32_t processor_id;
    uint32_t base;
    uint32_t size;
};

constexpr std::array<WormholeFirmwarePartition, 5> wormhole_firmware_partitions = {{
    {HalProcessorClassType::DM, 0, MEM_BRISC_FIRMWARE_BASE, MEM_BRISC_FIRMWARE_SIZE},
    {HalProcessorClassType::DM, 1, MEM_NCRISC_FIRMWARE_BASE, MEM_NCRISC_FIRMWARE_SIZE},
    {HalProcessorClassType::COMPUTE, 0, MEM_TRISC0_FIRMWARE_BASE, MEM_TRISC0_FIRMWARE_SIZE},
    {HalProcessorClassType::COMPUTE, 1, MEM_TRISC1_FIRMWARE_BASE, MEM_TRISC1_FIRMWARE_SIZE},
    {HalProcessorClassType::COMPUTE, 2, MEM_TRISC2_FIRMWARE_BASE, MEM_TRISC2_FIRMWARE_SIZE},
}};

TEST(WormholeFirmwareLayout, RedistributingFirmwareDoesNotMoveFollowingRegions) {
    // This is the existing reserved interval, not additional L1 capacity.
    EXPECT_EQ(wormhole_firmware_partitions.front().base, 0x3a60u);
    EXPECT_EQ(uint32_t(MEM_NOC_COUNTER_BASE), 0x7280u);
    uint32_t next = wormhole_firmware_partitions.front().base;
    uint32_t total = 0;
    for (const auto& partition : wormhole_firmware_partitions) {
        EXPECT_EQ(partition.base, next);
        EXPECT_EQ(partition.base % 32, 0u);
        EXPECT_EQ(partition.size % 32, 0u);
        total += partition.size;
        next += partition.size;
    }
    EXPECT_EQ(total, 14368u);
    EXPECT_EQ(next, uint32_t(MEM_NOC_COUNTER_BASE));
}

class WormholeFirmwareBuild : public ::testing::Test {
protected:
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "wormhole_firmware_XXXXXX").string();
        const auto* directory = mkdtemp(pattern.data());
        ASSERT_NE(directory, nullptr);
        scratch = directory;
    }

    void TearDown() override {
        if (!scratch.empty()) {
            std::filesystem::remove_all(scratch);
        }
    }

    void buildImages(DeviceProfilerMode mode) {
        // Discover the SDK resource root as other JIT tests do, but isolate the
        // build cache and select profiling explicitly. No device is opened.
        llrt::RunTimeOptions inherited(mode);
        llrt::RunTimeOptions options(
            llrt::RunTimeOptions::ExplicitBuildOptions{
                .root_dir = inherited.get_root_dir(), .cache_dir = (scratch / "jit").string(), .profiler_mode = mode});
        options.set_disable_precompiled_fw(true);
        bool visited = false;
        enumerate_offline_compile_device_configs(options, [&](const JitDeviceConfig& authored) {
            if (visited || authored.arch != tt::ARCH::WORMHOLE_B0) {
                return;
            }
            visited = true;
            // Offline catalog entries normally use profiling-disabled HALs.
            const uint32_t profiler_bytes = get_profiler_dram_bank_size_for_hal_allocation(options);
            Hal hal(
                authored.arch,
                authored.routing_fw_enabled,
                options.get_enable_2_erisc_mode(),
                profiler_bytes,
                false,
                false,
                true,
                true);
            auto config = authored;
            config.hal = &hal;
            config.profiler_dram_bank_size_per_risc_bytes = profiler_bytes;
            const auto core = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
            // A rebuilt HAL and regenerated linker scripts must agree. Never
            // accept an image cohort against stale host-side launch addresses.
            for (const auto& partition : wormhole_firmware_partitions) {
                ASSERT_EQ(
                    hal.get_jit_build_config(
                           core, static_cast<uint32_t>(partition.processor_class), partition.processor_id)
                        .fw_base_addr,
                    partition.base);
            }
            BuildEnvManager manager(hal);
            manager.add_build_env(0, config, options);
            ASSERT_FALSE(manager.get_device_build_env(0).firmware_precompiled);
            jit_build_cache_clear();
            manager.build_firmware(0, /*ignore_precompiled=*/true);
            for (const auto& partition : wormhole_firmware_partitions) {
                const auto path = manager.get_firmware_binary_path(
                    0, core, static_cast<uint32_t>(partition.processor_class), partition.processor_id);
                SCOPED_TRACE(path);
                ll_api::ElfFile image;
                // Exercise the production loader's .segments budget check,
                // rather than accepting an enlarged synthetic test image.
                ASSERT_NO_THROW(image.ReadImage(path));
                ASSERT_FALSE(image.GetSegments().empty());
                const auto& text = image.GetSegments().front();
                EXPECT_EQ(text.address, partition.base);
                EXPECT_EQ(text.lma, partition.base);
                EXPECT_LE(text.membytes, partition.size);
                for (const auto& segment : image.GetSegments()) {
                    if (segment.address < MEM_BRISC_FIRMWARE_BASE || segment.address >= MEM_NOC_COUNTER_BASE) {
                        continue;
                    }
                    EXPECT_GE(segment.address, partition.base);
                    EXPECT_LE(uint64_t(segment.address) + segment.membytes, uint64_t(partition.base) + partition.size);
                }
            }
        });
        ASSERT_TRUE(visited) << "offline configuration catalog has no Wormhole target";
    }

    std::filesystem::path scratch;
};

TEST_F(WormholeFirmwareBuild, DisabledFirmwareFitsAllFivePartitions) { buildImages(DeviceProfilerMode::Disabled); }

TEST_F(WormholeFirmwareBuild, ProgramProfilerFirmwareFitsAllFivePartitions) {
    buildImages(DeviceProfilerMode::Program);
}

}  // namespace
}  // namespace tt::tt_metal
