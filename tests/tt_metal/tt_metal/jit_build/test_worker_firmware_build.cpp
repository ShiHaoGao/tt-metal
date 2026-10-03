// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "hostdev/worker_stream_state_contract.h"
#include "impl/profiler/profiler_state_manager.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_device_config.hpp"
#include "jit_build/precompiled.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_elffile.hpp"
#include "llrt/tt_memory.h"

namespace tt::tt_metal {
namespace {

// No MetalContext, mock device, mesh or device open: both key construction and
// the real firmware compilation consume the existing offline configuration API.
class WorkerFirmwareBuild : public ::testing::Test {
protected:
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "worker_firmware_XXXXXX").string();
        const auto* directory = mkdtemp(pattern.data());
        ASSERT_NE(directory, nullptr);
        scratch = directory;
        if (const auto* prior = std::getenv("TT_METAL_CACHE")) previous_cache = prior;
        ASSERT_EQ(setenv("TT_METAL_CACHE", (scratch / "jit").c_str(), 1), 0);
    }
    void TearDown() override {
        if (previous_cache) setenv("TT_METAL_CACHE", previous_cache->c_str(), 1);
        else unsetenv("TT_METAL_CACHE");
        if (!scratch.empty()) std::filesystem::remove_all(scratch);
    }

    void withBlackhole(
        llrt::RunTimeOptions& options,
        const std::function<void(const JitDeviceConfig&)>& action) {
        bool visited = false;
        enumerate_offline_compile_device_configs(options, [&](const JitDeviceConfig& authored) {
            if (visited || authored.arch != tt::ARCH::BLACKHOLE) return;
            visited = true;
            // The ordinary precompile enumerator intentionally disables profiling.
            // Recreate its HAL with the same topology and the requested profiler
            // allocation before constructing actual firmware build states.
            const uint32_t profiler_bytes = options.get_profiler_enabled()
                ? get_profiler_dram_bank_size_per_risc_bytes(options) : 0;
            Hal hal(authored.arch, authored.routing_fw_enabled, options.get_enable_2_erisc_mode(),
                    profiler_bytes, false, false, true, true);
            auto config = authored;
            config.hal = &hal;
            config.profiler_dram_bank_size_per_risc_bytes = profiler_bytes;
            action(config);
        });
        ASSERT_TRUE(visited) << "offline configuration catalog has no Blackhole target";
    }

    void checkImage(const std::string& path, worker_stream_state::FirmwareRole role) {
        SCOPED_TRACE(path);
        ll_api::ElfFile elf;
        elf.ReadImage(path);
        uint64_t address = 0;
        const auto bytes = elf.GetSectionContents(".tt_worker_stream_state", address);
        ASSERT_EQ(bytes.size(), sizeof(worker_stream_state::ImageRecord));
        worker_stream_state::ImageRecord record{};
        std::memcpy(&record, bytes.data(), sizeof(record));
        EXPECT_TRUE(worker_stream_state::acceptsImage(record, role));
        // The section is host metadata and must never overlap a loaded segment.
        for (const auto& segment : elf.GetSegments()) {
            const uint64_t end = uint64_t(segment.address) + segment.membytes;
            EXPECT_TRUE(address + bytes.size() <= segment.address || address >= end);
        }
        // Also exercise the actual loader's retained-record decoder.
        const ll_api::memory selected(path, ll_api::memory::Loading::DISCRETE);
        EXPECT_TRUE(worker_stream_state::acceptsImage(selected.worker_stream_state_image(), role));
    }

    void buildImages(DeviceProfilerMode mode) {
        llrt::RunTimeOptions options(mode);
        EXPECT_EQ(options.get_profiler_enabled(), mode == DeviceProfilerMode::Program);
        options.set_disable_precompiled_fw(true);
        withBlackhole(options, [&](const JitDeviceConfig& config) {
            BuildEnvManager manager(*config.hal);
            manager.add_build_env(0, config, options);
            const auto& before = manager.get_device_build_env(0);
            ASSERT_FALSE(before.firmware_precompiled);
            ASSERT_EQ(before.build_env.get_firmware_binary_root(),
                      before.build_env.get_out_firmware_root_path());
            jit_build_cache_clear();
            manager.build_firmware(0, /*ignore_precompiled=*/true);
            const auto core = config.hal->get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
            checkImage(manager.get_firmware_binary_path(
                           0, core, static_cast<uint32_t>(HalProcessorClassType::DM), 0),
                       worker_stream_state::FirmwareRole::Brisc);
            checkImage(manager.get_firmware_binary_path(
                           0, core, static_cast<uint32_t>(HalProcessorClassType::COMPUTE), 0),
                       worker_stream_state::FirmwareRole::Trisc0);
        });
    }

    std::filesystem::path scratch;
    std::optional<std::string> previous_cache;
};

TEST_F(WorkerFirmwareBuild, ImplicitAbiUsesCanonicalJitAndPrecompiledIdentity) {
    llrt::RunTimeOptions options(DeviceProfilerMode::Disabled);
    withBlackhole(options, [&](const JitDeviceConfig& config) {
        JitBuildEnv implicit_abi;
        JitBuildEnv explicit_abi;
        constexpr uint64_t topology_key = 17;
        implicit_abi.init(topology_key, config, options, {});
        explicit_abi.init(topology_key, config, options,
                          {{"TT_WORKER_STREAM_STATE_ABI_VERSION", std::to_string(worker_stream_state::kVersion)}});
        EXPECT_EQ(implicit_abi.get_build_key(), explicit_abi.get_build_key());
        EXPECT_EQ(implicit_abi.get_out_firmware_root_path(), explicit_abi.get_out_firmware_root_path());
        EXPECT_EQ(implicit_abi.get_out_kernel_root_path(), explicit_abi.get_out_kernel_root_path());
        // Drive the same directory resolver used by add_build_env_locked with
        // the actual computed key, without recreating the private hash algorithm.
        const auto bundle = scratch / "bundle";
        const auto selected = bundle / "tt_metal/pre-compiled" / std::to_string(explicit_abi.get_build_key());
        std::filesystem::create_directories(selected);
        const auto result = precompiled::find_precompiled_dir(bundle.string() + "/", implicit_abi.get_build_key());
        ASSERT_TRUE(result);
        EXPECT_EQ(std::filesystem::path(*result).lexically_normal(), (selected / "").lexically_normal());
    });
}

TEST_F(WorkerFirmwareBuild, CallerCannotOverrideCanonicalWorkerAbi) {
    llrt::RunTimeOptions options(DeviceProfilerMode::Disabled);
    withBlackhole(options, [&](const JitDeviceConfig& config) {
        for (const auto& value : {std::string("0"), std::to_string(unsigned(worker_stream_state::kVersion) + 1)}) {
            SCOPED_TRACE(value);
            JitBuildEnv environment;
            EXPECT_THROW(environment.init(17, config, options, {{"TT_WORKER_STREAM_STATE_ABI_VERSION", value}}),
                         std::runtime_error);
        }
    });
}

TEST_F(WorkerFirmwareBuild, OfflineDisabledImagesCarryLoaderRecords) {
    buildImages(DeviceProfilerMode::Disabled);
}

TEST_F(WorkerFirmwareBuild, OfflineProgramProfilerImagesCarryLoaderRecords) {
    buildImages(DeviceProfilerMode::Program);
}

}  // namespace
}  // namespace tt::tt_metal
