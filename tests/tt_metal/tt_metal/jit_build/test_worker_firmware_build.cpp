// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <elf.h>
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
#include "hostdev/native_kernel_contract.h"
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
            const uint32_t profiler_bytes = get_profiler_dram_bank_size_for_hal_allocation(options);
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

    void checkProfileDataBuffer(const std::string& path, const Hal& hal) {
        SCOPED_TRACE(path);
        ll_api::ElfFile elf;
        elf.ReadImage(path);
        uint64_t address = 0;
        const auto symbols = elf.GetSectionContents(".symtab", address);
        const auto strings = elf.GetSectionContents(".strtab", address);
        ASSERT_FALSE(symbols.empty());
        ASSERT_FALSE(strings.empty());
        const auto core = HalProgrammableCoreType::TENSIX;
        const auto expected = hal.get_dev_addr(core, HalL1MemAddrType::PROFILER) +
            hal.get_dev_msgs_factory(core).offset_of<dev_msgs::profiler_msg_t>(dev_msgs::profiler_msg_t::Field::buffer);
        for (size_t offset = 0; offset + sizeof(Elf32_Sym) <= symbols.size(); offset += sizeof(Elf32_Sym)) {
            Elf32_Sym symbol{};
            std::memcpy(&symbol, symbols.data() + offset, sizeof(symbol));
            ASSERT_LT(symbol.st_name, strings.size());
            const auto* name = reinterpret_cast<const char*>(strings.data() + symbol.st_name);
            if (std::strcmp(name, "_ZN15kernel_profiler20profiler_data_bufferE") != 0) continue;
            ASSERT_EQ(ELF32_ST_TYPE(symbol.st_info), STT_OBJECT);
            ASSERT_EQ(symbol.st_size, sizeof(uint32_t));
            ASSERT_NE(symbol.st_shndx, SHN_UNDEF);
            for (const auto& segment : elf.GetSegments()) {
                if (symbol.st_value < segment.address) continue;
                const size_t byte_offset = symbol.st_value - segment.address;
                if (byte_offset + sizeof(uint32_t) > segment.contents.size_bytes()) continue;
                uint32_t pointer = 0;
                std::memcpy(&pointer, reinterpret_cast<const std::byte*>(segment.contents.data()) + byte_offset,
                            sizeof(pointer));
                EXPECT_EQ(pointer, expected);
                return;
            }
            FAIL() << "profiler data pointer has no loaded initialization bytes";
        }
        FAIL() << "profile firmware lacks resident profiler_data_buffer";
    }

    void checkNativeImage(const std::string& path, uint8_t role, DeviceProfilerMode mode) {
        SCOPED_TRACE(path);
        ll_api::ElfFile elf;
        elf.ReadImage(path);
        uint64_t address = 0;
        const auto bytes = elf.GetSectionContents(TT_NATIVE_IMAGE_SECTION, address);
        ASSERT_EQ(bytes.size(), sizeof(tt_native_image_record));
        tt_native_image_record record{};
        std::memcpy(&record, bytes.data(), sizeof(record));
        EXPECT_EQ(record.magic, TT_NATIVE_IMAGE_MAGIC);
        EXPECT_EQ(record.version, TT_NATIVE_IMAGE_ABI_VERSION);
        EXPECT_EQ(record.kind, TT_NATIVE_IMAGE_FIRMWARE);
        EXPECT_EQ(record.role, role);
        EXPECT_EQ(record.architecture, TT_NATIVE_ARCH_BLACKHOLE);
        EXPECT_EQ(record.loading, TT_NATIVE_LOADING_DISCRETE);
        EXPECT_EQ(record.profile, mode == DeviceProfilerMode::Program
            ? TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM : TT_NATIVE_PROFILE_ABSENT);
        EXPECT_EQ(record.print, TT_NATIVE_PRINT_TENSIX_SHARED_BUFFER);
        EXPECT_EQ(record.worker_stream_abi, worker_stream_state::kVersion);
        EXPECT_EQ(record.worker_stream_owner, TT_NATIVE_WORKER_STREAM_SDK);
        EXPECT_EQ(record.reserved, 0);
        for (const auto& segment : elf.GetSegments()) {
            const uint64_t end = uint64_t(segment.address) + segment.membytes;
            EXPECT_TRUE(address + bytes.size() <= segment.address || address >= end);
        }
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
            for (uint32_t role = 0; role != TT_NATIVE_ROLE_COUNT; ++role) {
                const auto path = manager.get_firmware_binary_path(
                    0, core, static_cast<uint32_t>(role < 2 ? HalProcessorClassType::DM : HalProcessorClassType::COMPUTE),
                    role < 2 ? role : role - 2);
                checkNativeImage(path, role, mode);
                if (mode == DeviceProfilerMode::Program) checkProfileDataBuffer(path, *config.hal);
            }
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

TEST_F(WorkerFirmwareBuild, ExplicitProfileModeDoesNotReadAmbientDeployment) {
    for (const auto mode : {DeviceProfilerMode::Disabled, DeviceProfilerMode::Program}) {
        llrt::RunTimeOptions options(llrt::RunTimeOptions::ExplicitBuildOptions{
            .root_dir = scratch.string(), .cache_dir = (scratch / "explicit-cache").string(), .profiler_mode = mode});
        EXPECT_EQ(options.get_profiler_enabled(), mode == DeviceProfilerMode::Program);
        EXPECT_FALSE(options.get_streaming_profiler_enabled());
        EXPECT_FALSE(options.get_watcher_enabled());
        options.validate_device_profiler_mode(mode);
    }
}

TEST_F(WorkerFirmwareBuild, OfflineDisabledImagesCarryLoaderRecords) {
    buildImages(DeviceProfilerMode::Disabled);
}

TEST_F(WorkerFirmwareBuild, OfflineProgramProfilerImagesCarryLoaderRecords) {
    buildImages(DeviceProfilerMode::Program);
}

}  // namespace
}  // namespace tt::tt_metal
