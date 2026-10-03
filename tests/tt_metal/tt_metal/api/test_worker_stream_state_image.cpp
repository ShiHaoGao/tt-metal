// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <elf.h>
#include <unistd.h>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>
#include "llrt/tt_memory.h"
#include "llrt/worker_stream_state_provider.hpp"
#include <type_traits>
#include <limits>
#include <optional>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <future>
#include <semaphore>
#include <thread>
#include <pthread.h>
#include "llrt/llrt.hpp"
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include "impl/context/context_descriptor.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/firmware/risc_firmware_initializer.hpp"
#include "llrt/hal.hpp"
#include "../api/metal2_host_api/test_helpers.hpp"

namespace {
class WorkerImage : public ::testing::Test {
protected:
    std::string path;
    void SetUp() override {
        char name[] = "/tmp/tt-worker-abi-XXXXXX";
        int fd = mkstemp(name);
        ASSERT_GE(fd, 0);
        close(fd);
        path = name;
    }
    void TearDown() override { unlink(path.c_str()); }
    void write(std::vector<uint8_t> record, bool present = true, uint32_t instruction = 0x13) {
        constexpr size_t textOffset = 0x100, stringsOffset = 0x110,
                         recordOffset = 0x180, sectionsOffset = 0x200;
        constexpr char names[] = "\0.text\0.shstrtab\0.tt_worker_stream_state\0";
        std::vector<uint8_t> bytes(sectionsOffset + 4 * sizeof(Elf32_Shdr), 0);
        Elf32_Ehdr header{};
        std::memcpy(header.e_ident, ELFMAG, SELFMAG);
        header.e_ident[EI_CLASS] = ELFCLASS32;
        header.e_ident[EI_DATA] = ELFDATA2LSB;
        header.e_ident[EI_VERSION] = EV_CURRENT;
        header.e_type = ET_EXEC;
        header.e_machine = EM_RISCV;
        header.e_version = EV_CURRENT;
        header.e_entry = 0x1000;
        header.e_ehsize = sizeof(header);
        header.e_phoff = sizeof(header);
        header.e_phentsize = sizeof(Elf32_Phdr);
        header.e_phnum = 1;
        header.e_shoff = sectionsOffset;
        header.e_shentsize = sizeof(Elf32_Shdr);
        header.e_shnum = present ? 4 : 3;
        header.e_shstrndx = 2;
        Elf32_Phdr load{};
        load.p_type = PT_LOAD;
        load.p_offset = textOffset;
        load.p_vaddr = load.p_paddr = 0x1000;
        load.p_filesz = load.p_memsz = 4;
        load.p_flags = PF_R | PF_X;
        load.p_align = 4;
        std::array<Elf32_Shdr, 4> sections{};
        sections[1].sh_name = 1;
        sections[1].sh_type = SHT_PROGBITS;
        sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
        sections[1].sh_addr = 0x1000;
        sections[1].sh_offset = textOffset;
        sections[1].sh_size = 4;
        sections[1].sh_addralign = 4;
        sections[2].sh_name = 7;
        sections[2].sh_type = SHT_STRTAB;
        sections[2].sh_offset = stringsOffset;
        sections[2].sh_size = sizeof(names);
        sections[3].sh_name = 17;
        sections[3].sh_type = SHT_PROGBITS;
        sections[3].sh_offset = recordOffset;
        sections[3].sh_size = record.size();
        std::memcpy(bytes.data(), &header, sizeof(header));
        std::memcpy(bytes.data() + header.e_phoff, &load, sizeof(load));
        std::memcpy(bytes.data() + stringsOffset, names, sizeof(names));
        std::memcpy(bytes.data() + recordOffset, record.data(), record.size());
        std::memcpy(bytes.data() + sectionsOffset, sections.data(), sizeof(sections));
        std::memcpy(bytes.data() + textOffset, &instruction, sizeof(instruction));
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ASSERT_TRUE(output.good());
    }
};
TEST_F(WorkerImage, CPU_ActualImageRetainsRoleAndVersion) {
    write({1, 1, 0, 0});
    ll_api::memory image(path, ll_api::memory::Loading::DISCRETE);
    EXPECT_EQ(image.worker_stream_state_image().version, 1);
    EXPECT_EQ(image.worker_stream_state_image().role, 1);
    write({1, 2, 0, 0});
    EXPECT_EQ(image.worker_stream_state_image().role, 1);
    ll_api::memory replaced(path, ll_api::memory::Loading::DISCRETE);
    EXPECT_EQ(replaced.worker_stream_state_image().role, 2);
}
TEST_F(WorkerImage, CPU_MissingOrMalformedImageCannotAdvertiseProvider) {
    for (const auto &record : std::vector<std::vector<uint8_t>>{
             {}, {1}, {1, 1, 0}, {1, 1, 0, 0, 0}, {1, 1, 1, 0},
             {0, 1, 0, 0}, {2, 1, 0, 0}, {1, 3, 0, 0}}) {
        write(record);
        ll_api::memory image(path, ll_api::memory::Loading::DISCRETE);
        EXPECT_EQ(image.worker_stream_state_image().version, 0);
    }
    write({1, 1, 0, 0}, false);
    ll_api::memory image(path, ll_api::memory::Loading::DISCRETE);
    EXPECT_EQ(image.worker_stream_state_image().version, 0);
}
TEST(WorkerStreamStateProvider, CPU_CallerCannotMintBootReceipt) {
    EXPECT_FALSE((std::is_constructible_v<tt::tt_metal::WorkerStreamStateProvider, uint8_t>));
    EXPECT_FALSE((std::is_default_constructible_v<tt::tt_metal::WorkerStreamStateProvider>));
}

class WorkerImageCache : public WorkerImage {
protected:
    void SetUp() override {
        WorkerImage::SetUp();
        tt::tt_metal::experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        (void)tt::tt_metal::MetalContext::instance(
            tt::tt_metal::DEFAULT_CONTEXT_ID, tt::tt_metal::DeviceProfilerMode::Disabled);
    }
    void TearDown() override {
        for (const auto& extra : extra_paths) unlink(extra.c_str());
        tt::tt_metal::MetalContext::destroy_all_instances(false);
        tt::tt_metal::experimental::disable_mock_mode();
        WorkerImage::TearDown();
    }
    std::string copyToFreshPath() {
        char name[] = "/tmp/tt-worker-cache-XXXXXX";
        int fd = mkstemp(name);
        if (fd < 0) throw std::runtime_error("cannot create cache test ELF");
        close(fd);
        extra_paths.emplace_back(name);
        std::filesystem::copy_file(path, name, std::filesystem::copy_options::overwrite_existing);
        return name;
    }
    [[noreturn]] void finishChildProbe(int code) {
        // Threadsafe death tests re-exec and create their own temporary files;
        // their deliberate _Exit does not run the fixture's TearDown.
        for (const auto& extra : extra_paths) unlink(extra.c_str());
        unlink(path.c_str());
        std::_Exit(code);
    }
    std::vector<std::string> extra_paths;
};

TEST_F(WorkerImageCache, CPU_PathReplacementRetainsActualCachedRecordAndBytes) {
    write({1, 1, 0, 0}, true, 0x13);
    const auto& first = tt::llrt::get_risc_binary(path);
    ASSERT_EQ(first.worker_stream_state_image().role, 1);
    ASSERT_EQ(first.data().front(), 0x13u);
    write({1, 2, 0, 0}, true, 0x00100093);
    const auto& cached = tt::llrt::get_risc_binary(path);
    EXPECT_EQ(&cached, &first);
    EXPECT_EQ(cached.worker_stream_state_image().role, 1);
    EXPECT_EQ(cached.data().front(), 0x13u);
    const auto& fresh = tt::llrt::get_risc_binary(copyToFreshPath());
    EXPECT_NE(&fresh, &first);
    EXPECT_EQ(fresh.worker_stream_state_image().role, 2);
    EXPECT_EQ(fresh.data().front(), 0x00100093u);
}

TEST_F(WorkerImageCache, CPU_MalformedCachedRecordCannotGainCapabilityFromPathReplacement) {
    write({1, 1, 1, 0}, true, 0x13);
    const auto& first = tt::llrt::get_risc_binary(path);
    ASSERT_EQ(first.worker_stream_state_image().version, 0);
    write({1, 1, 0, 0}, true, 0x00100093);
    const auto& cached = tt::llrt::get_risc_binary(path);
    EXPECT_EQ(&cached, &first);
    EXPECT_EQ(cached.worker_stream_state_image().version, 0);
    EXPECT_EQ(cached.data().front(), 0x13u);
    const auto& fresh = tt::llrt::get_risc_binary(copyToFreshPath());
    EXPECT_EQ(fresh.worker_stream_state_image().version, 1);
    EXPECT_EQ(fresh.data().front(), 0x00100093u);
}

TEST_F(WorkerImageCache, CPU_ReadbackEqualityDoesNotInventFileOnlyRecord) {
    write({1, 1, 0, 0});
    const auto& image = tt::llrt::get_risc_binary(path);
    ll_api::memory readback;
    size_t offset = 0;
    readback.fill_from_mem_template(image, [&](auto destination, uint64_t, uint32_t words) {
        std::copy_n(image.data().begin() + offset, words, destination);
        offset += words;
    });
    EXPECT_TRUE(image == readback);
    EXPECT_EQ(readback.worker_stream_state_image().version, 0);
    EXPECT_EQ(image.worker_stream_state_image().version, 1);
}

TEST_F(WorkerImageCache, CPU_ActualReadFailureAllowsSamePathRetry) {
    write({1, 1, 0, 0});
    const auto backup = copyToFreshPath();
    // A threadsafe death-test child bounds the old infinite wait without
    // detaching stuck threads into the actual SDK test process.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(([&] {
        std::filesystem::remove(path);
        bool rejected = false;
        try { (void)tt::llrt::get_risc_binary(path); }
        catch (const std::exception& error) {
            rejected = std::string(error.what()).find("cannot map elf") != std::string::npos;
        }
        if (!rejected) finishChildProbe(71);
        std::filesystem::copy_file(backup, path);
        std::promise<const ll_api::memory*> result;
        auto ready = result.get_future();
        std::thread retry([&] {
            try { result.set_value(&tt::llrt::get_risc_binary(path)); }
            catch (...) { result.set_exception(std::current_exception()); }
        });
        if (ready.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            std::fprintf(stderr, "cache retry blocked after actual ELF read failure\n");
            finishChildProbe(72);
        }
        const ll_api::memory* image = nullptr;
        try { image = ready.get(); } catch (...) { finishChildProbe(73); }
        retry.join();
        if (!image || image->worker_stream_state_image().role != 1 || image->data().front() != 0x13) finishChildProbe(74);
        std::fprintf(stderr, "actual ELF retry recovered\n");
        finishChildProbe(0);
    }()), ::testing::ExitedWithCode(0), "actual ELF retry recovered");
}

TEST_F(WorkerImageCache, CPU_ConcurrentFailedPublisherReleasesWaitersAndAllowsRetry) {
    write({1, 1, 0, 0});
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(([&] {
        std::binary_semaphore constructing{0}, release{0}, waiter_started{0};
        std::promise<bool> owner_result, waiter_result;
        auto owner_ready = owner_result.get_future();
        auto waiter_ready = waiter_result.get_future();
        auto rejected = [](const std::exception& error) {
            return std::string(error.what()) == "actual image update failed";
        };
        std::thread owner([&] {
            try {
                (void)tt::llrt::get_risc_binary(path, ll_api::memory::Loading::DISCRETE, [&](ll_api::memory&) {
                    constructing.release();
                    release.acquire();
                    throw std::runtime_error("actual image update failed");
                });
                owner_result.set_value(false);
            } catch (const std::exception& error) { owner_result.set_value(rejected(error)); }
        });
        constructing.acquire();
        std::thread waiter([&] {
            waiter_started.release();
            try {
                const auto& image = tt::llrt::get_risc_binary(path);
                waiter_result.set_value(image.worker_stream_state_image().role == 1 && image.data().front() == 0x13);
            } catch (const std::exception& error) { waiter_result.set_value(rejected(error)); }
        });
        waiter_started.acquire();
        release.release();
        if (owner_ready.wait_for(std::chrono::seconds(5)) != std::future_status::ready ||
            waiter_ready.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            std::fprintf(stderr, "actual cache publisher left a pending entry forever\n");
            finishChildProbe(75);
        }
        const bool passed = owner_ready.get() && waiter_ready.get();
        owner.join();
        waiter.join();
        if (!passed) finishChildProbe(76);
        const auto& retry = tt::llrt::get_risc_binary(path);
        if (retry.worker_stream_state_image().role != 1 || retry.data().front() != 0x13) finishChildProbe(77);
        std::fprintf(stderr, "concurrent actual cache failure recovered\n");
        finishChildProbe(0);
    }()), ::testing::ExitedWithCode(0), "concurrent actual cache failure recovered");
}
}  // namespace

namespace tt::tt_metal {
namespace {
class WorkerFirmwarePhase : public ::testing::Test {
protected:
    void SetUp() override {
        slow_dispatch.emplace();
        experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        auto& context = MetalContext::instance(DEFAULT_CONTEXT_ID, DeviceProfilerMode::Disabled);
        context.initialize({}, 1, {}, 0, true);
        descriptor = std::make_shared<ContextDescriptor>(
            &context.get_env(), &context, 1, 0, 0,
            context.hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED));
        initializer = std::make_unique<RiscFirmwareInitializer>(
            descriptor, [&context]() -> tt::tt_fabric::ControlPlane& { return context.get_control_plane(); },
            context.get_dispatch_core_manager());
    }
    void TearDown() override {
        initializer.reset();
        descriptor.reset();
        MetalContext::destroy_all_instances(false);
        experimental::disable_mock_mode();
        slow_dispatch.reset();
    }
    std::optional<experimental::test_helpers::ScopedSlowDispatchOverride> slow_dispatch;
    std::shared_ptr<ContextDescriptor> descriptor;
    std::unique_ptr<RiscFirmwareInitializer> initializer;
};

TEST_F(WorkerFirmwarePhase, CPU_RebuildWithdrawsPriorCompletedPhaseBeforeLaunch) {
    // Real phase API with a mock platform: lifecycle completion is observable,
    // but no actual firmware was loaded and no boot provider may be minted.
    initializer->run_launch_phase({0});
    ASSERT_TRUE(initializer->is_initialized());
    ASSERT_FALSE(initializer->worker_stream_state_provider(0));
    initializer->run_async_build_phase({});
    EXPECT_FALSE(initializer->is_initialized());
    EXPECT_FALSE(initializer->worker_stream_state_provider(0));
    initializer->run_launch_phase({0});
    EXPECT_TRUE(initializer->is_initialized());
    EXPECT_FALSE(initializer->worker_stream_state_provider(0));
}

TEST_F(WorkerFirmwarePhase, CPU_RealAsyncBuildFailureCannotRetainCompletedPhase) {
    initializer->run_launch_phase({0});
    ASSERT_TRUE(initializer->is_initialized());
    // This uses the real async build path and UMD mock chip lookup. Its missing
    // device exception reaches future::get(); no replacement phase model,
    // synthetic callback or private-state mutation supplies the failure.
    const auto missing = std::numeric_limits<tt::ChipId>::max();
    try {
        initializer->run_async_build_phase({missing});
        FAIL() << "Expected the actual mock cluster to reject a missing chip";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("not found in cluster"), std::string::npos) << error.what();
    }
    EXPECT_FALSE(initializer->is_initialized());
    EXPECT_FALSE(initializer->worker_stream_state_provider(0));
    EXPECT_FALSE(initializer->worker_stream_state_provider(missing));
}

TEST_F(WorkerFirmwarePhase, CPU_MockAndTeardownNeverAdvertiseBootReceipt) {
    initializer->run_launch_phase({0});
    ASSERT_TRUE(initializer->is_initialized());
    EXPECT_FALSE(initializer->worker_stream_state_provider(0));
    std::unordered_set<InitializerKey> completed;
    initializer->teardown(completed);
    EXPECT_FALSE(initializer->is_initialized());
    EXPECT_FALSE(initializer->worker_stream_state_provider(0));
}
}  // namespace
}  // namespace tt::tt_metal
