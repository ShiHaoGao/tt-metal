// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "impl/profiler/profiler_state_manager.hpp"
#include "published_deployment_test_support.hpp"
#include <cstdlib>
#include <gtest/gtest.h>

namespace tt::tt_metal {
namespace {
TEST(ProfilerAllocation, TypedModesNeedNoRuntimeOptionsOrContext) {
    EXPECT_EQ(get_profiler_dram_bank_size_for_hal_allocation(DeviceProfilerMode::Disabled), 0u);
    EXPECT_EQ(get_profiler_dram_bank_size_for_hal_allocation(DeviceProfilerMode::Program), 48000u);
    EXPECT_ANY_THROW(get_profiler_dram_bank_size_for_hal_allocation(static_cast<DeviceProfilerMode>(255)));
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(ProfilerAllocation, DisabledHasNoAllocationOrCapacityMutation) {
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = "/tmp", .profiler_mode = DeviceProfilerMode::Disabled}};
    EXPECT_EQ(get_profiler_dram_bank_size_for_hal_allocation(options), 0u);
    EXPECT_FALSE(options.get_profiler_program_support_count().has_value());
}

TEST(ProfilerAllocation, ProgramPreservesDefaultAllocation) {
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = "/tmp", .profiler_mode = DeviceProfilerMode::Program}};
    EXPECT_EQ(get_profiler_dram_bank_size_for_hal_allocation(options), 48000u);
    EXPECT_EQ(options.get_profiler_program_support_count(), 1000u);
}

TEST(ProfilerAllocation, ProgramPreservesExplicitCapacity) {
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = "/tmp", .profiler_mode = DeviceProfilerMode::Program}};
    options.set_profiler_program_support_count(4096);
    EXPECT_EQ(get_profiler_dram_bank_size_for_hal_allocation(options), 196608u);
    EXPECT_EQ(options.get_profiler_program_support_count(), 4096u);
}

TEST(ProfilerAllocation, DebugDumpPreservesTwoBufferAllocation) {
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = "/tmp", .profiler_mode = DeviceProfilerMode::Disabled}};
    options.set_experimental_noc_debug_dump_enabled(true);
    EXPECT_EQ(get_profiler_dram_bank_size_for_hal_allocation(options), 48000u);
    EXPECT_EQ(options.get_profiler_program_support_count(), 500u);
}

class ScopedEnvironment {
    const char* name;
    std::optional<std::string> previous;
public:
    ScopedEnvironment(const char* name, const char* value) : name(name) {
        if (const auto* current = std::getenv(name)) previous = current;
        setenv(name, value, 1);
    }
    ~ScopedEnvironment() {
        if (previous) setenv(name, previous->c_str(), 1);
        else unsetenv(name);
    }
};

TEST(ProfilerAllocation, StreamingPreservesSpoolWhenProgramProfilerIsDisabled) {
    const ScopedEnvironment root("TT_METAL_RUNTIME_ROOT", "/tmp");
    const ScopedEnvironment program("TT_METAL_DEVICE_PROFILER", "0");
    const ScopedEnvironment streaming("TT_METAL_STREAMING_PROFILER", "1");
    const ScopedEnvironment spool("TT_METAL_STREAMING_PROFILER_DRAM_MB", "8");
    llrt::RunTimeOptions options;
    ASSERT_FALSE(options.get_profiler_enabled());
    ASSERT_TRUE(options.get_streaming_profiler_enabled());
    EXPECT_EQ(get_profiler_dram_bank_size_for_hal_allocation(options), 83887u);
}

class ProfilerAllocationArchive : public experimental::test::PublishedDeploymentTest {};
TEST_F(ProfilerAllocationArchive, DisabledRuntimeHalReceivesPublishedArchive) {
    const auto config = configuration();
    experimental::test::PublicationInput images(config, hal);
    const auto publication = experimental::PublishedDeployment::admit(config, images.input);
    const auto archive = publication.to_archive();
    Hal actual_hal{tt::ARCH::BLACKHOLE, false, true,
        get_profiler_dram_bank_size_for_hal_allocation(options), false, false, true, true};
    EXPECT_NO_THROW({
        const auto received = experimental::PublishedDeployment::from_archive(archive, actual_hal);
        EXPECT_TRUE(received.configuration().matches(config));
    });
}
}  // namespace
}  // namespace tt::tt_metal
