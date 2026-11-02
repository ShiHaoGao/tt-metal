// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "tt-metalium/tt_metal.hpp"
#include "tt-metalium/experimental/mock_device/mock_device.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "../api/metal2_host_api/test_helpers.hpp"
#include <optional>

namespace tt::tt_metal {
namespace {
using Owner = tt::worker_stream_state::Owner;
class WorkerStreamStateProgramBase : public testing::Test {
protected:
    virtual tt::ARCH architecture() const = 0;
    void SetUp() override {
        slow_dispatch.emplace();
        experimental::configure_mock_mode(architecture(), 1);
        // Projection needs the actual HAL and host kernel builders, but no
        // device creation, firmware build, load or successful boot receipt.
        (void)MetalContext::instance(DEFAULT_CONTEXT_ID, DeviceProfilerMode::Disabled);
    }
    void TearDown() override {
        MetalContext::destroy_all_instances(false);
        experimental::disable_mock_mode();
        slow_dispatch.reset();
    }
    void kernel(Program &program, KernelBodyMode mode) {
        CreateKernelFromString(program, "void kernel_main() {}", CoreCoord{0, 0},
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                                  .body_mode = mode});
    }
    uint32_t workerIndex(Program &program) {
        return MetalContext::instance(program.impl().get_context_id()).hal().get_programmable_core_type_index(
            HalProgrammableCoreType::TENSIX);
    }
    void checkLaunch(Program &program, Owner expected) {
        auto &groups = program.impl().get_kernel_groups(workerIndex(program));
        ASSERT_EQ(groups.size(), 1u);
        auto config = groups[0]->launch_msg.view().kernel_config();
        EXPECT_EQ(config.worker_stream_state_abi_version(), tt::worker_stream_state::kVersion);
        EXPECT_EQ(config.worker_stream_state_owner(), static_cast<uint8_t>(expected));
        EXPECT_EQ(config.local_cb_mask(), 0u);
    }
    std::optional<experimental::test_helpers::ScopedSlowDispatchOverride> slow_dispatch;
};
class WorkerStreamStateProgram : public WorkerStreamStateProgramBase,
                                 public testing::WithParamInterface<tt::ARCH> {
protected:
    tt::ARCH architecture() const override { return GetParam(); }
};
TEST_P(WorkerStreamStateProgram, NativeBodyKeepsExplicitSdkDefaultWithZeroMask) {
    Program program;
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
    kernel(program, KernelBodyMode::Native);
    checkLaunch(program, Owner::SdkCircularBuffers);
}
TEST_P(WorkerStreamStateProgram, SdkBodyReceivesExplicitProgramOwner) {
    Program program;
    ASSERT_NO_THROW(program.set_worker_stream_state_owner(Owner::Program));
    kernel(program, KernelBodyMode::Sdk);
    checkLaunch(program, Owner::Program);
}
TEST_P(WorkerStreamStateProgram, ProjectedGroupsFreezeOwnerBeforeCompilation) {
    Program program;
    kernel(program, KernelBodyMode::Native);
    checkLaunch(program, Owner::SdkCircularBuffers);
    EXPECT_ANY_THROW(program.set_worker_stream_state_owner(Owner::Program));
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
    checkLaunch(program, Owner::SdkCircularBuffers);
}
TEST_P(WorkerStreamStateProgram, UnknownOwnerRejectsWithoutChangingProgram) {
    Program program;
    EXPECT_ANY_THROW(program.set_worker_stream_state_owner(static_cast<Owner>(255)));
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
}
TEST_P(WorkerStreamStateProgram, ProgramOwnerRejectsNewSdkDescriptors) {
    Program program;
    ASSERT_NO_THROW(program.set_worker_stream_state_owner(Owner::Program));
    auto config = CircularBufferConfig(4096, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 2048);
    EXPECT_ANY_THROW(CreateCircularBuffer(program, CoreCoord{0, 0}, config));
    EXPECT_TRUE(program.circular_buffers().empty());
}
TEST_P(WorkerStreamStateProgram, ExistingSdkDescriptorRejectsOwnerTransfer) {
    Program program;
    auto config = CircularBufferConfig(4096, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 2048);
    CreateCircularBuffer(program, CoreCoord{0, 0}, config);
    EXPECT_ANY_THROW(program.set_worker_stream_state_owner(Owner::Program));
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
    EXPECT_EQ(program.circular_buffers().size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(
    SupportedArchitectures, WorkerStreamStateProgram,
    testing::Values(tt::ARCH::BLACKHOLE, tt::ARCH::WORMHOLE_B0),
    [](const testing::TestParamInfo<tt::ARCH>& info) {
        return info.param == tt::ARCH::BLACKHOLE ? "Blackhole" : "Wormhole";
    });

class UnsupportedWorkerStreamStateProgram : public WorkerStreamStateProgramBase {
protected:
    tt::ARCH architecture() const override { return tt::ARCH::QUASAR; }
};
TEST_F(UnsupportedWorkerStreamStateProgram, ProgramOwnerRejectsWithoutChangingProgram) {
    Program program;
    EXPECT_ANY_THROW(program.set_worker_stream_state_owner(Owner::Program));
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
}
} // namespace
} // namespace tt::tt_metal
