// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "tt-metalium/tt_metal.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/experimental/mock_device/mock_device.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "../api/metal2_host_api/test_helpers.hpp"
#include <memory>
#include <optional>

namespace tt::tt_metal {
namespace {
using Owner = tt::worker_stream_state::Owner;
class WorkerStreamStateProgram : public testing::Test {
protected:
    void SetUp() override {
        slow_dispatch.emplace();
        experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        mesh = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        if (mesh) { mesh->close(); mesh.reset(); }
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
    std::shared_ptr<distributed::MeshDevice> mesh;
    std::optional<experimental::test_helpers::ScopedSlowDispatchOverride> slow_dispatch;
};
TEST_F(WorkerStreamStateProgram, NativeBodyKeepsExplicitSdkDefaultWithZeroMask) {
    Program program;
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
    kernel(program, KernelBodyMode::Native);
    checkLaunch(program, Owner::SdkCircularBuffers);
}
TEST_F(WorkerStreamStateProgram, SdkBodyReceivesExplicitProgramOwner) {
    Program program;
    program.set_worker_stream_state_owner(Owner::Program);
    kernel(program, KernelBodyMode::Sdk);
    checkLaunch(program, Owner::Program);
}
TEST_F(WorkerStreamStateProgram, ProjectedGroupsFreezeOwnerBeforeCompilation) {
    Program program;
    kernel(program, KernelBodyMode::Native);
    checkLaunch(program, Owner::SdkCircularBuffers);
    EXPECT_ANY_THROW(program.set_worker_stream_state_owner(Owner::Program));
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
    checkLaunch(program, Owner::SdkCircularBuffers);
}
TEST_F(WorkerStreamStateProgram, UnknownOwnerRejectsWithoutChangingProgram) {
    Program program;
    EXPECT_ANY_THROW(program.set_worker_stream_state_owner(static_cast<Owner>(255)));
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
}
TEST_F(WorkerStreamStateProgram, ProgramOwnerRejectsNewSdkDescriptors) {
    Program program;
    program.set_worker_stream_state_owner(Owner::Program);
    auto config = CircularBufferConfig(4096, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 2048);
    EXPECT_ANY_THROW(CreateCircularBuffer(program, CoreCoord{0, 0}, config));
    EXPECT_TRUE(program.circular_buffers().empty());
}
TEST_F(WorkerStreamStateProgram, ExistingSdkDescriptorRejectsOwnerTransfer) {
    Program program;
    auto config = CircularBufferConfig(4096, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 2048);
    CreateCircularBuffer(program, CoreCoord{0, 0}, config);
    EXPECT_ANY_THROW(program.set_worker_stream_state_owner(Owner::Program));
    EXPECT_EQ(program.get_worker_stream_state_owner(), Owner::SdkCircularBuffers);
    EXPECT_EQ(program.circular_buffers().size(), 1u);
}
} // namespace
} // namespace tt::tt_metal
