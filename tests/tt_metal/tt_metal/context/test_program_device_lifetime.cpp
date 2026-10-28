// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/experimental/context/device_profiler_config.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_cache.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/device/mock_device_util.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {
namespace {

MetalEnvDescriptor mock_descriptor() {
    MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1));
    descriptor.set_device_profiler_mode(DeviceProfilerMode::Disabled);
    return descriptor;
}

std::unique_ptr<Program> cb_program(distributed::MeshDevice& mesh) {
    auto program = std::make_unique<Program>(std::make_shared<detail::ProgramImpl>(extract_context_id(&mesh)));
    const CoreRangeSet cores(CoreRange({0, 0}, {0, 0}));
    const CircularBufferConfig cb = CircularBufferConfig(2048, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 2048);
    CreateCircularBuffer(*program, cores, cb);
    // Exercise production allocation/tracking without an authored kernel or
    // work submission. Mock initialization may compile firmware, but all
    // device-driver calls are mocked.
    program->impl().allocate_circular_buffers(&mesh);
    return program;
}

Device& physical_device(distributed::MeshDevice& mesh) { return dynamic_cast<Device&>(*mesh.get_devices().at(0)); }

bool cached_program_outlives_owned_context() {
    MetalEnv env(mock_descriptor());
    // Do not query get_system_mesh() first: that gives context ownership to
    // the environment and hides the close-before-cache-destruction order.
    auto mesh = env.create_unit_mesh_device(0);
    const auto context_id = extract_context_id(mesh.get());
    auto program = cb_program(*mesh);
    if (physical_device(*mesh).get_total_cb_allocated() != 2048) {
        return false;
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(*program));
    program.reset();
    mesh->enable_program_cache();
    program_cache::detail::CachedMeshWorkload<int> cached(std::move(workload), 0);
    mesh->get_program_cache().insert(
        {.hash = 0xCB11FE, .canonical = "owned-context-cb-lifetime"},
        program_cache::detail::CachedProgramFactory(std::move(cached), 0));
    if (mesh->num_program_cache_entries() != 1) {
        return false;
    }
    if (!mesh->close()) {
        return false;
    }
    if (MetalContext::instance_exists(context_id)) {
        return false;
    }
    // Cache retention is required by multihost hybrid-allocator teardown.
    if (mesh->num_program_cache_entries() != 1) {
        return false;
    }
    mesh.reset();
    return true;
}

bool two_devices_detach_independently() {
    MetalEnv first_env(mock_descriptor());
    MetalEnv second_env(mock_descriptor());
    auto first = first_env.create_unit_mesh_device(0);
    auto second = second_env.create_unit_mesh_device(0);
    auto program = cb_program(*first);
    program->impl().allocate_circular_buffers(second.get());
    if (physical_device(*first).get_total_cb_allocated() != 2048 ||
        physical_device(*second).get_total_cb_allocated() != 2048) {
        return false;
    }
    // Both devices have ordinal 0 but distinct context/object identities.
    // Closing one must retain the other device's program registration.
    if (!first->close()) {
        return false;
    }
    first.reset();
    if (physical_device(*second).get_total_cb_allocated() != 2048) {
        return false;
    }
    program.reset();
    if (physical_device(*second).get_total_cb_allocated() != 0) {
        return false;
    }
    if (!second->close()) {
        return false;
    }
    second.reset();
    return true;
}

bool program_outlives_both_devices() {
    MetalEnv first_env(mock_descriptor());
    MetalEnv second_env(mock_descriptor());
    auto first = first_env.create_unit_mesh_device(0);
    auto second = second_env.create_unit_mesh_device(0);
    auto program = cb_program(*first);
    program->impl().allocate_circular_buffers(second.get());
    if (!first->close() || !second->close()) {
        return false;
    }
    first.reset();
    second.reset();
    program.reset();
    return true;
}

bool empty_program_outlives_owned_context() {
    MetalEnv env(mock_descriptor());
    auto mesh = env.create_unit_mesh_device(0);
    auto program = std::make_unique<Program>(std::make_shared<detail::ProgramImpl>(extract_context_id(mesh.get())));
    // The default empty program needs no allocation refresh. Its tracked
    // device still has to detach before context destruction.
    program->impl().allocate_circular_buffers(mesh.get());
    if (!mesh->close()) {
        return false;
    }
    mesh.reset();
    program.reset();
    return true;
}

TEST(ProgramDeviceLifetime, CPU_LiveProgramDestructionUnregistersDevice) {
    MetalEnv env(mock_descriptor());
    auto mesh = env.create_unit_mesh_device(0);
    auto program = cb_program(*mesh);
    EXPECT_EQ(physical_device(*mesh).get_total_cb_allocated(), 2048u);
    program.reset();
    EXPECT_EQ(physical_device(*mesh).get_total_cb_allocated(), 0u);
    EXPECT_TRUE(mesh->close());
    mesh.reset();
}

TEST(ProgramDeviceLifetime, CPU_EmptyProgramDeallocationUnregistersDevice) {
    MetalEnv first_env(mock_descriptor());
    MetalEnv second_env(mock_descriptor());
    auto first = first_env.create_unit_mesh_device(0);
    auto second = second_env.create_unit_mesh_device(0);
    auto program = std::make_unique<Program>(std::make_shared<detail::ProgramImpl>(extract_context_id(first.get())));
    // Exercise allocation refresh even when the current program has no CBs.
    program->impl().invalidate_circular_buffer_allocation();
    program->impl().allocate_circular_buffers(first.get());
    program->impl().deallocate_circular_buffers();

    const CoreRangeSet cores(CoreRange({0, 0}, {0, 0}));
    const CircularBufferConfig cb = CircularBufferConfig(2048, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 2048);
    CreateCircularBuffer(*program, cores, cb);
    program->impl().allocate_circular_buffers(second.get());
    // An empty program must unregister too: a later allocation on another
    // device must not appear in the first device's live-program statistics.
    EXPECT_EQ(physical_device(*first).get_total_cb_allocated(), 0u);
    EXPECT_EQ(physical_device(*second).get_total_cb_allocated(), 2048u);
    // Keep teardown safe even on RED; the assertion above detects stale
    // registration without depending on a subsequent use-after-free crash.
    physical_device(*first).unregister_program(&program->impl());
    program.reset();
    EXPECT_EQ(physical_device(*second).get_total_cb_allocated(), 0u);
    EXPECT_TRUE(first->close());
    EXPECT_TRUE(second->close());
    first.reset();
    second.reset();
}

TEST(ProgramDeviceLifetime, CPU_PerDeviceDeallocationIsIdempotent) {
    MetalEnv env(mock_descriptor());
    auto mesh = env.create_unit_mesh_device(0);
    auto program = cb_program(*mesh);
    auto& device = physical_device(*mesh);
    program->impl().deallocate_circular_buffers(&device);
    program->impl().deallocate_circular_buffers(&device);
    EXPECT_EQ(device.get_total_cb_allocated(), 0u);
    // Reattaching cached CB addresses must create a fresh registration.
    program->impl().allocate_circular_buffers(mesh.get());
    EXPECT_EQ(device.get_total_cb_allocated(), 2048u);
    program->impl().deallocate_circular_buffers();
    program->impl().deallocate_circular_buffers();
    EXPECT_EQ(device.get_total_cb_allocated(), 0u);
    program.reset();
    EXPECT_TRUE(mesh->close());
    mesh.reset();
}

TEST(ProgramDeviceLifetime, CPU_CachedProgramOutlivesOwnedContext) {
    // Run destruction in a child so a use-after-free is a reported failure
    // instead of terminating the rest of the CPU regression executable.
    ASSERT_EXIT(std::exit(cached_program_outlives_owned_context() ? 0 : 1), ::testing::ExitedWithCode(0), "");
}

TEST(ProgramDeviceLifetime, CPU_TwoDevicesDetachIndependently) {
    ASSERT_EXIT(std::exit(two_devices_detach_independently() ? 0 : 1), ::testing::ExitedWithCode(0), "");
}

TEST(ProgramDeviceLifetime, CPU_ProgramOutlivesBothDevices) {
    ASSERT_EXIT(std::exit(program_outlives_both_devices() ? 0 : 1), ::testing::ExitedWithCode(0), "");
}

TEST(ProgramDeviceLifetime, CPU_EmptyProgramOutlivesOwnedContext) {
    ASSERT_EXIT(std::exit(empty_program_outlives_owned_context() ? 0 : 1), ::testing::ExitedWithCode(0), "");
}

}  // namespace
}  // namespace tt::tt_metal
