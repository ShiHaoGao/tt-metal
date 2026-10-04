// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <atomic>
#include <semaphore>
#include <thread>
#include <pthread.h>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/lightmetal/lightmetal_api.hpp>
#include "lightmetal/lightmetal_capture.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/device/device_manager.hpp"
#include "fabric/fabric_init.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_command_queue_base.hpp"
#include "distributed/sd_mesh_command_queue.hpp"
#include "../api/metal2_host_api/test_helpers.hpp"
#include "impl/context/worker_stream_state_client.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"

namespace {
std::binary_semaphore worker_client_publication_release{0};
std::atomic<bool> worker_client_publication_rejected{false};
std::atomic<bool> worker_client_publication_released{false};
std::binary_semaphore worker_client_capture_release{0};
std::atomic<bool> worker_client_capture_released{false};
}
// Test-only debugger seam. The SDK production path has no probe or callback.
extern "C" __attribute__((noinline)) void trex_worker_client_release_reader() {
    if (!worker_client_publication_released.exchange(true))
        worker_client_publication_release.release();
}
extern "C" __attribute__((noinline)) void trex_worker_client_reader_finished() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}
extern "C" __attribute__((noinline)) void trex_worker_client_release_capture() {
    if (!worker_client_capture_released.exchange(true))
        worker_client_capture_release.release();
}
extern "C" __attribute__((noinline)) void trex_worker_client_capture_finished() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}
namespace tt::tt_metal {
namespace {
template <typename Action>
void expectWorkerClientRejection(Action&& action, const char* diagnostic) {
    try {
        action();
        FAIL() << "Expected worker stream-state admission failure";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(diagnostic), std::string::npos) << error.what();
    }
}
class WorkerClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    }
    void TearDown() override {
        // Deliberate owner cleanup; tests never create native devices.
        WorkerStreamStateAccess access(client);
        MetalContext::destroy_all_instances(false);
        experimental::disable_mock_mode();
    }
    std::shared_ptr<const WorkerStreamStateClient> client;
};
MetalEnvDescriptor explicit_mock_env() {
    MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc().value());
    descriptor.set_device_profiler_mode(DeviceProfilerMode::Disabled);
    return descriptor;
}
TEST_F(WorkerClientTest, ExplicitEnvironmentOwnsNondefaultExclusiveContextAndTeardown) {
    std::optional<ContextId> id;
    {
        MetalEnv env(explicit_mock_env());
        auto token = env.acquire_worker_stream_state_client();
        id = ContextId{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
        EXPECT_NE(*id, DEFAULT_CONTEXT_ID);
        EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
        auto& context = MetalContext::instance(*id);
        EXPECT_NO_THROW(context.validate_worker_stream_state_client(token));
        EXPECT_THROW(context.validate_worker_stream_state_access(), std::runtime_error);
        EXPECT_THROW(env.acquire_worker_stream_state_client(), std::runtime_error);
        token.reset();
        EXPECT_THROW(env.acquire_worker_stream_state_client(), std::runtime_error);
    }
    ASSERT_TRUE(id.has_value());
    EXPECT_FALSE(MetalContext::instance_exists(*id));
    EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
}
TEST_F(WorkerClientTest, ExplicitEnvironmentClientsCannotCrossEnvironmentsOrDefault) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    MetalEnv first(explicit_mock_env()), second(explicit_mock_env());
    auto a = first.acquire_worker_stream_state_client();
    auto b = second.acquire_worker_stream_state_client();
    const auto a_id = ContextId{MetalEnvAccessor(first).impl().ensure_context_registered(first)};
    const auto b_id = ContextId{MetalEnvAccessor(second).impl().ensure_context_registered(second)};
    auto& a_context = MetalContext::instance(a_id);
    auto& b_context = MetalContext::instance(b_id);
    EXPECT_NE(a_id, b_id);
    EXPECT_THROW(a_context.validate_worker_stream_state_client(b), std::runtime_error);
    EXPECT_THROW(a_context.validate_worker_stream_state_client(client), std::runtime_error);
    EXPECT_THROW(b_context.validate_worker_stream_state_client(a), std::runtime_error);
    EXPECT_THROW(MetalContext::instance(DEFAULT_CONTEXT_ID).validate_worker_stream_state_client(a), std::runtime_error);
    { WorkerStreamStateAccess access(a);
      EXPECT_NO_THROW(a_context.validate_worker_stream_state_access());
      EXPECT_THROW(b_context.validate_worker_stream_state_access(), std::runtime_error); }
}
TEST_F(WorkerClientTest, ExplicitEnvironmentRejectsOrdinaryContextWithoutMutation) {
    MetalEnv env(explicit_mock_env());
    auto& impl = MetalEnvAccessor(env).impl();
    const auto id = ContextId{impl.ensure_context_registered(env)};
    EXPECT_THROW(env.acquire_worker_stream_state_client(), std::runtime_error);
    EXPECT_EQ(impl.ensure_context_registered(env), id.get());
    EXPECT_NO_THROW(MetalContext::instance(id).validate_worker_stream_state_client({}));
    EXPECT_NO_THROW(MetalContext::instance(id).validate_worker_stream_state_access());
}
TEST_F(WorkerClientTest, ExplicitEnvironmentRejectsUnregisteredOrdinaryContextWithoutMutation) {
    MetalEnv env(explicit_mock_env());
    const auto id = MetalContext::create_instance(env);
    EXPECT_THROW(env.acquire_worker_stream_state_client(), std::runtime_error);
    EXPECT_FALSE(MetalEnvAccessor(env).impl().has_registered_context());
    EXPECT_NO_THROW(MetalContext::instance(id).validate_worker_stream_state_client({}));
    MetalContext::destroy_instance(false, id);
}
TEST_F(WorkerClientTest, ExplicitEnvironmentRequiresSelectedProfileBeforeRegistration) {
    MetalEnv env(MetalEnvDescriptor{experimental::get_mock_cluster_desc().value()});
    auto& impl = MetalEnvAccessor(env).impl();
    EXPECT_THROW(env.acquire_worker_stream_state_client(), std::runtime_error);
    EXPECT_FALSE(impl.has_registered_context());
    EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
}
TEST_F(WorkerClientTest, ExplicitEnvironmentRejectsProfileConflictBeforeRegistration) {
    auto descriptor = explicit_mock_env();
    descriptor.set_device_profiler_mode(DeviceProfilerMode::Program);
    MetalEnv env(std::move(descriptor));
    auto& impl = MetalEnvAccessor(env).impl();
    impl.get_rtoptions().set_watcher_enabled(true);
    expectWorkerClientRejection([&] { env.acquire_worker_stream_state_client(); }, "conflicts with DPRINT or Watcher");
    EXPECT_FALSE(impl.has_registered_context());
    EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
    impl.get_rtoptions().set_watcher_enabled(false);
    EXPECT_NO_THROW(env.acquire_worker_stream_state_client());
}
TEST_F(WorkerClientTest, ExplicitEnvironmentMeshRetainsExactContextAndClient) {
    MetalEnv env(explicit_mock_env());
    auto token = env.acquire_worker_stream_state_client();
    const auto id = ContextId{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
    std::shared_ptr<distributed::MeshDevice> mesh;
    {
        WorkerStreamStateAccess access(token);
        mesh = env.create_unit_mesh_device(0);
    }
    EXPECT_EQ(mesh->impl().get_context_id(), id);
    EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
    token.reset();
    EXPECT_THROW(env.acquire_worker_stream_state_client(), std::runtime_error);
    EXPECT_NO_THROW(mesh.reset());
    EXPECT_TRUE(MetalContext::instance_exists(id));
    EXPECT_TRUE(MetalContext::instance(id).device_manager()->get_all_active_devices().empty());
}
TEST_F(WorkerClientTest, ExplicitEnvironmentRejectsSecondOrdinaryFactory) {
    MetalEnv env(explicit_mock_env());
    auto token = env.acquire_worker_stream_state_client();
    const auto id = ContextId{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
    EXPECT_THROW(MetalContext::create_instance(env), std::runtime_error);
    EXPECT_NO_THROW(MetalContext::instance(id).validate_worker_stream_state_client(token));
}
TEST_F(WorkerClientTest, ExplicitEnvironmentDoesNotDestroyRecycledForeignSlot) {
    MetalEnv foreign(explicit_mock_env());
    std::optional<ContextId> id;
    {
        MetalEnv env(explicit_mock_env());
        auto token = env.acquire_worker_stream_state_client();
        id = ContextId{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
        { WorkerStreamStateAccess access(token); MetalContext::destroy_instance(false, *id); }
        EXPECT_THROW(MetalEnvAccessor(env).impl().ensure_context_registered(env), std::runtime_error);
        const auto foreign_id = MetalContext::create_instance(foreign);
        ASSERT_EQ(*id, foreign_id);
        EXPECT_THROW(MetalEnvAccessor(env).impl().ensure_context_registered(env), std::runtime_error);
    }
    ASSERT_TRUE(MetalContext::instance_exists(*id));
    EXPECT_EQ(&MetalContext::instance(*id).get_env(), &foreign);
    MetalContext::destroy_instance(false, *id);
}
TEST_F(WorkerClientTest, ExplicitEnvironmentDoesNotDestroyRecycledOrdinaryGeneration) {
    MetalEnv env(explicit_mock_env());
    auto& impl = MetalEnvAccessor(env).impl();
    auto token = env.acquire_worker_stream_state_client();
    const auto id = ContextId{impl.ensure_context_registered(env)};
    { WorkerStreamStateAccess access(token); MetalContext::destroy_instance(false, id); }
    const auto replacement = MetalContext::create_instance(env);
    ASSERT_EQ(id, replacement);
    EXPECT_THROW(impl.ensure_context_registered(env), std::runtime_error);
    EXPECT_NO_THROW(impl.teardown_registered_context());
    EXPECT_TRUE(MetalContext::instance_exists(replacement));
    EXPECT_NO_THROW(MetalContext::instance(replacement).validate_worker_stream_state_client({}));
    MetalContext::destroy_instance(false, replacement);
}
TEST_F(WorkerClientTest, ExplicitEnvironmentConcurrentAcquisitionHasOneOwner) {
    MetalEnv env(explicit_mock_env());
    std::atomic<int> accepted{0}, rejected{0};
    std::counting_semaphore<2> start{0};
    auto acquire = [&] {
        start.acquire();
        try { (void)env.acquire_worker_stream_state_client(); ++accepted; }
        catch (const std::runtime_error&) { ++rejected; }
    };
    std::thread first(acquire), second(acquire);
    start.release();
    start.release();
    first.join(); second.join();
    EXPECT_EQ(accepted, 1);
    EXPECT_EQ(rejected, 1);
    EXPECT_TRUE(MetalEnvAccessor(env).impl().has_registered_context());
    EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
}
TEST_F(WorkerClientTest, ExplicitEnvironmentConcurrentRegistrationNeverConvertsOrdinaryContext) {
    MetalEnv env(explicit_mock_env());
    auto& impl = MetalEnvAccessor(env).impl();
    std::counting_semaphore<2> start{0};
    std::shared_ptr<const WorkerStreamStateClient> token;
    int registered = -1;
    std::thread ordinary([&] { start.acquire(); registered = impl.ensure_context_registered(env); });
    std::thread exclusive([&] {
        start.acquire();
        try { token = env.acquire_worker_stream_state_client(); }
        catch (const std::runtime_error&) {}
    });
    start.release(); start.release();
    ordinary.join(); exclusive.join();
    EXPECT_EQ(registered, impl.ensure_context_registered(env));
    auto& context = MetalContext::instance(ContextId{registered});
    EXPECT_NO_THROW(context.validate_worker_stream_state_client(token));
    EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
}
TEST_F(WorkerClientTest, ExplicitEnvironmentAndLightMetalCaptureExcludeEachOther) {
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
    MetalEnv env(explicit_mock_env());
    experimental::lightmetal::LightMetalBeginCapture();
    EXPECT_THROW(env.acquire_worker_stream_state_client(), std::runtime_error);
    EXPECT_FALSE(MetalEnvAccessor(env).impl().has_registered_context());
    (void)experimental::lightmetal::LightMetalEndCapture();
    auto token = env.acquire_worker_stream_state_client();
    EXPECT_THROW(experimental::lightmetal::LightMetalBeginCapture(), std::runtime_error);
    EXPECT_FALSE(LightMetalCaptureContext::get().is_tracing());
    EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
#else
    GTEST_SKIP() << "LightMetal capture is not built";
#endif
}
TEST_F(WorkerClientTest, ExistingForeignContextIsRejectedWithoutMutation) {
    auto& context = MetalContext::instance(DEFAULT_CONTEXT_ID, DeviceProfilerMode::Disabled);
    EXPECT_THROW(MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled), std::runtime_error);
    EXPECT_NO_THROW(context.validate_worker_stream_state_client({}));
    EXPECT_FALSE(context.worker_stream_state_provider(0));
}
TEST_F(WorkerClientTest, ExclusiveClientRejectsUnknownAndSecondClient) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    auto& context = MetalContext::instance();
    EXPECT_THROW(context.validate_worker_stream_state_client({}), std::runtime_error);
    EXPECT_THROW(context.validate_worker_stream_state_access(), std::runtime_error);
    EXPECT_THROW(context.validate_worker_stream_state_trace(), std::runtime_error);
    EXPECT_THROW(MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled), std::runtime_error);
    EXPECT_NO_THROW(context.validate_worker_stream_state_client(client));
    { WorkerStreamStateAccess access(client);
      EXPECT_NO_THROW(context.validate_worker_stream_state_access()); }
    EXPECT_THROW(context.validate_worker_stream_state_access(), std::runtime_error);
    EXPECT_FALSE(context.worker_stream_state_provider(0));
}
TEST_F(WorkerClientTest, ProgramBindingIsExplicitAndCannotBeReplaced) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    auto& context = MetalContext::instance();
    Program external;
    EXPECT_THROW(external.impl().validate_worker_stream_state_client(context), std::runtime_error);
    external.set_worker_stream_state_owner(tt::worker_stream_state::Owner::Program);
    EXPECT_THROW(external.impl().validate_worker_stream_state_client(context), std::runtime_error);
    external.impl().bind_worker_stream_state_client(client);
    EXPECT_NO_THROW(external.impl().validate_worker_stream_state_client(context));
    EXPECT_THROW(external.impl().bind_worker_stream_state_client({}), std::runtime_error);
    EXPECT_THROW(external.impl().bind_worker_stream_state_client(client), std::runtime_error);
}
TEST_F(WorkerClientTest, ActualQueueAndDirectDispatchRejectBeforeProgramPreparation) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    WorkerStreamStateAccess access(client);
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), Program{});
    auto& queue = mesh->mesh_command_queue();
    EXPECT_THROW(distributed::EnqueueMeshWorkload(queue, workload, false), std::runtime_error);
    EXPECT_THROW(queue.enqueue_mesh_workload(workload, false), std::runtime_error);
    EXPECT_THROW(queue.enqueue_trace(distributed::MeshTraceId{0}, false), std::runtime_error);
    auto& program = workload.get_programs().begin()->second;
    auto* device = mesh->get_view().get_devices().front();
    EXPECT_THROW(experimental::DispatchCompiledProgramToDevice(device, program), std::runtime_error);
    EXPECT_THROW(experimental::ConfigureProgramWithoutLaunch(device, program), std::runtime_error);
    expectWorkerClientRejection([&] { detail::WriteRuntimeArgsToDevice(device, program, false); }, "uncoordinated or stale SDK client");
    expectWorkerClientRejection([&] { detail::WriteRuntimeArgsToDevice(device, program, true); }, "uncoordinated or stale SDK client");
    expectWorkerClientRejection([&] { detail::ConfigureDeviceWithProgram(device, program, false); }, "uncoordinated or stale SDK client");
    expectWorkerClientRejection([&] { detail::ConfigureDeviceWithProgram(device, program, true); }, "uncoordinated or stale SDK client");
    expectWorkerClientRejection([&] { program.impl().compile_and_allocate(device, false); }, "uncoordinated or stale SDK client");
    expectWorkerClientRejection([&] { program.impl().compile_and_allocate(device, true); }, "uncoordinated or stale SDK client");
    expectWorkerClientRejection([&] { program.impl().compile(device, true); }, "uncoordinated or stale SDK client");
    EXPECT_FALSE(program.impl().is_compiled());
    EXPECT_NO_THROW(mesh->close());
}

TEST_F(WorkerClientTest, ContextLifecycleRejectsBeforeMutation) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    auto& context = MetalContext::instance();
    const bool fast_dispatch = context.rtoptions().get_fast_dispatch();
    expectWorkerClientRejection([&] { context.teardown(); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.initialize_control_plane(); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.set_default_fabric_topology(); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.set_custom_fabric_topology("unused", {}); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.set_fabric_config(tt::tt_fabric::FabricConfig::DISABLED); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.set_fabric_tensix_config(tt::tt_fabric::FabricTensixConfig::DISABLED); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.initialize_fabric_config(); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.initialize_fabric_tensix_datamover_config(); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.set_fast_dispatch_mode(!fast_dispatch); }, "lifecycle requires its exclusive client");
    EXPECT_EQ(context.rtoptions().get_fast_dispatch(), fast_dispatch);
    expectWorkerClientRejection([&] { context.device_manager()->close_devices({}); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.device_manager()->close_device(0); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.device_manager()->reset_dispatch_topology(); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.device_manager()->initialize_dispatch_firmware(false); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.device_manager()->initialize_profiler(); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { context.device_manager()->initialize_fabric_and_dispatch_fw(); }, "lifecycle requires its exclusive client");
    expectWorkerClientRejection([&] { experimental::DispatchContext::get().reset(); }, "lifecycle requires its exclusive client");
    WorkerStreamStateAccess access(client);
    EXPECT_NO_THROW(context.teardown());
}

TEST_F(WorkerClientTest, PhysicalDeviceAndMeshLifecycleRequireOwnerAccess) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    WorkerStreamStateAccess access(client);
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto* device = mesh->get_view().get_devices().front();
    auto& context = MetalContext::instance();
    {
        WorkerStreamStateAccess foreign({});
        expectWorkerClientRejection([&] { device->close(); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { tt::tt_fabric::create_and_compile_fabric_program(device); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { context.device_manager()->close_device(device->id()); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { context.device_manager()->close_devices({device}); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { mesh->close(); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { experimental::DispatchContext::get().initialize_fast_dispatch(mesh.get()); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { experimental::DispatchContext::get().terminate_fast_dispatch(mesh.get()); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { experimental::DispatchContext::get().set_configure_only(mesh.get(), true); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { experimental::DispatchContext::get().enable_asynchronous_slow_dispatch(mesh.get()); }, "lifecycle requires its exclusive client");
        expectWorkerClientRejection([&] { experimental::DispatchContext::get().disable_asynchronous_slow_dispatch(mesh.get()); }, "lifecycle requires its exclusive client");
        EXPECT_TRUE(device->is_initialized());
        EXPECT_TRUE(mesh->is_initialized());
    }
    EXPECT_NO_THROW(mesh->close());
}

TEST_F(WorkerClientTest, MeshRaiiCleanupRetainsOwnerOutsideCallerScope) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    std::shared_ptr<distributed::MeshDevice> mesh;
    {
        WorkerStreamStateAccess access(client);
        mesh = distributed::MeshDevice::create_unit_mesh(0);
    }
    client.reset();
    EXPECT_THROW(MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled), std::runtime_error);
    EXPECT_NO_THROW(mesh.reset());
    EXPECT_TRUE(MetalContext::instance().device_manager()->get_all_active_devices().empty());
    EXPECT_NO_THROW(client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled));
}

TEST_F(WorkerClientTest, OrdinarySdkProgramDoesNotRequireExclusiveClient) {
    auto& context = MetalContext::instance(DEFAULT_CONTEXT_ID, DeviceProfilerMode::Disabled);
    Program program;
    EXPECT_NO_THROW(program.impl().validate_worker_stream_state_client(context));
    EXPECT_NO_THROW(context.validate_worker_stream_state_access());
    EXPECT_NO_THROW(context.teardown());
}

TEST_F(WorkerClientTest, TraceCaptureRejectsBeforeSdkOrQueueMutation) {
    experimental::test_helpers::ScopedSlowDispatchOverride slow_dispatch;
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    WorkerStreamStateAccess access(client);
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& queue = mesh->mesh_command_queue();
    const distributed::MeshTraceId trace{123};
    const char* diagnostic = "SDK trace";
    expectWorkerClientRejection([&] { distributed::BeginTraceCapture(mesh.get(), 0); }, diagnostic);
    expectWorkerClientRejection([&] { mesh->begin_mesh_trace(queue, trace); }, diagnostic);
    expectWorkerClientRejection([&] { mesh->end_mesh_trace(queue, trace); }, diagnostic);
    expectWorkerClientRejection([&] { queue.record_begin(trace, {}); }, diagnostic);
    expectWorkerClientRejection([&] { queue.record_end(); }, diagnostic);
    EXPECT_FALSE(queue.trace_id().has_value());
    {
        WorkerStreamStateAccess foreign({});
        expectWorkerClientRejection([&] { distributed::BeginTraceCapture(mesh.get(), 0); }, diagnostic);
    }
    EXPECT_NO_THROW(mesh->close());
}

TEST_F(WorkerClientTest, SubdeviceAndQueueReconfigurationRequireActualClient) {
    experimental::test_helpers::ScopedSlowDispatchOverride slow_dispatch;
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    WorkerStreamStateAccess access(client);
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& queue = mesh->impl().mesh_command_queue_base(0);
    auto& slow_queue = dynamic_cast<distributed::SDMeshCommandQueue&>(queue);
    const auto original = mesh->get_active_sub_device_manager_id();
    const auto original_stall = mesh->get_sub_device_stall_group();
    const char* diagnostic = "lifecycle requires its exclusive client";
    {
        WorkerStreamStateAccess foreign({});
        expectWorkerClientRejection([&] { mesh->load_sub_device_manager(original); }, diagnostic);
        expectWorkerClientRejection([&] { mesh->clear_loaded_sub_device_manager(); }, diagnostic);
        expectWorkerClientRejection([&] { mesh->remove_sub_device_manager(original); }, diagnostic);
        expectWorkerClientRejection([&] { mesh->create_sub_device_manager(std::initializer_list<SubDevice>{}, 0); }, diagnostic);
        expectWorkerClientRejection([&] { mesh->set_sub_device_stall_group({}); }, diagnostic);
        expectWorkerClientRejection([&] { mesh->reset_sub_device_stall_group(); }, diagnostic);
        expectWorkerClientRejection([&] { queue.reset_worker_state(true, 0, {}, {}, {}); }, diagnostic);
        expectWorkerClientRejection([&] { slow_queue.set_configure_only(true); }, diagnostic);
        expectWorkerClientRejection([&] { slow_queue.enable_asynchronous_slow_dispatch(); }, diagnostic);
        expectWorkerClientRejection([&] { slow_queue.disable_asynchronous_slow_dispatch(); }, diagnostic);
    }
    EXPECT_EQ(mesh->get_active_sub_device_manager_id(), original);
    EXPECT_EQ(mesh->get_sub_device_stall_group(), original_stall);
    EXPECT_NO_THROW(mesh->reset_sub_device_stall_group());
    EXPECT_NO_THROW(slow_queue.set_configure_only(false));
    EXPECT_NO_THROW(slow_queue.enable_asynchronous_slow_dispatch());
    EXPECT_NO_THROW(slow_queue.disable_asynchronous_slow_dispatch());
    EXPECT_NO_THROW(mesh->close());
}

TEST_F(WorkerClientTest, PublishedContextAlreadyRejectsForeignThread) {
    worker_client_publication_rejected = false;
    worker_client_publication_released = false;
    std::binary_semaphore ready{0};
    std::thread foreign([&] {
        pthread_setname_np(pthread_self(), "worker-state-r");
        ready.release();
        worker_client_publication_release.acquire();
        try {
            auto& context = MetalContext::instance();
            context.validate_worker_stream_state_access();
        } catch (const std::runtime_error& error) {
            worker_client_publication_rejected =
                std::string(error.what()).find("lifecycle requires its exclusive client") != std::string::npos;
        }
        trex_worker_client_reader_finished();
    });
    ready.acquire();
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    // Without the debugger the reader starts after acquisition; the deterministic
    // debugger probe starts it at the real g_instances publication boundary.
    trex_worker_client_release_reader();
    foreign.join();
    EXPECT_TRUE(worker_client_publication_rejected.load());
}

TEST_F(WorkerClientTest, RetiredGenerationRemainsClosedToForeignThread) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    client.reset();
    std::atomic<bool> rejected{false};
    std::thread foreign([&] {
        try { MetalContext::instance().validate_worker_stream_state_access(); }
        catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("lifecycle requires its exclusive client") != std::string::npos;
        }
    });
    foreign.join();
    EXPECT_TRUE(rejected.load());
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    EXPECT_NO_THROW(MetalContext::instance().validate_worker_stream_state_client(client));
}

TEST_F(WorkerClientTest, LightMetalAndClientCannotBothWinPublicationRace) {
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
    worker_client_capture_released = false;
    std::binary_semaphore ready{0};
    std::atomic<bool> capture_started{false};
    std::atomic<bool> capture_rejected{false};
    std::thread foreign([&] {
        pthread_setname_np(pthread_self(), "worker-capture");
        ready.release();
        worker_client_capture_release.acquire();
        try {
            experimental::lightmetal::LightMetalBeginCapture();
            capture_started = LightMetalCaptureContext::get().is_tracing();
        } catch (const std::runtime_error& error) {
            capture_rejected = std::string(error.what()).find("SDK trace") != std::string::npos;
        }
        trex_worker_client_capture_finished();
    });
    ready.acquire();
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    trex_worker_client_release_capture();
    foreign.join();
    EXPECT_FALSE(client && capture_started.load());
    EXPECT_TRUE(capture_rejected.load());
    if (capture_started)
        (void)experimental::lightmetal::LightMetalEndCapture();
#else
    GTEST_SKIP() << "LightMetal capture is not built; not capture exclusion evidence";
#endif
}

TEST_F(WorkerClientTest, ActiveLightMetalCaptureRejectsClientBeforeContextCreation) {
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
    experimental::lightmetal::LightMetalBeginCapture();
    ASSERT_TRUE(LightMetalCaptureContext::get().is_tracing());
    expectWorkerClientRejection([&] {
        client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    }, "active LightMetal capture");
    EXPECT_FALSE(MetalContext::instance_exists());
    (void)experimental::lightmetal::LightMetalEndCapture();
    EXPECT_FALSE(LightMetalCaptureContext::get().is_tracing());
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    expectWorkerClientRejection([&] { experimental::lightmetal::LightMetalBeginCapture(); }, "SDK trace");
#else
    GTEST_SKIP() << "LightMetal capture is not built; not capture exclusion evidence";
#endif
}

TEST_F(WorkerClientTest, OrdinarySdkLightMetalCaptureRemainsAvailable) {
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
    auto& context = MetalContext::instance(DEFAULT_CONTEXT_ID, DeviceProfilerMode::Disabled);
    EXPECT_NO_THROW(experimental::lightmetal::LightMetalBeginCapture());
    EXPECT_TRUE(LightMetalCaptureContext::get().is_tracing());
    EXPECT_NO_THROW(context.validate_worker_stream_state_client({}));
    EXPECT_NO_THROW((void)experimental::lightmetal::LightMetalEndCapture());
    EXPECT_FALSE(LightMetalCaptureContext::get().is_tracing());
#else
    GTEST_SKIP() << "LightMetal capture is not built; not capture exclusion evidence";
#endif
}

TEST_F(WorkerClientTest, StaleContextGenerationCannotBorrowNewClient) {
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    auto old = client;
    { WorkerStreamStateAccess access(client); MetalContext::destroy_all_instances(false); }
    client = MetalContext::acquire_worker_stream_state_client(DeviceProfilerMode::Disabled);
    auto& context = MetalContext::instance();
    EXPECT_THROW(context.validate_worker_stream_state_client(old), std::runtime_error);
    EXPECT_NO_THROW(context.validate_worker_stream_state_client(client));
}
}  // namespace
}  // namespace tt::tt_metal
