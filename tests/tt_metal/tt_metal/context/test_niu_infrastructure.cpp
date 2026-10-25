// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "impl/context/niu_admission.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <memory>
#include <limits>
#include <future>
#include <chrono>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/sub_device.hpp>
#include "distributed/mesh_device_impl.hpp"
#include "distributed/fd_mesh_command_queue.hpp"
#include "distributed/pinned_memory_impl.hpp"
#include "../api/metal2_host_api/test_helpers.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/system_memory_manager.hpp"
#include "impl/internal/service/service_core_manager_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/program_execution.hpp"
#include "impl/threading/thread_pool.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {
namespace {

template <class Allocation>
constexpr bool mutable_managed_token = requires(Allocation& allocation) {
    allocation.token = NiuAdmission::Token{};
};
template <class Allocation>
constexpr bool mutable_managed_selections = requires(Allocation& allocation) {
    allocation.selections.clear();
};

TEST(NiuManagedInvocationContractTest, AllocationProjectionCannotBeForgedOrMutated) {
    EXPECT_FALSE(std::is_default_constructible_v<NiuAdmission::Allocation>);
    EXPECT_FALSE(std::is_aggregate_v<NiuAdmission::Allocation>);
    EXPECT_FALSE(mutable_managed_token<NiuAdmission::Allocation>);
    EXPECT_FALSE(mutable_managed_selections<NiuAdmission::Allocation>);
    EXPECT_FALSE(std::is_copy_assignable_v<NiuAdmission::Allocation>);
    EXPECT_FALSE(std::is_move_assignable_v<NiuAdmission::Allocation>);
    using SelectionView = decltype(std::declval<const NiuAdmission::Allocation&>().selections());
    EXPECT_TRUE((std::is_same_v<SelectionView, std::span<const detail::NiuSelection>>));
}

TEST(NiuManagedInvocationContractTest, PreparationHasOneHostOwnerAndSupportsNestedBorrowing) {
    detail::ProgramPreparation preparation;
    ASSERT_TRUE(preparation.acquire());
    ASSERT_TRUE(preparation.acquire());
    EXPECT_TRUE(preparation.owned());
    auto competing_acquire = [&] {
        bool acquired = false;
        std::thread contender([&] {
            acquired = preparation.acquire();
            if (acquired) preparation.release();
        });
        contender.join();
        return acquired;
    };
    EXPECT_FALSE(competing_acquire());
    EXPECT_TRUE(preparation.owned());
    preparation.release();
    EXPECT_FALSE(competing_acquire());
    preparation.release();
    EXPECT_FALSE(preparation.occupied());
    EXPECT_TRUE(competing_acquire());
    EXPECT_FALSE(preparation.occupied());
}

TEST(NiuManagedInvocationContractTest, PreparationCancellationWaitsForLastBorrow) {
    detail::ProgramPreparation preparation;
    ASSERT_TRUE(preparation.acquire());
    ASSERT_TRUE(preparation.acquire());
    preparation.cancel();
    EXPECT_TRUE(preparation.cancelled());
    EXPECT_FALSE(preparation.ready_to_cancel());
    EXPECT_FALSE(preparation.acquire());
    preparation.release();
    EXPECT_FALSE(preparation.ready_to_cancel());
    preparation.release();
    EXPECT_TRUE(preparation.ready_to_cancel());
    EXPECT_FALSE(preparation.acquire());
}

template <class Execution, class... Inputs>
void expect_unbooted_managed_rejection(Inputs&&... inputs) {
    if constexpr (requires { Execution::acquire_managed(std::forward<Inputs>(inputs)...); }) {
        try {
            auto invocation = Execution::acquire_managed(std::forward<Inputs>(inputs)...);
            FAIL() << "Managed invocation must require a successful actual SDK firmware boot";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("boot"), std::string::npos) << error.what();
        }
    } else {
        ADD_FAILURE() << "SDK has no token-bound managed invocation entrypoint";
    }
}

template <class Execution>
void expect_ordered_managed_rejection(
    distributed::MeshDevice& mesh,
    std::span<distributed::MeshWorkload* const> workloads,
    std::string_view expected_error) {
    if constexpr (requires { Execution::acquire_managed(mesh, workloads); }) {
        try {
            auto invocation = Execution::acquire_managed(mesh, workloads);
            FAIL() << "Managed ordered invocation must reject before publishing any member";
        } catch (const std::runtime_error& error) {
            const std::string diagnostic = error.what();
            EXPECT_NE(diagnostic.find(expected_error), std::string::npos) << diagnostic;
            if (expected_error != "boot") {
                EXPECT_EQ(diagnostic.find("boot"), std::string::npos)
                    << "Every workload must be structurally checked before looking up any firmware boot: "
                    << diagnostic;
            }
        }
    } else {
        ADD_FAILURE() << "SDK has no managed invocation entrypoint for an ordered workload collection";
    }
}

// Exercise the real SDK managers and SOC mapping. Minimal context/device setup
// stops before firmware compilation or launch and cannot publish LoadedFirmware.
class NiuInfrastructureTest : public ::testing::Test {
protected:
    void SetUp() override {
        MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1).value());
        descriptor.set_device_profiler_mode(DeviceProfilerMode::Disabled);
        environment = std::make_unique<MetalEnv>(std::move(descriptor));
        context_id = MetalContext::create_instance(*environment);
        context = &MetalContext::instance(*context_id);
        context->rtoptions().set_fast_dispatch(true);
        context->initialize(dispatch_config, 1, {}, DEFAULT_WORKER_L1_SIZE, true);
        const auto& cluster = MetalEnvAccessor(*environment).impl().get_cluster();
        ASSERT_EQ(cluster.get_target_device_type(), tt::TargetDevice::Mock);
        ASSERT_EQ(cluster.all_chip_ids().size(), 1u);
        chip = *cluster.all_chip_ids().begin();
        device = std::make_unique<Device>(
            environment.get(), context, chip, 1, 0, 0, ttsl::Span<const std::uint32_t>{}, true);
        ASSERT_FALSE(context->native_firmware(chip));
        free_cores = dispatch().get_available_dispatch_cores(chip);
        ASSERT_GE(free_cores.size(), 2u);
    }

    void TearDown() override {
        device.reset();
        if (context_id) {
            MetalContext::destroy_instance(false, *context_id);
        }
        environment.reset();
    }

    dispatch_core_manager& dispatch() { return context->get_dispatch_core_manager(); }
    internal::ServiceCoreManager& services() { return context->get_service_core_manager(); }

    void quarantine(CoreCoord logical_core, detail::NiuNoc noc = detail::NiuNoc::Noc0) {
        const auto& cluster = MetalEnvAccessor(*environment).impl().get_cluster();
        const auto physical = cluster.get_soc_desc(chip).get_physical_tensix_core_from_logical(logical_core);
        const std::array keys{detail::NiuKey{chip, physical, noc}};
        std::string error;
        auto token = context->niu_admission().reserve_ordinary(keys, error);
        ASSERT_TRUE(token) << error;
        ASSERT_TRUE(context->niu_admission().quarantine(*token, error)) << error;
    }

    template <typename Action>
    void expect_quarantined_program_rejection(Action&& action) {
        const CoreCoord worker{0, 0};
        ASSERT_NO_FATAL_FAILURE(quarantine(worker, detail::NiuNoc::Noc1));
        Program program;
        CreateKernelFromString(
            program,
            "void kernel_main() {}",
            worker,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
        ASSERT_FALSE(program.impl().is_compiled());
        ASSERT_FALSE(program.impl().is_finalized());
        try {
            action(program);
            FAIL() << "Quarantined physical NIU must reject before Program preparation or device configuration";
        } catch (const std::runtime_error& error) {
            const std::string diagnostic = error.what();
            EXPECT_NE(diagnostic.find("NIU"), std::string::npos) << diagnostic;
            EXPECT_NE(diagnostic.find("quarantin"), std::string::npos) << diagnostic;
        }
        EXPECT_FALSE(program.impl().is_compiled());
        EXPECT_FALSE(program.impl().is_finalized());
        EXPECT_FALSE(context->native_firmware(chip));
    }

    Program worker_program(CoreCoord core = {0, 0}) {
        Program program(std::make_shared<detail::ProgramImpl>(*context_id));
        EXPECT_EQ(program.impl().get_context_id(), *context_id);
        CreateKernelFromString(program, "void kernel_main() {}", core,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
        return program;
    }

    void open_device() {
        ASSERT_TRUE(device->initialize(1, 0, 0, DEFAULT_WORKER_L1_SIZE, {}, false));
        ASSERT_TRUE(device->is_initialized());
        ASSERT_FALSE(context->native_firmware(chip));
    }

    const DispatchCoreConfig dispatch_config{DispatchCoreType::WORKER, DispatchCoreAxis::COL};
    std::unique_ptr<MetalEnv> environment;
    std::optional<ContextId> context_id;
    MetalContext* context = nullptr;
    std::unique_ptr<Device> device;
    ChipId chip = -1;
    std::vector<CoreCoord> free_cores;
};

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsLaunchBeforeCompilation) {
    expect_quarantined_program_rejection(
        [&](Program& program) { detail::LaunchProgram(device.get(), program, false, true); });
}

TEST_F(NiuInfrastructureTest, ManagedDirectInvocationRequiresActualBootBeforePreparation) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    const std::array<IDevice*, 1> devices{device.get()};
    expect_unbooted_managed_rejection<detail::ProgramExecution>(program, std::span<IDevice* const>(devices));
    // A rejected acquisition must not publish a stale Program binding or keep
    // any ordinary/full-NIU ownership behind.
    EXPECT_NO_THROW(detail::ProgramExecution::reserve(program, devices));
    expect_unbooted_managed_rejection<detail::ProgramExecution>(program, std::span<IDevice* const>(devices));
    EXPECT_FALSE(program.impl().is_compiled());
    EXPECT_FALSE(program.impl().is_finalized());
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsConfigureOnlyBeforeCompilation) {
    expect_quarantined_program_rejection(
        [&](Program& program) { experimental::ConfigureProgramWithoutLaunch(device.get(), program); });
}

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsDirectDispatchBeforePreparedProgramChecks) {
    expect_quarantined_program_rejection(
        [&](Program& program) { experimental::DispatchCompiledProgramToDevice(device.get(), program); });
}

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsDeviceConfigurationBeforeAllocation) {
    expect_quarantined_program_rejection(
        [&](Program& program) { detail::ConfigureDeviceWithProgram(device.get(), program, true); });
}

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsRuntimeArgumentsBeforeDeviceWrites) {
    expect_quarantined_program_rejection(
        [&](Program& program) { detail::WriteRuntimeArgsToDevice(device.get(), program, true); });
}

TEST_F(NiuInfrastructureTest, ActualDeviceCloseRetiresOrdinaryTokensWithoutNativeBoot) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    const auto physical = context->get_cluster().get_soc_desc(chip).get_physical_tensix_core_from_logical({0, 0});
    const std::array keys{detail::NiuKey{chip, physical, detail::NiuNoc::Noc0}};
    std::string error;
    auto token = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(token) << error;
    ASSERT_TRUE(device->close());
    EXPECT_FALSE(context->niu_admission().mark_issued(*token, error));
    EXPECT_FALSE(context->niu_admission().cancel_unissued(*token, error));
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, ClosedActualDeviceRejectsNewExecutionReservation) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    ASSERT_TRUE(device->close());
    EXPECT_THROW(detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()}), std::runtime_error);
    EXPECT_THROW(detail::WriteRuntimeArgsToDevice(device.get(), program, true), std::runtime_error);
    EXPECT_FALSE(program.impl().is_compiled());
}

TEST_F(NiuInfrastructureTest, ExecutionReservationCanOutliveItsContextWithoutFirmware) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    std::shared_ptr<detail::ProgramExecution> execution;
    {
        auto program = worker_program();
        execution = detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()});
    }
    device.reset();
    MetalContext::destroy_instance(false, *context_id);
    context_id.reset();
    environment.reset();
    EXPECT_NO_THROW(execution.reset());
}

TEST_F(NiuInfrastructureTest, DirectPreparationFailureDoesNotPoisonEitherPhysicalNiu) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    EXPECT_THROW(experimental::DispatchCompiledProgramToDevice(device.get(), program), std::runtime_error);
    const auto physical = context->get_cluster().get_soc_desc(chip).get_physical_tensix_core_from_logical({0, 0});
    const std::array keys{detail::NiuKey{chip, physical, detail::NiuNoc::Noc0},
                          detail::NiuKey{chip, physical, detail::NiuNoc::Noc1}};
    std::string error;
    auto next = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(next) << error;
    EXPECT_TRUE(context->niu_admission().cancel_unissued(*next, error)) << error;
    EXPECT_FALSE(program.impl().is_compiled());
}

TEST_F(NiuInfrastructureTest, EthernetProgramDoesNotClaimWorkerNius) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    ASSERT_NO_FATAL_FAILURE(quarantine({0, 0}, detail::NiuNoc::Noc0));
    ASSERT_NO_FATAL_FAILURE(quarantine({1, 0}, detail::NiuNoc::Noc1));
    const auto ethernet = device->get_inactive_ethernet_cores();
    ASSERT_FALSE(ethernet.empty());
    Program program;
    CreateKernelFromString(program, "void kernel_main() {}", *ethernet.begin(), EthernetConfig{.eth_mode = Eth::IDLE});
    EXPECT_NO_THROW(detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()}));
    EXPECT_FALSE(program.impl().is_compiled());
}

class NiuExecutionMeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value());
        descriptor.set_device_profiler_mode(DeviceProfilerMode::Disabled);
        environment = std::make_unique<MetalEnv>(std::move(descriptor));
        context_id = MetalContext::create_instance(*environment);
        context = &MetalContext::instance(*context_id);
        const DispatchCoreConfig dispatch_config{DispatchCoreType::WORKER, DispatchCoreAxis::COL};
        // Use the existing minimal context seam before creating actual Mesh,
        // Device and command-queue objects. Matching initialization parameters
        // preserve that context and never enter the firmware JIT/build phase.
        context->initialize(dispatch_config, 1, {}, DEFAULT_WORKER_L1_SIZE, true);
        mesh = distributed::MeshDeviceImpl::create(
            *context_id,
            distributed::MeshDeviceConfig(distributed::MeshShape{1, 2}),
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            1,
            dispatch_config,
            {},
            DEFAULT_WORKER_L1_SIZE);
        ASSERT_EQ(context->get_cluster().get_target_device_type(), tt::TargetDevice::Mock);
        ASSERT_EQ(mesh->get_devices().size(), 2u);
        for (auto* device : mesh->get_devices()) {
            ASSERT_FALSE(context->native_firmware(device->id()));
        }
    }
    void TearDown() override {
        if (mesh) {
            const auto devices = mesh->get_devices();
            // Actual-chip withdrawal tests deliberately close one member in
            // the body. Complete physical close before releasing the uniform
            // mesh facade; its state queries require matching member states.
            if (std::any_of(devices.begin(), devices.end(), [](auto* device) { return !device->is_initialized(); })) {
                for (auto* device : devices) {
                    if (device->is_initialized()) {
                        EXPECT_TRUE(device->close());
                    }
                }
            }
            mesh->close();
            mesh.reset();
        }
        if (context_id) {
            MetalContext::destroy_instance(false, *context_id);
        }
        environment.reset();
    }
    void quarantine(IDevice* device, CoreCoord logical) {
        const auto core =
            context->get_cluster().get_soc_desc(device->id()).get_physical_tensix_core_from_logical(logical);
        const std::array keys{detail::NiuKey{device->id(), core, detail::NiuNoc::Noc1}};
        std::string error;
        auto token = context->niu_admission().reserve_ordinary(keys, error);
        ASSERT_TRUE(token) << error;
        ASSERT_TRUE(context->niu_admission().quarantine(*token, error)) << error;
    }
    distributed::MeshWorkload workload() {
        Program program(std::make_shared<detail::ProgramImpl>(*context_id));
        EXPECT_EQ(program.impl().get_context_id(), *context_id);
        CreateKernelFromString(
            program,
            "void kernel_main() {}",
            CoreCoord{0, 0},
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
        distributed::MeshWorkload result;
        result.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 0}), std::move(program));
        return result;
    }
    void expect_rejected_before_compile(distributed::MeshWorkload& work) {
        try {
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), work, false);
            FAIL() << "Quarantined FD GO recipient must reject before compilation";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("NIU"), std::string::npos) << error.what();
            EXPECT_NE(std::string(error.what()).find("quarantin"), std::string::npos) << error.what();
        }
        for (const auto& [range, program] : work.get_programs()) {
            EXPECT_FALSE(program.impl().is_compiled());
        }
    }
    void expect_ordered_rejection_before_preparation(
        std::span<distributed::MeshWorkload* const> workloads,
        std::string_view expected_error,
        std::span<distributed::MeshWorkload* const> intact_workloads) {
        for (unsigned attempt = 0; attempt != 2; ++attempt) {
            SCOPED_TRACE(attempt);
            expect_ordered_managed_rejection<detail::ProgramExecution>(*mesh, workloads, expected_error);
            for (auto* work : intact_workloads) {
                EXPECT_EQ(detail::ProgramExecution::validate_workload(*mesh, *work), nullptr);
                EXPECT_NO_THROW(
                    detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{0}}, *work));
                for (const auto& [range, program] : work->get_programs()) {
                    EXPECT_FALSE(program.impl().is_compiled());
                    EXPECT_FALSE(program.impl().is_finalized());
                }
            }
            EXPECT_NO_THROW(detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{0}}));
            for (auto* device : mesh->get_devices()) EXPECT_FALSE(context->native_firmware(device->id()));
        }
    }
    std::shared_ptr<distributed::MeshDevice> mesh;
    std::unique_ptr<MetalEnv> environment;
    std::optional<ContextId> context_id;
    MetalContext* context = nullptr;
};

TEST_F(NiuExecutionMeshTest, EventAllocationRejectsExhaustionWithoutReusingZero) {
    auto& manager = mesh->get_devices().front()->sysmem_manager();
    constexpr auto maximum = std::numeric_limits<uint32_t>::max();
    manager.set_current_and_last_completed_event(0, maximum - 2, maximum - 2);
    EXPECT_EQ(manager.get_next_event(0), maximum - 1);
    for (unsigned attempt = 0; attempt != 2; ++attempt) {
        EXPECT_THROW(manager.get_next_event(0), std::runtime_error);
        EXPECT_EQ(manager.get_current_event(0), maximum - 1);
        EXPECT_EQ(manager.get_last_completed_event(0), maximum - 2);
    }
}

TEST_F(NiuExecutionMeshTest, EventIncrementRejectsOverflowWithoutChangingTheNextEvent) {
    auto& manager = mesh->get_devices().front()->sysmem_manager();
    constexpr auto maximum = std::numeric_limits<uint32_t>::max();
    manager.set_current_and_last_completed_event(0, maximum - 2, maximum - 2);
    EXPECT_THROW(manager.increment_event_id(0, 3), std::runtime_error);
    EXPECT_EQ(manager.get_current_event(0), maximum - 2);
    EXPECT_EQ(manager.get_next_event(0), maximum - 1);
    EXPECT_THROW(manager.increment_event_id(0, 1), std::runtime_error);
    EXPECT_EQ(manager.get_current_event(0), maximum - 1);
}

TEST_F(NiuExecutionMeshTest, QuiescenceWaitSerializesWithTheQueueApiBeforeInspectingPendingWork) {
    // The production constructor accepts its real API lock owner. Use that
    // public seam, with actual SDK queue/pools, without changing device state.
    std::mutex api_mutex;
    auto pool = create_passthrough_thread_pool(*context_id);
    auto shared_state = std::make_shared<CQSharedState>();
    distributed::FDMeshCommandQueue queue(
        mesh.get(), 0, pool, pool, shared_state,
        [&] { return std::lock_guard<std::mutex>(api_mutex); }, nullptr);
    std::promise<void> entered;
    auto started = entered.get_future();
    std::future<void> wait;
    {
        std::lock_guard api_lock(api_mutex);
        wait = std::async(std::launch::async, [&] {
            entered.set_value();
            queue.wait_for_completion(false);
        });
        started.wait();
        EXPECT_EQ(wait.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    }
    EXPECT_NO_THROW(wait.get());
}

TEST_F(NiuExecutionMeshTest, NumericEventProgressCannotPrunePinnedMemoryBeforeSynchronization) {
    context->rtoptions().set_fast_dispatch(true);
    std::array<uint32_t, 16> host_data{};
    experimental::PinnedMemoryImpl pinned(
        MetalEnvAccessor(*environment).impl(), mesh->get_devices(), host_data.data(), sizeof(host_data));
    const distributed::MeshEvent event(
        7, mesh->mesh_command_queue(), distributed::MeshCoordinateRange(mesh->shape()));
    // This updates ordinary host bookkeeping only. No device response or
    // firmware completion has been observed by the private event receiver.
    for (auto* device : mesh->get_devices())
        device->sysmem_manager().set_last_completed_event(0, 7);
    pinned.add_barrier_event(event);
    EXPECT_TRUE(pinned.lock_may_block());
    EXPECT_EQ(pinned.lock(), host_data.data());
    EXPECT_FALSE(pinned.lock_may_block());
    pinned.unlock();
    for (auto* device : mesh->get_devices()) EXPECT_FALSE(context->native_firmware(device->id()));
}

TEST_F(NiuExecutionMeshTest, EventEpochResetSerializesWithTheQueueApiBeforeInspectingPendingWork) {
    std::mutex api_mutex;
    auto pool = create_passthrough_thread_pool(*context_id);
    auto shared_state = std::make_shared<CQSharedState>();
    distributed::FDMeshCommandQueue queue(
        mesh.get(), 0, pool, pool, shared_state,
        [&] { return std::lock_guard<std::mutex>(api_mutex); }, nullptr);
    std::promise<void> entered;
    auto started = entered.get_future();
    std::future<void> reset;
    {
        std::lock_guard api_lock(api_mutex);
        reset = std::async(std::launch::async, [&] {
            entered.set_value();
            queue.finish_and_reset_in_use();
        });
        started.wait();
        EXPECT_EQ(reset.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    }
    EXPECT_NO_THROW(reset.get());
}

TEST_F(NiuExecutionMeshTest, FastDispatchAdmitsIdleWorkerGoRecipientsBeforeCompilation) {
    auto work = workload();
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().front(), {1, 0}));
    expect_rejected_before_compile(work);
}

TEST_F(NiuExecutionMeshTest, ManagedMeshInvocationRequiresActualBootBeforePreparation) {
    auto work = workload();
    Program second(std::make_shared<detail::ProgramImpl>(*context_id));
    ASSERT_EQ(second.impl().get_context_id(), *context_id);
    CreateKernelFromString(second, "void kernel_main() {}", CoreCoord{1, 0},
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
    work.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 1}), std::move(second));
    const std::array workloads{&work};
    expect_unbooted_managed_rejection<detail::ProgramExecution>(*mesh, workloads);
    EXPECT_EQ(detail::ProgramExecution::validate_workload(*mesh, work), nullptr);
    EXPECT_NO_THROW(detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{0}}, work));
    expect_unbooted_managed_rejection<detail::ProgramExecution>(*mesh, workloads);
    for (const auto& [range, program] : work.get_programs()) {
        EXPECT_FALSE(program.impl().is_compiled());
        EXPECT_FALSE(program.impl().is_finalized());
    }
    for (auto* device : mesh->get_devices()) EXPECT_FALSE(context->native_firmware(device->id()));
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRequireActualBootBeforePreparingEitherGroup) {
    auto first = workload();
    auto second = workload();
    const std::array ordered{&first, &second};
    expect_ordered_rejection_before_preparation(ordered, "boot", ordered);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsValidateTheLastGroupBeforeBootLookup) {
    auto first = workload();
    distributed::MeshWorkload empty;
    const std::array ordered{&first, &empty};
    expect_ordered_rejection_before_preparation(ordered, "empty workload", ordered);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectAnEmptyCollection) {
    expect_ordered_rejection_before_preparation({}, "nonempty workload collection", {});
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectALaterNullWorkload) {
    auto first = workload();
    const std::array ordered{&first, static_cast<distributed::MeshWorkload*>(nullptr)};
    const std::array intact{&first};
    expect_ordered_rejection_before_preparation(ordered, "null workload", intact);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectALaterMovedFromWorkload) {
    auto first = workload();
    auto source = workload();
    auto moved = std::move(source);
    const std::array ordered{&first, &source};
    const std::array intact{&first, &moved};
    expect_ordered_rejection_before_preparation(ordered, "moved-from workload", intact);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectARepeatedWorkloadOwner) {
    auto first = workload();
    const std::array ordered{&first, &first};
    const std::array intact{&first};
    expect_ordered_rejection_before_preparation(ordered, "duplicate workload", intact);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectTheSameProgramOwnerInDifferentGroups) {
    auto first = workload();
    auto& program = first.get_programs().begin()->second;
    Program alias(program.impl().shared_from_this());
    distributed::MeshWorkload second;
    second.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 1}), std::move(alias));
    const std::array ordered{&first, &second};
    expect_ordered_rejection_before_preparation(ordered, "distinct", ordered);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectAForeignProgramContextBeforeBootLookup) {
    auto first = workload();
    Program foreign;
    ASSERT_NE(foreign.impl().get_context_id(), *context_id);
    CreateKernelFromString(
        foreign, "void kernel_main() {}", CoreCoord{0, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
    distributed::MeshWorkload second;
    second.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 1}), std::move(foreign));
    const std::array ordered{&first, &second};
    expect_ordered_rejection_before_preparation(ordered, "another context", ordered);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectOverlappingRangesInALaterGroupBeforeBootLookup) {
    auto first = workload();
    auto second = workload();
    Program overlapping(std::make_shared<detail::ProgramImpl>(*context_id));
    CreateKernelFromString(
        overlapping, "void kernel_main() {}", CoreCoord{1, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
    // The public mutable map can bypass add_program's range check. Acquisition
    // must revalidate the actual complete workload, including late groups.
    second.get_programs().emplace(distributed::MeshCoordinateRange(mesh->shape()), std::move(overlapping));
    const std::array ordered{&first, &second};
    expect_ordered_rejection_before_preparation(ordered, "overlap", ordered);
}

TEST_F(NiuExecutionMeshTest, FastDispatchAdmitsLocalChipWithoutAProgramBeforeCompilation) {
    auto work = workload();
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().back(), {0, 0}));
    expect_rejected_before_compile(work);
}

TEST_F(NiuExecutionMeshTest, DispatchReservationExcludesUnrelatedSubdevice) {
    SubDevice first(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    SubDevice second(std::array{CoreRangeSet(CoreRange({1, 0}, {1, 0}))});
    auto manager = mesh->create_sub_device_manager({first, second}, 0);
    mesh->load_sub_device_manager(manager);
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().back(), {1, 0}));
    EXPECT_NO_THROW(detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{0}}));
    EXPECT_THROW(detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{1}}), std::runtime_error);
}

TEST_F(NiuExecutionMeshTest, SubmeshDispatchDoesNotAdmitUnrelatedLocalChip) {
    auto local = mesh->create_submesh(distributed::MeshShape{1, 1}, distributed::MeshCoordinate{0, 0});
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().back(), {0, 0}));
    EXPECT_NO_THROW(detail::ProgramExecution::reserve_dispatch(*local, std::array{SubDeviceId{0}}));
    local->close();
}

TEST_F(NiuExecutionMeshTest, ManagerResetAdmitsOldWorkerAbsentFromNewManager) {
    SubDevice first(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    SubDevice second(std::array{CoreRangeSet(CoreRange({1, 0}, {1, 0}))});
    const auto old_manager = mesh->create_sub_device_manager({first, second}, 0);
    const auto new_manager = mesh->create_sub_device_manager({first}, 0);
    mesh->load_sub_device_manager(old_manager);
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().back(), {1, 0}));
    try {
        mesh->load_sub_device_manager(new_manager);
        FAIL() << "Reset GO must admit old-manager workers before remapping";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("quarantin"), std::string::npos) << error.what();
    }
    EXPECT_EQ(mesh->num_sub_devices(), 2u);
}

TEST_F(NiuExecutionMeshTest, MockManagerReductionDoesNotMintIssueOrStopOnClose) {
    SubDevice first(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    SubDevice second(std::array{CoreRangeSet(CoreRange({1, 0}, {1, 0}))});
    const auto old_manager = mesh->create_sub_device_manager({first, second}, 0);
    const auto new_manager = mesh->create_sub_device_manager({first}, 0);
    mesh->load_sub_device_manager(old_manager);
    mesh->load_sub_device_manager(new_manager);
    mesh->mesh_command_queue().finish();
    auto* device = mesh->get_devices().front();
    const auto physical =
        context->get_cluster().get_soc_desc(device->id()).get_physical_tensix_core_from_logical({1, 0});
    const std::array keys{
        detail::NiuKey{device->id(), physical, detail::NiuNoc::Noc0},
        detail::NiuKey{device->id(), physical, detail::NiuNoc::Noc1}};
    ASSERT_TRUE(device->close());
    std::string error;
    auto next = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(next) << error;
    EXPECT_TRUE(context->niu_admission().cancel_unissued(*next, error)) << error;
}

TEST_F(NiuExecutionMeshTest, ServiceManagerRejectsMeshIdentityBeforeChipLookup) {
    auto& services = context->get_service_core_manager();
    try {
        services.get_claimable_cores(mesh.get());
        FAIL() << "Mesh identity must not be interpreted as ChipId";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("actual local Device"), std::string::npos) << error.what();
    }
    EXPECT_THROW(services.claim(mesh.get(), {{0, 0}}), std::runtime_error);
    EXPECT_THROW(services.release(mesh.get(), {{0, 0}}), std::runtime_error);
    EXPECT_THROW(services.allocate_l1(mesh.get(), {0, 0}, 32), std::runtime_error);
}

TEST_F(NiuExecutionMeshTest, ActualChipCloseQuarantinesIssuedCrossChipOwnerOnly) {
    const auto devices = mesh->get_devices();
    auto key = [&](IDevice* device, CoreCoord logical) {
        return detail::NiuKey{
            device->id(),
            context->get_cluster().get_soc_desc(device->id()).get_physical_tensix_core_from_logical(logical),
            detail::NiuNoc::Noc0};
    };
    const std::array cross{key(devices.front(), {0, 0}), key(devices.back(), {0, 0})};
    const std::array unrelated{key(devices.back(), {1, 0})};
    std::string error;
    auto issued = context->niu_admission().reserve_ordinary(cross, error);
    auto separate = context->niu_admission().reserve_ordinary(unrelated, error);
    ASSERT_TRUE(issued) << error;
    ASSERT_TRUE(separate) << error;
    // Coordinator bookkeeping deliberately models unknown issue; no native
    // firmware or successful hardware completion is manufactured.
    ASSERT_TRUE(context->niu_admission().mark_issued(*issued, error)) << error;
    ASSERT_TRUE(devices.front()->close());
    EXPECT_FALSE(context->niu_admission().mark_issued(*issued, error));
    EXPECT_FALSE(context->niu_admission().cancel_unissued(*issued, error));
    EXPECT_FALSE(context->niu_admission().reserve_ordinary(std::span(cross).last(1), error));
    EXPECT_TRUE(context->niu_admission().cancel_unissued(*separate, error)) << error;
}

class NiuExecutionSlowMeshTest : public NiuExecutionMeshTest {
protected:
    void SetUp() override {
        slow.emplace();
        NiuExecutionMeshTest::SetUp();
    }
    void TearDown() override {
        NiuExecutionMeshTest::TearDown();
        slow.reset();
    }
    std::optional<experimental::test_helpers::ScopedSlowDispatchOverride> slow;
};

TEST_F(NiuExecutionSlowMeshTest, MockQueueReturnDoesNotCreateIssuedQuarantineOnClose) {
    auto work = workload();
    ASSERT_NO_THROW(mesh->mesh_command_queue().enqueue_mesh_workload(work, false));
    auto* device = mesh->get_devices().front();
    const auto physical =
        context->get_cluster().get_soc_desc(device->id()).get_physical_tensix_core_from_logical({0, 0});
    const std::array keys{
        detail::NiuKey{device->id(), physical, detail::NiuNoc::Noc0},
        detail::NiuKey{device->id(), physical, detail::NiuNoc::Noc1}};
    ASSERT_TRUE(device->close());
    std::string error;
    auto next = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(next) << error;
    EXPECT_TRUE(context->niu_admission().cancel_unissued(*next, error)) << error;
}

TEST_F(NiuInfrastructureTest, QuarantinedPhysicalNiuRejectsWholeServiceClaim) {
    ASSERT_NO_FATAL_FAILURE(quarantine(free_cores[1], detail::NiuNoc::Noc1));
    EXPECT_THROW(services().claim(device.get(), {free_cores[0], free_cores[1]}), std::runtime_error);
    EXPECT_TRUE(services().claimed_cores(chip).empty());
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
    ASSERT_NO_THROW(services().claim(device.get(), {free_cores[0]}));
    EXPECT_EQ(services().claimed_cores(chip), std::unordered_set<CoreCoord>{free_cores[0]});
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, RejectedDuplicateClaimDoesNotPublishEarlierCore) {
    ASSERT_NO_THROW(services().claim(device.get(), {free_cores[1]}));
    EXPECT_THROW(services().claim(device.get(), {free_cores[0], free_cores[1]}), std::runtime_error);
    EXPECT_EQ(services().claimed_cores(chip), std::unordered_set<CoreCoord>{free_cores[1]});
    ASSERT_NO_THROW(services().claim(device.get(), {free_cores[0]}));
}

TEST_F(NiuInfrastructureTest, ServiceClaimRevalidatesDispatchAllocationAfterQuery) {
    const auto claimable = services().get_claimable_cores(device.get());
    ASSERT_FALSE(claimable.empty());
    const auto assigned = dispatch().prefetcher_core(chip, 0, 0);
    ASSERT_EQ(CoreCoord(assigned.x, assigned.y), claimable.front());
    EXPECT_THROW(services().claim(device.get(), {claimable.front()}), std::runtime_error);
    EXPECT_TRUE(services().claimed_cores(chip).empty());
}

TEST_F(NiuInfrastructureTest, QuarantinedPhysicalNiuRejectsDispatchWithoutConsumingPool) {
    ASSERT_NO_FATAL_FAILURE(quarantine(free_cores.front()));
    EXPECT_THROW(dispatch().prefetcher_core(chip, 0, 0), std::runtime_error);
    EXPECT_FALSE(dispatch().is_prefetcher_core_allocated(chip, 0, 0));
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
}

TEST_F(NiuInfrastructureTest, QuarantinedPhysicalNiuRejectsFabricMuxWithoutPublishingAssignment) {
    ASSERT_NO_FATAL_FAILURE(quarantine(free_cores.front(), detail::NiuNoc::Noc1));
    EXPECT_THROW(dispatch().fabric_mux_core(chip, 0, 0, 0), std::runtime_error);
    EXPECT_FALSE(dispatch().is_fabric_mux_core_allocated(chip, 0, 0, 0));
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
}

TEST_F(NiuInfrastructureTest, CachedDispatchAssignmentCannotBypassLaterQuarantine) {
    const auto assigned = dispatch().prefetcher_core(chip, 0, 0);
    ASSERT_NO_FATAL_FAILURE(quarantine({assigned.x, assigned.y}));
    EXPECT_THROW(dispatch().prefetcher_core(chip, 0, 0), std::runtime_error);
}

TEST_F(NiuInfrastructureTest, QuarantinedProfilerCoreRejectsDispatchManagerConstruction) {
    const auto profiler = dispatch().get_reserved_realtime_profiler_core(chip);
    ASSERT_TRUE(profiler);
    ASSERT_NO_FATAL_FAILURE(quarantine({profiler->x, profiler->y}));
    auto& env_impl = MetalEnvAccessor(*environment).impl();
    EXPECT_THROW((dispatch_core_manager{dispatch_config, 1, env_impl, *context}), std::runtime_error);
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, UnlaunchedServiceReleaseRestoresDispatchAvailability) {
    const CoreCoord core = free_cores.front();
    ASSERT_NO_THROW(services().claim(device.get(), {core}));
    const auto available = dispatch().get_available_dispatch_cores(chip);
    EXPECT_EQ(std::find(available.begin(), available.end(), core), available.end());
    ASSERT_NO_THROW(services().release(device.get(), {core}));
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
    const auto assigned = dispatch().prefetcher_core(chip, 0, 0);
    EXPECT_EQ(CoreCoord(assigned.x, assigned.y), core);
}

TEST_F(NiuInfrastructureTest, ServiceClaimKeepsFastDispatchGridAfterSlowDispatchSwitch) {
    const auto fd_grid = device->compute_with_storage_grid_size();
    ASSERT_NO_THROW(services().claim(device.get(), {free_cores.front()}));
    context->rtoptions().set_fast_dispatch(false);
    EXPECT_EQ(services().impl().get_safe_compute_grid(chip), fd_grid);
    EXPECT_EQ(device->compute_with_storage_grid_size(), fd_grid);
    ASSERT_NO_THROW(services().release(device.get(), {free_cores.front()}));
    EXPECT_FALSE(services().impl().get_safe_compute_grid(chip));
}

TEST_F(NiuInfrastructureTest, ServiceClaimsSurviveDispatchManagerReconstruction) {
    const CoreCoord core = free_cores.front();
    const auto fd_grid = device->compute_with_storage_grid_size();
    ASSERT_NO_THROW(services().claim(device.get(), {core}));
    ASSERT_NO_THROW(context->set_fast_dispatch_mode(false));
    EXPECT_EQ(services().impl().get_safe_compute_grid(chip), fd_grid);
    EXPECT_EQ(device->compute_with_storage_grid_size(), fd_grid);
    ASSERT_NO_THROW(context->set_fast_dispatch_mode(true));
    EXPECT_EQ(services().claimed_cores(chip), std::unordered_set<CoreCoord>{core});
    const auto available = dispatch().get_available_dispatch_cores(chip);
    EXPECT_EQ(std::find(available.begin(), available.end(), core), available.end());
    ASSERT_NO_THROW(services().release(device.get(), {core}));
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, RejectedProfilerReconstructionPreservesPublishedManager) {
    auto* published = &dispatch();
    const auto profiler = dispatch().get_reserved_realtime_profiler_core(chip);
    ASSERT_TRUE(profiler);
    ASSERT_NO_FATAL_FAILURE(quarantine({profiler->x, profiler->y}));
    EXPECT_THROW(context->set_fast_dispatch_mode(true), std::runtime_error);
    EXPECT_EQ(&dispatch(), published);
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
    EXPECT_THROW(dispatch().get_reserved_realtime_profiler_core(chip), std::runtime_error);
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, LaunchedServiceReleaseDoesNotCertifyNiuClean) {
    const CoreCoord core = free_cores.front();
    ASSERT_NO_THROW(services().claim(device.get(), {core}));
    // A real coordinator bookkeeping owner is lost after issue. This is not a
    // device launch receipt; releasing an unrelated planning claim must retain
    // its unknown NIU exclusion.
    const auto physical = context->get_cluster().get_soc_desc(chip).get_physical_tensix_core_from_logical(core);
    const std::array keys{detail::NiuKey{chip, physical, detail::NiuNoc::Noc0}};
    std::string error;
    auto issued = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(issued) << error;
    ASSERT_TRUE(context->niu_admission().mark_issued(*issued, error)) << error;
    issued.reset();
    ASSERT_NO_THROW(services().release(device.get(), {core}));
    EXPECT_TRUE(services().claimed_cores(chip).empty());
    EXPECT_THROW(services().claim(device.get(), {core}), std::runtime_error);
    EXPECT_THROW(dispatch().prefetcher_core(chip, 0, 0), std::runtime_error);
}

TEST_F(NiuInfrastructureTest, EthernetDispatchDoesNotReserveWorkerNius) {
    auto& env_impl = MetalEnvAccessor(*environment).impl();
    dispatch_core_manager ethernet(
        DispatchCoreConfig{DispatchCoreType::ETH, DispatchCoreAxis::COL}, 1, env_impl, *context);
    ASSERT_EQ(ethernet.get_dispatch_core_type(), CoreType::ETH);
    const auto available = ethernet.get_available_dispatch_cores(chip);
    ASSERT_FALSE(available.empty());
    // Poison the worker mapping at the same logical coordinate. It is a
    // different physical resource and must not govern an ethernet assignment.
    ASSERT_NO_FATAL_FAILURE(quarantine(available.front()));
    EXPECT_NO_THROW(ethernet.prefetcher_core(chip, 0, 0));
    EXPECT_FALSE(ethernet.get_reserved_realtime_profiler_core(chip));
}

}  // namespace
}  // namespace tt::tt_metal
