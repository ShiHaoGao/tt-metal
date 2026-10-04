// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include "impl/device/firmware/firmware_images.hpp"
#include "impl/device/firmware/risc_firmware_initializer.hpp"
#include "impl/context/context_descriptor.hpp"
#include "impl/kernels/external_binary_kernel.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include "metal2_host_api/test_helpers.hpp"
#include <map>

namespace tt::tt_metal::experimental::test {
class PublishedFirmwareTest : public PublishedDeploymentTest {};

TEST_F(PublishedFirmwareTest, OwnsAllProcessorImagesWithoutContextOrBuildEnvironment) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto images = [&] {
        auto deployment = PublishedDeployment::admit(config, input.input);
        return FirmwareImages(deployment, JitDeviceConfig{device, &hal}, options);
    }();
    for (auto processor : config.firmware_processors()) {
        const auto& memory = images.image(processor);
        EXPECT_EQ(memory.get_text_size(), 4u);
    }
    EXPECT_TRUE(std::ranges::equal(images.tensix().image_bytes(TensixKernelRole::Brisc),
                                  input.input[0].bytes));
    // Input bytes may disappear; the admitted owner retains its loading image.
    input.images.clear();
    EXPECT_EQ(images.image(config.firmware_processors()[0]).get_text_size(), 4u);
    std::map<uint64_t, uint32_t> loaded;
    images.image(config.firmware_processors()[0]).process_spans(
        [&](auto words, uint64_t address, uint32_t count) {
            for (uint32_t i = 0; i != count; ++i) loaded.emplace(address + i * sizeof(uint32_t), words[i]);
        });
    const auto text_address = hal.get_jit_build_config(
        hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX),
        static_cast<uint32_t>(HalProcessorClassType::DM), 0).fw_base_addr;
    const auto data_address = hal.get_processor_image_regions(config.firmware_processors()[0])->local_data.base;
    EXPECT_EQ(loaded, (std::map<uint64_t, uint32_t>{{text_address, 0x00008067u}, {data_address, 0x12345678u}}));
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedFirmwareTest, RejectsActualHalAndBuildFlagsBeforeLoading) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto deployment = PublishedDeployment::admit(config, input.input);
    Hal changed_hal{tt::ARCH::BLACKHOLE, false, false, 0, false, false, true, true};
    EXPECT_THROW((FirmwareImages(deployment, JitDeviceConfig{device, &changed_hal}, options)), std::invalid_argument);
    options.set_enable_2_erisc_mode(false);
    EXPECT_THROW((FirmwareImages(deployment, JitDeviceConfig{device, &hal}, options)), std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedFirmwareTest, RejectsActualAllocatorAndProfilerMismatchBeforeLoading) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto deployment = PublishedDeployment::admit(config, input.input);
    auto changed = device;
    changed.num_l1_banks = device.num_l1_banks - 1;
    EXPECT_THROW((FirmwareImages(deployment, JitDeviceConfig{changed, &hal}, options)), std::invalid_argument);
    llrt::RunTimeOptions profiled{llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = "/tmp", .profiler_mode = DeviceProfilerMode::Program}};
    profiled.set_enable_2_erisc_mode(true);
    EXPECT_THROW((FirmwareImages(deployment, JitDeviceConfig{device, &hal}, profiled)), std::exception);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedFirmwareTest, RejectsUnknownProcessorWithoutSourceFallback) {
    auto config = configuration();
    PublicationInput input(config, hal);
    FirmwareImages images(PublishedDeployment::admit(config, input.input), JitDeviceConfig{device, &hal}, options);
    EXPECT_THROW(images.image({HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 7}),
                 std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}

class PublishedFirmwarePhaseTest : public PublishedDeploymentTest {
protected:
    void SetUp() override {
        slow_dispatch.emplace();
        configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    }
    void TearDown() override {
        disable_mock_mode();
        slow_dispatch.reset();
        EXPECT_FALSE(MetalContext::instance_exists());
    }
    template<class F> void with_initializer(const std::optional<PublishedDeployment>& publication, F&& body) {
        MetalEnvDescriptor selection(get_mock_cluster_desc().value());
        selection.set_device_profiler_mode(DeviceProfilerMode::Disabled);
        if (publication) selection.set_published_deployment(*publication);
        MetalEnv env(std::move(selection));
        const auto id = ContextId{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
        auto& context = MetalContext::instance(id);
        context.initialize({}, 1, {}, 0, true);
        auto descriptor = std::make_shared<ContextDescriptor>(
            &env, &context, 1, 0, 0,
            context.hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED));
        RiscFirmwareInitializer initializer(
            descriptor, [&context]() -> tt::tt_fabric::ControlPlane& { return context.get_control_plane(); },
            context.get_dispatch_core_manager());
        body(context, initializer);
    }
    PublishedDeployment publication_for_mock() {
        std::optional<PublishedDeployment> publication;
        with_initializer(std::nullopt, [&](auto& context, auto&) {
            const auto actual = create_jit_device_config(0, 1, context.get_context_id());
            auto inputs = program;
            inputs.workers.count = actual.num_l1_banks;
            inputs.hal = capture_dispatch_hal(*actual.hal);
            const auto config = DeploymentConfiguration::from_sdk(
                actual, context.rtoptions(), plan_dispatch_program(inputs));
            PublicationInput input(config, *actual.hal);
            publication.emplace(PublishedDeployment::admit(config, input.input));
        });
        return std::move(*publication);
    }
    std::optional<test_helpers::ScopedSlowDispatchOverride> slow_dispatch;
};

TEST_F(PublishedFirmwarePhaseTest, RejectsLaunchWithoutDeviceAdmission) {
    const auto publication = publication_for_mock();
    with_initializer(publication, [&](auto&, auto& initializer) {
        EXPECT_THROW(initializer.run_launch_phase({0}), std::exception);
        EXPECT_FALSE(initializer.is_initialized());
        EXPECT_FALSE(initializer.worker_stream_state_provider(0));
        EXPECT_FALSE(initializer.native_firmware(0));
    });
}

TEST_F(PublishedFirmwarePhaseTest, RetainsAdmissionAcrossLaunchButNeverClaimsMockBoot) {
    const auto publication = publication_for_mock();
    with_initializer(publication, [&](auto&, auto& initializer) {
        initializer.run_async_build_phase({0});
        EXPECT_FALSE(initializer.is_initialized());
        EXPECT_NO_THROW(initializer.run_launch_phase({0}));
        EXPECT_TRUE(initializer.is_initialized());
        EXPECT_FALSE(initializer.worker_stream_state_provider(0));
        EXPECT_FALSE(initializer.native_firmware(0));
        EXPECT_NO_THROW(initializer.run_launch_phase({0}));
        EXPECT_TRUE(initializer.is_initialized());
    });
}

TEST_F(PublishedFirmwarePhaseTest, RebuildAndTeardownDiscardPreviousAdmission) {
    const auto publication = publication_for_mock();
    with_initializer(publication, [&](auto&, auto& initializer) {
        initializer.run_async_build_phase({0});
        initializer.run_launch_phase({0});
        initializer.run_async_build_phase({});
        EXPECT_THROW(initializer.run_launch_phase({0}), std::exception);
        EXPECT_FALSE(initializer.is_initialized());
        initializer.run_async_build_phase({0});
        initializer.run_launch_phase({0});
        std::unordered_set<InitializerKey> completed;
        initializer.teardown(completed);
        EXPECT_THROW(initializer.run_launch_phase({0}), std::exception);
        EXPECT_FALSE(initializer.is_initialized());
    });
}

TEST_F(PublishedFirmwarePhaseTest, FailedAdmissionDoesNotRetainPartialDeviceSet) {
    const auto publication = publication_for_mock();
    with_initializer(publication, [&](auto&, auto& initializer) {
        EXPECT_THROW(initializer.run_async_build_phase({0, std::numeric_limits<tt::ChipId>::max()}), std::exception);
        EXPECT_THROW(initializer.run_launch_phase({0}), std::exception);
        EXPECT_FALSE(initializer.is_initialized());
        EXPECT_FALSE(initializer.native_firmware(0));
    });
}

TEST_F(PublishedFirmwarePhaseTest, MeshDeploymentUsesPhysicalDevicesWithoutInventingBootReceipts) {
    const auto publication = publication_for_mock();
    with_initializer(publication, [&](auto& context, auto&) {
        auto mesh = context.get_env().create_unit_mesh_device(0);
        auto submesh = mesh->create_submesh(distributed::MeshShape{1, 1});
        const auto physical = mesh->get_devices();
        ASSERT_EQ(physical.size(), 1u);
        ASSERT_EQ(physical.front()->id(), 0);
        ASSERT_NE(submesh->id(), physical.front()->id());
        EXPECT_EQ(ExternalBinaryKernel::physical_devices(*mesh), physical);
        EXPECT_EQ(ExternalBinaryKernel::physical_devices(*submesh), physical);
        EXPECT_EQ(ExternalBinaryKernel::physical_devices(*physical.front()), physical);
        EXPECT_FALSE(context.native_firmware(physical.front()->id()));
        EXPECT_TRUE(submesh->close());
        submesh.reset();
        EXPECT_TRUE(mesh->close());
    });
}
} // namespace tt::tt_metal::experimental::test
