// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include "impl/experimental/published_deployment/dispatch_kernel.hpp"
#include "impl/experimental/published_deployment/storage.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/device/device_impl.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include <gtest/gtest.h>
namespace tt::tt_metal::experimental {
namespace {
using PublishedDispatchKernelTest = test::PublishedDeploymentTest;
TEST_F(PublishedDispatchKernelTest, RetainsPublishedBytesWithoutNativeRecordOrSourceContext) {
    auto c = configuration();
    const auto nodes = c.dispatch_nodes();
    test::PublicationInput input(c, hal);
    auto deployment = std::make_shared<const PublishedDeployment>(PublishedDeployment::admit(c, input.input));
    KernelBuildContext context{hal, options, nullptr};
    PublishedDispatchKernel kernel(context, DEFAULT_CONTEXT_ID, deployment, 7,
        std::span(&nodes[0].processor, 1));
    EXPECT_TRUE(kernel.is_external_binary());
    EXPECT_TRUE(kernel.is_published_dispatch());
    EXPECT_FALSE(kernel.is_native());
    EXPECT_EQ(kernel.expected_num_binaries(), 1);
    EXPECT_EQ(kernel.get_kernel_processor_type(0), 0);
    EXPECT_EQ(kernel.core_range_set(), CoreRangeSet({CoreRange({0, 0}, {0, 0})}));
    EXPECT_EQ(kernel.owned_binaries().front(),
        &deployment_detail::Access::dispatch(*deployment, 7, nodes[0].processor));
    EXPECT_EQ(std::get<DataMovementConfig>(kernel.config()).noc,
        nodes[0].nocs.non_dispatch_noc);
    std::fill(input.images.back().bytes.begin(), input.images.back().bytes.end(), std::byte{});
    deployment.reset();
    EXPECT_GT(kernel.get_binary_packed_size(nullptr, 0), 0u);
    EXPECT_THROW(kernel.kernel_source(), std::bad_variant_access);
    EXPECT_THROW(kernel.native_images(), std::bad_variant_access);
    EXPECT_THROW(kernel.compute_hash(), std::logic_error);
    EXPECT_THROW(kernel.binaries(uint64_t{17}), std::exception);
    auto binaries = kernel.owned_binaries();
    EXPECT_THROW(kernel.set_binaries(uint64_t{17}, std::move(binaries)), std::exception);
    EXPECT_THROW(kernel.read_binaries(nullptr, "/absent"), std::logic_error);
    EXPECT_THROW(kernel.get_compiler_opt_level(), std::logic_error);
    EXPECT_THROW(kernel.get_linker_opt_level(), std::logic_error);
    EXPECT_THROW(kernel.prepare(nullptr, tt::worker_stream_state::Owner::SdkCircularBuffers), std::exception);
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST_F(PublishedDispatchKernelTest, SubordinateRequiresWholeCanonicalComputeGroup) {
    auto c = configuration();
    const auto& subordinate = c.dispatch_nodes().back();
    test::PublicationInput input(c, hal);
    auto deployment = std::make_shared<const PublishedDeployment>(PublishedDeployment::admit(c, input.input));
    const auto plan = plan_dispatch_kernel(tt::ARCH::BLACKHOLE, subordinate);
    ASSERT_EQ(plan.kernels.size(), 2u);
    const auto& processors = plan.kernels[1].processors;
    KernelBuildContext context{hal, options, nullptr};
    PublishedDispatchKernel kernel(context, DEFAULT_CONTEXT_ID, deployment, subordinate.node_id, processors);
    EXPECT_EQ(kernel.expected_num_binaries(), 3);
    for (size_t i = 0; i != 3; ++i) {
        EXPECT_EQ(kernel.get_kernel_processor_type(i), i);
        EXPECT_EQ(kernel.owned_binaries()[i], &deployment_detail::Access::dispatch(*deployment, subordinate.node_id, processors[i]));
    }
    EXPECT_THROW((PublishedDispatchKernel(context, DEFAULT_CONTEXT_ID, deployment, subordinate.node_id,
        std::span(processors).subspan(0, 2))), std::exception);
    auto permuted = processors; std::swap(permuted[1], permuted[2]);
    EXPECT_THROW((PublishedDispatchKernel(context, DEFAULT_CONTEXT_ID, deployment, subordinate.node_id, permuted)), std::exception);
    EXPECT_THROW((PublishedDispatchKernel(context, DEFAULT_CONTEXT_ID, deployment, 8, processors)), std::exception);
    EXPECT_THROW((PublishedDispatchKernel(context, DEFAULT_CONTEXT_ID, {}, 7, processors)), std::exception);
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST_F(PublishedDispatchKernelTest, FactoryRejectsForeignSelectionAndConflictsBeforeMutation) {
    auto c = configuration(); test::PublicationInput input(c, hal);
    const auto nodes = c.dispatch_nodes();
    auto deployment = std::make_shared<const PublishedDeployment>(PublishedDeployment::admit(c, input.input));
    experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
    {
        MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc().value());
        descriptor.set_published_deployment(*deployment);
        MetalEnv env(std::move(descriptor));
        const auto id = ContextId{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
        Program program(std::make_shared<detail::ProgramImpl>(id));
        EXPECT_THROW(CreateKernelFromPublishedDeployment(program, deployment, 7,
            std::span(&nodes[0].processor, 1)), std::exception);
        EXPECT_EQ(program.impl().num_kernels(), 0u);
        deployment = env.get_descriptor().published_deployment();
        const auto handle = CreateKernelFromPublishedDeployment(program, deployment, 7,
            std::span(&nodes[0].processor, 1));
        EXPECT_TRUE(program.impl().get_kernel(handle)->is_published_dispatch());
        auto& kernel = static_cast<PublishedDispatchKernel&>(*program.impl().get_kernel(handle));
        auto& actual_context = MetalContext::instance(id);
        actual_context.initialize(DispatchCoreConfig{}, 1, {}, 0, true);
        tt::tt_metal::Device unbooted(&env, &actual_context, 0, 1, 0, 0, {}, true);
        try {
            kernel.validate_deployment(unbooted, tt::worker_stream_state::Owner::SdkCircularBuffers);
            FAIL() << "Publication ownership cannot manufacture firmware boot";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("successful actual firmware boot"), std::string::npos);
        }
        EXPECT_THROW(kernel.validate_deployment(unbooted, tt::worker_stream_state::Owner::Program), std::exception);
        EXPECT_EQ(program.impl().num_kernels(), 1u);
        EXPECT_THROW(CreateKernelFromPublishedDeployment(program, deployment, 7,
            std::span(&nodes[0].processor, 1)), std::exception);
        EXPECT_EQ(program.impl().num_kernels(), 1u);
        EXPECT_THROW(program.impl().validate_worker_stream_state_client(MetalContext::instance(id), 0), std::exception);
        EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
    }
    experimental::disable_mock_mode();
    EXPECT_FALSE(MetalContext::instance_exists());
}
}
}
