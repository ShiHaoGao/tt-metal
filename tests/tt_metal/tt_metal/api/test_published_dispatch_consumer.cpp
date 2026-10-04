// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include "impl/experimental/published_deployment/dispatch_kernel.hpp"
#include "impl/experimental/published_deployment/storage.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/program.hpp>

namespace tt::tt_metal::experimental::test {
namespace {
class PublishedDispatchConsumerTest : public PublishedDeploymentTest {
protected:
    void SetUp() override { configure_mock_mode(tt::ARCH::BLACKHOLE, 1); }
    void TearDown() override {
        disable_mock_mode();
        EXPECT_FALSE(MetalContext::instance_exists());
    }
    template<class F> void with_environment(F&& body, bool select = true) {
        auto config = configuration();
        PublicationInput input(config, hal);
        auto deployment = PublishedDeployment::admit(config, input.input);
        MetalEnvDescriptor descriptor(get_mock_cluster_desc().value());
        if (select) descriptor.set_published_deployment(deployment);
        MetalEnv env(std::move(descriptor));
        const auto id = ContextId{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
        Program target(std::make_shared<detail::ProgramImpl>(id));
        auto owner = select ? env.get_descriptor().published_deployment()
                            : std::make_shared<const PublishedDeployment>(deployment);
        body(target, owner);
    }
    static void add_semaphores(Program& target, const DispatchProgramPlan& plan) {
        for (const auto& semaphore : plan.semaphores())
            target.impl().add_semaphore(CoreRangeSet(CoreRange(semaphore.logical)), semaphore.id,
                semaphore.initial_value, CoreType::WORKER);
    }
};

TEST_F(PublishedDispatchConsumerTest, CreatesAllFourGroupsAndBindsOnlyDeclaredArguments) {
    const auto actual = plan();
    with_environment([&](Program& target, auto owner) {
        add_semaphores(target, actual);
        CreateDispatchKernelsFromPublishedDeployment(target, owner, actual);
        ASSERT_EQ(target.impl().num_kernels(), 4u);
        const std::array<uint32_t, 4> nodes{7, 8, 9, 9};
        const std::array<uint8_t, 4> binary_counts{1, 1, 1, 3};
        for (uint32_t i = 0; i != 4; ++i) {
            auto kernel = target.impl().get_kernel(i);
            ASSERT_TRUE(kernel->is_published_dispatch());
            EXPECT_TRUE(kernel->is_external_binary());
            EXPECT_FALSE(kernel->is_native());
            EXPECT_EQ(kernel->published_dispatch_images().deployment, owner);
            EXPECT_EQ(kernel->published_dispatch_images().node, nodes[i]);
            EXPECT_EQ(kernel->expected_num_binaries(), binary_counts[i]);
            EXPECT_THROW(kernel->kernel_source(), std::bad_variant_access);
            EXPECT_THROW(kernel->compute_hash(), std::logic_error);
            if (i < 2) {
                const CoreCoord core{i, 0};
                EXPECT_EQ(kernel->runtime_args(core), (std::vector<uint32_t>{0, 0, 0}));
                EXPECT_EQ(kernel->cores_with_runtime_args(), (std::set<CoreCoord>{core}));
            } else {
                EXPECT_TRUE(kernel->cores_with_runtime_args().empty());
            }
        }
        EXPECT_EQ(target.impl().semaphores().size(), 6u);
    });
}

TEST_F(PublishedDispatchConsumerTest, RejectsChangedWholePlanBeforeAddingAnyKernel) {
    const auto actual = plan();
    with_environment([&](Program& target, auto owner) {
        add_semaphores(target, actual);
        auto changed = program;
        std::swap(changed.topology.nodes.front().downstream[0], changed.topology.nodes.front().downstream[1]);
        auto other = plan_dispatch_program(changed);
        // This changes a whole-program topology fact without changing the
        // processor groups or per-node source configuration.
        EXPECT_THROW(CreateDispatchKernelsFromPublishedDeployment(target, owner, other), std::exception);
        EXPECT_EQ(target.impl().num_kernels(), 0u);
        EXPECT_EQ(target.impl().semaphores().size(), 6u);
    });
}

TEST_F(PublishedDispatchConsumerTest, RejectsMissingAndAlteredSemaphoreProjectionBeforeMutation) {
    const auto actual = plan();
    with_environment([&](Program& target, auto owner) {
        EXPECT_THROW(CreateDispatchKernelsFromPublishedDeployment(target, owner, actual), std::exception);
        EXPECT_EQ(target.impl().num_kernels(), 0u);
        for (const auto& semaphore : actual.semaphores())
            target.impl().add_semaphore(CoreRangeSet(CoreRange(semaphore.logical)), semaphore.id,
                semaphore.initial_value + 1, CoreType::WORKER);
        EXPECT_THROW(CreateDispatchKernelsFromPublishedDeployment(target, owner, actual), std::exception);
        EXPECT_EQ(target.impl().num_kernels(), 0u);
    });
}

TEST_F(PublishedDispatchConsumerTest, RejectsAbsentAndForeignEnvironmentSelectionBeforeMutation) {
    const auto actual = plan();
    with_environment([&](Program& target, auto owner) {
        add_semaphores(target, actual);
        EXPECT_THROW(CreateDispatchKernelsFromPublishedDeployment(target, owner, actual), std::exception);
        EXPECT_EQ(target.impl().num_kernels(), 0u);
    }, false);
    with_environment([&](Program& target, auto owner) {
        add_semaphores(target, actual);
        auto foreign = std::make_shared<const PublishedDeployment>(*owner);
        EXPECT_THROW(CreateDispatchKernelsFromPublishedDeployment(target, foreign, actual), std::exception);
        EXPECT_THROW(CreateDispatchKernelsFromPublishedDeployment(target, {}, actual), std::exception);
        EXPECT_EQ(target.impl().num_kernels(), 0u);
    });
}

TEST_F(PublishedDispatchConsumerTest, RejectsExistingKernelAndRepeatedMaterializationWithoutPartialAdd) {
    const auto actual = plan();
    with_environment([&](Program& target, auto owner) {
        add_semaphores(target, actual);
        const auto& node = actual.dispatch_nodes().front();
        CreateKernelFromPublishedDeployment(target, owner, node.node_id, std::span(&node.processor, 1));
        EXPECT_THROW(CreateDispatchKernelsFromPublishedDeployment(target, owner, actual), std::exception);
        EXPECT_EQ(target.impl().num_kernels(), 1u);
    });
    with_environment([&](Program& target, auto owner) {
        add_semaphores(target, actual);
        CreateDispatchKernelsFromPublishedDeployment(target, owner, actual);
        EXPECT_THROW(CreateDispatchKernelsFromPublishedDeployment(target, owner, actual), std::exception);
        EXPECT_EQ(target.impl().num_kernels(), 4u);
    });
}
} // namespace
} // namespace tt::tt_metal::experimental::test
