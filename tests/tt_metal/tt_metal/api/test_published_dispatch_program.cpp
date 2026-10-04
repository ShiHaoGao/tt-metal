// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include "impl/experimental/published_deployment/dispatch_program_plan.hpp"

namespace tt::tt_metal::experimental::test {
template<class T> concept PublicationInputAccepted = requires(
    const JitDeviceConfig& device, const llrt::RunTimeOptions& options, const T& input) {
    DeploymentConfiguration::from_sdk(device, options, input);
};
static_assert(PublicationInputAccepted<DispatchProgramPlan>);
static_assert(!PublicationInputAccepted<std::span<const DispatchKernelConfiguration>>);

TEST_F(PublishedDeploymentTest, ConfigurationRetainsTheWholeProgramOwner) {
    auto config = configuration();
    EXPECT_EQ(config.dispatch_program().semaphores().size(), 6u);
    EXPECT_EQ(config.dispatch_program().runtime_arguments().size(), 2u);
    EXPECT_EQ(config.dispatch_nodes().data(), config.dispatch_program().dispatch_nodes().data());
    EXPECT_FALSE(config.dispatch_program().initialization().empty());
}
} // namespace tt::tt_metal::experimental::test
