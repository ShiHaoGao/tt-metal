// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include <gtest/gtest.h>
#include <reflect>
#include <elf.h>
#include <cstring>
#include <algorithm>
#include <limits>
#include "tt-metalium/experimental/published_deployment.hpp"
#include "tt-metalium/experimental/context/metal_env.hpp"
#include "impl/experimental/published_deployment/configuration.hpp"
#include "impl/context/metal_context.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::experimental {
namespace {
using namespace test;

TEST_F(PublishedDeploymentTest, PublishesEverySubordinateCompanionOnItsOwnProcessor) {
    auto config = configuration();
    std::vector<std::vector<std::byte>> records;
    for (auto processor : dispatch_processors(config.dispatch_nodes().back())) {
        ASSERT_NO_THROW(records.push_back(config.image_record(PublishedImageKind::Dispatch, processor, 9)));
    }
    ASSERT_EQ(records.size(), 4u);
    for (size_t i = 0; i != records.size(); ++i)
        for (size_t j = i + 1; j != records.size(); ++j) EXPECT_NE(records[i], records[j]);
    PublicationInput input(config, hal);
    ASSERT_EQ(input.input.size(), config.firmware_processors().size() + 6);
    auto deployment = PublishedDeployment::admit(config, input.input);
    for (const auto& image : input.input)
        if (image.kind == PublishedImageKind::Dispatch)
            EXPECT_TRUE(std::ranges::equal(deployment.dispatch_image(image.dispatch_node, image.processor), image.bytes));
    input.input.pop_back();
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    input.input.push_back(input.input.back());
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, RejectsUnsupportedFabricAtConfigurationAdmission) {
    program.topology.fabric = true;
    EXPECT_THROW(configuration(), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, PlannedRolesOwnTheirExactProcessorGroups) {
    auto config = configuration();
    auto nodes = config.dispatch_nodes();
    ASSERT_EQ(nodes.size(), 3u);
    EXPECT_EQ(nodes.front().processor.processor_type, 0);
    EXPECT_EQ(nodes.back().processor.processor_type, 1);
    const auto prefetch_processors = dispatch_processors(nodes.front());
    const auto subordinate_processors = dispatch_processors(nodes.back());
    ASSERT_EQ(prefetch_processors.size(), 1u);
    ASSERT_EQ(subordinate_processors.size(), 4u);
    EXPECT_EQ(prefetch_processors.front(), nodes.front().processor);
    EXPECT_EQ(subordinate_processors.front(), nodes.back().processor);
    for (size_t i = 1; i != subordinate_processors.size(); ++i)
        EXPECT_EQ(subordinate_processors[i].processor_class, HalProcessorClassType::COMPUTE);
}

TEST_F(PublishedDeploymentTest, CopiesConfigurationAndIncludesAllHalFirmwareProcessors) {
    ASSERT_FALSE(MetalContext::instance_exists());
    auto config = configuration();
    // Blackhole: five Tensix, two active Ethernet, two idle Ethernet, one DRAM firmware image.
    EXPECT_EQ(config.firmware_processors().size(), 10u);
    EXPECT_EQ(config.device().num_l1_banks, 120u);
    EXPECT_EQ(config.dispatch_nodes().size(), 3u);
    auto original = config.image_record(PublishedImageKind::Dispatch, config.dispatch_nodes()[0].processor, 7);
    queue_size(0x40000);
    device.num_l1_banks = 112;
    program.workers.count = 112;
    EXPECT_EQ(config.device().num_l1_banks, 120u);
    EXPECT_EQ(config.image_record(PublishedImageKind::Dispatch, config.dispatch_nodes()[0].processor, 7), original);
    EXPECT_FALSE(config.matches(configuration()));
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedDeploymentTest, KernelRendererRejectsMissingResolvedConfiguration) {
    auto p = plan();
    auto node = p.dispatch_nodes().front();
    node.resolved.reset();
    EXPECT_THROW(plan_dispatch_kernel(device.arch, node), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, KernelRendererRejectsMissingCanonicalDispatchField) {
    auto p = plan();
    auto node = p.dispatch_nodes().front();
    std::get<PrefetchConfiguration>(node.kernel).static_config.pcie_size.reset();
    EXPECT_THROW(plan_dispatch_kernel(device.arch, node), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, RejectsInvalidProcessorDuplicateNodeAndCq) {
    ASSERT_NO_THROW(configuration());
    const auto role = program.topology.nodes[0].role;
    program.topology.nodes[0].role = static_cast<DispatchWorkerType>(27);
    EXPECT_THROW(configuration(), std::invalid_argument);
    program.topology.nodes[0].role = role;
    program.topology.nodes.push_back(program.topology.nodes[0]);
    EXPECT_THROW(configuration(), std::invalid_argument);
    program.topology.nodes.pop_back();
    program.topology.nodes[0].cq = 1;
    EXPECT_THROW(configuration(), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, RejectsMismatchedHalAndMissingFirmwareConfiguration) {
    ASSERT_NO_THROW(configuration());
    device.arch = tt::ARCH::WORMHOLE_B0;
    EXPECT_THROW(configuration(), std::invalid_argument);
    device.arch = tt::ARCH::BLACKHOLE;
    device.num_dram_banks = 0;
    EXPECT_THROW(configuration(), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, RejectsDispatchProcessorDifferentFromConfiguredCoreType) {
    ASSERT_NO_THROW(configuration());
    device.dispatch_core_type = DispatchCoreType::ETH;
    device.resolved_dispatch_core_type = tt::CoreType::ETH;
    EXPECT_THROW(configuration(), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, RejectsTwoDispatchNodesOnTheSameProcessorPlacement) {
    ASSERT_NO_THROW(configuration());
    program.topology.nodes[0].core = program.topology.nodes[1].core;
    EXPECT_THROW(configuration(), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, OwnsDispatcherAndSubordinateCanonicalFields) {
    auto original = configuration();
    EXPECT_EQ(original.dispatch_nodes().size(), 3u);
    queue_size(0x80000);
    EXPECT_FALSE(original.matches(configuration()));
    std::get<HostQueueBacking>(program.queue.inputs.backing).cq_size_override.reset();
    program.queue.plan = plan_system_memory_queues(program.queue.inputs);
    program.memory.subordinate_buffer_size *= 2;
    program.memory.subordinate_buffer_pages *= 2;
    EXPECT_FALSE(original.matches(configuration()));
    program.memory.subordinate_buffer_size = 0;
    EXPECT_THROW(configuration(), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, AdmitsAndOwnsCompleteFirmwareAndIndependentDispatchBytes) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto deployment = PublishedDeployment::admit(config, input.input);
    const auto prefetch_index = config.firmware_processors().size();
    auto expected = input.images[prefetch_index].bytes;
    auto firmware_expected = input.images[0].bytes;
    std::fill(input.images[prefetch_index].bytes.begin(), input.images[prefetch_index].bytes.end(), std::byte{});
    std::fill(input.images[0].bytes.begin(), input.images[0].bytes.end(), std::byte{});
    EXPECT_TRUE(std::ranges::equal(deployment.dispatch_image(7, config.dispatch_nodes()[0].processor), expected));
    EXPECT_TRUE(std::ranges::equal(deployment.firmware_image(config.firmware_processors()[0]), firmware_expected));
    EXPECT_TRUE(std::ranges::equal(deployment.tensix_firmware().image_bytes(TensixKernelRole::Brisc), firmware_expected));
    EXPECT_NO_THROW(deployment.validate_configuration(config));
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedDeploymentTest, RejectsMissingNonTensixFirmwareAndDuplicateProcessor) {
    auto config = configuration();
    PublicationInput input(config, hal);
    ASSERT_NO_THROW(PublishedDeployment::admit(config, input.input));
    input.input.erase(input.input.begin() + 9); // DRAM firmware is as mandatory as Tensix.
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    input.input.insert(input.input.begin() + 9, input.input.front());
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, RejectsMissingDispatchAndUnknownKindOrProcessor) {
    auto config = configuration();
    PublicationInput input(config, hal);
    ASSERT_NO_THROW(PublishedDeployment::admit(config, input.input));
    auto dispatch = input.input.back(); input.input.pop_back();
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    input.input.push_back(dispatch); input.input.back().kind = static_cast<PublishedImageKind>(7);
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    input.input.back() = dispatch; input.input.back().processor.processor_type = 27;
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, RejectsBareElfLoadedMetadataAndWrongRecordVersion) {
    auto config = configuration();
    PublicationInput input(config, hal);
    ASSERT_NO_THROW(PublishedDeployment::admit(config, input.input));
    auto& elf = input.images.back();
    auto original = elf.bytes;
    elf.put(elf.sections_offset + 8 * sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_size), uint32_t{0});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::exception);
    std::copy(original.begin(), original.end(), elf.bytes.begin());
    elf.put(elf.sections_offset + 8 * sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_flags), uint32_t{SHF_ALLOC});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::exception);
    std::copy(original.begin(), original.end(), elf.bytes.begin());
    elf.put(elf.publication_offset + 8, uint64_t{1});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::exception);
}

TEST_F(PublishedDeploymentTest, RejectsMalformedElfAndUnresolvedOrStaleFirmwareImports) {
    auto config = configuration();
    PublicationInput input(config, hal);
    ASSERT_NO_THROW(PublishedDeployment::admit(config, input.input));
    auto& elf = input.images.back();
    auto original = elf.bytes;
    elf.put(offsetof(Elf32_Ehdr, e_machine), Elf32_Half{EM_X86_64});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::exception);
    std::copy(original.begin(), original.end(), elf.bytes.begin());
    elf.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_shndx), Elf32_Half{SHN_UNDEF});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::exception);
    std::copy(original.begin(), original.end(), elf.bytes.begin());
    elf.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value), uint32_t{0x2004});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::exception);
}

TEST_F(PublishedDeploymentTest, RejectsRealConfigurationChangeEvenWithIdenticalElfInputs) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto deployment = PublishedDeployment::admit(config, input.input);
    queue_size(0x80000);
    auto changed = configuration();
    EXPECT_THROW(deployment.validate_configuration(changed), std::invalid_argument);
    EXPECT_THROW(PublishedDeployment::admit(changed, input.input), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, DescriptorRetainsTypedPublicationAndRejectsProfileConflict) {
    MetalEnvDescriptor descriptor;
    EXPECT_FALSE(descriptor.published_deployment()); // Independent source SDK baseline remains legal.
    auto config = configuration();
    PublicationInput input(config, hal);
    descriptor.set_published_deployment(PublishedDeployment::admit(config, input.input));
    EXPECT_TRUE(descriptor.published_deployment()->configuration().matches(config));
    EXPECT_EQ(descriptor.device_profiler_mode(), DeviceProfilerMode::Disabled);
    EXPECT_THROW(descriptor.set_device_profiler_mode(DeviceProfilerMode::Program), std::invalid_argument);
    EXPECT_EQ(descriptor.device_profiler_mode(), DeviceProfilerMode::Disabled);
    EXPECT_FALSE(MetalContext::instance_exists());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
