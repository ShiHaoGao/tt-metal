// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_kernel.hpp"
#include "impl/experimental/published_deployment/storage.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/llrt.hpp"
#include <tt-metalium/program.hpp>
#include <algorithm>
#include <stdexcept>
namespace tt::tt_metal {
namespace {
using experimental::DispatchKernelConfiguration;
using experimental::PublishedDeployment;
[[noreturn]] void reject_source() { throw std::logic_error("Published dispatch ELF has no source/JIT compilation input"); }
const DispatchKernelConfiguration& node_configuration(const PublishedDispatchImages& images) {
    TT_FATAL(images.deployment, "Published dispatch requires an admitted deployment");
    const auto nodes = images.deployment->configuration().dispatch_nodes();
    const auto found = std::find_if(nodes.begin(), nodes.end(), [&](const auto& node) { return node.node_id == images.node; });
    TT_FATAL(found != nodes.end(), "Published dispatch node is absent from the admitted deployment");
    return *found;
}
PublishedDispatchImages select_images(std::shared_ptr<const PublishedDeployment> deployment,
                                     uint32_t node, std::span<const HalProcessorIdentifier> processors) {
    PublishedDispatchImages images{std::move(deployment), node, {processors.begin(), processors.end()}, {}};
    const auto& configuration = node_configuration(images);
    const auto plan = experimental::plan_dispatch_kernel(images.deployment->configuration().device().arch, configuration);
    TT_FATAL(std::any_of(plan.kernels.begin(), plan.kernels.end(), [&](const auto& kernel) {
                 return std::ranges::equal(kernel.processors, processors);
             }), "Published dispatch requires one complete canonical processor group");
    images.label = "published_dispatch_" + std::to_string(node) + "_" +
        std::to_string(static_cast<uint32_t>(processors.front().processor_class)) + "_" +
        std::to_string(processors.front().processor_type);
    return images;
}
Kernel::Config placement_configuration(const PublishedDispatchImages& images) {
    const auto& node = node_configuration(images);
    if (images.processors.front().processor_class == HalProcessorClassType::COMPUTE) return ComputeConfig{};
    return DataMovementConfig{.processor = static_cast<DataMovementProcessor>(images.processors.front().processor_type),
                              .noc = node.nocs.non_dispatch_noc, .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
}
void validate_environment(MetalContext& context, const PublishedDispatchImages& images) {
    const auto& selected = context.get_env().get_descriptor().published_deployment();
    TT_FATAL(selected && selected == images.deployment,
             "Published dispatch deployment differs from the actual environment selection");
}
void validate_stream_owner(tt::worker_stream_state::Owner owner) {
    TT_FATAL(owner == tt::worker_stream_state::Owner::SdkCircularBuffers,
             "Published dispatch requires SDK worker stream ownership");
}
}

PublishedDispatchKernel::PublishedDispatchKernel(const KernelBuildContext& context, ContextId id,
    std::shared_ptr<const PublishedDeployment> deployment, uint32_t node,
    std::span<const HalProcessorIdentifier> processors) :
    PublishedDispatchKernel(context, id, select_images(std::move(deployment), node, processors)) {}

PublishedDispatchKernel::PublishedDispatchKernel(const KernelBuildContext& context, ContextId id,
                                                 PublishedDispatchImages images) :
    ExternalBinaryKernel(context, id, images.processors.front().core_type, images.processors.front().processor_class,
        images, CoreRangeSet{CoreRange{node_configuration(images).logical_core, node_configuration(images).logical_core}},
        {}, {}, {}), config_(placement_configuration(images)) {
    TT_FATAL(context.hal.get_arch() == images.deployment->configuration().device().arch,
             "Published dispatch architecture differs from actual HAL");
    for (const auto processor : images.processors) {
        const auto& memory = experimental::deployment_detail::Access::dispatch(*images.deployment, images.node, processor);
        const auto& layout = context.hal.get_jit_build_config(context.hal.get_programmable_core_type_index(processor.core_type),
            static_cast<uint32_t>(processor.processor_class), processor.processor_type);
        TT_FATAL(memory.get_loading() == layout.memory_load, "Published dispatch loading differs from actual processor HAL");
        owned_binaries_.push_back(&memory);
    }
}
uint8_t PublishedDispatchKernel::expected_num_binaries() const { return published_dispatch_images().processors.size(); }
uint32_t PublishedDispatchKernel::get_kernel_processor_type(int index) const {
    return published_dispatch_images().processors.at(index).processor_type;
}
void PublishedDispatchKernel::generate_binaries(IDevice*, JitBuildOptions&) const { reject_source(); }
void PublishedDispatchKernel::read_binaries(IDevice*, const std::string&) { reject_source(); }
std::string_view PublishedDispatchKernel::get_compiler_opt_level() const { reject_source(); }
std::string_view PublishedDispatchKernel::get_linker_opt_level() const { reject_source(); }
std::string PublishedDispatchKernel::config_hash() const { reject_source(); }

std::shared_ptr<const experimental::native_detail::LoadedFirmware>
PublishedDispatchKernel::validate_deployment(IDevice& device, tt::worker_stream_state::Owner owner) const {
    validate_stream_owner(owner);
    const auto& images = published_dispatch_images();
    TT_FATAL(extract_context_id(&device) == get_context_id(), "Published dispatch belongs to another context");
    TT_FATAL(MetalContext::instance_exists(get_context_id()), "Published dispatch requires an existing context");
    auto& context = MetalContext::instance(get_context_id());
    validate_environment(context, images);
    TT_FATAL(device.id() == node_configuration(images).device_id, "Published dispatch belongs to another device");
    TT_FATAL(device.arch() == images.deployment->configuration().device().arch &&
                 context.get_cluster().get_target_device_type() != tt::TargetDevice::Emule,
             "Published dispatch requires its actual hardware architecture");
    auto loaded = context.native_firmware(device.id());
    TT_FATAL(loaded && loaded->live(), "Published dispatch requires successful actual firmware boot");
    TT_FATAL(images.deployment->tensix_firmware().matches(loaded->bundle()),
             "Published dispatch linked firmware differs from actual boot bytes");
    for (size_t i = 0; i != images.processors.size(); ++i) {
        const auto processor = images.processors[i];
        const auto& layout = context.hal().get_jit_build_config(context.hal().get_programmable_core_type_index(processor.core_type),
            static_cast<uint32_t>(processor.processor_class), processor.processor_type);
        TT_FATAL(owned_binaries_[i]->get_loading() == layout.memory_load,
                 "Published dispatch loading differs from actual processor HAL");
    }
    std::lock_guard lock(deployment_mutex_);
    if (auto found = deployments_.find(device.id()); found != deployments_.end())
        TT_FATAL(found->second.lock() == loaded, "Published dispatch firmware generation changed");
    TT_FATAL(loaded->live(), "Published dispatch firmware was withdrawn during admission");
    deployments_.insert_or_assign(device.id(), loaded);
    return loaded;
}

void PublishedDispatchKernel::prepare(IDevice* device, tt::worker_stream_state::Owner owner) {
    validate_stream_owner(owner);
    TT_FATAL(MetalContext::instance_exists(get_context_id()), "Published dispatch requires an existing context");
    TT_FATAL(device, "Published dispatch preparation requires a device");
    (void)validate_deployment(*device, owner);
}

bool PublishedDispatchKernel::configure(IDevice* device, const CoreCoord& logical, uint32_t base,
                                        const uint32_t offsets[]) const {
    TT_FATAL(device && offsets, "Published dispatch configuration requires a device and offsets");
    (void)validate_deployment(*device, tt::worker_stream_state::Owner::SdkCircularBuffers);
    TT_FATAL(is_on_logical_core(logical), "Published dispatch is not placed on requested core");
    auto& context = MetalContext::instance(get_context_id());
    auto& env = MetalEnvAccessor(context.get_env()).impl();
    const auto core = device->worker_core_from_logical_core(logical);
    const auto& processors = published_dispatch_images().processors;
    for (size_t i = 0; i != processors.size(); ++i) {
        const auto processor = processors[i];
        const uint32_t role = processor.processor_class == HalProcessorClassType::DM ? processor.processor_type : 2 + processor.processor_type;
        llrt::write_binary_to_address(env, *owned_binaries_[i], device->id(), core, base + offsets[role]);
    }
    return true;
}

KernelHandle experimental::CreateKernelFromPublishedDeployment(Program& program,
    std::shared_ptr<const PublishedDeployment> deployment, uint32_t node,
    std::span<const HalProcessorIdentifier> processors) {
    auto images = select_images(std::move(deployment), node, processors);
    const auto id = program.impl().get_context_id();
    TT_FATAL(MetalContext::instance_exists(id), "Published dispatch requires the Program's exact existing context");
    auto& context = MetalContext::instance(id);
    validate_environment(context, images);
    validate_stream_owner(program.impl().get_worker_stream_state_owner());
    const auto& configuration = node_configuration(images);
    const auto type = images.processors.front().core_type;
    const auto index = context.hal().get_programmable_core_type_index(type);
    for (const auto& [existing_id, kernel] : program.impl().get_kernels(index)) {
        if (!kernel->is_on_logical_core(configuration.logical_core)) continue;
        for (auto processor : images.processors) {
            if (processor.processor_class != kernel->get_kernel_processor_class()) continue;
            for (uint32_t i = 0; i != kernel->expected_num_binaries(); ++i)
                TT_FATAL(processor.processor_type != kernel->get_kernel_processor_type(i),
                         "Published dispatch processor placement conflicts with an existing kernel");
        }
    }
    auto kernel = std::make_shared<PublishedDispatchKernel>(KernelBuildContext::from_runtime(id), id,
        images.deployment, images.node, images.processors);
    return program.impl().add_kernel(std::move(kernel), type);
}
void experimental::CreateDispatchKernelsFromPublishedDeployment(Program& program,
    std::shared_ptr<const PublishedDeployment> deployment, const DispatchProgramPlan& plan) {
    TT_FATAL(deployment, "Published dispatch requires an admitted deployment");
    TT_FATAL(std::ranges::equal(plan.canonical_record(),
                 deployment->configuration().dispatch_program().canonical_record()),
             "Published dispatch program differs from the admitted complete plan");
    auto& target = program.impl();
    TT_FATAL(!target.is_compiled() && !target.is_finalized() && target.num_kernels() == 0 &&
                 target.circular_buffers().empty(),
             "Published dispatch requires an unprepared Program without kernels or circular buffers");
    const auto& semaphores = target.semaphores();
    TT_FATAL(std::ranges::equal(semaphores, plan.semaphores(), [](const auto& actual, const auto& expected) {
                 return actual.id() == expected.id && actual.initial_value() == expected.initial_value &&
                        actual.core_type() == CoreType::WORKER &&
                        actual.core_range_set() == CoreRangeSet(CoreRange(expected.logical));
             }), "Published dispatch semaphores differ from the admitted complete plan");
    const auto id = target.get_context_id();
    TT_FATAL(MetalContext::instance_exists(id), "Published dispatch requires the Program's exact existing context");
    auto& context = MetalContext::instance(id);
    TT_FATAL(context.get_env().get_descriptor().published_deployment() == deployment,
             "Published dispatch deployment differs from the actual environment selection");
    validate_stream_owner(target.get_worker_stream_state_owner());

    // Validate every image group and bind its arguments before the first
    // Program mutation. The immutable full plan owns the complete inventory;
    // no source string, compiler defines, or JIT options are consumed here.
    std::vector<std::shared_ptr<PublishedDispatchKernel>> kernels;
    size_t bound_arguments = 0;
    for (const auto& node : plan.kernel_plans()) {
        for (const auto& group : node.kernels) {
            auto kernel = std::make_shared<PublishedDispatchKernel>(KernelBuildContext::from_runtime(id), id,
                deployment, node.configuration.node_id, group.processors);
            for (const auto& args : plan.runtime_arguments()) {
                if (args.node == node.configuration.node_id && group.processors.size() == 1 &&
                    args.processor == group.processors.front()) {
                    TT_FATAL(args.logical == node.configuration.logical_core,
                             "Published dispatch argument placement differs from its kernel");
                    kernel->set_runtime_args(args.logical, args.words);
                    ++bound_arguments;
                }
            }
            kernels.push_back(std::move(kernel));
        }
    }
    TT_FATAL(bound_arguments == plan.runtime_arguments().size(),
             "Published dispatch kernel groups do not cover the complete argument inventory");
    for (auto& kernel : kernels) {
        const auto type = kernel->published_dispatch_images().processors.front().core_type;
        target.add_kernel(std::move(kernel), type);
    }
}
} // namespace tt::tt_metal
