// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "storage.hpp"
#include "record_writer.hpp"
#include "dispatch_plan.hpp"
#include "impl/experimental/published_deployment/dispatch_program_adapter.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/rtoptions.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <tuple>

namespace tt::tt_metal::experimental {
namespace {
deployment_detail::BuildConfiguration build_configuration(const llrt::RunTimeOptions& options) {
    const auto profile = options.get_profiler_enabled() ? DeviceProfilerMode::Program : DeviceProfilerMode::Disabled;
    options.validate_device_profiler_mode(profile);
    // Other instrumentation requires its own typed protocol. Reject it rather
    // than silently dropping a baked semantic input from the snapshot.
    if (options.get_streaming_profiler_enabled() || options.get_profiler_noc_events_enabled() ||
        options.get_profiler_sync_events_enabled() || options.get_profiler_perf_counter_mode() ||
        options.get_profiler_do_dispatch_cores() || options.get_profiler_trace_only() ||
        options.get_profiler_sum() || options.get_profiler_accumulate() ||
        options.get_watcher_enabled() || options.get_watcher_noinline() ||
        options.get_watcher_noc_sanitize_linked_transaction() || !options.get_watcher_disabled_features().empty() ||
        options.get_checkpoint_enabled() || options.get_record_noc_transfers() || options.get_kernels_nullified() ||
        options.get_kernels_early_return() || options.get_measure_dfb_init_time_enabled() ||
        options.get_watcher_debug_delay() || options.get_sanitizer_settings().enabled ||
        options.get_experimental_noc_debug_dump_enabled() ||
        options.get_brisc_firmware_variant() != llrt::BriscFirmwareVariant::Default)
        throw std::invalid_argument("unsupported published deployment instrumentation");
    for (int i = 0; i != llrt::RunTimeDebugFeatureCount; ++i)
        if (options.get_feature_enabled(static_cast<llrt::RunTimeDebugFeatures>(i)))
            throw std::invalid_argument("unsupported published deployment debug feature");
    return {profile, options.get_enable_2_erisc_mode(), options.get_erisc_iram_enabled(),
        options.get_dram_backed_cq(), options.get_eth_ptp_trace(), options.get_disable_fabric_2_erisc_mode(),
        options.get_hw_cache_invalidation_enabled(), options.get_relaxed_memory_ordering_disabled(),
        options.get_gathering_enabled(), options.get_lightweight_kernel_asserts(), options.get_llk_asserts(),
        options.get_disable_sfploadmacro()};
}
} // namespace
struct DeploymentConfiguration::Impl : deployment_detail::ConfigurationStorage {
    using ConfigurationStorage::ConfigurationStorage;
};
DeploymentConfiguration::DeploymentConfiguration(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
DeploymentConfiguration DeploymentConfiguration::from_sdk(
    const JitDeviceConfig& config, const llrt::RunTimeOptions& options,
    const DispatchProgramPlan& program) {
    if (!config.hal) throw std::invalid_argument("published device configuration has no HAL");
    const auto& observation = program.inputs().observation;
    if (observation.watcher_enabled != options.get_watcher_enabled() ||
        observation.watcher_noinline != options.get_watcher_noinline() ||
        observation.watcher_dispatch_disabled != options.watcher_dispatch_disabled() ||
        observation.telemetry_disabled != options.get_dispatch_telemetry_disabled())
        throw std::invalid_argument("published dispatch observation differs from build options");
    return deployment_detail::Access::admit_configuration(config, build_configuration(options), program, *config.hal);
}
DeploymentConfiguration deployment_detail::Access::admit_configuration(
    const DeviceConfiguration& config, const BuildConfiguration& build,
    const DispatchProgramPlan& program, const Hal& hal) {
    if (config.arch != tt::ARCH::BLACKHOLE || hal.get_arch() != config.arch ||
        !config.num_dram_banks || !config.num_l1_banks || !config.num_hw_cqs ||
        config.max_cbs != hal.get_arch_num_circular_buffers() ||
        config.coordinate_virtualization_enabled != hal.is_coordinate_virtualization_enabled() ||
        !config.dispatch_message_addr ||
        (config.dispatch_core_axis != DispatchCoreAxis::ROW && config.dispatch_core_axis != DispatchCoreAxis::COL) ||
        !((config.dispatch_core_type == DispatchCoreType::WORKER && config.resolved_dispatch_core_type == tt::CoreType::WORKER) ||
          (config.dispatch_core_type == DispatchCoreType::ETH && config.resolved_dispatch_core_type == tt::CoreType::ETH)) ||
        (build.profile != DeviceProfilerMode::Disabled && build.profile != DeviceProfilerMode::Program))
        throw std::invalid_argument("invalid or incompatible published device configuration");
    const auto& input = program.inputs();
    deployment_detail::RecordWriter expected_hal, stored_hal;
    expected_hal.append(capture_dispatch_hal(hal)); stored_hal.append(input.hal);
    if (input.topology.arch != config.arch || input.topology.core_type != config.resolved_dispatch_core_type ||
        input.topology.num_hw_cqs != config.num_hw_cqs || input.workers.count != config.num_l1_banks ||
        std::holds_alternative<DramQueueBacking>(input.queue.inputs.backing) != build.dram_backed_cq ||
        expected_hal.bytes != stored_hal.bytes ||
        (input.queue.dram_bank && *input.queue.dram_bank >= config.num_dram_banks))
        throw std::invalid_argument("published dispatch program differs from device, HAL or build configuration");
    auto owner = std::make_shared<DeploymentConfiguration::Impl>(program);
    owner->device = config;
    owner->build = build;
    if (owner->build.profile == DeviceProfilerMode::Program && !config.profiler_dram_bank_size_per_risc_bytes)
        throw std::invalid_argument("program profiler has no deployment allocation");
    for (uint32_t core = 0; core != hal.get_programmable_core_type_count(); ++core) {
        const auto type = hal.get_programmable_core_type(core);
        deployment_detail::CoreLayout layout{type, {}, {}};
        for (unsigned address = 0; address != static_cast<unsigned>(HalL1MemAddrType::COUNT); ++address) {
            const auto slot = static_cast<HalL1MemAddrType>(address);
            // HAL deliberately has no fixed Tensix UNRESERVED allocation or
            // KERNEL_CONFIG size: these belong to the device/program planner.
            const bool dynamic = type == HalProgrammableCoreType::TENSIX;
            layout.bases.push_back(dynamic && slot == HalL1MemAddrType::UNRESERVED
                ? std::nullopt : std::optional<DeviceAddr>(hal.get_dev_addr(type, slot)));
            layout.sizes.push_back(dynamic && (slot == HalL1MemAddrType::UNRESERVED || slot == HalL1MemAddrType::KERNEL_CONFIG)
                ? std::nullopt : std::optional<uint32_t>(hal.get_dev_size(type, slot)));
        }
        owner->cores.push_back(std::move(layout));
        for (uint32_t cls = 0; cls != hal.get_processor_classes_count(type); ++cls) {
            for (uint32_t processor = 0; processor != hal.get_processor_class_num_fw_binaries(core, cls); ++processor) {
                HalProcessorIdentifier id{type, static_cast<HalProcessorClassType>(cls), static_cast<int>(processor)};
                owner->processors.push_back(id);
                const auto regions = hal.get_processor_image_regions(id);
                if (!regions) throw std::invalid_argument("published processor has no image region contract");
                owner->layouts.push_back({id, hal.get_jit_build_config(core, cls, processor), *regions});
            }
        }
    }
    for (unsigned address = 0; address != static_cast<unsigned>(HalDramMemAddrType::COUNT); ++address) {
        owner->dram_bases.push_back(hal.get_dev_addr(static_cast<HalDramMemAddrType>(address)));
        owner->dram_sizes.push_back(hal.get_dev_size(static_cast<HalDramMemAddrType>(address)));
    }
    std::set<uint32_t> seen;
    std::set<std::tuple<ChipId, uint32_t, uint32_t, HalProgrammableCoreType, int>> placements;
    for (const auto& node : owner->program.dispatch_nodes()) {
        const auto expected_core = config.dispatch_core_type == DispatchCoreType::WORKER
            ? HalProgrammableCoreType::TENSIX : HalProgrammableCoreType::IDLE_ETH;
        if (node.node_id == NoDispatchNode || !seen.insert(node.node_id).second || node.cq_id >= config.num_hw_cqs ||
            std::find(owner->processors.begin(), owner->processors.end(), node.processor) == owner->processors.end() ||
            node.processor.processor_class != HalProcessorClassType::DM ||
            node.processor.core_type != expected_core ||
            !placements.emplace(node.device_id, node.logical_core.x, node.logical_core.y,
                                node.processor.core_type, node.processor.processor_type).second ||
            node.nocs.non_dispatch_noc > NOC_1 || node.nocs.upstream_noc > NOC_1 || node.nocs.downstream_noc > NOC_1 ||
            !node.resolved || !node.resolved->command_queue_size || !node.resolved->worker_count ||
            node.resolved->dram_backed_cq != owner->build.dram_backed_cq)
            throw std::invalid_argument("invalid published dispatch node or unresolved placement");
        validate_published_dispatch_configuration(node);
    }
    deployment_detail::RecordWriter record;
    record.append(owner->device); record.append(owner->build); record.append(owner->layouts);
    record.append(owner->cores); record.append(owner->dram_bases); record.append(owner->dram_sizes);
    const auto program_record = owner->program.canonical_record();
    record.append(program_record.size());
    record.bytes.insert(record.bytes.end(), program_record.begin(), program_record.end());
    owner->canonical = std::move(record.bytes);
    return DeploymentConfiguration(std::move(owner));
}
const DeviceConfiguration& DeploymentConfiguration::device() const { return impl_->device; }
DeviceProfilerMode DeploymentConfiguration::profiler_mode() const { return impl_->build.profile; }
std::span<const HalProcessorIdentifier> DeploymentConfiguration::firmware_processors() const { return impl_->processors; }
std::span<const DispatchKernelConfiguration> DeploymentConfiguration::dispatch_nodes() const { return impl_->program.dispatch_nodes(); }
const DispatchProgramPlan& DeploymentConfiguration::dispatch_program() const { return impl_->program; }
bool DeploymentConfiguration::matches(const DeploymentConfiguration& other) const { return impl_->canonical == other.impl_->canonical; }
std::vector<std::byte> DeploymentConfiguration::image_record(PublishedImageKind kind, HalProcessorIdentifier processor, uint32_t node) const {
    if (std::find(impl_->processors.begin(), impl_->processors.end(), processor) == impl_->processors.end())
        throw std::invalid_argument("publication processor is not required by HAL");
    if (kind == PublishedImageKind::Firmware) {
        if (node != NoDispatchNode) throw std::invalid_argument("firmware cannot have a dispatch node");
    } else if (kind == PublishedImageKind::Dispatch) {
        const auto nodes = dispatch_nodes();
        auto found = std::find_if(nodes.begin(), nodes.end(), [&](const auto& value) { return value.node_id == node; });
        if (found == nodes.end())
            throw std::invalid_argument("dispatch image does not belong to configured node/processor");
        const auto processors = dispatch_processors(*found);
        if (std::find(processors.begin(), processors.end(), processor) == processors.end())
            throw std::invalid_argument("dispatch image does not belong to configured node/processor");
    } else throw std::invalid_argument("invalid publication image kind");
    deployment_detail::RecordWriter record;
    record.append(uint32_t{0x54545044}); record.append(uint32_t{2});
    record.append(kind); record.append(processor); record.append(node);
    record.append(impl_->canonical.size());
    record.bytes.insert(record.bytes.end(), impl_->canonical.begin(), impl_->canonical.end());
    return std::move(record.bytes);
}
const deployment_detail::ConfigurationStorage& deployment_detail::Access::configuration(const DeploymentConfiguration& config) {
    return *config.impl_;
}
} // namespace tt::tt_metal::experimental
