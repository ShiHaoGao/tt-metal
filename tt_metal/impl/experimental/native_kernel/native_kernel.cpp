// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "native_kernel.hpp"
#include "impl/allocator/allocator_types.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/llrt.hpp"
#include "llrt/zone_meta.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tt::tt_metal {
namespace {
[[noreturn]] void reject_source_compilation() {
    throw std::logic_error("Native ELF has no source/JIT compilation input");
}
NativeElfImages compute_images(const experimental::NativeComputeConfig& config) {
    experimental::native_detail::NativeImageAccess::validate_compute(config);
    return {{config.trisc0, config.trisc1, config.trisc2}};
}
}  // namespace

NativeElfKernel::NativeElfKernel(
    const KernelBuildContext& context,
    ContextId id,
    const CoreRangeSet& cores,
    const experimental::NativeDataMovementConfig& config) :
    ExternalBinaryKernel(
        context,
        id,
        HalProgrammableCoreType::TENSIX,
        HalProcessorClassType::DM,
        NativeElfImages{{config.image}},
        cores,
        {},
        {},
        {}),
    config_(config) {
    TT_FATAL(
        config.image.role() == experimental::TensixKernelRole::Brisc ||
            config.image.role() == experimental::TensixKernelRole::Ncrisc,
        "Native DM requires BRISC or NCRISC");
    retain_binaries();
}
NativeElfKernel::NativeElfKernel(
    const KernelBuildContext& context,
    ContextId id,
    const CoreRangeSet& cores,
    const experimental::NativeComputeConfig& config) :
    ExternalBinaryKernel(
        context,
        id,
        HalProgrammableCoreType::TENSIX,
        HalProcessorClassType::COMPUTE,
        compute_images(config),
        cores,
        {},
        {},
        {}),
    config_(config) {
    retain_binaries();
}

uint32_t NativeElfKernel::get_kernel_processor_type(int index) const {
    const auto role = static_cast<uint32_t>(native_images().images.at(index).role());
    return role < 2 ? role : role - 2;
}
void NativeElfKernel::generate_binaries(IDevice*, JitBuildOptions&) const { reject_source_compilation(); }
void NativeElfKernel::read_binaries(IDevice*, const std::string&) { reject_source_compilation(); }
std::string_view NativeElfKernel::get_compiler_opt_level() const { reject_source_compilation(); }
std::string_view NativeElfKernel::get_linker_opt_level() const { reject_source_compilation(); }
std::string NativeElfKernel::config_hash() const { reject_source_compilation(); }

std::shared_ptr<const experimental::native_detail::LoadedFirmware> NativeElfKernel::validate_deployment(
    IDevice& device, tt::worker_stream_state::Owner owner) const {
    TT_FATAL(extract_context_id(&device) == get_context_id(), "Native ELF belongs to another device context");
    TT_FATAL(MetalContext::instance_exists(get_context_id()), "Native ELF requires an existing device context");
    auto& context = MetalContext::instance(extract_context_id(&device));
    TT_FATAL(device.arch() == tt::ARCH::BLACKHOLE, "Native ELF currently requires Blackhole");
    TT_FATAL(
        context.get_cluster().get_target_device_type() != tt::TargetDevice::Emule,
        "Native RISC-V ELF cannot execute through source-based emulation");
    auto loaded = context.native_firmware(device.id());
    TT_FATAL(loaded && loaded->live(), "Native ELF requires a successful boot of retained firmware bytes");
    validate_stream_owner(owner);
    for (const auto& image : native_images().images) {
        TT_FATAL(image.firmware().matches(loaded->bundle()), "Native ELF linked firmware differs from actual boot");
        const auto type = static_cast<uint32_t>(image.role());
        const auto expected = context.hal()
                                  .get_jit_build_config(
                                      context.hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX),
                                      static_cast<uint32_t>(get_kernel_processor_class()),
                                      type < 2 ? type : type - 2)
                                  .memory_load;
        TT_FATAL(
            experimental::native_detail::NativeImageAccess::memory(image).get_loading() == expected,
            "Native ELF loading mode differs from processor HAL");
    }
    std::lock_guard lock(deployment_mutex_);
    const auto previous = deployments_.find(device.id());
    if (previous != deployments_.end()) {
        TT_FATAL(previous->second.lock() == loaded, "Native ELF firmware generation changed");
    }
    TT_FATAL(loaded->live(), "Native ELF firmware was withdrawn during admission");
    deployments_.insert_or_assign(device.id(), loaded);
    return loaded;
}

void NativeElfKernel::validate_stream_owner(tt::worker_stream_state::Owner owner) const {
    uint8_t required;
    switch (owner) {
        case tt::worker_stream_state::Owner::SdkCircularBuffers: required = TT_NATIVE_WORKER_STREAM_SDK; break;
        case tt::worker_stream_state::Owner::Program: required = TT_NATIVE_WORKER_STREAM_PROGRAM; break;
        default: TT_THROW("Native ELF received an unknown worker stream owner");
    }
    for (const auto& image : native_images().images) {
        TT_FATAL(
            image.contract().worker_stream_owner == required,
            "Native ELF worker stream ownership differs from Program");
    }
}

void NativeElfKernel::validate_managed_traffic() const {
    if (const auto* dm = std::get_if<experimental::NativeDataMovementConfig>(&config_)) {
        TT_FATAL(
            dm->noc_mode == NOC_MODE::DM_DYNAMIC_NOC,
            "Managed native DM requires the actual dynamic NoC counter producer");
    }
    for (const auto& image : native_images().images) {
        const auto algorithm = image.noc_contract();
        const auto firmware = image.firmware().niu_handoff();
        TT_FATAL(algorithm, "Managed native algorithm has no verified NoC traffic declaration");
        TT_FATAL(
            firmware && firmware->traffic_contract == algorithm->traffic_contract,
            "Managed native algorithm requires matching firmware NIU handoff support");
        TT_FATAL(
            image.noc_windows().empty(),
            "Managed native relative L1 windows require original live allocation bindings");
    }
}

void NativeElfKernel::validate_managed_traffic(const AllocatorConfig& layout) const {
    validate_managed_traffic();
    TT_FATAL(
        layout.l1_unreserved_base && layout.l1_unreserved_base < layout.worker_l1_size,
        "Managed native traffic requires the actual configured L1 data region");
    for (const auto& image : native_images().images) {
        const auto algorithm = *image.noc_contract();
        TT_FATAL(
            (algorithm.l1_begin == 0 && algorithm.l1_end == 0) ||
                (algorithm.l1_begin >= layout.l1_unreserved_base && algorithm.l1_end <= layout.worker_l1_size),
            "Managed native memory accesses overlap the actual SDK load region or leave L1");
    }
}

void NativeElfKernel::validate_managed_endpoints(std::span<const tt_native_noc_endpoint> allowed) const {
    for (const auto& image : native_images().images) {
        for (const auto& endpoint : image.noc_endpoints()) {
            const auto found = std::find_if(allowed.begin(), allowed.end(), [&](const auto& candidate) {
                return candidate.noc == endpoint.noc && candidate.x == endpoint.x && candidate.y == endpoint.y;
            });
            TT_FATAL(
                found != allowed.end(),
                "Managed native algorithm endpoint is outside the actual invocation Worker footprint");
        }
    }
}

void NativeElfKernel::validate_managed_group(std::span<const NativeElfKernel* const> kernels) {
    TT_FATAL(!kernels.empty(), "Managed native group requires actual algorithms");
    std::set<CoreCoord> active;
    std::set<CoreCoord> dynamic;
    for (const auto* kernel : kernels) {
        TT_FATAL(kernel, "Managed native group contains no actual algorithm");
        kernel->validate_managed_traffic();
        const auto& cores = kernel->logical_cores();
        active.insert(cores.begin(), cores.end());
        if (std::holds_alternative<experimental::NativeDataMovementConfig>(kernel->config_)) {
            dynamic.insert(cores.begin(), cores.end());
        }
    }
    TT_FATAL(
        std::includes(dynamic.begin(), dynamic.end(), active.begin(), active.end()),
        "Managed native group requires a dynamic DM configuration on every active Worker");
}

void NativeElfKernel::retain_binaries() {
    owned_binaries_.reserve(native_images().images.size());
    for (const auto& image : native_images().images) {
        owned_binaries_.push_back(&experimental::native_detail::NativeImageAccess::memory(image));
    }
}

void NativeElfKernel::prepare(IDevice* device, tt::worker_stream_state::Owner owner) {
    TT_FATAL(MetalContext::instance_exists(get_context_id()), "Native ELF requires an existing device context");
    TT_FATAL(device != nullptr, "Native ELF preparation requires a device");
    validate_deployment(*device, owner);
    for (const auto& image : native_images().images) {
        if (image.contract().profile) {
            tt::llrt::ZoneMetaRegistry::instance().ingest_elf(image.image_bytes());
        }
    }
}

void experimental::PrepareNativeProgram(Program& program, IDevice& device) {
    // Keep all native-only admission and preparation checks in the SDK's
    // ProgramImpl owner.  This wrapper is the installed typed boundary; it
    // intentionally does not reproduce or weaken those checks here.
    program.impl().prepare_native(&device);
}

bool NativeElfKernel::configure(
    IDevice* device, const CoreCoord& logical, uint32_t base, const uint32_t offsets[]) const {
    TT_FATAL(is_on_logical_core(logical), "Native kernel is not placed on requested core");
    auto& context = MetalContext::instance(extract_context_id(device));
    auto& env = MetalEnvAccessor(context.get_env()).impl();
    const auto core = device->worker_core_from_logical_core(logical);
    for (size_t i = 0; i != native_images().images.size(); ++i) {
        const auto role = static_cast<uint32_t>(native_images().images[i].role());
        llrt::write_binary_to_address(env, *owned_binaries()[i], device->id(), core, base + offsets[role]);
    }
    return true;
}

experimental::FirmwareBundle experimental::GetLoadedFirmwareBundle(IDevice& device) {
    auto context_id = extract_context_id(&device);
    TT_FATAL(MetalContext::instance_exists(context_id), "Native firmware query requires an existing device context");
    auto loaded = MetalContext::instance(context_id).native_firmware(device.id());
    TT_FATAL(loaded && loaded->live(), "Device has no live native firmware boot receipt");
    return loaded->bundle();
}
}  // namespace tt::tt_metal
