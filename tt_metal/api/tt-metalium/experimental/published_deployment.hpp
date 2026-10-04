// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/experimental/native_kernel.hpp>
#include <tt-metalium/experimental/context/device_profiler_config.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>
namespace tt::llrt { class RunTimeOptions; }
namespace tt::tt_metal {
struct JitDeviceConfig;
class Hal;
// Canonical value owner shared by source compilation and published deployment.
// HAL remains a construction/query dependency, never part of the owned value.
struct DeviceConfiguration {
    tt::ARCH arch = tt::ARCH::Invalid;

    size_t num_dram_banks = 0;
    size_t num_l1_banks = 0;
    CoreCoord pcie_core{0, 0};

    uint32_t harvesting_mask = 0;
    DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
    // Effective dispatch placement (Quasar: DISPATCH vs WORKER from soc/env, not DispatchCoreConfig alone).
    tt::CoreType resolved_dispatch_core_type = tt::CoreType::WORKER;
    DispatchCoreAxis dispatch_core_axis = DispatchCoreAxis::ROW;
    bool coordinate_virtualization_enabled = false;

    uint32_t dispatch_message_addr = 0;
    uint32_t max_cbs = 0;
    uint8_t num_hw_cqs = 0;

    bool routing_fw_enabled = false;

    // Pre-computed in the factory so that JitBuildEnv::init can consume it without
    // calling get_profiler_dram_bank_size_per_risc_bytes(), which has a side-effect
    // of mutating rtoptions (set_profiler_program_support_count). The build module
    // must only observe const RunTimeOptions; any mutation belongs in the factory or
    // in the profiler subsystem itself.
    uint32_t profiler_dram_bank_size_per_risc_bytes = 0;
};
namespace experimental {
struct DispatchKernelConfiguration;
class DispatchProgramPlan;
namespace deployment_detail { class Access; }

enum class PublishedImageKind : uint8_t { Firmware, Dispatch };
inline constexpr uint32_t NoDispatchNode = UINT32_MAX;
inline constexpr char PublishedImageSection[] = ".tt_published_image";

class DeploymentConfiguration {
public:
    // Pure snapshot of the existing SDK owners. No context, filesystem or compiler.
    static DeploymentConfiguration from_sdk(const JitDeviceConfig&, const llrt::RunTimeOptions&,
                                            const DispatchProgramPlan&);
    const DeviceConfiguration& device() const;
    DeviceProfilerMode profiler_mode() const;
    std::span<const HalProcessorIdentifier> firmware_processors() const;
    std::span<const DispatchKernelConfiguration> dispatch_nodes() const;
    const DispatchProgramPlan& dispatch_program() const;
    bool matches(const DeploymentConfiguration&) const;
    // Offline publisher uses this complete, versioned record as host-only ELF
    // metadata. It contains configuration values, not a hash or boot receipt.
    std::vector<std::byte> image_record(PublishedImageKind, HalProcessorIdentifier,
                                      uint32_t dispatch_node = NoDispatchNode) const;
private:
    struct Impl;
    explicit DeploymentConfiguration(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
    friend class deployment_detail::Access;
};
struct PublishedImageInput {
    PublishedImageKind kind;
    HalProcessorIdentifier processor;
    uint32_t dispatch_node;
    std::span<const std::byte> bytes;
};
class PublishedDeployment {
public:
    static PublishedDeployment admit(const DeploymentConfiguration&, std::span<const PublishedImageInput>);
    std::vector<std::byte> to_archive() const;
    // Decode compiler-owned expectations using the HAL requirements already in
    // the canonical archive. This does not assert compatibility with a device.
    static PublishedDeployment from_archive(std::span<const std::byte>);
    // Explicit actual-HAL admission additionally requires an exact match.
    static PublishedDeployment from_archive(std::span<const std::byte>, const Hal& actual_hal);
    const DeploymentConfiguration& configuration() const;
    const FirmwareBundle& tensix_firmware() const;
    std::span<const std::byte> firmware_image(HalProcessorIdentifier) const;
    std::span<const std::byte> dispatch_image(uint32_t node, HalProcessorIdentifier) const;
    void validate_configuration(const DeploymentConfiguration&) const;
private:
    struct Impl;
    explicit PublishedDeployment(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
    friend class deployment_detail::Access;
};
} // namespace experimental
} // namespace tt::tt_metal
