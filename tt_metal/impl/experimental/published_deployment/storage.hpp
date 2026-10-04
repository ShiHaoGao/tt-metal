// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "configuration.hpp"
#include "impl/experimental/published_deployment/dispatch_program_plan.hpp"
#include "llrt/hal.hpp"
namespace tt::tt_metal::experimental::deployment_detail {
struct BuildConfiguration {
    DeviceProfilerMode profile;
    bool two_erisc;
    bool erisc_iram;
    bool dram_backed_cq;
    bool eth_ptp_trace;
    bool disable_fabric_two_erisc;
    bool hw_cache_invalidation;
    bool relaxed_memory_ordering_disabled;
    bool gathering;
    bool lightweight_asserts;
    bool llk_asserts;
    bool disable_sfploadmacro;
};
struct ProcessorLayout { HalProcessorIdentifier processor; HalJitBuildConfig layout; HalProcessorImageRegions regions; };
struct CoreLayout {
    HalProgrammableCoreType type;
    std::vector<std::optional<DeviceAddr>> bases;
    std::vector<std::optional<uint32_t>> sizes;
};
struct ConfigurationStorage {
    explicit ConfigurationStorage(const DispatchProgramPlan& program) : program(program) {}
    DeviceConfiguration device;
    BuildConfiguration build;
    std::vector<HalProcessorIdentifier> processors;
    std::vector<ProcessorLayout> layouts;
    std::vector<CoreLayout> cores;
    std::vector<DeviceAddr> dram_bases;
    std::vector<uint32_t> dram_sizes;
    DispatchProgramPlan program;
    std::vector<std::byte> canonical;
};
class Access {
public:
    static const ConfigurationStorage& configuration(const DeploymentConfiguration&);
    static DeploymentConfiguration admit_configuration(const DeviceConfiguration&, const BuildConfiguration&,
        const DispatchProgramPlan&, const Hal& actual_hal);
    static const ll_api::memory& firmware(const PublishedDeployment&, HalProcessorIdentifier);
    static const ll_api::memory& dispatch(const PublishedDeployment&, uint32_t node, HalProcessorIdentifier);
};
} // namespace tt::tt_metal::experimental::deployment_detail
