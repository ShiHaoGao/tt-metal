// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "impl/kernels/external_binary_kernel.hpp"
#include "impl/experimental/published_deployment/dispatch_plan.hpp"
#include <mutex>
#include <unordered_map>
namespace tt::tt_metal::experimental {
class DispatchProgramPlan;
// Materialize the complete admitted kernel/argument inventory after the
// topology has materialized the exact plan semaphores. Never enters source/JIT.
void CreateDispatchKernelsFromPublishedDeployment(
    Program&, std::shared_ptr<const PublishedDeployment>, const DispatchProgramPlan&);
// Internal SDK entrypoint. Placement and NoC come from the admitted node; the
// processor list must select a complete kernel group from its canonical plan.
KernelHandle CreateKernelFromPublishedDeployment(
    Program&, std::shared_ptr<const PublishedDeployment>, uint32_t node,
    std::span<const HalProcessorIdentifier> processors);
}
namespace tt::tt_metal {
class PublishedDispatchKernel final : public ExternalBinaryKernel {
public:
    PublishedDispatchKernel(const KernelBuildContext&, ContextId,
                            std::shared_ptr<const experimental::PublishedDeployment>, uint32_t node,
                            std::span<const HalProcessorIdentifier> processors);
    Config config() const override { return config_; }
    uint8_t expected_num_binaries() const override;
    uint32_t get_kernel_processor_type(int index) const override;
    bool configure(IDevice*, const CoreCoord&, uint32_t, const uint32_t[]) const override;
    void generate_binaries(IDevice*, JitBuildOptions&) const override;
    void read_binaries(IDevice*, const std::string&) override;
    std::string_view get_compiler_opt_level() const override;
    std::string_view get_linker_opt_level() const override;
    std::shared_ptr<const experimental::native_detail::LoadedFirmware>
    validate_deployment(IDevice&, tt::worker_stream_state::Owner) const override;
    void prepare(IDevice*, tt::worker_stream_state::Owner) override;
    const std::vector<const ll_api::memory*>& owned_binaries() const override { return owned_binaries_; }
private:
    PublishedDispatchKernel(const KernelBuildContext&, ContextId, PublishedDispatchImages);
    std::string config_hash() const override;
    Config config_;
    std::vector<const ll_api::memory*> owned_binaries_;
    mutable std::mutex deployment_mutex_;
    mutable std::unordered_map<ChipId, std::weak_ptr<const experimental::native_detail::LoadedFirmware>> deployments_;
};
} // namespace tt::tt_metal
