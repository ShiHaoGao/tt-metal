// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "impl/kernels/external_binary_kernel.hpp"
#include <mutex>
#include <unordered_map>

namespace tt::tt_metal {
class NativeElfKernel final : public ExternalBinaryKernel {
public:
    NativeElfKernel(const KernelBuildContext&, ContextId, const CoreRangeSet&,
                    const experimental::NativeDataMovementConfig&);
    NativeElfKernel(const KernelBuildContext&, ContextId, const CoreRangeSet&,
                    const experimental::NativeComputeConfig&);
    Config config() const override { return config_; }
    uint8_t expected_num_binaries() const override { return native_images().images.size(); }
    uint32_t get_kernel_processor_type(int index) const override;
    bool configure(IDevice*, const CoreCoord&, uint32_t, const uint32_t[]) const override;
    void generate_binaries(IDevice*, JitBuildOptions&) const override;
    void read_binaries(IDevice*, const std::string&) override;
    std::string_view get_compiler_opt_level() const override;
    std::string_view get_linker_opt_level() const override;
    std::shared_ptr<const experimental::native_detail::LoadedFirmware>
    validate_deployment(IDevice&, tt::worker_stream_state::Owner) const override;
    void validate_stream_owner(tt::worker_stream_state::Owner) const;
    void prepare(IDevice*, tt::worker_stream_state::Owner) override;
    const std::vector<const ll_api::memory*>& owned_binaries() const override { return owned_binaries_; }
private:
    void retain_binaries();
    std::string config_hash() const override;
    Config config_;
    std::vector<const ll_api::memory*> owned_binaries_;
    mutable std::mutex deployment_mutex_;
    mutable std::unordered_map<ChipId, std::weak_ptr<const experimental::native_detail::LoadedFirmware>> deployments_;
};
}  // namespace tt::tt_metal
