// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <map>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "tt-metalium/experimental/native_kernel.hpp"
#include "llrt/tt_memory.h"

namespace tt::tt_metal::experimental::native_detail {
class NativeImageAccess {
public:
    static const ll_api::memory& memory(const KernelElfImage&);
    static const ll_api::memory& memory(const FirmwareBundle&, TensixKernelRole);
    static void validate_compute(const NativeComputeConfig&);
};

// Constructor is reachable only at the successful firmware boot seam. A
// retained state becomes invalid on teardown even while Programs retain it.
class LoadedFirmware {
public:
    const FirmwareBundle& bundle() const { return bundle_; }
    bool live() const { return live_.load(); }
private:
    explicit LoadedFirmware(FirmwareBundle bundle) : bundle_(std::move(bundle)) {}
    void withdraw() const { live_.store(false); }
    FirmwareBundle bundle_;
    mutable std::atomic<bool> live_{true};
    friend class ::tt::tt_metal::RiscFirmwareInitializer;
};
}  // namespace tt::tt_metal::experimental::native_detail

namespace tt::tt_metal {
// Physical chip IDs retain every boot generation for a mesh deployment.
using FirmwareDeployments =
    std::map<ChipId, std::shared_ptr<const experimental::native_detail::LoadedFirmware>>;
}
