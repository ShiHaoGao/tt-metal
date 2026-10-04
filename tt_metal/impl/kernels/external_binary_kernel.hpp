// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "impl/kernels/kernel.hpp"
namespace tt::tt_metal {
// Shared preparation boundary for admitted numerical and SDK deployment ELF.
// The distinct typed inputs retain their original admission/ownership contracts.
class ExternalBinaryKernel : public Kernel {
public:
    // Mesh IDs are logical object identities, never firmware chip IDs. Retain
    // every physical boot generation, including when a mesh has one device.
    static std::vector<IDevice*> physical_devices(IDevice&);
    FirmwareDeployments validate_deployments(IDevice&, tt::worker_stream_state::Owner) const;
    void prepare_deployments(IDevice&, tt::worker_stream_state::Owner);
    virtual const std::vector<const ll_api::memory*>& owned_binaries() const = 0;
    virtual std::shared_ptr<const experimental::native_detail::LoadedFirmware>
    validate_deployment(IDevice&, tt::worker_stream_state::Owner) const = 0;
    virtual void prepare(IDevice*, tt::worker_stream_state::Owner) = 0;
protected:
    using Kernel::Kernel;
};
} // namespace tt::tt_metal
