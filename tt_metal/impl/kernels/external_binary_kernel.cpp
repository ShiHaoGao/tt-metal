// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "external_binary_kernel.hpp"
#include <tt-metalium/mesh_device.hpp>
#include "impl/context/metal_env_accessor.hpp"
#include <set>

namespace tt::tt_metal {
std::vector<IDevice*> ExternalBinaryKernel::physical_devices(IDevice& device) {
    if (auto* mesh = dynamic_cast<distributed::MeshDevice*>(&device)) {
        auto devices = mesh->get_devices();
        TT_FATAL(!devices.empty(), "External ELF requires local physical devices");
        std::set<ChipId> ids;
        for (auto* physical : devices) {
            TT_FATAL(physical && !dynamic_cast<distributed::MeshDevice*>(physical) &&
                         extract_context_id(physical) == extract_context_id(&device) &&
                         ids.insert(physical->id()).second,
                     "External ELF mesh has invalid physical device ownership");
        }
        return devices;
    }
    return {&device};
}

FirmwareDeployments ExternalBinaryKernel::validate_deployments(
    IDevice& device, tt::worker_stream_state::Owner owner) const {
    FirmwareDeployments result;
    for (auto* physical : physical_devices(device)) {
        auto firmware = validate_deployment(*physical, owner);
        TT_FATAL(firmware && firmware->live(), "External ELF requires each physical firmware boot");
        result.emplace(physical->id(), std::move(firmware));
    }
    return result;
}

void ExternalBinaryKernel::prepare_deployments(IDevice& device, tt::worker_stream_state::Owner owner) {
    // Admit the full device set before any individual preparation.
    validate_deployments(device, owner);
    for (auto* physical : physical_devices(device)) prepare(physical, owner);
}
} // namespace tt::tt_metal
