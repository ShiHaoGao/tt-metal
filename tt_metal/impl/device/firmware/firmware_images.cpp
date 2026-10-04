// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "firmware_images.hpp"

#include "impl/experimental/published_deployment/storage.hpp"
#include <utility>

namespace tt::tt_metal {

FirmwareImages::FirmwareImages(
    experimental::PublishedDeployment deployment, const JitDeviceConfig& device, const llrt::RunTimeOptions& options) :
    deployment_(std::move(deployment)) {
    // This boot owner checks actual device/HAL/options against the retained
    // plan. The topology owner separately compares a fresh live capture before
    // dispatch side effects; stored topology is never a claim of live state.
    deployment_.validate_configuration(experimental::DeploymentConfiguration::from_sdk(
        device, options, deployment_.configuration().dispatch_program()));
}

const ll_api::memory& FirmwareImages::image(HalProcessorIdentifier processor) const {
    return experimental::deployment_detail::Access::firmware(deployment_, processor);
}

const experimental::FirmwareBundle& FirmwareImages::tensix() const {
    return deployment_.tensix_firmware();
}

} // namespace tt::tt_metal
