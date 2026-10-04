// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tt-metalium/experimental/published_deployment.hpp"
#include "llrt/tt_memory.h"

namespace tt::tt_metal {

// Holds the admitted bytes used by one device's boot. Construction validates
// the actual device, HAL and build options; it does not prove firmware boot.
class FirmwareImages {
public:
    FirmwareImages(experimental::PublishedDeployment, const JitDeviceConfig&, const llrt::RunTimeOptions&);
    const ll_api::memory& image(HalProcessorIdentifier) const;
    const experimental::FirmwareBundle& tensix() const;

private:
    experimental::PublishedDeployment deployment_;
};

} // namespace tt::tt_metal
