// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Deployment selected before the environment creates firmware/JIT resources.
// Program uses the ordinary start/end marker protocol, without trace-only,
// summed, NoC-event, or performance-counter overlays.
enum class DeviceProfilerMode : std::uint8_t { Disabled, Program };

}  // namespace tt::tt_metal
