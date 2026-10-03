// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace tt::tt_metal {
class RiscFirmwareInitializer;
// Identity is tied to one completed load/boot, never a path or SDK build option.
class WorkerStreamStateProvider final {
public:
    uint8_t version() const { return version_; }
private:
    friend class RiscFirmwareInitializer;
    explicit WorkerStreamStateProvider(uint8_t version) : version_(version) {}
    uint8_t version_;
};
}  // namespace tt::tt_metal
