// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only HAL validation; constructing this object does not open a device.
#include "llrt/hal.hpp"

#include <cstdio>

int main() {
    unsigned failures = 0;
    for (tt::ARCH architecture : {tt::ARCH::BLACKHOLE, tt::ARCH::WORMHOLE_B0}) {
        const tt::tt_metal::Hal hal(architecture, false, false, 0, false);
        for (uint32_t address : {0xffb121f0u, 0xffb121f4u}) {
            if (!hal.valid_reg_addr(address)) {
                std::fprintf(stderr, "architecture %d clock register 0x%x is not admitted\n",
                             static_cast<int>(architecture), address);
                ++failures;
            }
        }
        // Keep Blackhole's existing streaming-profiler latch access, without
        // broadening Wormhole beyond the two coherent-reader registers.
        if (static_cast<bool>(hal.valid_reg_addr(0xffb121f8u)) !=
            (architecture == tt::ARCH::BLACKHOLE)) {
            std::fprintf(stderr, "architecture %d latched-high admission changed\n",
                         static_cast<int>(architecture));
            ++failures;
        }
        for (uint32_t address : {0xffb121ecu, 0xffb121f1u, 0xffb121f5u, 0xffb121fcu}) {
            if (hal.valid_reg_addr(address)) {
                std::fprintf(stderr, "architecture %d unowned register 0x%x is admitted\n",
                             static_cast<int>(architecture), address);
                ++failures;
            }
        }
    }
    std::printf("profiler clock registers: %u passed, %u failed\n", 14 - failures, failures);
    return failures ? 1 : 0;
}
