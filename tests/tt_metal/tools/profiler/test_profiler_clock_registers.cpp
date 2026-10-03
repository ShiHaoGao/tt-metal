// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only HAL validation; constructing this object does not open a device.
#include "llrt/hal.hpp"

#include <cstdio>

int main() {
    const tt::tt_metal::Hal hal(tt::ARCH::BLACKHOLE, false, false, 0, false);
    unsigned failures = 0;
    for (uint32_t address : {0xffb121f0u, 0xffb121f4u, 0xffb121f8u}) {
        if (!hal.valid_reg_addr(address)) {
            std::fprintf(stderr, "clock register 0x%x is not admitted\n", address);
            ++failures;
        }
    }
    for (uint32_t address : {0xffb121ecu, 0xffb121f1u, 0xffb121f5u, 0xffb121fcu}) {
        if (hal.valid_reg_addr(address)) {
            std::fprintf(stderr, "unowned register 0x%x is admitted\n", address);
            ++failures;
        }
    }
    std::printf("profiler clock registers: %u passed, %u failed\n", 7 - failures, failures);
    return failures ? 1 : 0;
}
