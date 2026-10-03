// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace kernel_profiler {

struct WallClockSample {
    uint32_t high;
    uint32_t low;

    uint64_t value() const { return (static_cast<uint64_t>(high) << 32) | low; }
};

// Blackhole and Wormhole share a low-word-triggered high latch between readers.
// Use live high on both sides of low so another reader cannot corrupt a sample.
// The production callbacks must perform sequenced volatile register reads.
template <class ReadHigh, class ReadLow>
inline __attribute__((always_inline)) WallClockSample read_coherent_wall_clock(ReadHigh&& read_high, ReadLow&& read_low) {
    WallClockSample sample;
    uint32_t final_high;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    do {
        sample.high = read_high();
        sample.low = read_low();
        final_high = read_high();
    } while (sample.high != final_high);
    return sample;
}

}  // namespace kernel_profiler
