// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "tools/profiler/kernel_profiler.hpp"

#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS 10000
#endif

void kernel_main() {
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("CBP_AUX_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("CBP_AUX_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("CBP_AUX_COMPUTE_PACK");
#endif

    volatile uint32_t sink = 0;
    for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
        asm volatile("" ::: "memory");
        sink ^= i + 0x6d2b79f5u;
    }
}
