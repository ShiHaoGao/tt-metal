// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include "hostdev/dev_msgs.h"

// The same physical counter layout is consumed by firmware and by host-side
// native support publication. Do not pull device operations or RVTT pointers
// into host layout projection.
#ifdef HAL_BUILD
namespace HAL_BUILD {
#endif

enum class NocBarrierType : uint8_t {
    READS_NUM_ISSUED,
    NONPOSTED_WRITES_NUM_ISSUED,
    NONPOSTED_WRITES_ACKED,
    NONPOSTED_ATOMICS_ACKED,
    POSTED_WRITES_NUM_ISSUED,
    COUNT
};

static constexpr uint8_t NUM_BARRIER_TYPES = static_cast<uint32_t>(NocBarrierType::COUNT);

struct BarrierCounter {
    uint32_t barrier[NUM_BARRIER_TYPES];
};

struct RiscBarrierCounter {
    BarrierCounter risc[MaxDMProcessorsPerCoreType];
};

struct NocBarrierCounter {
    RiscBarrierCounter noc[NUM_NOCS];
};

// Must update the allocated size for the counters in dev_mem_map.h AND base FW if this changes
static_assert(sizeof(NocBarrierCounter) == 80, "NocBarrierCounter size is not 80 bytes");

#ifdef HAL_BUILD
} // namespace HAL_BUILD
#endif
