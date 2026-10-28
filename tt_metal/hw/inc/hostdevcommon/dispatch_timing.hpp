// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "hostdev/realtime_profiler_msgs.h"

namespace tt::dispatch_timing {

inline constexpr uint32_t capacity = 32;
inline constexpr uint32_t marker = 0x44544731;  // DTG1: dispatch GO -> matching completion.
enum class State : uint32_t { Pending = 1, Complete = 2, Invalid = 3 };

// The queue has one dispatch RISC producer, one co-located TRISC completion
// monitor, and one remote profiler consumer. Publish/retire ordinals only
// after their slots, using uncached L1 pointers on architectures with a cache.
inline void publish_fence() {
#if defined(__riscv)
    asm volatile("fence rw, rw" ::: "memory");
#endif
}

inline bool begin(volatile realtime_profiler_msg_t& msg, uint32_t runtime_id, uint64_t occurrence_id,
                  uint32_t stream, uint32_t expected_count, uint64_t start) {
    if (runtime_id == 0 || runtime_id > 0xffff || occurrence_id == 0 || start == 0) {
        msg.dispatch_timing_dropped = msg.dispatch_timing_dropped + 1;
        return false;
    }
    const uint32_t ordinal = msg.dispatch_timing_published;
    if (ordinal - msg.dispatch_timing_consumed >= capacity) {
        msg.dispatch_timing_dropped = msg.dispatch_timing_dropped + 1;
        return false;
    }
    volatile auto& slot = msg.dispatch_timing_records[ordinal % capacity];
    slot.start_hi = static_cast<uint32_t>(start >> 32);
    slot.start_lo = static_cast<uint32_t>(start);
    slot.runtime_id = runtime_id;
    slot.marker = marker;
    slot.end_hi = 0;
    slot.end_lo = 0;
    slot.occurrence_lo = static_cast<uint32_t>(occurrence_id);
    slot.occurrence_hi = static_cast<uint32_t>(occurrence_id >> 32);
    slot.stream = stream;
    slot.expected_count = expected_count;
    slot.observed_count = 0;
    slot.state = static_cast<uint32_t>(State::Pending);
    slot.dropped = msg.dispatch_timing_dropped;
    slot.reserved0 = 0;
    slot.reserved1 = 0;
    slot.reserved2 = 0;
    publish_fence();
    msg.dispatch_timing_published = ordinal + 1;
    return true;
}

inline bool count_after(uint32_t observed, uint32_t expected, uint32_t width) {
    const uint32_t shift = 32 - width;
    return static_cast<int32_t>((observed - expected) << shift) > 0;
}

// A count beyond the target does not prove when the target was reached. Retire
// that row as Invalid instead of attributing a later occurrence's timestamp.
// Inspect every pending slot so independent completion streams do not block one
// another. The completed ordinal advances only through sealed contiguous rows.
inline bool complete(volatile realtime_profiler_msg_t& msg, uint32_t stream, uint32_t observed_count,
                     uint64_t end, uint32_t width = 32) {
    bool changed = false;
    const uint32_t published = msg.dispatch_timing_published;
    for (uint32_t ordinal = msg.dispatch_timing_completed; ordinal != published; ++ordinal) {
        volatile auto& slot = msg.dispatch_timing_records[ordinal % capacity];
        if (slot.state != static_cast<uint32_t>(State::Pending) || slot.stream != stream) continue;
        const uint32_t expected = slot.expected_count;
        if (observed_count != expected && !count_after(observed_count, expected, width)) continue;
        const uint64_t start = (static_cast<uint64_t>(slot.start_hi) << 32) | slot.start_lo;
        slot.end_hi = static_cast<uint32_t>(end >> 32);
        slot.end_lo = static_cast<uint32_t>(end);
        slot.observed_count = observed_count;
        slot.dropped = msg.dispatch_timing_dropped;
        publish_fence();
        slot.state = static_cast<uint32_t>(observed_count == expected && end > start ? State::Complete : State::Invalid);
        changed = true;
    }
    uint32_t completed = msg.dispatch_timing_completed;
    while (completed != published &&
           msg.dispatch_timing_records[completed % capacity].state != static_cast<uint32_t>(State::Pending)) ++completed;
    publish_fence();
    msg.dispatch_timing_completed = completed;
    return changed;
}

struct Decoded {
    bool valid = false;
    uint32_t runtime_id = 0;
    uint64_t occurrence_id = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    uint32_t dropped = 0;
};

inline Decoded decode(const volatile realtime_profiler_dispatch_timing_t& slot) {
    Decoded result;
    result.runtime_id = slot.runtime_id;
    result.occurrence_id = (static_cast<uint64_t>(slot.occurrence_hi) << 32) | slot.occurrence_lo;
    result.start = (static_cast<uint64_t>(slot.start_hi) << 32) | slot.start_lo;
    result.end = (static_cast<uint64_t>(slot.end_hi) << 32) | slot.end_lo;
    result.dropped = slot.dropped;
    result.valid = slot.marker == marker && slot.state == static_cast<uint32_t>(State::Complete) &&
                   result.runtime_id != 0 && result.runtime_id <= 0xffff && result.occurrence_id != 0 &&
                   result.start != 0 && result.end > result.start && slot.expected_count == slot.observed_count &&
                   slot.reserved0 == 0 && slot.reserved1 == 0 && slot.reserved2 == 0;
    return result;
}

static_assert(sizeof(realtime_profiler_dispatch_timing_t) == 64);

}  // namespace tt::dispatch_timing
