// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef BENCH_STATIC_PROTOCOL
#define BENCH_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_USE_STREAM_REG_CBREGS
#define BENCH_USE_STREAM_REG_CBREGS 0
#endif

#ifndef BENCH_PROTOCOL_START_VALUE
#define BENCH_PROTOCOL_START_VALUE 1
#endif

#ifndef BENCH_PROTOCOL_START_SEM_ADDR
#define BENCH_PROTOCOL_START_SEM_ADDR 0
#endif

#ifndef BENCH_STREAM_REG_START_STREAM_ID
#define BENCH_STREAM_REG_START_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_VALUE_MASK
#define BENCH_STREAM_REG_VALUE_MASK 0x00ffffffu
#endif

#ifndef BENCH_STREAM_SYNC_REG_INDEX
#ifdef STREAM_SCRATCH32_REG_INDEX
#define BENCH_STREAM_SYNC_REG_INDEX STREAM_SCRATCH32_REG_INDEX
#else
#define BENCH_STREAM_SYNC_REG_INDEX STREAM_SCRATCH_1_REG_INDEX
#endif
#endif

namespace {

constexpr auto kCbOutput = tt::CBIndex::c_0;

#if BENCH_STATIC_PROTOCOL
inline void wait_equal_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] == value) {
            return;
        }
    }
}

inline uint32_t read_stream_sync(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, BENCH_STREAM_SYNC_REG_INDEX) & BENCH_STREAM_REG_VALUE_MASK;
}

inline void wait_equal_stream(uint32_t stream_id, uint32_t value) {
    value &= BENCH_STREAM_REG_VALUE_MASK;
    while (read_stream_sync(stream_id) != value) {
    }
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

#if BENCH_STATIC_PROTOCOL
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(1);
#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TKVL_STATIC_STREAMREG_CBREGS_WRITER");
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TKVL_STATIC_RUNTIME_WRITER");
#endif
#else
    CircularBuffer cb_output(kCbOutput);
    DeviceZoneScopedN("TKVL_CB_WRITER");
    cb_output.wait_front(num_tiles);
#endif
}
