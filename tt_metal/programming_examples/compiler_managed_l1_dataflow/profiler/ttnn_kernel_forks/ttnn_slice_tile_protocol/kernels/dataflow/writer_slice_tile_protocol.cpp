// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
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

#ifndef BENCH_RING_ADDR
#define BENCH_RING_ADDR 0
#endif

#ifndef BENCH_PAGE_SIZE
#define BENCH_PAGE_SIZE 0
#endif

#ifndef BENCH_NUM_PAGES
#define BENCH_NUM_PAGES 1
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

constexpr auto kCbSlice = tt::CBIndex::c_0;
constexpr uint32_t kOneTile = 1;

#if BENCH_STATIC_PROTOCOL
inline void wait_equal_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] == value) {
            return;
        }
    }
}

inline void wait_min_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] < value) {
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

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const uint32_t tile_bytes = get_tile_size(kCbSlice);
    const auto dst = TensorAccessor(dst_args, dst_addr);

    CircularBuffer cb_slice(kCbSlice);
    Noc noc;

#if BENCH_STATIC_PROTOCOL
    const uint32_t ring_addr = get_arg_val<uint32_t>(3);
    const uint32_t page_size = get_arg_val<uint32_t>(4);
    const uint32_t num_pages = get_arg_val<uint32_t>(5);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(6);

    volatile tt_reg_ptr uint32_t* ready_reg = reg_ptr_from_cb(kCbSlice, true);
    volatile tt_reg_ptr uint32_t* consumed_reg = reg_ptr_from_cb(kCbSlice, false);

#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TTSL_STATIC_STREAMREG_CBREGS_WRITER");
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TTSL_STATIC_RUNTIME_WRITER");
#endif

    for (uint32_t i = 0; i < num_tiles; ++i) {
        const uint32_t generation = i + 1;
        wait_min_reg(ready_reg, generation);
        const uint32_t src_l1_addr = ring_addr + (i % num_pages) * page_size;
        noc_async_write_tile(start_id + i, dst, src_l1_addr);
        noc_async_write_barrier();
        consumed_reg[0] = generation;
    }
#else
    DeviceZoneScopedN("TTSL_CB_WRITER");
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_slice.wait_front(kOneTile);
        noc.async_write(cb_slice, dst, tile_bytes, {}, {.page_id = start_id + i});
        noc.async_writes_flushed();
        cb_slice.pop_front(kOneTile);
    }
    noc.async_write_barrier();
#endif
}
