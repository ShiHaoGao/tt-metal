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

#ifndef BENCH_USE_STREAM_REG_SYNC
#define BENCH_USE_STREAM_REG_SYNC 0
#endif

#ifndef BENCH_USE_STREAM_REG_CBREGS
#define BENCH_USE_STREAM_REG_CBREGS 0
#endif

#ifndef BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
#define BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS 0
#endif

#ifndef BENCH_PROTOCOL_START_VALUE
#define BENCH_PROTOCOL_START_VALUE 1
#endif

#ifndef BENCH_SRC_RING_ADDR
#define BENCH_SRC_RING_ADDR 0
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

#ifndef BENCH_STREAM_REG_INPUT_READY_STREAM_ID
#define BENCH_STREAM_REG_INPUT_READY_STREAM_ID 4
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED_STREAM_ID
#define BENCH_STREAM_REG_INPUT_CONSUMED_STREAM_ID 5
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID
#define BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID 6
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID
#define BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID 7
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

constexpr auto kCbSrc = tt::CBIndex::c_0;
constexpr uint32_t kOneTile = 1;

#if BENCH_STATIC_PROTOCOL
inline void wait_min_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] < value) {
    }
}

inline uint32_t read_stream_sync(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, BENCH_STREAM_SYNC_REG_INDEX) & BENCH_STREAM_REG_VALUE_MASK;
}

inline void wait_min_stream(uint32_t stream_id, uint32_t value) {
    while (read_stream_sync(stream_id) < value) {
    }
}

inline void set_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    asm volatile("fence" ::: "memory");
    noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(reinterpret_cast<uint32_t>(sem)), value);
    noc_async_write_barrier();
}

inline void set_stream_sync(uint32_t stream_id, uint32_t value) {
    asm volatile("fence" ::: "memory");
    NOC_STREAM_WRITE_REG(stream_id, BENCH_STREAM_SYNC_REG_INDEX, value & BENCH_STREAM_REG_VALUE_MASK);
    asm volatile("fence" ::: "memory");
}

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t start_ht = get_arg_val<uint32_t>(3);
    uint32_t start_wt = get_arg_val<uint32_t>(4);
    const uint32_t Ht = get_arg_val<uint32_t>(5);
    const uint32_t Wt = get_arg_val<uint32_t>(6);
    const uint32_t HtWt = get_arg_val<uint32_t>(7);

    constexpr auto src_args = TensorAccessorArgs<0>();
    const uint32_t src_tile_bytes = get_tile_size(kCbSrc);
    const auto src = TensorAccessor(src_args, src_addr);

    Noc noc;
    CircularBuffer cb_src(kCbSrc);

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t src_ring_addr = BENCH_SRC_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t src_ring_addr = get_arg_val<uint32_t>(8);
    const uint32_t page_size = get_arg_val<uint32_t>(9);
    const uint32_t num_pages = get_arg_val<uint32_t>(10);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(11);
#endif

#if BENCH_USE_STREAM_REG_SYNC
    set_stream_sync(BENCH_STREAM_REG_INPUT_READY_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_reg_ptr uint32_t* input_ready_reg = reg_ptr_from_cb(kCbSrc, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = reg_ptr_from_cb(kCbSrc, false);
    input_ready_reg[0] = 0;
    input_consumed_reg[0] = 0;
#if BENCH_USE_STREAM_REG_CBREGS
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#endif

#if BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    DeviceZoneScopedN("TTWH_STATIC_STREAMREG_CBREGS_COMPILETIME_READER");
#elif BENCH_USE_STREAM_REG_SYNC
    DeviceZoneScopedN("TTWH_STATIC_STREAMREG_READER");
#elif BENCH_USE_STREAM_REG_CBREGS
    DeviceZoneScopedN("TTWH_STATIC_STREAMREG_CBREGS_READER");
#else
    DeviceZoneScopedN("TTWH_STATIC_RUNTIME_READER");
#endif

    uint32_t ht = start_ht;
    uint32_t wt = start_wt;
    uint32_t i_tile = start_id;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        const uint32_t generation = i + 1;
        if (generation > num_pages) {
#if BENCH_USE_STREAM_REG_SYNC
            wait_min_stream(BENCH_STREAM_REG_INPUT_CONSUMED_STREAM_ID, generation - num_pages);
#else
            wait_min_reg(input_consumed_reg, generation - num_pages);
#endif
        }

        const uint32_t dst_l1_addr = src_ring_addr + (i % num_pages) * page_size;
        noc_async_read_tile(i_tile, src, dst_l1_addr);
        noc_async_read_barrier();
#if BENCH_USE_STREAM_REG_SYNC
        set_stream_sync(BENCH_STREAM_REG_INPUT_READY_STREAM_ID, generation);
#else
        input_ready_reg[0] = generation;
#endif
        i_tile += Wt;
        ++ht;
        if (ht == Ht) {
            ht = 0;
            i_tile += 1;
            ++wt;
            if (wt == Wt) {
                wt = 0;
                i_tile -= Wt;
            } else {
                i_tile -= HtWt;
            }
        }
    }
#else
    DeviceZoneScopedN("TTWH_CB_READER");
    uint32_t ht = start_ht;
    uint32_t wt = start_wt;
    uint32_t i_tile = start_id;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_src.reserve_back(kOneTile);
        noc.async_read(src, cb_src, src_tile_bytes, {.page_id = i_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_src.push_back(kOneTile);
        i_tile += Wt;
        ++ht;
        if (ht == Ht) {
            ht = 0;
            i_tile += 1;
            ++wt;
            if (wt == Wt) {
                wt = 0;
                i_tile -= Wt;
            } else {
                i_tile -= HtWt;
            }
        }
    }
#endif
}
