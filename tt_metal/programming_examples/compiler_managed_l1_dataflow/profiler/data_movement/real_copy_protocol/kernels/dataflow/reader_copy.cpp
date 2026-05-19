// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef BENCH_STATIC_PROTOCOL
#define BENCH_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_USE_COMPILE_TIME_ARGS
#define BENCH_USE_COMPILE_TIME_ARGS 0
#endif

#ifndef BENCH_USE_STREAM_REG_SYNC
#define BENCH_USE_STREAM_REG_SYNC 0
#endif

#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS 1
#endif

#ifndef BENCH_START_TILE
#define BENCH_START_TILE 0
#endif

#ifndef BENCH_PAGE_SIZE
#define BENCH_PAGE_SIZE 0
#endif

#ifndef BENCH_NUM_PAGES
#define BENCH_NUM_PAGES 1
#endif

#ifndef BENCH_SEM_SLOT_BYTES
#define BENCH_SEM_SLOT_BYTES 64
#endif

#ifndef BENCH_SRC_DRAM_ADDR
#define BENCH_SRC_DRAM_ADDR 0
#endif

#ifndef BENCH_RING_ADDR
#define BENCH_RING_ADDR 0
#endif

#ifndef BENCH_READY_SEM_ADDR
#define BENCH_READY_SEM_ADDR 0
#endif

#ifndef BENCH_CONSUMED_SEM_ADDR
#define BENCH_CONSUMED_SEM_ADDR 0
#endif

#ifndef BENCH_START_SEM_ADDR
#define BENCH_START_SEM_ADDR 0
#endif

#ifndef BENCH_START_VALUE
#define BENCH_START_VALUE 1
#endif

#ifndef BENCH_STREAM_REG_START_STREAM_ID
#define BENCH_STREAM_REG_START_STREAM_ID 0
#endif

#ifndef BENCH_STREAM_REG_READY_STREAM_ID
#define BENCH_STREAM_REG_READY_STREAM_ID 1
#endif

#ifndef BENCH_STREAM_REG_CONSUMED_STREAM_ID
#define BENCH_STREAM_REG_CONSUMED_STREAM_ID 2
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

#ifndef BENCH_MODE_NAME
#if BENCH_USE_COMPILE_TIME_ARGS && BENCH_USE_STREAM_REG_SYNC
#define BENCH_MODE_NAME "STATIC_STREAMREG_SCRATCH_COMPILETIME"
#elif BENCH_USE_COMPILE_TIME_ARGS
#define BENCH_MODE_NAME "STATIC_COMPILETIME"
#elif BENCH_USE_STREAM_REG_SYNC
#define BENCH_MODE_NAME "STATIC_STREAMREG_SCRATCH"
#else
#define BENCH_MODE_NAME "STATIC_RUNTIME"
#endif
#endif

namespace {

constexpr uint32_t kCbIn = tt::CBIndex::c_0;
constexpr uint32_t kOneTile = 1;

inline void wait_min_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] >= value) {
            return;
        }
    }
}

inline void set_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    asm volatile("fence" ::: "memory");
    noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(reinterpret_cast<uint32_t>(sem)), value);
    noc_async_write_barrier();
}

inline uint32_t read_stream_sync(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, BENCH_STREAM_SYNC_REG_INDEX) & BENCH_STREAM_REG_VALUE_MASK;
}

inline void wait_min_stream(uint32_t stream_id, uint32_t value) {
    while (read_stream_sync(stream_id) < value) {
    }
}

inline void wait_equal_stream(uint32_t stream_id, uint32_t value) {
    value &= BENCH_STREAM_REG_VALUE_MASK;
    while (read_stream_sync(stream_id) != value) {
    }
}

inline void set_stream_sync(uint32_t stream_id, uint32_t value) {
    asm volatile("fence" ::: "memory");
    NOC_STREAM_WRITE_REG(stream_id, BENCH_STREAM_SYNC_REG_INDEX, value & BENCH_STREAM_REG_VALUE_MASK);
    asm volatile("fence" ::: "memory");
}

}  // namespace

void kernel_main() {
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t iterations = BENCH_ITERATIONS;
    constexpr uint32_t src_dram_addr = BENCH_SRC_DRAM_ADDR;
    constexpr uint32_t start_tile_id = BENCH_START_TILE;
#else
    const uint32_t iterations = get_arg_val<uint32_t>(0);
    const uint32_t src_dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
#endif

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t ring_addr = BENCH_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t ready_sem_addr = BENCH_READY_SEM_ADDR;
    constexpr uint32_t consumed_sem_addr = BENCH_CONSUMED_SEM_ADDR;
    constexpr uint32_t start_sem_addr = BENCH_START_SEM_ADDR;
#else
    const uint32_t ring_addr = get_arg_val<uint32_t>(3);
    const uint32_t page_size = get_arg_val<uint32_t>(4);
    const uint32_t num_pages = get_arg_val<uint32_t>(5);
    const uint32_t ready_sem_addr = get_arg_val<uint32_t>(6);
    const uint32_t consumed_sem_addr = get_arg_val<uint32_t>(7);
    const uint32_t start_sem_addr = get_arg_val<uint32_t>(8);
#endif

    InterleavedAddrGen<true> src_addrgen = {.bank_base_address = src_dram_addr, .page_size = page_size};

#if BENCH_USE_STREAM_REG_SYNC
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);
    volatile tt_l1_ptr uint32_t* consumed_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem_addr);
    volatile tt_l1_ptr uint32_t* start_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_sem_addr);
    set_local(start_sem, BENCH_START_VALUE);
#endif

    DeviceZoneScopedN("RTCOPY_" BENCH_MODE_NAME "_READER");
    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t generation = i + 1;
        if (i >= num_pages) {
#if BENCH_USE_STREAM_REG_SYNC
            wait_min_stream(BENCH_STREAM_REG_CONSUMED_STREAM_ID, generation - num_pages);
#else
            wait_min_local(consumed_sem, generation - num_pages);
#endif
        }

        const uint32_t slot = i % num_pages;
        const uint32_t dst_l1_addr = ring_addr + slot * page_size;
        const uint32_t global_tile_id = start_tile_id + i;
        noc_async_read(get_noc_addr(global_tile_id, src_addrgen), dst_l1_addr, page_size);
        noc_async_read_barrier();
#if BENCH_USE_STREAM_REG_SYNC
        set_stream_sync(BENCH_STREAM_REG_READY_STREAM_ID, generation);
#else
        set_local(ready_sem, generation);
#endif
    }
#else
    const uint32_t page_size = get_tile_size(kCbIn);
    InterleavedAddrGen<true> src_addrgen = {.bank_base_address = src_dram_addr, .page_size = page_size};

    DeviceZoneScopedN("RTCOPY_CB_READER");
    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t global_tile_id = start_tile_id + i;
        cb_reserve_back(kCbIn, kOneTile);
        const uint32_t dst_l1_addr = get_write_ptr(kCbIn);
        noc_async_read(get_noc_addr(global_tile_id, src_addrgen), dst_l1_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(kCbIn, kOneTile);
    }
#endif
}
