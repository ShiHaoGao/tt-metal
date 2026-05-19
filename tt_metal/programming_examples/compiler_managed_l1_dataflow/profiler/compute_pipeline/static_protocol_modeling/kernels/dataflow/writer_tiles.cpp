// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef BENCH_STATIC_PROTOCOL
#define BENCH_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_SERIAL_STATIC_PROTOCOL
#define BENCH_SERIAL_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_USE_COMPILE_TIME_ARGS
#define BENCH_USE_COMPILE_TIME_ARGS 0
#endif

#ifndef BENCH_USE_STREAM_REG_SYNC
#define BENCH_USE_STREAM_REG_SYNC 0
#endif

#ifndef BENCH_USE_STREAM_REG_CBREGS
#define BENCH_USE_STREAM_REG_CBREGS 0
#endif

#ifndef BENCH_TRACE_STATIC_PROTOCOL
#define BENCH_TRACE_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_PROFILE_CASE_LABELS
#define BENCH_PROFILE_CASE_LABELS 0
#endif

#ifndef BENCH_OP_ELTWISE_CHAIN
#define BENCH_OP_ELTWISE_CHAIN 0
#endif

#ifndef BENCH_OP_MATMUL
#define BENCH_OP_MATMUL 0
#endif

#ifndef BENCH_OP_MATMUL_BLOCK
#define BENCH_OP_MATMUL_BLOCK 0
#endif

#ifndef BENCH_CHAIN_DEPTH
#define BENCH_CHAIN_DEPTH 1
#endif

#ifndef BENCH_MATMUL_GLOBAL_MT
#define BENCH_MATMUL_GLOBAL_MT 1
#endif

#ifndef BENCH_MATMUL_NT
#define BENCH_MATMUL_NT BENCH_ITERATIONS
#endif

#ifndef BENCH_MATMUL_GLOBAL_NT
#define BENCH_MATMUL_GLOBAL_NT BENCH_MATMUL_NT
#endif

#ifndef BENCH_MATMUL_BASE_MT
#define BENCH_MATMUL_BASE_MT 0
#endif

#ifndef BENCH_MATMUL_BASE_NT
#define BENCH_MATMUL_BASE_NT 0
#endif

#ifndef BENCH_MATMUL_KT
#define BENCH_MATMUL_KT 1
#endif

#ifndef BENCH_CORE_GRID_X
#define BENCH_CORE_GRID_X 1
#endif

#ifndef BENCH_CORE_GRID_Y
#define BENCH_CORE_GRID_Y 1
#endif

#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS 1
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

#ifndef BENCH_DST_DRAM_ADDR
#define BENCH_DST_DRAM_ADDR 0
#endif

#ifndef BENCH_DST_RING_ADDR
#define BENCH_DST_RING_ADDR 0
#endif

#ifndef BENCH_OUTPUT_READY_SEM_ADDR
#define BENCH_OUTPUT_READY_SEM_ADDR 0
#endif

#ifndef BENCH_OUTPUT_CONSUMED_SEM_ADDR
#define BENCH_OUTPUT_CONSUMED_SEM_ADDR 0
#endif

#ifndef BENCH_PROTOCOL_START_SEM_ADDR
#define BENCH_PROTOCOL_START_SEM_ADDR 0
#endif

#ifndef BENCH_PROTOCOL_START_VALUE
#define BENCH_PROTOCOL_START_VALUE 1
#endif

#ifndef BENCH_STREAM_REG_START_STREAM_ID
#define BENCH_STREAM_REG_START_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID
#define BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID 8
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID
#define BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID 9
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

constexpr uint32_t kCbOut = tt::CBIndex::c_16;
constexpr uint32_t kOneTile = 1;

#define SPM_STRINGIZE_IMPL(x) #x
#define SPM_STRINGIZE(x) SPM_STRINGIZE_IMPL(x)

#if BENCH_OP_MATMUL
#if BENCH_OP_MATMUL_BLOCK
#define SPM_OP_PREFIX "SPM_MATMUL_BLOCK"
#else
#define SPM_OP_PREFIX "SPM_MATMUL_SINGLE"
#endif
#if BENCH_PROFILE_CASE_LABELS
#define SPM_CASE_PREFIX \
    SPM_OP_PREFIX "_M" SPM_STRINGIZE(BENCH_MATMUL_GLOBAL_MT) "_N" SPM_STRINGIZE(BENCH_MATMUL_GLOBAL_NT) \
        "_K" SPM_STRINGIZE(BENCH_MATMUL_KT) "_S" SPM_STRINGIZE(BENCH_NUM_PAGES) "_G" \
            SPM_STRINGIZE(BENCH_CORE_GRID_X) "X" SPM_STRINGIZE(BENCH_CORE_GRID_Y)
#else
#define SPM_CASE_PREFIX SPM_OP_PREFIX
#endif
#elif BENCH_OP_ELTWISE_CHAIN
#if BENCH_PROFILE_CASE_LABELS
#define SPM_CASE_PREFIX \
    "SPM_ELTWISE_CHAIN_T" SPM_STRINGIZE(BENCH_ITERATIONS) "_C" SPM_STRINGIZE(BENCH_CHAIN_DEPTH) "_S" \
        SPM_STRINGIZE(BENCH_NUM_PAGES)
#else
#define SPM_CASE_PREFIX "SPM_ELTWISE_CHAIN"
#endif
#else
#if BENCH_PROFILE_CASE_LABELS
#define SPM_CASE_PREFIX \
    "SPM_TILE_ADD_T" SPM_STRINGIZE(BENCH_ITERATIONS) "_C" SPM_STRINGIZE(BENCH_CHAIN_DEPTH) "_S" \
        SPM_STRINGIZE(BENCH_NUM_PAGES)
#else
#define SPM_CASE_PREFIX "SPM_TILE_ADD"
#endif
#endif

#define SPM_ZONE_STATIC_COMPILETIME_WRITER SPM_CASE_PREFIX "_STATIC_COMPILETIME_WRITER"
#define SPM_ZONE_STATIC_RUNTIME_WRITER SPM_CASE_PREFIX "_STATIC_RUNTIME_WRITER"
#define SPM_ZONE_STATIC_SERIALIZED_WRITER SPM_CASE_PREFIX "_STATIC_SERIALIZED_WRITER"
#define SPM_ZONE_STATIC_STREAMREG_WRITER SPM_CASE_PREFIX "_STATIC_STREAMREG_WRITER"
#define SPM_ZONE_STATIC_STREAMREG_CBREGS_WRITER SPM_CASE_PREFIX "_STATIC_STREAMREG_CBREGS_WRITER"
#define SPM_ZONE_CB_WRITER SPM_CASE_PREFIX "_CB_WRITER"

inline void wait_min_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] >= value) {
            return;
        }
    }
}

inline void wait_equal_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] != value) {
    }
}

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

inline void wait_equal_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] == value) {
            return;
        }
    }
}

inline void set_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    asm volatile("fence" ::: "memory");
    noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(reinterpret_cast<uint32_t>(sem)), value);
    noc_async_write_barrier();
}

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}

inline volatile tt_l1_ptr uint32_t* sem_slot(uint32_t sem_base_addr, uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_base_addr + slot * BENCH_SEM_SLOT_BYTES);
}

inline uint32_t output_tile_index(uint32_t local_index, uint32_t local_nt, uint32_t global_nt, uint32_t base_mt, uint32_t base_nt) {
#if BENCH_OP_MATMUL
    const uint32_t local_m = local_index / local_nt;
    const uint32_t local_n = local_index % local_nt;
    return (base_mt + local_m) * global_nt + base_nt + local_n;
#else
    return local_index;
#endif
}

}  // namespace

void kernel_main() {
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t iterations = BENCH_ITERATIONS;
    constexpr uint32_t dst_dram_addr = BENCH_DST_DRAM_ADDR;
    constexpr uint32_t local_nt = BENCH_MATMUL_NT;
    constexpr uint32_t global_nt = BENCH_MATMUL_GLOBAL_NT;
    constexpr uint32_t base_mt = BENCH_MATMUL_BASE_MT;
    constexpr uint32_t base_nt = BENCH_MATMUL_BASE_NT;
#else
    const uint32_t iterations = get_arg_val<uint32_t>(0);
    const uint32_t dst_dram_addr = get_arg_val<uint32_t>(1);
#endif

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t dst_ring_addr = BENCH_DST_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t output_ready_sem_addr = BENCH_OUTPUT_READY_SEM_ADDR;
    constexpr uint32_t output_consumed_sem_addr = BENCH_OUTPUT_CONSUMED_SEM_ADDR;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t dst_ring_addr = get_arg_val<uint32_t>(2);
    const uint32_t page_size = get_arg_val<uint32_t>(3);
    const uint32_t num_pages = get_arg_val<uint32_t>(4);
    const uint32_t output_ready_sem_addr = get_arg_val<uint32_t>(5);
    const uint32_t output_consumed_sem_addr = get_arg_val<uint32_t>(6);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(7);
#if BENCH_OP_MATMUL
    const uint32_t local_nt = get_arg_val<uint32_t>(8);
    const uint32_t global_nt = get_arg_val<uint32_t>(9);
    const uint32_t base_mt = get_arg_val<uint32_t>(10);
    const uint32_t base_nt = get_arg_val<uint32_t>(11);
#else
    const uint32_t local_nt = iterations;
    const uint32_t global_nt = iterations;
    const uint32_t base_mt = 0;
    const uint32_t base_nt = 0;
#endif
#endif

#if BENCH_USE_STREAM_REG_SYNC
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbOut, false);
#endif

    InterleavedAddrGen<true> dst_addrgen = {.bank_base_address = dst_dram_addr, .page_size = page_size};

#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#elif !BENCH_USE_STREAM_REG_SYNC
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
    DEVICE_PRINT("spm writer start value={} slots={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
#endif

#if BENCH_USE_COMPILE_TIME_ARGS
    DeviceZoneScopedN(SPM_ZONE_STATIC_COMPILETIME_WRITER);
#elif BENCH_SERIAL_STATIC_PROTOCOL
    DeviceZoneScopedN(SPM_ZONE_STATIC_SERIALIZED_WRITER);
#elif BENCH_USE_STREAM_REG_CBREGS
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_CBREGS_WRITER);
#elif BENCH_USE_STREAM_REG_SYNC
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_WRITER);
#else
    DeviceZoneScopedN(SPM_ZONE_STATIC_RUNTIME_WRITER);
#endif
    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t slot = i % num_pages;
        const uint32_t generation = i + 1;
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT("spm writer wait i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
#if BENCH_USE_STREAM_REG_SYNC
        wait_min_stream(BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID, generation);
#else
        wait_equal_reg(output_ready_reg, generation);
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT("spm writer got i={} slot={} gen={}\n", i, slot, generation);
        }
#endif

        const uint32_t src_l1_addr = dst_ring_addr + slot * page_size;
        noc_async_write(src_l1_addr, get_noc_addr(output_tile_index(i, local_nt, global_nt, base_mt, base_nt), dst_addrgen), page_size);
        noc_async_write_barrier();
#if BENCH_USE_STREAM_REG_SYNC
        set_stream_sync(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, generation);
#else
        output_consumed_reg[0] = generation;
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT("spm writer consumed i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
    }
#else
    const uint32_t page_size = get_tile_size(kCbOut);
#if BENCH_OP_MATMUL
    const uint32_t local_nt = get_arg_val<uint32_t>(2);
    const uint32_t global_nt = get_arg_val<uint32_t>(3);
    const uint32_t base_mt = get_arg_val<uint32_t>(4);
    const uint32_t base_nt = get_arg_val<uint32_t>(5);
#else
    const uint32_t local_nt = iterations;
    const uint32_t global_nt = iterations;
    const uint32_t base_mt = 0;
    const uint32_t base_nt = 0;
#endif
    InterleavedAddrGen<true> dst_addrgen = {.bank_base_address = dst_dram_addr, .page_size = page_size};

    DeviceZoneScopedN(SPM_ZONE_CB_WRITER);
    for (uint32_t i = 0; i < iterations; ++i) {
        cb_wait_front(kCbOut, kOneTile);
        const uint32_t src_l1_addr = get_read_ptr(kCbOut);
        noc_async_write(src_l1_addr, get_noc_addr(output_tile_index(i, local_nt, global_nt, base_mt, base_nt), dst_addrgen), page_size);
        noc_async_write_barrier();
        cb_pop_front(kCbOut, kOneTile);
    }
#endif
}
