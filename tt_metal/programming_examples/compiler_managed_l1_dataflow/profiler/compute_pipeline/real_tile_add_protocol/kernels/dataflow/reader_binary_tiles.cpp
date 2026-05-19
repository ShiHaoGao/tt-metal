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

#ifndef BENCH_TRACE_STATIC_PROTOCOL
#define BENCH_TRACE_STATIC_PROTOCOL 0
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

#ifndef BENCH_LEVEL_C_GENERATED_STATIC
#define BENCH_LEVEL_C_GENERATED_STATIC 0
#endif

#ifndef BENCH_LEVEL_C_LLK_DIRECT
#define BENCH_LEVEL_C_LLK_DIRECT 0
#endif

#ifndef BENCH_LEVEL_C_FW_SKIP_CB_INIT
#define BENCH_LEVEL_C_FW_SKIP_CB_INIT 0
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

#ifndef BENCH_SRC0_DRAM_ADDR
#define BENCH_SRC0_DRAM_ADDR 0
#endif

#ifndef BENCH_SRC1_DRAM_ADDR
#define BENCH_SRC1_DRAM_ADDR 0
#endif

#ifndef BENCH_SRC0_RING_ADDR
#define BENCH_SRC0_RING_ADDR 0
#endif

#ifndef BENCH_SRC1_RING_ADDR
#define BENCH_SRC1_RING_ADDR 0
#endif

#ifndef BENCH_INPUT_READY_SEM_ADDR
#define BENCH_INPUT_READY_SEM_ADDR 0
#endif

#ifndef BENCH_INPUT_CONSUMED_SEM_ADDR
#define BENCH_INPUT_CONSUMED_SEM_ADDR 0
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

#ifndef BENCH_STREAM_REG_INPUT_READY0_STREAM_ID
#define BENCH_STREAM_REG_INPUT_READY0_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_INPUT_READY1_STREAM_ID
#define BENCH_STREAM_REG_INPUT_READY1_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID
#define BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID
#define BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID
#define BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID
#define BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID 3
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

#ifndef BENCH_STREAM_REG_START_REG_INDEX
#ifdef STREAM_SCRATCH32_REG_INDEX
#define BENCH_STREAM_REG_START_REG_INDEX STREAM_SCRATCH32_REG_INDEX
#else
#define BENCH_STREAM_REG_START_REG_INDEX STREAM_SCRATCH_5_REG_INDEX
#endif
#endif

#ifndef BENCH_STREAM_REG_INPUT_READY0_REG_INDEX
#define BENCH_STREAM_REG_INPUT_READY0_REG_INDEX STREAM_SCRATCH_0_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_INPUT_READY1_REG_INDEX
#define BENCH_STREAM_REG_INPUT_READY1_REG_INDEX STREAM_SCRATCH_1_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED0_REG_INDEX
#define BENCH_STREAM_REG_INPUT_CONSUMED0_REG_INDEX STREAM_SCRATCH_2_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED1_REG_INDEX
#define BENCH_STREAM_REG_INPUT_CONSUMED1_REG_INDEX STREAM_SCRATCH_3_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_READY_REG_INDEX
#define BENCH_STREAM_REG_OUTPUT_READY_REG_INDEX STREAM_SCRATCH_4_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_CONSUMED_REG_INDEX
#define BENCH_STREAM_REG_OUTPUT_CONSUMED_REG_INDEX STREAM_SCRATCH_5_REG_INDEX
#endif

namespace {

constexpr uint32_t kCbIn0 = tt::CBIndex::c_0;
constexpr uint32_t kCbIn1 = tt::CBIndex::c_1;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;
constexpr uint32_t kOneTile = 1;

inline void wait_min_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] >= value) {
            return;
        }
    }
}

inline void wait_min_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] < value) {
    }
}

inline uint32_t read_stream_sync(uint32_t stream_id, uint32_t reg_index) {
    return NOC_STREAM_READ_REG(stream_id, reg_index) & BENCH_STREAM_REG_VALUE_MASK;
}

inline void wait_min_stream(uint32_t stream_id, uint32_t reg_index, uint32_t value) {
    while (read_stream_sync(stream_id, reg_index) < value) {
    }
}

inline void set_stream_sync(uint32_t stream_id, uint32_t reg_index, uint32_t value) {
    asm volatile("fence" ::: "memory");
    NOC_STREAM_WRITE_REG(stream_id, reg_index, value & BENCH_STREAM_REG_VALUE_MASK);
    asm volatile("fence" ::: "memory");
}

inline void set_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    asm volatile("fence" ::: "memory");
    noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(reinterpret_cast<uint32_t>(sem)), value);
    noc_async_write_barrier();
}

#if BENCH_LEVEL_C_FW_SKIP_CB_INIT
inline volatile tt_reg_ptr uint32_t* protocol_counter_ptr(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        STREAM_REG_ADDR(
            OPERAND_START_STREAM + cbid,
            received ? STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX : STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}
#else
inline volatile tt_reg_ptr uint32_t* protocol_counter_ptr(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}
#endif

inline volatile tt_l1_ptr uint32_t* sem_slot(uint32_t sem_base_addr, uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_base_addr + slot * BENCH_SEM_SLOT_BYTES);
}

}  // namespace

void kernel_main() {
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t iterations = BENCH_ITERATIONS;
    constexpr uint32_t start_tile_id = BENCH_START_TILE;
    constexpr uint32_t src0_dram_addr = BENCH_SRC0_DRAM_ADDR;
    constexpr uint32_t src1_dram_addr = BENCH_SRC1_DRAM_ADDR;
#else
    const uint32_t iterations = get_arg_val<uint32_t>(0);
    const uint32_t src0_dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t src1_dram_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(3);
#endif

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t src0_ring_addr = BENCH_SRC0_RING_ADDR;
    constexpr uint32_t src1_ring_addr = BENCH_SRC1_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t input_ready_sem_addr = BENCH_INPUT_READY_SEM_ADDR;
    constexpr uint32_t input_consumed_sem_addr = BENCH_INPUT_CONSUMED_SEM_ADDR;
    constexpr uint32_t output_ready_sem_addr = BENCH_OUTPUT_READY_SEM_ADDR;
    constexpr uint32_t output_consumed_sem_addr = BENCH_OUTPUT_CONSUMED_SEM_ADDR;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t src0_ring_addr = get_arg_val<uint32_t>(4);
    const uint32_t src1_ring_addr = get_arg_val<uint32_t>(5);
    const uint32_t page_size = get_arg_val<uint32_t>(6);
    const uint32_t num_pages = get_arg_val<uint32_t>(7);
    const uint32_t input_ready_sem_addr = get_arg_val<uint32_t>(8);
    const uint32_t input_consumed_sem_addr = get_arg_val<uint32_t>(9);
    const uint32_t output_ready_sem_addr = get_arg_val<uint32_t>(10);
    const uint32_t output_consumed_sem_addr = get_arg_val<uint32_t>(11);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(12);
#endif

#if BENCH_USE_STREAM_REG_SYNC
    set_stream_sync(BENCH_STREAM_REG_INPUT_READY0_STREAM_ID, BENCH_STREAM_REG_INPUT_READY0_REG_INDEX, 0);
    set_stream_sync(BENCH_STREAM_REG_INPUT_READY1_STREAM_ID, BENCH_STREAM_REG_INPUT_READY1_REG_INDEX, 0);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID, BENCH_STREAM_REG_INPUT_CONSUMED0_REG_INDEX, 0);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID, BENCH_STREAM_REG_INPUT_CONSUMED1_REG_INDEX, 0);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID, BENCH_STREAM_REG_OUTPUT_READY_REG_INDEX, 0);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, BENCH_STREAM_REG_OUTPUT_CONSUMED_REG_INDEX, 0);
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_REG_INDEX, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_reg_ptr uint32_t* input_ready_reg = protocol_counter_ptr(kCbIn0, true);
    volatile tt_reg_ptr uint32_t* input1_ready_reg = protocol_counter_ptr(kCbIn1, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = protocol_counter_ptr(kCbIn0, false);
    volatile tt_reg_ptr uint32_t* input1_consumed_reg = protocol_counter_ptr(kCbIn1, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = protocol_counter_ptr(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = protocol_counter_ptr(kCbOut, false);
#endif

    InterleavedAddrGen<true> src0_addrgen = {.bank_base_address = src0_dram_addr, .page_size = page_size};
    InterleavedAddrGen<true> src1_addrgen = {.bank_base_address = src1_dram_addr, .page_size = page_size};
#if BENCH_USE_STREAM_REG_CBREGS
    input_ready_reg[0] = 0;
    input1_ready_reg[0] = 0;
    input_consumed_reg[0] = 0;
    input1_consumed_reg[0] = 0;
    output_ready_reg[0] = 0;
    output_consumed_reg[0] = 0;
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_REG_INDEX, BENCH_PROTOCOL_START_VALUE);
#elif !BENCH_USE_STREAM_REG_SYNC
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
    DEVICE_PRINT("rtadd reader start value={} pages={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
#endif

#if BENCH_LEVEL_C_FW_SKIP_CB_INIT
    DeviceZoneScopedN("RTADD_LEVEL_C_LLK_DIRECT_FW_SKIP_CB_INIT_READER");
#elif BENCH_LEVEL_C_LLK_DIRECT
    DeviceZoneScopedN("RTADD_LEVEL_C_LLK_DIRECT_READER");
#elif BENCH_LEVEL_C_GENERATED_STATIC
    DeviceZoneScopedN("RTADD_LEVEL_C_GENERATED_STATIC_READER");
#elif BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_ARGS
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_CBREGS_COMPILETIME_READER");
#elif BENCH_USE_COMPILE_TIME_ARGS
    DeviceZoneScopedN("RTADD_STATIC_COMPILETIME_READER");
#elif BENCH_USE_STREAM_REG_CBREGS
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_CBREGS_READER");
#elif BENCH_USE_STREAM_REG_SYNC
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_READER");
#else
    DeviceZoneScopedN("RTADD_STATIC_RUNTIME_READER");
#endif
    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t slot = i % num_pages;
        const uint32_t generation = i + 1;
#if BENCH_SERIAL_STATIC_PROTOCOL
        if (i > 0) {
            const uint32_t prev = i - 1;
#if BENCH_USE_STREAM_REG_SYNC
            wait_min_stream(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, BENCH_STREAM_REG_OUTPUT_CONSUMED_REG_INDEX, prev + 1);
#else
            wait_min_reg(output_consumed_reg, prev + 1);
#endif
        }
#endif
        if (i >= num_pages) {
#if BENCH_USE_STREAM_REG_SYNC
            wait_min_stream(BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID, BENCH_STREAM_REG_INPUT_CONSUMED0_REG_INDEX, generation - num_pages);
            wait_min_stream(BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID, BENCH_STREAM_REG_INPUT_CONSUMED1_REG_INDEX, generation - num_pages);
#else
            wait_min_reg(input_consumed_reg, generation - num_pages);
            wait_min_reg(input1_consumed_reg, generation - num_pages);
#endif
        }

        const uint32_t dst0_l1_addr = src0_ring_addr + slot * page_size;
        const uint32_t dst1_l1_addr = src1_ring_addr + slot * page_size;
        const uint32_t global_tile_id = start_tile_id + i;
        noc_async_read(get_noc_addr(global_tile_id, src0_addrgen), dst0_l1_addr, page_size);
        noc_async_read(get_noc_addr(global_tile_id, src1_addrgen), dst1_l1_addr, page_size);
        noc_async_read_barrier();
#if BENCH_USE_STREAM_REG_SYNC
        set_stream_sync(BENCH_STREAM_REG_INPUT_READY0_STREAM_ID, BENCH_STREAM_REG_INPUT_READY0_REG_INDEX, generation);
        set_stream_sync(BENCH_STREAM_REG_INPUT_READY1_STREAM_ID, BENCH_STREAM_REG_INPUT_READY1_REG_INDEX, generation);
#else
        input_ready_reg[0] = generation;
        input1_ready_reg[0] = generation;
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT("rtadd reader ready i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
    }
#else
    const uint32_t page_size = get_tile_size(kCbIn0);
    InterleavedAddrGen<true> src0_addrgen = {.bank_base_address = src0_dram_addr, .page_size = page_size};
    InterleavedAddrGen<true> src1_addrgen = {.bank_base_address = src1_dram_addr, .page_size = page_size};

    DeviceZoneScopedN("RTADD_CB_READER");
    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t global_tile_id = start_tile_id + i;
        cb_reserve_back(kCbIn0, kOneTile);
        const uint32_t src0_l1_addr = get_write_ptr(kCbIn0);
        noc_async_read(get_noc_addr(global_tile_id, src0_addrgen), src0_l1_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(kCbIn0, kOneTile);

        cb_reserve_back(kCbIn1, kOneTile);
        const uint32_t src1_l1_addr = get_write_ptr(kCbIn1);
        noc_async_read(get_noc_addr(global_tile_id, src1_addrgen), src1_l1_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(kCbIn1, kOneTile);
    }
#endif
}
