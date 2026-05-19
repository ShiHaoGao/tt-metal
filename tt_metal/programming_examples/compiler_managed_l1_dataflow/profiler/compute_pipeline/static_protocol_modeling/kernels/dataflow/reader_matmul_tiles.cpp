// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
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

#ifndef BENCH_PROFILE_CASE_LABELS
#define BENCH_PROFILE_CASE_LABELS 0
#endif

#ifndef BENCH_OP_MATMUL_BLOCK
#define BENCH_OP_MATMUL_BLOCK 0
#endif

#ifndef BENCH_MATMUL_MT
#define BENCH_MATMUL_MT 1
#endif

#ifndef BENCH_MATMUL_GLOBAL_MT
#define BENCH_MATMUL_GLOBAL_MT BENCH_MATMUL_MT
#endif

#ifndef BENCH_MATMUL_KT
#define BENCH_MATMUL_KT 1
#endif

#ifndef BENCH_MATMUL_NT
#define BENCH_MATMUL_NT 1
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

#ifndef BENCH_CORE_GRID_X
#define BENCH_CORE_GRID_X 1
#endif

#ifndef BENCH_CORE_GRID_Y
#define BENCH_CORE_GRID_Y 1
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
#define BENCH_STREAM_REG_INPUT_READY0_STREAM_ID 4
#endif

#ifndef BENCH_STREAM_REG_INPUT_READY1_STREAM_ID
#define BENCH_STREAM_REG_INPUT_READY1_STREAM_ID 5
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID
#define BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID 6
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID
#define BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID 7
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

constexpr uint32_t kCbIn0 = tt::CBIndex::c_0;
constexpr uint32_t kCbIn1 = tt::CBIndex::c_1;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;
constexpr uint32_t kOneTile = 1;

#define SPM_STRINGIZE_IMPL(x) #x
#define SPM_STRINGIZE(x) SPM_STRINGIZE_IMPL(x)

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

inline void set_stream_sync(uint32_t stream_id, uint32_t value) {
    asm volatile("fence" ::: "memory");
    NOC_STREAM_WRITE_REG(stream_id, BENCH_STREAM_SYNC_REG_INDEX, value & BENCH_STREAM_REG_VALUE_MASK);
    asm volatile("fence" ::: "memory");
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

}  // namespace

void kernel_main() {
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t src0_dram_addr = BENCH_SRC0_DRAM_ADDR;
    constexpr uint32_t src1_dram_addr = BENCH_SRC1_DRAM_ADDR;
    constexpr uint32_t Mt = BENCH_MATMUL_MT;
    constexpr uint32_t Kt = BENCH_MATMUL_KT;
    constexpr uint32_t Nt = BENCH_MATMUL_NT;
    constexpr uint32_t global_Nt = BENCH_MATMUL_GLOBAL_NT;
    constexpr uint32_t base_mt = BENCH_MATMUL_BASE_MT;
    constexpr uint32_t base_nt = BENCH_MATMUL_BASE_NT;
#else
    const uint32_t src0_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t Mt = get_arg_val<uint32_t>(2);
    const uint32_t Kt = get_arg_val<uint32_t>(3);
    const uint32_t Nt = get_arg_val<uint32_t>(4);
#endif

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t src0_ring_addr = BENCH_SRC0_RING_ADDR;
    constexpr uint32_t src1_ring_addr = BENCH_SRC1_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t src0_ring_addr = get_arg_val<uint32_t>(5);
    const uint32_t src1_ring_addr = get_arg_val<uint32_t>(6);
    const uint32_t page_size = get_arg_val<uint32_t>(7);
    const uint32_t num_pages = get_arg_val<uint32_t>(8);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(13);
    const uint32_t global_Nt = get_arg_val<uint32_t>(14);
    const uint32_t base_mt = get_arg_val<uint32_t>(15);
    const uint32_t base_nt = get_arg_val<uint32_t>(16);
#endif

#if BENCH_USE_STREAM_REG_SYNC
    set_stream_sync(BENCH_STREAM_REG_INPUT_READY0_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_INPUT_READY1_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, 0);
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_reg_ptr uint32_t* input_ready_reg = reg_ptr_from_cb(kCbIn0, true);
    volatile tt_reg_ptr uint32_t* input1_ready_reg = reg_ptr_from_cb(kCbIn1, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = reg_ptr_from_cb(kCbIn0, false);
    volatile tt_reg_ptr uint32_t* input1_consumed_reg = reg_ptr_from_cb(kCbIn1, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbOut, false);
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
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#elif !BENCH_USE_STREAM_REG_SYNC
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif

#if BENCH_TRACE_STATIC_PROTOCOL
    DEVICE_PRINT("spm matmul reader start value={} slots={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
#endif

#if BENCH_USE_COMPILE_TIME_ARGS
    DeviceZoneScopedN(SPM_CASE_PREFIX "_STATIC_COMPILETIME_READER");
#elif BENCH_SERIAL_STATIC_PROTOCOL
    DeviceZoneScopedN(SPM_CASE_PREFIX "_STATIC_SERIALIZED_READER");
#elif BENCH_USE_STREAM_REG_CBREGS
    DeviceZoneScopedN(SPM_CASE_PREFIX "_STATIC_STREAMREG_CBREGS_READER");
#elif BENCH_USE_STREAM_REG_SYNC
    DeviceZoneScopedN(SPM_CASE_PREFIX "_STATIC_STREAMREG_READER");
#else
    DeviceZoneScopedN(SPM_CASE_PREFIX "_STATIC_RUNTIME_READER");
#endif

    uint32_t pair_generation = 0;
    uint32_t output_generation = 0;
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            ++output_generation;
#if BENCH_SERIAL_STATIC_PROTOCOL
            if (output_generation > 1) {
#if BENCH_USE_STREAM_REG_SYNC
                wait_min_stream(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, output_generation - 1);
#else
                wait_min_reg(output_consumed_reg, output_generation - 1);
#endif
            }
#endif
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                ++pair_generation;
                const uint32_t slot = (pair_generation - 1) % num_pages;
                if (pair_generation > num_pages) {
#if BENCH_USE_STREAM_REG_SYNC
                    wait_min_stream(BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID, pair_generation - num_pages);
                    wait_min_stream(BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID, pair_generation - num_pages);
#else
                    wait_min_reg(input_consumed_reg, pair_generation - num_pages);
                    wait_min_reg(input1_consumed_reg, pair_generation - num_pages);
#endif
                }

                const uint32_t global_mt = base_mt + mt;
                const uint32_t global_nt = base_nt + nt;
                const uint32_t a_tile_index = global_mt * Kt + kt;
                const uint32_t b_tile_index = kt * global_Nt + global_nt;
                const uint32_t dst0_l1_addr = src0_ring_addr + slot * page_size;
                const uint32_t dst1_l1_addr = src1_ring_addr + slot * page_size;
                noc_async_read(get_noc_addr(a_tile_index, src0_addrgen), dst0_l1_addr, page_size);
                noc_async_read(get_noc_addr(b_tile_index, src1_addrgen), dst1_l1_addr, page_size);
                noc_async_read_barrier();
#if BENCH_USE_STREAM_REG_SYNC
                set_stream_sync(BENCH_STREAM_REG_INPUT_READY0_STREAM_ID, pair_generation);
                set_stream_sync(BENCH_STREAM_REG_INPUT_READY1_STREAM_ID, pair_generation);
#else
                input_ready_reg[0] = pair_generation;
                input1_ready_reg[0] = pair_generation;
#endif
            }
        }
    }
#else
    const uint32_t page_size = get_tile_size(kCbIn0);
    const uint32_t global_Nt = get_arg_val<uint32_t>(5);
    const uint32_t base_mt = get_arg_val<uint32_t>(6);
    const uint32_t base_nt = get_arg_val<uint32_t>(7);
    InterleavedAddrGen<true> src0_addrgen = {.bank_base_address = src0_dram_addr, .page_size = page_size};
    InterleavedAddrGen<true> src1_addrgen = {.bank_base_address = src1_dram_addr, .page_size = page_size};

    DeviceZoneScopedN(SPM_CASE_PREFIX "_CB_READER");
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                const uint32_t global_mt = base_mt + mt;
                const uint32_t global_nt = base_nt + nt;
                const uint32_t a_tile_index = global_mt * Kt + kt;
                cb_reserve_back(kCbIn0, kOneTile);
                const uint32_t l1_write_addr_in0 = get_write_ptr(kCbIn0);
                noc_async_read(get_noc_addr(a_tile_index, src0_addrgen), l1_write_addr_in0, page_size);
                noc_async_read_barrier();
                cb_push_back(kCbIn0, kOneTile);

                const uint32_t b_tile_index = kt * global_Nt + global_nt;
                cb_reserve_back(kCbIn1, kOneTile);
                const uint32_t l1_write_addr_in1 = get_write_ptr(kCbIn1);
                noc_async_read(get_noc_addr(b_tile_index, src1_addrgen), l1_write_addr_in1, page_size);
                noc_async_read_barrier();
                cb_push_back(kCbIn1, kOneTile);
            }
        }
    }
#endif
}
