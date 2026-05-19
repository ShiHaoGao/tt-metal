// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "tt-metalium/circular_buffer_constants.h"

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

#ifndef BENCH_SRC0_RING_ADDR
#define BENCH_SRC0_RING_ADDR 0
#endif

#ifndef BENCH_SRC1_RING_ADDR
#define BENCH_SRC1_RING_ADDR 0
#endif

#ifndef BENCH_DST_RING_ADDR
#define BENCH_DST_RING_ADDR 0
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

constexpr auto kCbPreLhs = tt::CBIndex::c_0;
constexpr auto kCbPreRhs = tt::CBIndex::c_1;
constexpr auto kCbOut = tt::CBIndex::c_2;
constexpr auto kCbPostLhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : kCbPreLhs;
constexpr auto kCbPostRhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : kCbPreRhs;

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

inline void wait_equal_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] != value) {
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

inline volatile tt_l1_ptr uint32_t* tensix_store_ptr(volatile tt_reg_ptr uint32_t* reg) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ((reinterpret_cast<uint32_t>(reg) >> 2) & 0x3ffff));
}

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}

inline uint32_t to_cb_addr(uint32_t l1_addr) {
    return l1_addr >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
}

inline void set_static_read_base(uint32_t cb_id, uint32_t l1_addr) {
#ifdef TRISC_UNPACK
    get_local_cb_interface(cb_id).fifo_rd_ptr = to_cb_addr(l1_addr);
#endif
}

inline void set_static_write_base(uint32_t cb_id, uint32_t l1_addr) {
#ifdef TRISC_PACK
    get_local_cb_interface(cb_id).fifo_wr_ptr = to_cb_addr(l1_addr);
    get_local_cb_interface(cb_id).fifo_wr_tile_ptr = 0;
#endif
}

inline void publish_after_pack(
    volatile tt_reg_ptr uint32_t* input_consumed_reg,
    volatile tt_reg_ptr uint32_t* input1_consumed_reg,
    volatile tt_reg_ptr uint32_t* output_ready_reg,
    uint32_t value) {
#ifdef TRISC_PACK
    TT_SETDMAREG(0, value, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    input_consumed_reg[0] = value;
    input1_consumed_reg[0] = value;
    auto* output_ready_tensix = tensix_store_ptr(output_ready_reg);
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, reinterpret_cast<uint32_t>(&output_ready_tensix[0]));
#endif
}

inline void publish_after_pack_stream(uint32_t value) {
#ifdef TRISC_PACK
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID, value);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID, value);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID, value);
#endif
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "This static protocol fork currently models one tile per compute cycle.");

    unary_op_init_common(kCbPostLhs, kCbOut);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT;
#endif
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT;
#endif

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t src0_ring_addr = BENCH_SRC0_RING_ADDR;
    constexpr uint32_t src1_ring_addr = BENCH_SRC1_RING_ADDR;
    constexpr uint32_t dst_ring_addr = BENCH_DST_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t src0_ring_addr = get_arg_val<uint32_t>(1);
    const uint32_t src1_ring_addr = get_arg_val<uint32_t>(2);
    const uint32_t dst_ring_addr = get_arg_val<uint32_t>(3);
    const uint32_t page_size = get_arg_val<uint32_t>(4);
    const uint32_t num_pages = get_arg_val<uint32_t>(5);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(6);
#endif

#if BENCH_USE_STREAM_REG_SYNC
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_reg_ptr uint32_t* input_ready_reg = reg_ptr_from_cb(kCbPreLhs, true);
    volatile tt_reg_ptr uint32_t* input1_ready_reg = reg_ptr_from_cb(kCbPreRhs, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = reg_ptr_from_cb(kCbPreLhs, false);
    volatile tt_reg_ptr uint32_t* input1_consumed_reg = reg_ptr_from_cb(kCbPreRhs, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbOut, false);
#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#endif

#if BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_PACK");
#endif
#elif BENCH_USE_STREAM_REG_SYNC
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_COMPUTE_PACK");
#endif
#elif BENCH_USE_STREAM_REG_CBREGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_COMPUTE_PACK");
#endif
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBNG_STATIC_RUNTIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBNG_STATIC_RUNTIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBNG_STATIC_RUNTIME_COMPUTE_PACK");
#endif
#endif

    for (uint32_t i = 0; i < num_tiles; ++i) {
        const uint32_t slot = i % num_pages;
        const uint32_t generation = i + 1;
        const uint32_t src0_l1_addr = src0_ring_addr + slot * page_size;
        const uint32_t src1_l1_addr = src1_ring_addr + slot * page_size;
        const uint32_t dst_l1_addr = dst_ring_addr + slot * page_size;

#ifdef TRISC_UNPACK
#if BENCH_USE_STREAM_REG_SYNC
        wait_equal_stream(BENCH_STREAM_REG_INPUT_READY0_STREAM_ID, generation);
        wait_equal_stream(BENCH_STREAM_REG_INPUT_READY1_STREAM_ID, generation);
#else
        wait_equal_reg(input_ready_reg, generation);
        wait_equal_reg(input1_ready_reg, generation);
#endif
#endif
        set_static_read_base(kCbPostLhs, src0_l1_addr);
        set_static_read_base(kCbPostRhs, src1_l1_addr);

        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(kCbPostRhs, kCbPostLhs);
        copy_tile(kCbPostLhs, 0, 0);
        copy_tile_to_dst_init_short_with_dt(kCbPostLhs, kCbPostRhs);
        copy_tile(kCbPostRhs, 0, 1);
#if HAS_ACTIVATIONS(POST)
        BINARY_SFPU_INIT;
#endif
        BINARY_SFPU_OP(0, 1, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

#ifdef TRISC_PACK
        if (generation > num_pages) {
#if BENCH_USE_STREAM_REG_SYNC
            wait_min_stream(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, generation - num_pages);
#else
            wait_min_reg(output_consumed_reg, generation - num_pages);
#endif
        }
#endif
        tile_regs_wait();
        set_static_write_base(kCbOut, dst_l1_addr);
        pack_tile(0, kCbOut);
        tile_regs_release();

#if BENCH_USE_STREAM_REG_SYNC
        publish_after_pack_stream(generation);
#else
        publish_after_pack(input_consumed_reg, input1_consumed_reg, output_ready_reg, generation);
#endif
    }
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBNG_CB_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBNG_CB_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBNG_CB_COMPUTE_PACK");
#endif

    auto process_tiles = [&](uint32_t n) {
        PREPROCESS(LHS, kCbPreLhs, kCbPostLhs, kCbOut, n);
        cb_wait_front(kCbPostLhs, n);

        PREPROCESS(RHS, kCbPreRhs, kCbPostRhs, kCbOut, n);
        cb_wait_front(kCbPostRhs, n);

        cb_reserve_back(kCbOut, n);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT;
#endif
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(kCbPostRhs, kCbPostLhs);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(kCbPostLhs, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(kCbPostLhs, kCbPostRhs);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(kCbPostRhs, i, i * 2 + 1);
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT;
#endif
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i * 2, kCbOut);
        }
        tile_regs_release();

        cb_push_back(kCbOut, n);
        cb_pop_front(kCbPostLhs, n);
        cb_pop_front(kCbPostRhs, n);
    };

    uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_tiles(num_tiles_per_cycle);
    }

    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles(remainder);
    }
#endif
}
