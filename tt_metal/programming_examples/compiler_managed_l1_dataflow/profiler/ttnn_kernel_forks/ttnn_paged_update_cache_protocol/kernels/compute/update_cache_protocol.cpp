// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file is forked from:
// ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp
//
// CB mode keeps the original helper path. Static modes keep the same compute
// transform, but explicitly drive the hot cache/intermediate/output CB counters
// so dynamic FIFO reserve/wait/push/pop is removed from the critical handoff.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "internal/circular_buffer_interface.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

#ifndef BENCH_STATIC_PROTOCOL
#define BENCH_STATIC_PROTOCOL 0
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

#ifndef BENCH_CACHE_RING_ADDR
#define BENCH_CACHE_RING_ADDR 0
#endif

#ifndef BENCH_INTERMED_RING_ADDR
#define BENCH_INTERMED_RING_ADDR 0
#endif

#ifndef BENCH_INPUT_UNTILIZED_ADDR
#define BENCH_INPUT_UNTILIZED_ADDR 0
#endif

#ifndef BENCH_OUTPUT_RING_ADDR
#define BENCH_OUTPUT_RING_ADDR 0
#endif

#ifndef BENCH_INPUT_TILES_ADDR
#define BENCH_INPUT_TILES_ADDR 0
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

#if BENCH_STATIC_PROTOCOL
inline void wait_equal_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] == value) {
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

inline void wait_equal_stream(uint32_t stream_id, uint32_t value) {
    value &= BENCH_STREAM_REG_VALUE_MASK;
    while (read_stream_sync(stream_id) != value) {
    }
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

inline void publish_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
#ifdef TRISC_PACK
    TT_SETDMAREG(0, value, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    auto* tensix_ptr = tensix_store_ptr(reg);
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, reinterpret_cast<uint32_t>(&tensix_ptr[0]));
#endif
}

inline void set_reg_after_pack(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
#ifdef TRISC_PACK
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    reg[0] = value;
#endif
}
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t cache_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads = get_compile_time_arg_val(7);
    static_assert(Wt <= 8, "Static protocol path currently supports Wt <= 8.");

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t cache_ring_addr = BENCH_CACHE_RING_ADDR;
    constexpr uint32_t intermed_ring_addr = BENCH_INTERMED_RING_ADDR;
    constexpr uint32_t input_untilized_addr = BENCH_INPUT_UNTILIZED_ADDR;
    constexpr uint32_t output_ring_addr = BENCH_OUTPUT_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
    constexpr uint32_t input_tiles_addr = BENCH_INPUT_TILES_ADDR;
#else
    const uint32_t cache_ring_addr = get_arg_val<uint32_t>(0);
    const uint32_t intermed_ring_addr = get_arg_val<uint32_t>(1);
    const uint32_t input_untilized_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_ring_addr = get_arg_val<uint32_t>(3);
    const uint32_t page_size = get_arg_val<uint32_t>(4);
    const uint32_t num_pages = get_arg_val<uint32_t>(5);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(6);
    const uint32_t input_tiles_addr = get_arg_val<uint32_t>(7);
#endif

    volatile tt_reg_ptr uint32_t* cache_ready_reg = reg_ptr_from_cb(cache_cb, true);
    volatile tt_reg_ptr uint32_t* cache_consumed_reg = reg_ptr_from_cb(cache_cb, false);
    volatile tt_reg_ptr uint32_t* intermed0_ready_reg = reg_ptr_from_cb(untilized_cache_cb, true);
    volatile tt_reg_ptr uint32_t* intermed0_consumed_reg = reg_ptr_from_cb(untilized_cache_cb, false);
    volatile tt_reg_ptr uint32_t* intermed1_ready_reg = reg_ptr_from_cb(untilized_cache2_cb, true);
    volatile tt_reg_ptr uint32_t* intermed1_consumed_reg = reg_ptr_from_cb(untilized_cache2_cb, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(out_cb, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(out_cb, false);
    volatile tt_reg_ptr uint32_t* input_untilized_ready_reg = reg_ptr_from_cb(untilized_in_cb, true);

#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif

#if BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TPUC_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TPUC_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TPUC_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_PACK");
#endif
#elif BENCH_USE_STREAM_REG_CBREGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TPUC_STATIC_STREAMREG_CBREGS_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TPUC_STATIC_STREAMREG_CBREGS_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TPUC_STATIC_STREAMREG_CBREGS_COMPUTE_PACK");
#endif
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TPUC_STATIC_RUNTIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TPUC_STATIC_RUNTIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TPUC_STATIC_RUNTIME_COMPUTE_PACK");
#endif
#endif

    pack_untilize_init<Wt, Wt>(in_cb, untilized_in_cb);
    set_static_read_base(in_cb, input_tiles_addr);
    set_static_write_base(untilized_in_cb, input_untilized_addr);
    pack_untilize_block<Wt, Wt>(in_cb, 1, untilized_in_cb, 0);
    pack_untilize_uninit(untilized_in_cb);
    publish_reg(input_untilized_ready_reg, 1);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        const uint32_t generation = cur_head + 1;
        const uint32_t slot = cur_head % num_pages;
        const uint32_t cache_l1 = cache_ring_addr + slot * Wt * page_size;
        const uint32_t intermed0_l1 = intermed_ring_addr + slot * Wt * page_size;
        const uint32_t intermed1_l1 = intermed_ring_addr + slot * Wt * page_size;
        const uint32_t out_l1 = output_ring_addr + slot * Wt * page_size;

        wait_equal_reg(cache_ready_reg, generation);
        if (generation > num_pages) {
            wait_min_reg(intermed1_consumed_reg, generation - num_pages);
        }
        set_static_read_base(cache_cb, cache_l1);
        set_static_write_base(untilized_cache_cb, intermed0_l1);
        pack_untilize_init<Wt, Wt>(cache_cb, untilized_cache_cb);
        pack_untilize_block<Wt, Wt>(cache_cb, 1, untilized_cache_cb, 0);
        pack_untilize_uninit(untilized_cache_cb);
        publish_reg(intermed0_ready_reg, generation);
        set_reg_after_pack(cache_consumed_reg, generation);

        wait_equal_reg(intermed1_ready_reg, generation);
        if (generation > num_pages) {
            wait_min_reg(output_consumed_reg, generation - num_pages);
        }
        set_static_read_base(untilized_cache2_cb, intermed1_l1);
        set_static_write_base(out_cb, out_l1);
        tilize_init(untilized_cache2_cb, Wt, out_cb);
        tilize_block(untilized_cache2_cb, Wt, out_cb);
        tilize_uninit(untilized_cache2_cb, out_cb);
        publish_reg(output_ready_reg, generation);
        set_reg_after_pack(intermed1_consumed_reg, generation);
    }
#else
    DeviceZoneScopedN("TPUC_CB_COMPUTE_INPUT_UNTILIZE");
    compute_kernel_lib::untilize<
        Wt,
        in_cb,
        untilized_in_cb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
#if defined(TRISC_UNPACK)
        DeviceZoneScopedN("TPUC_CB_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
        DeviceZoneScopedN("TPUC_CB_COMPUTE_MATH");
#elif defined(TRISC_PACK)
        DeviceZoneScopedN("TPUC_CB_COMPUTE_PACK");
#endif
        compute_kernel_lib::untilize<Wt, cache_cb, untilized_cache_cb>(1);
        compute_kernel_lib::tilize<Wt, untilized_cache2_cb, out_cb>(1);
    }
#endif
}
