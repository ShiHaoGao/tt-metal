// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-metalium/circular_buffer_constants.h"

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

#ifndef BENCH_SRC_RING_ADDR
#define BENCH_SRC_RING_ADDR 0
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
constexpr auto kCbDst = tt::CBIndex::c_1;

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

inline void wait_equal_stream(uint32_t stream_id, uint32_t value) {
    value &= BENCH_STREAM_REG_VALUE_MASK;
    while (read_stream_sync(stream_id) != value) {
    }
}

inline volatile tt_l1_ptr uint32_t* tensix_store_ptr(volatile tt_reg_ptr uint32_t* reg) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>((reinterpret_cast<uint32_t>(reg) >> 2) & 0x3ffff);
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
    volatile tt_reg_ptr uint32_t* output_ready_reg,
    uint32_t value) {
#ifdef TRISC_PACK
    TT_SETDMAREG(0, value, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    input_consumed_reg[0] = value;
    auto* output_ready_tensix = tensix_store_ptr(output_ready_reg);
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, reinterpret_cast<uint32_t>(&output_ready_tensix[0]));
#endif
}
#endif

}  // namespace

void kernel_main() {
    uint32_t arg_index = 0;
    uint32_t start_n = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_c = get_arg_val<uint32_t>(arg_index++);
    const uint32_t start_t = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_th = get_arg_val<uint32_t>(arg_index++);
    uint32_t start_tw = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(arg_index++);
    const uint32_t n_stride = get_arg_val<uint32_t>(arg_index++);
    const uint32_t c_stride = get_arg_val<uint32_t>(arg_index++);
    const uint32_t N = get_arg_val<uint32_t>(arg_index++);
    const uint32_t C = get_arg_val<uint32_t>(arg_index++);
    const uint32_t Ht = get_arg_val<uint32_t>(arg_index++);
    const uint32_t Wt = get_arg_val<uint32_t>(arg_index++);
    (void)start_t;
    (void)n_stride;
    (void)c_stride;

    constexpr auto cb_id_src = get_compile_time_arg_val(0);
    constexpr auto cb_id_dst = get_compile_time_arg_val(1);
    unary_bcast_init<BroadcastType::ROW>(cb_id_src, cb_id_dst);

    uint32_t num_tiles_read = 0;

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t src_ring_addr = BENCH_SRC_RING_ADDR;
    constexpr uint32_t dst_ring_addr = BENCH_DST_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t src_ring_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t dst_ring_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t page_size = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_pages = get_arg_val<uint32_t>(arg_index++);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(arg_index++);
#endif

    volatile tt_reg_ptr uint32_t* input_ready_reg = reg_ptr_from_cb(kCbSrc, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = reg_ptr_from_cb(kCbSrc, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbDst, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbDst, false);

#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif

#if BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBCAST_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBCAST_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBCAST_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_PACK");
#endif
#elif BENCH_USE_STREAM_REG_CBREGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBCAST_STATIC_STREAMREG_CBREGS_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBCAST_STATIC_STREAMREG_CBREGS_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBCAST_STATIC_STREAMREG_CBREGS_COMPUTE_PACK");
#endif
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBCAST_STATIC_RUNTIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBCAST_STATIC_RUNTIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBCAST_STATIC_RUNTIME_COMPUTE_PACK");
#endif
#endif

    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th, start_tw = 0) {
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_read < num_tiles; ++tw, ++num_tiles_read) {
                    const uint32_t generation = num_tiles_read + 1;
                    const uint32_t slot = num_tiles_read % num_pages;
                    const uint32_t src_l1_addr = src_ring_addr + slot * page_size;
                    const uint32_t dst_l1_addr = dst_ring_addr + slot * page_size;

#ifdef TRISC_UNPACK
                    wait_equal_reg(input_ready_reg, generation);
#endif
                    set_static_read_base(cb_id_src, src_l1_addr);
                    set_static_write_base(cb_id_dst, dst_l1_addr);

                    tile_regs_acquire();
                    unary_bcast<BroadcastType::ROW>(cb_id_src, 0, 0);
                    tile_regs_commit();

#ifdef TRISC_PACK
                    if (generation > num_pages) {
                        wait_min_reg(output_consumed_reg, generation - num_pages);
                    }
#endif
                    tile_regs_wait();
                    pack_tile(0, cb_id_dst);
                    tile_regs_release();

                    publish_after_pack(input_consumed_reg, output_ready_reg, generation);
                }
            }
        }
    }
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("TBCAST_CB_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("TBCAST_CB_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("TBCAST_CB_COMPUTE_PACK");
#endif

    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th, start_tw = 0) {
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_read < num_tiles; ++tw, ++num_tiles_read) {
                    cb_wait_front(cb_id_src, 1);
                    tile_regs_acquire();
                    unary_bcast<BroadcastType::ROW>(cb_id_src, 0, 0);
                    tile_regs_commit();

                    cb_pop_front(cb_id_src, 1);
                    cb_reserve_back(cb_id_dst, 1);
                    tile_regs_wait();
                    pack_tile(0, cb_id_dst);

                    cb_push_back(cb_id_dst, 1);
                    tile_regs_release();
                }
            }
        }
    }
#endif
}
