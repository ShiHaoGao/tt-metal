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

#ifndef BENCH_TRACE_STATIC_PROTOCOL
#define BENCH_TRACE_STATIC_PROTOCOL 0
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

constexpr auto kCbSrc0 = tt::CBIndex::c_0;
constexpr auto kCbSrc1 = tt::CBIndex::c_1;
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
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(4);
    const uint32_t nD_stride = get_arg_val<uint32_t>(5);
    const uint32_t d_stride = get_arg_val<uint32_t>(6);
    const uint32_t n_stride = get_arg_val<uint32_t>(7);
    const uint32_t c_stride = get_arg_val<uint32_t>(8);
    const uint32_t D = get_arg_val<uint32_t>(9);
    const uint32_t N = get_arg_val<uint32_t>(10);
    const uint32_t C = get_arg_val<uint32_t>(11);
    const uint32_t Ht = get_arg_val<uint32_t>(12);
    const uint32_t Wt = get_arg_val<uint32_t>(13);
    const uint32_t cND = get_arg_val<uint32_t>(14);
    const uint32_t src_addr_b = get_arg_val<uint32_t>(15);
    const uint32_t nD_stride_b = get_arg_val<uint32_t>(16);
    const uint32_t d_stride_b = get_arg_val<uint32_t>(17);
    const uint32_t n_stride_b = get_arg_val<uint32_t>(18);
    const uint32_t c_stride_b = get_arg_val<uint32_t>(19);
    const uint32_t src_num_tiles_b = get_arg_val<uint32_t>(20);

    constexpr auto src_args = TensorAccessorArgs<0, 0>();
    constexpr auto src_b_args =
        TensorAccessorArgs<src_args.next_compile_time_args_offset(), src_args.next_common_runtime_args_offset()>();
    constexpr bool has_sharding = get_compile_time_arg_val(src_b_args.next_compile_time_args_offset()) == 1;

    Noc noc;
    CircularBuffer cb_src(kCbSrc0);
    CircularBuffer cb_src_b(kCbSrc1);

#if SRC_SHARDED
    cb_src.reserve_back(src_num_tiles);
    cb_src.push_back(src_num_tiles);
#else
    const uint32_t src_tile_bytes = get_tile_size(kCbSrc0);
    const auto src = TensorAccessor(src_args, src_addr);
#endif
#if SRC_SHARDED_B
    cb_src_b.reserve_back(src_num_tiles_b);
    cb_src_b.push_back(src_num_tiles_b);
#else
    const uint32_t src_tile_bytes_b = get_tile_size(kCbSrc1);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b);
#endif

#if !SRC_SHARDED || !SRC_SHARDED_B
    const uint32_t HtWt = Ht * Wt;

    const uint32_t tiles_per_n = C * HtWt;
    const uint32_t tiles_per_d = N * tiles_per_n;
    const uint32_t tiles_per_nd = D * tiles_per_d;
    const uint32_t offset_nd = start_tile_id % tiles_per_nd;
    const uint32_t offset_d = offset_nd % tiles_per_d;
    const uint32_t offset_n = offset_d % tiles_per_n;
    const uint32_t offset_c = offset_n % HtWt;
    uint32_t start_nd = start_tile_id / tiles_per_nd;
    uint32_t start_d = offset_nd / tiles_per_d;
    uint32_t start_n = offset_d / tiles_per_n;
    uint32_t start_c = offset_n / HtWt;
    uint32_t start_th = offset_c / Wt;
    uint32_t start_tw = offset_c % Wt;
    uint32_t end_tw = has_sharding ? start_tw + dst_shard_width : Wt;

    uint32_t tile_offset =
        start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt;
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b + start_th * Wt;
    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t src0_ring_addr = BENCH_SRC0_RING_ADDR;
    constexpr uint32_t src1_ring_addr = BENCH_SRC1_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t src0_ring_addr = get_arg_val<uint32_t>(21);
    const uint32_t src1_ring_addr = get_arg_val<uint32_t>(22);
    const uint32_t page_size = get_arg_val<uint32_t>(23);
    const uint32_t num_pages = get_arg_val<uint32_t>(24);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(25);
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
    volatile tt_reg_ptr uint32_t* input_ready_reg = reg_ptr_from_cb(kCbSrc0, true);
    volatile tt_reg_ptr uint32_t* input1_ready_reg = reg_ptr_from_cb(kCbSrc1, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = reg_ptr_from_cb(kCbSrc0, false);
    volatile tt_reg_ptr uint32_t* input1_consumed_reg = reg_ptr_from_cb(kCbSrc1, false);

#if BENCH_USE_STREAM_REG_CBREGS
    input_ready_reg[0] = 0;
    input1_ready_reg[0] = 0;
    input_consumed_reg[0] = 0;
    input1_consumed_reg[0] = 0;
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#endif
#if BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_COMPILETIME_READER");
#elif BENCH_USE_STREAM_REG_SYNC
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_READER");
#elif BENCH_USE_STREAM_REG_CBREGS
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_READER");
#else
    DeviceZoneScopedN("TBNG_STATIC_RUNTIME_READER");
#endif

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
                            const uint32_t generation = num_tiles_read + 1;
                            const uint32_t slot = num_tiles_read % num_pages;
                            if (generation > num_pages) {
#if BENCH_USE_STREAM_REG_SYNC
                                wait_min_stream(BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID, generation - num_pages);
                                wait_min_stream(BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID, generation - num_pages);
#else
                                wait_min_reg(input_consumed_reg, generation - num_pages);
                                wait_min_reg(input1_consumed_reg, generation - num_pages);
#endif
                            }

                            const uint32_t dst0_l1_addr = src0_ring_addr + slot * page_size;
                            const uint32_t dst1_l1_addr = src1_ring_addr + slot * page_size;
#if !SRC_SHARDED
                            noc_async_read_tile(tile_offset + tw, src, dst0_l1_addr);
#endif
#if !SRC_SHARDED_B
                            noc_async_read_tile(tile_offset_b + tw, src_b, dst1_l1_addr);
#endif
                            noc_async_read_barrier();
#if BENCH_USE_STREAM_REG_SYNC
                            set_stream_sync(BENCH_STREAM_REG_INPUT_READY0_STREAM_ID, generation);
                            set_stream_sync(BENCH_STREAM_REG_INPUT_READY1_STREAM_ID, generation);
#else
                            input_ready_reg[0] = generation;
                            input1_ready_reg[0] = generation;
#endif
                        }
                        if constexpr (!has_sharding) {
                            start_tw = 0;
                        }
                        tile_offset += Wt;
                        tile_offset_b += Wt;
                    }
                    tile_offset += next_c_shift;
                    tile_offset_b += next_c_shift_b;
                }
                tile_offset += next_n_shift;
                tile_offset_b += next_n_shift_b;
            }
            tile_offset += next_d_shift;
            tile_offset_b += next_d_shift_b;
        }
        tile_offset += next_nd_shift;
        tile_offset_b += next_nd_shift_b;
    }
#else
    DeviceZoneScopedN("TBNG_CB_READER");

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_SHARDED
                            cb_src.reserve_back(kOneTile);
                            noc.async_read(
                                src, cb_src, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_B
                            cb_src_b.reserve_back(kOneTile);
                            noc.async_read(
                                src_b,
                                cb_src_b,
                                src_tile_bytes_b,
                                {.page_id = tile_offset_b + tw},
                                {.offset_bytes = 0});
#endif
                            noc.async_read_barrier();
#if !SRC_SHARDED
                            cb_src.push_back(kOneTile);
#endif
#if !SRC_SHARDED_B
                            cb_src_b.push_back(kOneTile);
#endif
                        }
                        if constexpr (!has_sharding) {
                            start_tw = 0;
                        }
                        tile_offset += Wt;
                        tile_offset_b += Wt;
                    }
                    tile_offset += next_c_shift;
                    tile_offset_b += next_c_shift_b;
                }
                tile_offset += next_n_shift;
                tile_offset_b += next_n_shift_b;
            }
            tile_offset += next_d_shift;
            tile_offset_b += next_d_shift_b;
        }
        tile_offset += next_nd_shift;
        tile_offset_b += next_nd_shift_b;
    }
#endif
#endif
}
