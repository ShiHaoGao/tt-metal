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
    uint32_t arg_index = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(arg_index++);
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

    constexpr auto cb_id_src = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr);

    const uint32_t HtWt = Ht * Wt;
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
    const uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_read = 0;

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t src_ring_addr = BENCH_SRC_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t src_ring_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t page_size = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_pages = get_arg_val<uint32_t>(arg_index++);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(arg_index++);
#endif

    volatile tt_reg_ptr uint32_t* input_ready_reg = reg_ptr_from_cb(kCbSrc, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = reg_ptr_from_cb(kCbSrc, false);
    input_ready_reg[0] = 0;
    input_consumed_reg[0] = 0;

#if BENCH_USE_STREAM_REG_CBREGS
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    DeviceZoneScopedN("TBCAST_STATIC_STREAMREG_CBREGS_COMPILETIME_READER");
#else
    DeviceZoneScopedN("TBCAST_STATIC_STREAMREG_CBREGS_READER");
#endif
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TBCAST_STATIC_RUNTIME_READER");
#endif

    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th, start_tw = 0) {
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_read < num_tiles; ++tw, ++num_tiles_read) {
                    const uint32_t generation = num_tiles_read + 1;
                    const uint32_t slot = num_tiles_read % num_pages;
                    if (generation > num_pages) {
                        wait_min_reg(input_consumed_reg, generation - num_pages);
                    }

                    const uint32_t l1_write_addr = src_ring_addr + slot * page_size;
                    noc_async_read_tile(tile_offset + tw, src, l1_write_addr);
                    noc_async_read_barrier();
                    input_ready_reg[0] = generation;
                }
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
#else
    CircularBuffer cb_src(kCbSrc);
    Noc noc;
    const uint32_t src_tile_bytes = get_tile_size(kCbSrc);

    DeviceZoneScopedN("TBCAST_CB_READER");

    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th, start_tw = 0) {
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_read < num_tiles; ++tw, ++num_tiles_read) {
                    cb_src.reserve_back(kOneTile);
                    noc.async_read(src, cb_src, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    cb_src.push_back(kOneTile);
                }
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
#endif

    (void)HtWt;
}
