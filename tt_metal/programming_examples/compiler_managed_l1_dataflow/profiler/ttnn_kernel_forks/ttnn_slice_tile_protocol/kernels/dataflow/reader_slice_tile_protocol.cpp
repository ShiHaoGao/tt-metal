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

#ifndef BENCH_PROTOCOL_START_VALUE
#define BENCH_PROTOCOL_START_VALUE 1
#endif

#ifndef BENCH_RING_ADDR
#define BENCH_RING_ADDR 0
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

constexpr auto kCbSlice = tt::CBIndex::c_0;
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
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t src_tile_id = get_arg_val<uint32_t>(2);
    uint32_t start_out_h = get_arg_val<uint32_t>(3);
    uint32_t start_out_w = get_arg_val<uint32_t>(4);
    const uint32_t output_height_tiles = get_arg_val<uint32_t>(5);
    const uint32_t output_width_tiles = get_arg_val<uint32_t>(6);
    const uint32_t input_width_tiles = get_arg_val<uint32_t>(7);

    constexpr auto src_args = TensorAccessorArgs<0>();
    const uint32_t tile_bytes = get_tile_size(kCbSlice);
    const auto src = TensorAccessor(src_args, src_addr);

    CircularBuffer cb_slice(kCbSlice);
    Noc noc;

#if BENCH_STATIC_PROTOCOL
    const uint32_t ring_addr = get_arg_val<uint32_t>(8);
    const uint32_t page_size = get_arg_val<uint32_t>(9);
    const uint32_t num_pages = get_arg_val<uint32_t>(10);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(11);

    volatile tt_reg_ptr uint32_t* ready_reg = reg_ptr_from_cb(kCbSlice, true);
    volatile tt_reg_ptr uint32_t* consumed_reg = reg_ptr_from_cb(kCbSlice, false);
    ready_reg[0] = 0;
    consumed_reg[0] = 0;

#if BENCH_USE_STREAM_REG_CBREGS
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TTSL_STATIC_STREAMREG_CBREGS_READER");
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TTSL_STATIC_RUNTIME_READER");
#endif

    uint32_t out_h = start_out_h;
    uint32_t out_w = start_out_w;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        const uint32_t generation = i + 1;
        if (generation > num_pages) {
            wait_min_reg(consumed_reg, generation - num_pages);
        }

        const uint32_t dst_l1_addr = ring_addr + (i % num_pages) * page_size;
        noc_async_read_tile(src_tile_id, src, dst_l1_addr);
        noc_async_read_barrier();
        ready_reg[0] = generation;

        ++out_w;
        ++src_tile_id;
        if (out_w == output_width_tiles) {
            out_w = 0;
            ++out_h;
            src_tile_id += input_width_tiles - output_width_tiles;
            if (out_h == output_height_tiles) {
                out_h = 0;
            }
        }
    }
#else
    DeviceZoneScopedN("TTSL_CB_READER");
    uint32_t out_h = start_out_h;
    uint32_t out_w = start_out_w;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_slice.reserve_back(kOneTile);
        noc.async_read(src, cb_slice, tile_bytes, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_slice.push_back(kOneTile);

        ++out_w;
        ++src_tile_id;
        if (out_w == output_width_tiles) {
            out_w = 0;
            ++out_h;
            src_tile_id += input_width_tiles - output_width_tiles;
            if (out_h == output_height_tiles) {
                out_h = 0;
            }
        }
    }
#endif
}
