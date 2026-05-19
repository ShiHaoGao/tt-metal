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

#ifndef BENCH_PROTOCOL_START_VALUE
#define BENCH_PROTOCOL_START_VALUE 1
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

constexpr auto kCbDst = tt::CBIndex::c_2;
constexpr uint32_t kOneTile = 1;

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

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}
#endif

}  // namespace

void kernel_main() {
    uint32_t index = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(index++);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(index++);
    const uint32_t D = get_arg_val<uint32_t>(index++);
    const uint32_t N = get_arg_val<uint32_t>(index++);
    const uint32_t C = get_arg_val<uint32_t>(index++);
    const uint32_t Ht = get_arg_val<uint32_t>(index++);
    const uint32_t Wt = get_arg_val<uint32_t>(index++);
    const uint32_t cND = get_arg_val<uint32_t>(index++);
    const uint32_t unused_runtime_shape = get_arg_val<uint32_t>(index++);
    (void)unused_runtime_shape;

    constexpr auto dst_args = TensorAccessorArgs<0, 0>();
    constexpr bool has_sharding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) == 1;
    const uint32_t dst_tile_bytes = get_tile_size(kCbDst);
    const auto dst = TensorAccessor(dst_args, dst_addr);

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

    uint32_t num_tiles_written = 0;
    uint32_t dst_tile_offset = start_tile_id;

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t dst_ring_addr = BENCH_DST_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t dst_ring_addr = get_arg_val<uint32_t>(index++);
    const uint32_t page_size = get_arg_val<uint32_t>(index++);
    const uint32_t num_pages = get_arg_val<uint32_t>(index++);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(index++);
#endif

#if BENCH_USE_STREAM_REG_SYNC
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbDst, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbDst, false);

#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#endif
#if BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_COMPILETIME_WRITER");
#elif BENCH_USE_STREAM_REG_SYNC
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_WRITER");
#elif BENCH_USE_STREAM_REG_CBREGS
    DeviceZoneScopedN("TBNG_STATIC_STREAMREG_CBREGS_WRITER");
#else
    DeviceZoneScopedN("TBNG_STATIC_RUNTIME_WRITER");
#endif

    for (uint32_t nd = start_nd; nd < cND && num_tiles_written < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_written < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_written < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_written < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_written < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_written < dst_num_tiles;
                             ++tw, ++num_tiles_written) {
                            const uint32_t generation = num_tiles_written + 1;
                            const uint32_t slot = num_tiles_written % num_pages;
#if BENCH_USE_STREAM_REG_SYNC
                            wait_min_stream(BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID, generation);
#else
                            wait_min_reg(output_ready_reg, generation);
#endif

                            const uint32_t src_l1_addr = dst_ring_addr + slot * page_size;
                            noc_async_write_tile(dst_tile_offset + num_tiles_written, dst, src_l1_addr);
                            noc_async_write_barrier();
#if BENCH_USE_STREAM_REG_SYNC
                            set_stream_sync(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, generation);
#else
                            output_consumed_reg[0] = generation;
#endif
                        }
                        if constexpr (has_sharding) {
                            dst_tile_offset += (Wt - dst_shard_width);
                        } else {
                            start_tw = 0;
                        }
                    }
                }
            }
        }
    }
#else
    CircularBuffer cb_dst(kCbDst);
    Noc noc;

    DeviceZoneScopedN("TBNG_CB_WRITER");

    for (uint32_t nd = start_nd; nd < cND && num_tiles_written < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_written < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_written < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_written < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_written < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_written < dst_num_tiles;
                             ++tw, ++num_tiles_written) {
                            cb_dst.wait_front(kOneTile);
                            noc.async_write(
                                cb_dst, dst, dst_tile_bytes, {}, {.page_id = dst_tile_offset + num_tiles_written});
                            noc.async_write_barrier();
                            cb_dst.pop_front(kOneTile);
                        }
                        if constexpr (has_sharding) {
                            dst_tile_offset += (Wt - dst_shard_width);
                        } else {
                            start_tw = 0;
                        }
                    }
                }
            }
        }
    }
#endif
}
