// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file is intentionally forked from:
// ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/
// writer_update_cache_interleaved_start_id.cpp

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

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(1);
    uint32_t cache_tile_offset_B = get_arg_val<uint32_t>(2);
    const uint32_t my_batch_idx = get_arg_val<uint32_t>(3);
    const bool send_signal = get_arg_val<uint32_t>(4) == 1;
    const uint32_t send_core_x = get_arg_val<uint32_t>(5);
    const uint32_t send_core_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t untilized_cache_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache2_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_input_cb_id = get_compile_time_arg_val(3);
    constexpr bool use_index_tensor = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(5);
    constexpr uint32_t cache_batch_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t Wbytes = get_compile_time_arg_val(8);

    constexpr bool is_paged_cache = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t num_heads = get_compile_time_arg_val(10);
    constexpr uint32_t block_size = get_compile_time_arg_val(11);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(12);
    constexpr uint32_t max_blocks_per_seq = get_compile_time_arg_val(13);
    constexpr uint32_t page_table_cb_id = get_compile_time_arg_val(14);
    (void)max_blocks_per_seq;

    constexpr uint32_t St = get_compile_time_arg_val(15);
    uint32_t semaphore_addr = get_semaphore(get_compile_time_arg_val(16));

    constexpr auto s0_args = TensorAccessorArgs<17>();
    constexpr uint32_t head_offset_t = Wt * St;
    constexpr uint32_t TILE_HEIGHT = 32;

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const auto s0 = TensorAccessor(s0_args, cache_addr);

    uint32_t cache_id = cache_start_id;
    bool skip_update = false;

    if constexpr (use_index_tensor) {
        CircularBuffer cb_index(cb_index_id);
        cb_index.wait_front(1);
        uint32_t index_cb_ptr = get_read_ptr(cb_index_id);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
        const uint32_t update_idx = index_ptr[my_batch_idx];

        if (update_idx == static_cast<uint32_t>(-1)) {
            skip_update = true;
        } else {
            if constexpr (is_paged_cache) {
                CircularBuffer cb_page_table(page_table_cb_id);
                cb_page_table.wait_front(1);
                uint32_t page_table_cb_rd_ptr = get_read_ptr(page_table_cb_id);
                volatile tt_l1_ptr uint32_t* page_table_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_rd_ptr);

                const uint32_t virtual_block_id = update_idx / block_size;
                const uint32_t physical_block_id = page_table_ptr[virtual_block_id];
                const uint32_t block_start_id = physical_block_id * num_heads * block_size_t * Wt;
                const uint32_t block_row_tile = (update_idx % block_size) / TILE_HEIGHT;
                const uint32_t block_offset = block_row_tile * Wt;
                cache_id = block_start_id + block_offset;
            } else {
                const uint32_t cache_batch_tile_offset = my_batch_idx * cache_batch_num_tiles;
                cache_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
            }
            cache_tile_offset_B = update_idx % TILE_HEIGHT * Wbytes;
        }
    }

#if BENCH_STATIC_PROTOCOL
    const uint32_t intermed_ring_addr = get_arg_val<uint32_t>(7);
    const uint32_t input_untilized_addr = get_arg_val<uint32_t>(8);
    const uint32_t output_ring_addr = get_arg_val<uint32_t>(9);
    const uint32_t intermed_page_size = get_arg_val<uint32_t>(10);
    const uint32_t output_page_size = get_arg_val<uint32_t>(11);
    const uint32_t num_pages = get_arg_val<uint32_t>(12);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(13);

#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif

    volatile tt_reg_ptr uint32_t* intermed0_ready_reg = reg_ptr_from_cb(untilized_cache_cb_id, true);
    volatile tt_reg_ptr uint32_t* intermed0_consumed_reg = reg_ptr_from_cb(untilized_cache_cb_id, false);
    volatile tt_reg_ptr uint32_t* intermed1_ready_reg = reg_ptr_from_cb(untilized_cache2_cb_id, true);
    volatile tt_reg_ptr uint32_t* intermed1_consumed_reg = reg_ptr_from_cb(untilized_cache2_cb_id, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(cache_cb_id, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(cache_cb_id, false);
    volatile tt_reg_ptr uint32_t* input_untilized_ready_reg = reg_ptr_from_cb(untilized_input_cb_id, true);
    wait_equal_reg(input_untilized_ready_reg, 1);

#if BENCH_USE_STREAM_REG_CBREGS
    DeviceZoneScopedN("TPUC_STATIC_STREAMREG_CBREGS_WRITER");
#else
    DeviceZoneScopedN("TPUC_STATIC_RUNTIME_WRITER");
#endif

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        const uint32_t generation = cur_head + 1;
        const uint32_t slot = cur_head % num_pages;
        const uint32_t intermed_l1 = intermed_ring_addr + slot * Wt * intermed_page_size;
        const uint32_t output_l1 = output_ring_addr + slot * Wt * output_page_size;

        wait_equal_reg(intermed0_ready_reg, generation);
        uint32_t cache_l1_write_addr = intermed_l1 + cache_tile_offset_B;
        const uint64_t input_l1_read_addr = get_noc_addr(input_untilized_addr + cur_head * Wbytes);
        noc_async_read(input_l1_read_addr, cache_l1_write_addr, Wbytes);
        noc_async_read_barrier();
        intermed1_ready_reg[0] = generation;
        intermed0_consumed_reg[0] = generation;

        wait_equal_reg(output_ready_reg, generation);
        if (!skip_update) {
            uint32_t out_l1_read_addr = output_l1;
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
                out_l1_read_addr += cache_tile_bytes;
            }
            noc_async_writes_flushed();
        }
        output_consumed_reg[0] = generation;
        if (!skip_update) {
            noc_async_write_barrier();
        }
        wait_min_reg(intermed1_consumed_reg, generation);
        cache_id += head_offset_t;
    }
#else
    CircularBuffer cb_untilized_input(untilized_input_cb_id);
    cb_untilized_input.wait_front(Wt);
    uint64_t input_l1_read_addr = get_noc_addr(get_read_ptr(untilized_input_cb_id));

    DeviceZoneScopedN("TPUC_CB_WRITER");
    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        CircularBuffer cb_untilized_cache(untilized_cache_cb_id);
        CircularBuffer cb_untilized_cache2(untilized_cache2_cb_id);
        CircularBuffer cb_cache(cache_cb_id);

        cb_untilized_cache.wait_front(Wt);
        cb_untilized_cache2.reserve_back(Wt);

        uint32_t cache_l1_write_addr = get_read_ptr(untilized_cache_cb_id) + cache_tile_offset_B;
        noc_async_read(input_l1_read_addr, cache_l1_write_addr, Wbytes);
        noc_async_read_barrier();
        cb_untilized_cache2.push_back(Wt);
        cb_untilized_cache.pop_front(Wt);

        cb_cache.wait_front(Wt);
        if (!skip_update) {
            uint32_t out_l1_read_addr = get_read_ptr(cache_cb_id);
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
                out_l1_read_addr += cache_tile_bytes;
            }
            noc_async_writes_flushed();
        }
        cb_cache.pop_front(Wt);
        if (!skip_update) {
            noc_async_write_barrier();
        }
        input_l1_read_addr += Wbytes;
        cache_id += head_offset_t;
    }

    cb_untilized_input.pop_front(Wt);
#endif

    if (send_signal) {
        const uint64_t in0_sender_semaphore_noc_addr = get_noc_addr(send_core_x, send_core_y, semaphore_addr);
        noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
        noc_async_atomic_barrier();
    }
}
