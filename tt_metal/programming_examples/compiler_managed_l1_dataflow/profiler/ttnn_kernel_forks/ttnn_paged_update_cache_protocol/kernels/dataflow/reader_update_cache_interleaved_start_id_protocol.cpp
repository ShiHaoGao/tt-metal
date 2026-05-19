// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file is intentionally forked from:
// ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/
// reader_update_cache_interleaved_start_id.cpp
//
// The CB path preserves the original TTNN behavior. The static paths keep the
// same page-table/cache addressing and replace the hot cache-row CB FIFO handoff
// with an explicit static ring protocol.

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

constexpr uint32_t kOneTile = 1;

#if BENCH_STATIC_PROTOCOL
inline void wait_min_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] < value) {
    }
}

inline uint32_t read_stream_sync(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, BENCH_STREAM_SYNC_REG_INDEX) & BENCH_STREAM_REG_VALUE_MASK;
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
#endif

}  // namespace

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(1);
    const uint32_t index_tensor_addr = get_arg_val<uint32_t>(2);
    const uint32_t my_batch_idx = get_arg_val<uint32_t>(3);
    const uint32_t page_table_tensor_addr = get_arg_val<uint32_t>(4);
    const bool wait_to_start_signal = get_arg_val<uint32_t>(5) == 1;

    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(1);
    constexpr bool use_index_tensor = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(3);
    constexpr uint32_t cache_batch_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    const uint32_t log_base_2_of_page_size = get_compile_time_arg_val(6);
    const uint32_t index_stick_size_B = get_compile_time_arg_val(7);
    (void)log_base_2_of_page_size;

    constexpr bool is_paged_cache = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_heads = get_compile_time_arg_val(9);
    constexpr uint32_t block_size = get_compile_time_arg_val(10);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(11);
    constexpr uint32_t max_blocks_per_seq = get_compile_time_arg_val(12);
    constexpr uint32_t log2_page_table_stick_size = get_compile_time_arg_val(13);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(14);
    constexpr uint32_t page_table_cb_id = get_compile_time_arg_val(15);
    (void)max_blocks_per_seq;
    (void)log2_page_table_stick_size;

    constexpr uint32_t St = get_compile_time_arg_val(16);
    uint32_t semaphore_addr = get_semaphore(get_compile_time_arg_val(17));

    constexpr auto s0_args = TensorAccessorArgs<18>();
    constexpr auto index_tensor_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<index_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t head_offset_t = Wt * St;

    CircularBuffer cb_input(input_cb_id);
    cb_input.reserve_back(Wt);
    cb_input.push_back(Wt);

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    constexpr uint32_t TILE_HEIGHT = 32;

    uint32_t cache_id = cache_start_id;
    const auto s0 = TensorAccessor(s0_args, cache_addr);
    bool skip_update = false;

    if constexpr (use_index_tensor) {
        const auto addrg = TensorAccessor(index_tensor_args, index_tensor_addr);

        CircularBuffer cb_index(cb_index_id);
        cb_index.reserve_back(1);
        uint32_t index_cb_wr_ptr = get_write_ptr(cb_index_id);
        uint64_t tensor_index_noc_addr = addrg.get_noc_addr(0);
        noc_async_read(tensor_index_noc_addr, index_cb_wr_ptr, index_stick_size_B);
        noc_async_read_barrier();
        cb_index.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);

        const uint32_t update_idx = index_ptr[my_batch_idx];
        if (update_idx == static_cast<uint32_t>(-1)) {
            skip_update = true;
        } else {
            if constexpr (is_paged_cache) {
                const auto page_table_gen = TensorAccessor(page_table_args, page_table_tensor_addr);
                CircularBuffer cb_page_table(page_table_cb_id);
                cb_page_table.reserve_back(1);
                uint32_t page_table_cb_wr_ptr = get_write_ptr(page_table_cb_id);
                uint64_t page_table_noc_addr = page_table_gen.get_noc_addr(my_batch_idx);
                noc_async_read(page_table_noc_addr, page_table_cb_wr_ptr, page_table_stick_size);
                noc_async_read_barrier();
                cb_page_table.push_back(1);
                volatile tt_l1_ptr uint32_t* page_table_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);

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
        }
    }

    if (wait_to_start_signal) {
        volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, 1);
        noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);
    }

#if BENCH_STATIC_PROTOCOL
    const uint32_t cache_ring_addr = get_arg_val<uint32_t>(6);
    const uint32_t page_size = get_arg_val<uint32_t>(7);
    const uint32_t num_pages = get_arg_val<uint32_t>(8);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(9);

    volatile tt_reg_ptr uint32_t* cache_ready_reg = reg_ptr_from_cb(cache_cb_id, true);
    volatile tt_reg_ptr uint32_t* cache_consumed_reg = reg_ptr_from_cb(cache_cb_id, false);
    volatile tt_reg_ptr uint32_t* intermed0_ready_reg = reg_ptr_from_cb(tt::CBIndex::c_24, true);
    volatile tt_reg_ptr uint32_t* intermed0_consumed_reg = reg_ptr_from_cb(tt::CBIndex::c_24, false);
    volatile tt_reg_ptr uint32_t* intermed1_ready_reg = reg_ptr_from_cb(tt::CBIndex::c_25, true);
    volatile tt_reg_ptr uint32_t* intermed1_consumed_reg = reg_ptr_from_cb(tt::CBIndex::c_25, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(tt::CBIndex::c_16, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(tt::CBIndex::c_16, false);

    cache_ready_reg[0] = 0;
    cache_consumed_reg[0] = 0;
    intermed0_ready_reg[0] = 0;
    intermed0_consumed_reg[0] = 0;
    intermed1_ready_reg[0] = 0;
    intermed1_consumed_reg[0] = 0;
    output_ready_reg[0] = 0;
    output_consumed_reg[0] = 0;

#if BENCH_USE_STREAM_REG_CBREGS
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif

#if BENCH_USE_STREAM_REG_CBREGS
    DeviceZoneScopedN("TPUC_STATIC_STREAMREG_CBREGS_READER");
#else
    DeviceZoneScopedN("TPUC_STATIC_RUNTIME_READER");
#endif

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        const uint32_t generation = cur_head + 1;
        if (generation > num_pages) {
            wait_min_reg(cache_consumed_reg, generation - num_pages);
        }
        if (!skip_update) {
            const uint32_t slot = cur_head % num_pages;
            uint32_t cache_l1_write_addr = cache_ring_addr + slot * Wt * page_size;
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc_async_read_tile(curr_cache_id, s0, cache_l1_write_addr);
                cache_l1_write_addr += cache_tile_bytes;
            }
            noc_async_read_barrier();
        }
        cache_ready_reg[0] = generation;
        cache_id += head_offset_t;
    }
#else
    CircularBuffer cb_cache(cache_cb_id);
    DeviceZoneScopedN("TPUC_CB_READER");
    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        cb_cache.reserve_back(Wt);
        if (!skip_update) {
            uint32_t cache_l1_write_addr = get_write_ptr(cache_cb_id);
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc_async_read_tile(curr_cache_id, s0, cache_l1_write_addr);
                cache_l1_write_addr += cache_tile_bytes;
            }
            noc_async_read_barrier();
        }
        cb_cache.push_back(Wt);
        cache_id += head_offset_t;
    }
#endif
}
