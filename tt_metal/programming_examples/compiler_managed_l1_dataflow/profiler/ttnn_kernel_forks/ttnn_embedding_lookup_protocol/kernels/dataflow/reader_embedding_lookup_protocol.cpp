// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
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

#ifndef BENCH_ROW_RING_ADDR
#define BENCH_ROW_RING_ADDR 0
#endif

#ifndef BENCH_ROW_BYTES
#define BENCH_ROW_BYTES 0
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

constexpr auto kCbRows = tt::CBIndex::c_0;
constexpr auto kCbIndexScratch = tt::CBIndex::c_1;
constexpr uint32_t kOnePage = 1;

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
    const uint32_t index_addr = get_arg_val<uint32_t>(0);
    const uint32_t weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t start_row = get_arg_val<uint32_t>(3);
    const uint32_t row_bytes = get_arg_val<uint32_t>(4);
    const uint32_t vocab_size = get_arg_val<uint32_t>(5);

    constexpr uint32_t index_page_bytes = get_compile_time_arg_val(0);
    constexpr auto index_args = TensorAccessorArgs<1>();
    constexpr auto weights_args =
        TensorAccessorArgs<index_args.next_compile_time_args_offset(), index_args.next_common_runtime_args_offset()>();
    const auto index_tensor = TensorAccessor(index_args, index_addr, index_page_bytes);
    const auto weights = TensorAccessor(weights_args, weights_addr, row_bytes);

    CircularBuffer cb_rows(kCbRows);
    CircularBuffer cb_index(kCbIndexScratch);
    cb_index.reserve_back(kOnePage);
    const uint32_t index_l1_addr = cb_index.get_write_ptr();
    volatile tt_l1_ptr uint32_t* index_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_l1_addr);

#if BENCH_STATIC_PROTOCOL
    const uint32_t row_ring_addr = get_arg_val<uint32_t>(6);
    const uint32_t num_pages = get_arg_val<uint32_t>(7);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(8);

    volatile tt_reg_ptr uint32_t* rows_ready_reg = reg_ptr_from_cb(kCbRows, true);
    volatile tt_reg_ptr uint32_t* rows_consumed_reg = reg_ptr_from_cb(kCbRows, false);
    rows_ready_reg[0] = 0;
    rows_consumed_reg[0] = 0;

#if BENCH_USE_STREAM_REG_CBREGS
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TEMB_STATIC_STREAMREG_CBREGS_READER");
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TEMB_STATIC_RUNTIME_READER");
#endif

    for (uint32_t row = 0; row < num_rows; ++row) {
        const uint32_t generation = row + 1;
        if (generation > num_pages) {
            wait_min_reg(rows_consumed_reg, generation - num_pages);
        }

        noc_async_read(get_noc_addr(start_row + row, index_tensor), index_l1_addr, index_page_bytes);
        noc_async_read_barrier();
        const uint32_t token = index_l1_ptr[0] % vocab_size;

        const uint32_t dst_l1_addr = row_ring_addr + (row % num_pages) * row_bytes;
        noc_async_read(get_noc_addr(token, weights), dst_l1_addr, row_bytes);
        noc_async_read_barrier();
        rows_ready_reg[0] = generation;
    }
#else
    DeviceZoneScopedN("TEMB_CB_READER");
    for (uint32_t row = 0; row < num_rows; ++row) {
        noc_async_read(get_noc_addr(start_row + row, index_tensor), index_l1_addr, index_page_bytes);
        noc_async_read_barrier();
        const uint32_t token = index_l1_ptr[0] % vocab_size;

        cb_rows.reserve_back(kOnePage);
        const uint32_t row_l1_addr = cb_rows.get_write_ptr();
        noc_async_read(get_noc_addr(token, weights), row_l1_addr, row_bytes);
        noc_async_read_barrier();
        cb_rows.push_back(kOnePage);
    }
#endif
}
