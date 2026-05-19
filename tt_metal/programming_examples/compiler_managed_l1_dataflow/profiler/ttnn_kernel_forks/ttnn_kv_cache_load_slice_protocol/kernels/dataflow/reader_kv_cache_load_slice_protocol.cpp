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

#ifndef BENCH_PROTOCOL_START_SEM_ADDR
#define BENCH_PROTOCOL_START_SEM_ADDR 0
#endif

#ifndef BENCH_OUTPUT_L1_ADDR
#define BENCH_OUTPUT_L1_ADDR 0
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

template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

namespace {

constexpr auto kCbOutput = tt::CBIndex::c_0;

#if BENCH_STATIC_PROTOCOL
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
#endif

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_unpadded_tiles_head_dim = get_compile_time_arg_val(1);
    constexpr uint32_t num_unpadded_tiles_seqlen_dim = get_compile_time_arg_val(2);
    constexpr uint32_t num_padded_tiles_seqlen_dim = get_compile_time_arg_val(3);
    constexpr uint32_t num_readers = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    constexpr uint32_t tile_size = get_tile_size(kCbOutput);
    const auto src = TensorAccessor(src_args, src_addr);

#if BENCH_STATIC_PROTOCOL
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_l1_addr = get_arg_val<uint32_t>(3);

#if BENCH_USE_STREAM_REG_CBREGS
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TKVL_STATIC_STREAMREG_CBREGS_READER");
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
    DeviceZoneScopedN("TKVL_STATIC_RUNTIME_READER");
#endif
#else
    DeviceZoneScopedN("TKVL_CB_READER");
#endif

#if BENCH_STATIC_PROTOCOL
    uint32_t dst_l1_addr = output_l1_addr;
#else
    CircularBuffer cb_output(kCbOutput);
    cb_output.reserve_back(num_tiles);
    uint32_t dst_l1_addr = cb_output.get_write_ptr();
#endif

    uint32_t src_tile_id = start_id;
    uint32_t seqlen_dim_id = 0;
    uint32_t barrier_count = 0;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_size, num_readers>();

    const uint32_t num_iterations = num_tiles / num_unpadded_tiles_head_dim;
    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_unpadded_tiles_head_dim; j++) {
            noc_async_read_tile(src_tile_id, src, dst_l1_addr);
            dst_l1_addr += tile_size;
            src_tile_id++;
            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
        seqlen_dim_id++;
        if (seqlen_dim_id == num_unpadded_tiles_seqlen_dim) {
            seqlen_dim_id = 0;
            src_tile_id += num_padded_tiles_seqlen_dim;
        }
    }

    noc_async_read_barrier();
#if !BENCH_STATIC_PROTOCOL
    cb_output.push_back(num_tiles);
#endif
}
