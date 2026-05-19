// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef BENCH_STATIC_PROTOCOL
#define BENCH_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_STATIC_INPUT_PROTOCOL
#define BENCH_STATIC_INPUT_PROTOCOL 0
#endif

#ifndef BENCH_STATIC_OUTPUT_PROTOCOL
#define BENCH_STATIC_OUTPUT_PROTOCOL 0
#endif

#ifndef BENCH_PROTOCOL_START_VALUE
#define BENCH_PROTOCOL_START_VALUE 1
#endif

namespace {

constexpr uint32_t kCbOut = tt::CBIndex::c_16;

#if BENCH_STATIC_INPUT_PROTOCOL && BENCH_STATIC_OUTPUT_PROTOCOL
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_INPUT_OUTPUT"
#elif BENCH_STATIC_INPUT_PROTOCOL
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_INPUT_ONLY"
#elif BENCH_STATIC_OUTPUT_PROTOCOL
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_OUTPUT_ONLY"
#else
#define RMP_MODE_PREFIX "RMP_REUSE_CB"
#endif

#define RMP_ZONE(name) RMP_MODE_PREFIX "_" name

#if BENCH_STATIC_OUTPUT_PROTOCOL
inline void wait_min_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] < value) {
    }
}

#if BENCH_STATIC_INPUT_PROTOCOL
inline void wait_equal_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] == value) {
            return;
        }
    }
}
#endif

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}
#endif

}  // namespace

void kernel_main() {
    // out tensor args
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(10);

    // batch args
    uint32_t MtNt = get_arg_val<uint32_t>(11);
    uint32_t batch = get_arg_val<uint32_t>(12);

    const uint32_t single_tile_size_bytes = get_tile_size(kCbOut);

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, out_tensor_addr);

#if BENCH_STATIC_OUTPUT_PROTOCOL
    const uint32_t out_ring_addr = get_arg_val<uint32_t>(13);
    const uint32_t out_slot_bytes = get_arg_val<uint32_t>(14);
    const uint32_t out_num_pages = get_arg_val<uint32_t>(15);
#if BENCH_STATIC_INPUT_PROTOCOL
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(16);

    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
#endif
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbOut, false);

#if BENCH_STATIC_INPUT_PROTOCOL
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
    DeviceZoneScopedN(RMP_ZONE("WRITER"));

    uint32_t generation = 0;
    for (uint32_t b = 0; b < batch; b++) {
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                ++generation;
                const uint32_t slot = (generation - 1) % out_num_pages;
                wait_min_reg(output_ready_reg, generation);

                uint32_t l1_read_addr = out_ring_addr + slot * out_slot_bytes;
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                        l1_read_addr += single_tile_size_bytes;
                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                output_consumed_reg[0] = generation;
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
    }
#else
    DeviceZoneScopedN(RMP_ZONE("WRITER"));

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                cb_wait_front(kCbOut, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(kCbOut);

                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                        l1_read_addr += single_tile_size_bytes;
                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                cb_pop_front(kCbOut, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
    }
#endif
}
