// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

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

#ifndef BENCH_USE_STREAM_REG_CBREGS
#define BENCH_USE_STREAM_REG_CBREGS 0
#endif

#ifndef BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
#define BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS 0
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

#ifndef BENCH_STREAM_REG_START_REG_INDEX
#ifdef STREAM_SCRATCH32_REG_INDEX
#define BENCH_STREAM_REG_START_REG_INDEX STREAM_SCRATCH32_REG_INDEX
#else
#define BENCH_STREAM_REG_START_REG_INDEX STREAM_SCRATCH_5_REG_INDEX
#endif
#endif

namespace {

constexpr uint32_t kCbIn0 = tt::CBIndex::c_0;
constexpr uint32_t kCbIn1 = tt::CBIndex::c_1;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;

#if BENCH_STATIC_INPUT_PROTOCOL && BENCH_STATIC_OUTPUT_PROTOCOL && BENCH_USE_STREAM_REG_CBREGS && \
    BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_INPUT_OUTPUT_CBREGS_COMPILETIME"
#elif BENCH_STATIC_INPUT_PROTOCOL && BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_INPUT_ONLY_CBREGS_COMPILETIME"
#elif BENCH_STATIC_OUTPUT_PROTOCOL && BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_OUTPUT_ONLY_CBREGS_COMPILETIME"
#elif BENCH_STATIC_INPUT_PROTOCOL && BENCH_STATIC_OUTPUT_PROTOCOL && BENCH_USE_STREAM_REG_CBREGS
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_INPUT_OUTPUT_CBREGS"
#elif BENCH_STATIC_INPUT_PROTOCOL && BENCH_USE_STREAM_REG_CBREGS
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_INPUT_ONLY_CBREGS"
#elif BENCH_STATIC_OUTPUT_PROTOCOL && BENCH_USE_STREAM_REG_CBREGS
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_OUTPUT_ONLY_CBREGS"
#elif BENCH_STATIC_INPUT_PROTOCOL && BENCH_STATIC_OUTPUT_PROTOCOL
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_INPUT_OUTPUT"
#elif BENCH_STATIC_INPUT_PROTOCOL
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_INPUT_ONLY"
#elif BENCH_STATIC_OUTPUT_PROTOCOL
#define RMP_MODE_PREFIX "RMP_REUSE_STATIC_OUTPUT_ONLY"
#else
#define RMP_MODE_PREFIX "RMP_REUSE_CB"
#endif

#define RMP_ZONE(name) RMP_MODE_PREFIX "_" name

#if BENCH_STATIC_INPUT_PROTOCOL || BENCH_USE_STREAM_REG_CBREGS
inline void wait_min_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] < value) {
    }
}

inline void set_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    asm volatile("fence" ::: "memory");
    noc_inline_dw_write<InlineWriteDst::L1>(get_noc_addr(reinterpret_cast<uint32_t>(sem)), value);
    noc_async_write_barrier();
}

inline void set_stream_sync(uint32_t stream_id, uint32_t reg_index, uint32_t value) {
    asm volatile("fence" ::: "memory");
    NOC_STREAM_WRITE_REG(stream_id, reg_index, value & BENCH_STREAM_REG_VALUE_MASK);
    asm volatile("fence" ::: "memory");
}

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}
#endif

}  // namespace

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks = get_arg_val<uint32_t>(16);

    // batch args
    uint32_t MtKt = get_arg_val<uint32_t>(17);
    uint32_t KtNt = get_arg_val<uint32_t>(18);
    uint32_t batch = get_arg_val<uint32_t>(19);
    uint32_t bcast_B = get_arg_val<uint32_t>(20);

    const uint32_t in0_single_tile_size_bytes = get_tile_size(kCbIn0);
    const uint32_t in1_single_tile_size_bytes = get_tile_size(kCbIn1);

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, in0_tensor_addr);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, in1_tensor_addr);

#if BENCH_USE_STREAM_REG_CBREGS && !BENCH_STATIC_INPUT_PROTOCOL
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbOut, false);
    output_ready_reg[0] = 0;
    output_consumed_reg[0] = 0;
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_REG_INDEX, BENCH_PROTOCOL_START_VALUE);
#endif

#if BENCH_STATIC_INPUT_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t in0_ring_addr = BENCH_SRC0_RING_ADDR;
    constexpr uint32_t in1_ring_addr = BENCH_SRC1_RING_ADDR;
    constexpr uint32_t in0_slot_bytes = BENCH_SRC0_SLOT_BYTES;
    constexpr uint32_t in1_slot_bytes = BENCH_SRC1_SLOT_BYTES;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
#if !BENCH_USE_STREAM_REG_CBREGS
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#endif
#else
    const uint32_t in0_ring_addr = get_arg_val<uint32_t>(21);
    const uint32_t in1_ring_addr = get_arg_val<uint32_t>(22);
    const uint32_t in0_slot_bytes = get_arg_val<uint32_t>(23);
    const uint32_t in1_slot_bytes = get_arg_val<uint32_t>(24);
    const uint32_t num_pages = get_arg_val<uint32_t>(25);
#if !BENCH_USE_STREAM_REG_CBREGS
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(26);
#endif
#endif

    volatile tt_reg_ptr uint32_t* input_ready_reg = reg_ptr_from_cb(kCbIn0, true);
    volatile tt_reg_ptr uint32_t* input1_ready_reg = reg_ptr_from_cb(kCbIn1, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = reg_ptr_from_cb(kCbIn0, false);
    volatile tt_reg_ptr uint32_t* input1_consumed_reg = reg_ptr_from_cb(kCbIn1, false);

#if BENCH_USE_STREAM_REG_CBREGS
    input_ready_reg[0] = 0;
    input1_ready_reg[0] = 0;
    input_consumed_reg[0] = 0;
    input1_consumed_reg[0] = 0;
#if BENCH_STATIC_OUTPUT_PROTOCOL
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbOut, false);
    output_ready_reg[0] = 0;
    output_consumed_reg[0] = 0;
#endif
    set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_REG_INDEX, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    set_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
    DeviceZoneScopedN(RMP_ZONE("READER"));

    uint32_t generation = 0;
    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; block++) {
            ++generation;
            const uint32_t slot = (generation - 1) % num_pages;
            if (generation > num_pages) {
                wait_min_reg(input_consumed_reg, generation - num_pages);
                wait_min_reg(input1_consumed_reg, generation - num_pages);
            }

            uint32_t l1_write_addr_in0 = in0_ring_addr + slot * in0_slot_bytes;
            uint32_t l1_write_addr_in1 = in1_ring_addr + slot * in1_slot_bytes;

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; h++) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; h++) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; w++) {
                    noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
                    l1_write_addr_in1 += in1_single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            noc_async_read_barrier();
            input_ready_reg[0] = generation;
            input1_ready_reg[0] = generation;
        }
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }
        in0_tensor_start_tile_id += MtKt;
    }
#else
    DeviceZoneScopedN(RMP_ZONE("READER"));

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_reserve_back(kCbIn0, in0_block_num_tiles);
            cb_reserve_back(kCbIn1, in1_block_num_tiles);

            uint32_t l1_write_addr_in0 = get_write_ptr(kCbIn0);
            uint32_t l1_write_addr_in1 = get_write_ptr(kCbIn1);

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; h++) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; h++) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; w++) {
                    noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
                    l1_write_addr_in1 += in1_single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            noc_async_read_barrier();

            cb_push_back(kCbIn0, in0_block_num_tiles);
            cb_push_back(kCbIn1, in1_block_num_tiles);
        }
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }
        in0_tensor_start_tile_id += MtKt;
    }
#endif
}
