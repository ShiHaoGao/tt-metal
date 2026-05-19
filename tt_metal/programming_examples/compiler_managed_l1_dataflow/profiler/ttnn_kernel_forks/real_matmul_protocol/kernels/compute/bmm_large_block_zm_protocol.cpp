// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-metalium/circular_buffer_constants.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_matmul.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack.h"
#endif

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

#ifndef BENCH_LEVEL_C_LLK_DIRECT
#define BENCH_LEVEL_C_LLK_DIRECT 0
#endif

#ifndef BENCH_LEVEL_C_FW_SKIP_CB_INIT
#define BENCH_LEVEL_C_FW_SKIP_CB_INIT 0
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
constexpr uint32_t kCbInterm = tt::CBIndex::c_24;
constexpr uint32_t kBfp16Format = static_cast<uint32_t>(DataFormat::Float16_b);

#if BENCH_LEVEL_C_FW_SKIP_CB_INIT
#define RMP_MODE_PREFIX "RMP_REUSE_LEVEL_C_LLK_DIRECT_FW_SKIP_CB_INIT"
#elif BENCH_LEVEL_C_LLK_DIRECT
#define RMP_MODE_PREFIX "RMP_REUSE_LEVEL_C_LLK_DIRECT"
#elif BENCH_STATIC_INPUT_PROTOCOL && BENCH_STATIC_OUTPUT_PROTOCOL && BENCH_USE_STREAM_REG_CBREGS && \
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

inline uint32_t read_stream_sync(uint32_t stream_id, uint32_t reg_index) {
    return NOC_STREAM_READ_REG(stream_id, reg_index) & BENCH_STREAM_REG_VALUE_MASK;
}

inline void wait_equal_stream(uint32_t stream_id, uint32_t reg_index, uint32_t value) {
    value &= BENCH_STREAM_REG_VALUE_MASK;
    while (read_stream_sync(stream_id, reg_index) != value) {
    }
}

inline volatile tt_l1_ptr uint32_t* tensix_store_ptr(volatile tt_reg_ptr uint32_t* reg) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ((reinterpret_cast<uint32_t>(reg) >> 2) & 0x3ffff));
}

#if BENCH_LEVEL_C_FW_SKIP_CB_INIT
inline volatile tt_reg_ptr uint32_t* protocol_counter_ptr(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        STREAM_REG_ADDR(
            OPERAND_START_STREAM + cbid,
            received ? STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX : STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}
#else
inline volatile tt_reg_ptr uint32_t* protocol_counter_ptr(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}
#endif

inline uint32_t to_cb_addr(uint32_t l1_addr) {
    return l1_addr >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
}

inline uint32_t to_llk_addr(uint32_t l1_addr) {
    return to_cb_addr(l1_addr) - 1;
}

inline void raw_matmul_init(uint32_t /*out_subblock_num_tiles*/) {
#if BENCH_LEVEL_C_LLK_DIRECT
#ifdef TRISC_UNPACK
    _llk_unpack_hw_configure_<DST_ACCUM_MODE>(
        kBfp16Format,
        kBfp16Format,
        kBfp16Format,
        kBfp16Format,
        FACE_R_DIM,
        FACE_R_DIM,
        4,
        4,
        BENCH_SRC0_TILE_WORDS,
        BENCH_SRC1_TILE_WORDS);
    _llk_unpack_AB_matmul_init_(
        0,
        1,
        1,
        1,
        FACE_R_DIM,
        FACE_R_DIM,
        4,
        4,
        false,
        false);
#endif
#ifdef TRISC_MATH
    _llk_math_pack_sync_init_<DST_SYNC_MODE, DST_ACCUM_MODE>();
    _llk_math_hw_configure_<DST_ACCUM_MODE>(kBfp16Format, kBfp16Format);
    _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE>(
        TILE_R_DIM,
        TILE_C_DIM,
        TILE_R_DIM,
        TILE_C_DIM,
        false,
        0,
        1,
        1);
#endif
#ifdef TRISC_PACK
    _llk_pack_hw_configure_<DST_ACCUM_MODE, PackMode::Default>(
        kBfp16Format,
        kBfp16Format,
        BENCH_OUT_TILE_WORDS,
        FACE_R_DIM,
        TILE_C_DIM,
        4,
        false,
        0);
    _llk_pack_init_<PackMode::Default, false, false>(FACE_R_DIM, TILE_C_DIM, 4, 1);
    _llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE>();
#endif
#else
    mm_init(kCbIn0, kCbIn1, kCbOut);
#endif
}

inline void raw_matmul_tile(
    uint32_t in0_l1_base,
    uint32_t in1_l1_base,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t dst_index) {
#if BENCH_LEVEL_C_LLK_DIRECT
#ifdef TRISC_UNPACK
    _llk_unpack_AB_matmul_(
        to_llk_addr(in0_l1_base),
        to_llk_addr(in1_l1_base),
        in0_tile_index,
        in1_tile_index,
        BENCH_SRC0_TILE_WORDS,
        BENCH_SRC1_TILE_WORDS,
        false,
        false,
        1,
        1,
        1);
#endif
#ifdef TRISC_MATH
    _llk_math_matmul_<MATH_FIDELITY, MM_THROTTLE>(dst_index);
#endif
#else
    matmul_tiles(kCbIn0, kCbIn1, in0_tile_index, in1_tile_index, dst_index);
#endif
}

inline void raw_pack_tile(uint32_t dst_index, uint32_t out_l1_addr) {
#if BENCH_LEVEL_C_LLK_DIRECT
#ifdef TRISC_PACK
    _llk_pack_<DST_SYNC_MODE, DST_ACCUM_MODE, PackMode::Default>(dst_index, to_llk_addr(out_l1_addr));
#endif
#else
    pack_tile(dst_index, kCbOut);
#endif
}

inline void set_static_read_base(uint32_t cb_id, uint32_t l1_addr) {
#if !BENCH_LEVEL_C_LLK_DIRECT
#ifdef TRISC_UNPACK
    get_local_cb_interface(cb_id).fifo_rd_ptr = to_cb_addr(l1_addr);
#endif
#endif
}

inline void set_static_write_base(uint32_t cb_id, uint32_t l1_addr) {
#if !BENCH_LEVEL_C_LLK_DIRECT
#ifdef TRISC_PACK
    get_local_cb_interface(cb_id).fifo_wr_ptr = to_cb_addr(l1_addr);
    get_local_cb_interface(cb_id).fifo_wr_tile_ptr = 0;
#endif
#endif
}

inline void publish_input_consumed(
    volatile tt_reg_ptr uint32_t* input_consumed_reg,
    volatile tt_reg_ptr uint32_t* input1_consumed_reg,
    uint32_t value) {
#ifdef TRISC_MATH
    input_consumed_reg[0] = value;
    input1_consumed_reg[0] = value;
#endif
}

inline void publish_output_after_pack(volatile tt_reg_ptr uint32_t* output_ready_reg, uint32_t value) {
#ifdef TRISC_PACK
    TT_SETDMAREG(0, value, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    auto* output_ready_tensix = tensix_store_ptr(output_ready_reg);
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, reinterpret_cast<uint32_t>(&output_ready_tensix[0]));
#endif
}
#endif

}  // namespace

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);
    uint32_t num_blocks = get_compile_time_arg_val(7);
    uint32_t out_subblock_h = get_compile_time_arg_val(8);
    uint32_t out_subblock_w = get_compile_time_arg_val(9);
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);
    uint32_t batch = get_compile_time_arg_val(11);

#if BENCH_STATIC_PROTOCOL
    raw_matmul_init(out_subblock_num_tiles);
#else
    mm_init(kCbIn0, kCbIn1, kCbOut);
#endif

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t in0_ring_addr = BENCH_SRC0_RING_ADDR;
    constexpr uint32_t in1_ring_addr = BENCH_SRC1_RING_ADDR;
    constexpr uint32_t out_ring_addr = BENCH_OUT_RING_ADDR;
    constexpr uint32_t in0_slot_bytes = BENCH_SRC0_SLOT_BYTES;
    constexpr uint32_t in1_slot_bytes = BENCH_SRC1_SLOT_BYTES;
    constexpr uint32_t out_slot_bytes = BENCH_OUT_SLOT_BYTES;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t out_num_pages = BENCH_OUT_NUM_PAGES;
#else
    const uint32_t in0_ring_addr = get_arg_val<uint32_t>(0);
    const uint32_t in1_ring_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_ring_addr = get_arg_val<uint32_t>(2);
    const uint32_t in0_slot_bytes = get_arg_val<uint32_t>(3);
    const uint32_t in1_slot_bytes = get_arg_val<uint32_t>(4);
    const uint32_t out_slot_bytes = get_arg_val<uint32_t>(5);
    const uint32_t num_pages = get_arg_val<uint32_t>(6);
    const uint32_t out_num_pages = get_arg_val<uint32_t>(7);
#endif

#if BENCH_STATIC_INPUT_PROTOCOL
#if !BENCH_USE_STREAM_REG_CBREGS
#if BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(8);
#endif
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
#endif
    volatile tt_reg_ptr uint32_t* input_ready_reg = protocol_counter_ptr(kCbIn0, true);
    volatile tt_reg_ptr uint32_t* input1_ready_reg = protocol_counter_ptr(kCbIn1, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = protocol_counter_ptr(kCbIn0, false);
    volatile tt_reg_ptr uint32_t* input1_consumed_reg = protocol_counter_ptr(kCbIn1, false);
#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_REG_INDEX, BENCH_PROTOCOL_START_VALUE);
#else
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#endif

#if BENCH_STATIC_OUTPUT_PROTOCOL
    volatile tt_reg_ptr uint32_t* output_ready_reg = protocol_counter_ptr(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = protocol_counter_ptr(kCbOut, false);
#if BENCH_USE_STREAM_REG_CBREGS && !BENCH_STATIC_INPUT_PROTOCOL
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_REG_INDEX, BENCH_PROTOCOL_START_VALUE);
#endif
#endif

#if defined(TRISC_UNPACK)
    DeviceZoneScopedN(RMP_ZONE("COMPUTE_UNPACK"));
#elif defined(TRISC_MATH)
    DeviceZoneScopedN(RMP_ZONE("COMPUTE_MATH"));
#elif defined(TRISC_PACK)
    DeviceZoneScopedN(RMP_ZONE("COMPUTE_PACK"));
#endif

#if BENCH_STATIC_INPUT_PROTOCOL
    uint32_t input_generation = 0;
#endif
#if BENCH_STATIC_OUTPUT_PROTOCOL
    uint32_t output_generation = 0;
#endif

    for (uint32_t b = 0; b < batch; b++) {
#if !BENCH_STATIC_OUTPUT_PROTOCOL
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
#endif
        for (uint32_t block = 0; block < num_blocks; block++) {
#if BENCH_STATIC_INPUT_PROTOCOL
            ++input_generation;
            const uint32_t input_slot = (input_generation - 1) % num_pages;
            const uint32_t in0_l1_addr = in0_ring_addr + input_slot * in0_slot_bytes;
            const uint32_t in1_l1_addr = in1_ring_addr + input_slot * in1_slot_bytes;
#ifdef TRISC_UNPACK
            wait_min_reg(input_ready_reg, input_generation);
            wait_min_reg(input1_ready_reg, input_generation);
#endif
            set_static_read_base(kCbIn0, in0_l1_addr);
            set_static_read_base(kCbIn1, in1_l1_addr);
#else
            const uint32_t in0_l1_addr = 0;
            const uint32_t in1_l1_addr = 0;
            cb_wait_front(kCbIn0, in0_block_num_tiles);
            cb_wait_front(kCbIn1, in1_block_num_tiles);
#endif

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    const uint32_t local_subblock_index = in0_subblock * in1_num_subblocks + in1_subblock;

                    if (block > 0) {
#if BENCH_STATIC_OUTPUT_PROTOCOL
                        const uint32_t interm_l1_addr = out_ring_addr + local_subblock_index * out_slot_bytes;
                        set_static_read_base(kCbInterm, interm_l1_addr);
                        copy_tile_to_dst_init_short(kCbInterm);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(kCbInterm, i, i);
                        }
                        mm_init_short(kCbIn0, kCbIn1);
#else
                        copy_tile_to_dst_init_short(kCbInterm);
                        cb_wait_front(kCbInterm, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(kCbInterm, i, i);
                        }
                        cb_pop_front(kCbInterm, out_subblock_num_tiles);
                        mm_init_short(kCbIn0, kCbIn1);
#endif
                    }

                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                raw_matmul_tile(
                                    in0_l1_addr,
                                    in1_l1_addr,
                                    in0_index,
                                    in1_index,
                                    dst_index);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    const bool last_out = block == (num_blocks - 1);
                    if (last_out) {
#if BENCH_STATIC_OUTPUT_PROTOCOL
                        ++output_generation;
                        const uint32_t output_slot = (output_generation - 1) % out_num_pages;
                        const uint32_t out_l1_addr = out_ring_addr + output_slot * out_slot_bytes;
#ifdef TRISC_PACK
                        if (output_generation > out_num_pages) {
                            wait_min_reg(output_consumed_reg, output_generation - out_num_pages);
                        }
#endif
                        set_static_write_base(kCbOut, out_l1_addr);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            raw_pack_tile(i, out_l1_addr + i * BENCH_OUT_TILE_BYTES);
                        }
                        publish_output_after_pack(output_ready_reg, output_generation);
#else
                        cb_reserve_back(kCbOut, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, kCbOut);
                        }
                        cb_push_back(kCbOut, out_subblock_num_tiles);
#endif
                    } else {
#if BENCH_STATIC_OUTPUT_PROTOCOL
                        const uint32_t interm_l1_addr = out_ring_addr + local_subblock_index * out_slot_bytes;
                        set_static_write_base(kCbInterm, interm_l1_addr);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, kCbInterm);
                        }
#else
                        if (block == 0) {
                            cb_reserve_back(kCbOut, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        cb_reserve_back(kCbInterm, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, kCbInterm);
                        }
                        cb_push_back(kCbInterm, out_subblock_num_tiles);
#endif
                    }

                    release_dst();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

#if BENCH_STATIC_INPUT_PROTOCOL
            publish_input_consumed(input_consumed_reg, input1_consumed_reg, input_generation);
#else
            cb_pop_front(kCbIn0, in0_block_num_tiles);
            cb_pop_front(kCbIn1, in1_block_num_tiles);
#endif
        }
    }
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RMP_REUSE_CB_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RMP_REUSE_CB_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RMP_REUSE_CB_COMPUTE_PACK");
#endif

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            cb_wait_front(kCbIn0, in0_block_num_tiles);
            cb_wait_front(kCbIn1, in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    if (enable_reload) {
                        copy_tile_to_dst_init_short(kCbInterm);
                        cb_wait_front(kCbInterm, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(kCbInterm, i, i);
                        }
                        cb_pop_front(kCbInterm, out_subblock_num_tiles);
                        mm_init_short(kCbIn0, kCbIn1);
                    }

                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                matmul_tiles(kCbIn0, kCbIn1, in0_index, in1_index, dst_index);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    if (last_out) {
                        cb_reserve_back(kCbOut, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, kCbOut);
                        }
                        cb_push_back(kCbOut, out_subblock_num_tiles);
                    } else {
                        if (block == 0) {
                            cb_reserve_back(kCbOut, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        cb_reserve_back(kCbInterm, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, kCbInterm);
                        }
                        cb_push_back(kCbInterm, out_subblock_num_tiles);
                    }

                    release_dst();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }

            cb_pop_front(kCbIn0, in0_block_num_tiles);
            cb_pop_front(kCbIn1, in1_block_num_tiles);
        }
    }
#endif
}
