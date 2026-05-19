// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "tt-metalium/circular_buffer_constants.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifdef TRISC_UNPACK
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack.h"
#endif

#ifndef BENCH_STATIC_PROTOCOL
#define BENCH_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_USE_COMPILE_TIME_ARGS
#define BENCH_USE_COMPILE_TIME_ARGS 0
#endif

#ifndef BENCH_USE_STREAM_REG_SYNC
#define BENCH_USE_STREAM_REG_SYNC 0
#endif

#ifndef BENCH_USE_STREAM_REG_CBREGS
#define BENCH_USE_STREAM_REG_CBREGS 0
#endif

#ifndef BENCH_LEVEL_C_GENERATED_STATIC
#define BENCH_LEVEL_C_GENERATED_STATIC 0
#endif

#ifndef BENCH_LEVEL_C_LLK_DIRECT
#define BENCH_LEVEL_C_LLK_DIRECT 0
#endif

#ifndef BENCH_LEVEL_C_FW_SKIP_CB_INIT
#define BENCH_LEVEL_C_FW_SKIP_CB_INIT 0
#endif

#ifndef BENCH_TRACE_STATIC_PROTOCOL
#define BENCH_TRACE_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS 1
#endif

#ifndef BENCH_PAGE_SIZE
#define BENCH_PAGE_SIZE 0
#endif

#ifndef BENCH_NUM_PAGES
#define BENCH_NUM_PAGES 1
#endif

#ifndef BENCH_SEM_SLOT_BYTES
#define BENCH_SEM_SLOT_BYTES 64
#endif

#ifndef BENCH_SRC0_RING_ADDR
#define BENCH_SRC0_RING_ADDR 0
#endif

#ifndef BENCH_SRC1_RING_ADDR
#define BENCH_SRC1_RING_ADDR 0
#endif

#ifndef BENCH_DST_RING_ADDR
#define BENCH_DST_RING_ADDR 0
#endif

#ifndef BENCH_INPUT_READY_SEM_ADDR
#define BENCH_INPUT_READY_SEM_ADDR 0
#endif

#ifndef BENCH_INPUT_CONSUMED_SEM_ADDR
#define BENCH_INPUT_CONSUMED_SEM_ADDR 0
#endif

#ifndef BENCH_OUTPUT_READY_SEM_ADDR
#define BENCH_OUTPUT_READY_SEM_ADDR 0
#endif

#ifndef BENCH_OUTPUT_CONSUMED_SEM_ADDR
#define BENCH_OUTPUT_CONSUMED_SEM_ADDR 0
#endif

#ifndef BENCH_PROTOCOL_START_SEM_ADDR
#define BENCH_PROTOCOL_START_SEM_ADDR 0
#endif

#ifndef BENCH_PROTOCOL_START_VALUE
#define BENCH_PROTOCOL_START_VALUE 1
#endif

#ifndef BENCH_STREAM_REG_START_STREAM_ID
#define BENCH_STREAM_REG_START_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_INPUT_READY0_STREAM_ID
#define BENCH_STREAM_REG_INPUT_READY0_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_INPUT_READY1_STREAM_ID
#define BENCH_STREAM_REG_INPUT_READY1_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID
#define BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID
#define BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID
#define BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID 3
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID
#define BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID 3
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

#ifndef BENCH_STREAM_REG_START_REG_INDEX
#ifdef STREAM_SCRATCH32_REG_INDEX
#define BENCH_STREAM_REG_START_REG_INDEX STREAM_SCRATCH32_REG_INDEX
#else
#define BENCH_STREAM_REG_START_REG_INDEX STREAM_SCRATCH_5_REG_INDEX
#endif
#endif

#ifndef BENCH_STREAM_REG_INPUT_READY0_REG_INDEX
#define BENCH_STREAM_REG_INPUT_READY0_REG_INDEX STREAM_SCRATCH_0_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_INPUT_READY1_REG_INDEX
#define BENCH_STREAM_REG_INPUT_READY1_REG_INDEX STREAM_SCRATCH_1_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED0_REG_INDEX
#define BENCH_STREAM_REG_INPUT_CONSUMED0_REG_INDEX STREAM_SCRATCH_2_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_INPUT_CONSUMED1_REG_INDEX
#define BENCH_STREAM_REG_INPUT_CONSUMED1_REG_INDEX STREAM_SCRATCH_3_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_READY_REG_INDEX
#define BENCH_STREAM_REG_OUTPUT_READY_REG_INDEX STREAM_SCRATCH_4_REG_INDEX
#endif

#ifndef BENCH_STREAM_REG_OUTPUT_CONSUMED_REG_INDEX
#define BENCH_STREAM_REG_OUTPUT_CONSUMED_REG_INDEX STREAM_SCRATCH_5_REG_INDEX
#endif

namespace {

constexpr uint32_t kCbIn0 = tt::CBIndex::c_0;
constexpr uint32_t kCbIn1 = tt::CBIndex::c_1;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;
constexpr uint32_t kDstReg = 0;
constexpr uint32_t kOneTile = 1;
constexpr uint32_t kBfp16Format = static_cast<uint32_t>(DataFormat::Float16_b);

#if BENCH_STATIC_PROTOCOL
inline void wait_min_local(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    while (true) {
        invalidate_l1_cache();
        if (sem[0] >= value) {
            return;
        }
    }
}

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

inline void wait_equal_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    while (reg[0] != value) {
    }
}

inline uint32_t read_stream_sync(uint32_t stream_id, uint32_t reg_index) {
    return NOC_STREAM_READ_REG(stream_id, reg_index) & BENCH_STREAM_REG_VALUE_MASK;
}

inline void wait_min_stream(uint32_t stream_id, uint32_t reg_index, uint32_t value) {
    while (read_stream_sync(stream_id, reg_index) < value) {
    }
}

inline void wait_equal_stream(uint32_t stream_id, uint32_t reg_index, uint32_t value) {
    value &= BENCH_STREAM_REG_VALUE_MASK;
    while (read_stream_sync(stream_id, reg_index) != value) {
    }
}

inline void set_stream_sync(uint32_t stream_id, uint32_t reg_index, uint32_t value) {
    asm volatile("fence" ::: "memory");
    NOC_STREAM_WRITE_REG(stream_id, reg_index, value & BENCH_STREAM_REG_VALUE_MASK);
    asm volatile("fence" ::: "memory");
}

inline void set_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    reg[0] = value;
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

inline void publish_after_pack(
    volatile tt_reg_ptr uint32_t* input_consumed_reg,
    volatile tt_reg_ptr uint32_t* output_ready_reg,
    uint32_t value) {
#ifdef TRISC_PACK
    TT_SETDMAREG(0, value, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    input_consumed_reg[0] = value;
    auto* output_ready_tensix = tensix_store_ptr(output_ready_reg);
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, reinterpret_cast<uint32_t>(&output_ready_tensix[0]));
#endif
}

inline void publish_after_pack_stream(uint32_t value) {
#ifdef TRISC_PACK
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID, BENCH_STREAM_REG_INPUT_CONSUMED0_REG_INDEX, value);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID, BENCH_STREAM_REG_INPUT_CONSUMED1_REG_INDEX, value);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID, BENCH_STREAM_REG_OUTPUT_READY_REG_INDEX, value);
#endif
}

inline volatile tt_l1_ptr uint32_t* sem_slot(uint32_t sem_base_addr, uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_base_addr + slot * BENCH_SEM_SLOT_BYTES);
}

inline void raw_tile_add_init(uint32_t /*page_size_bytes*/) {
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
        BENCH_PAGE_SIZE >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT,
        BENCH_PAGE_SIZE >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT);
    _llk_unpack_AB_init_<BroadcastType::NONE>(ckernel::DEFAULT_TENSOR_SHAPE, ckernel::Transpose::None);
#endif
#ifdef TRISC_MATH
    _llk_math_pack_sync_init_<DST_SYNC_MODE, DST_ACCUM_MODE>();
    _llk_math_hw_configure_<DST_ACCUM_MODE>(kBfp16Format, kBfp16Format);
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi>(
        ckernel::DEFAULT_TENSOR_SHAPE, false);
#endif
#ifdef TRISC_PACK
    _llk_pack_hw_configure_<DST_ACCUM_MODE, PackMode::Default>(
        kBfp16Format,
        kBfp16Format,
        BENCH_PAGE_SIZE >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT,
        FACE_R_DIM,
        TILE_C_DIM,
        4,
        false,
        0);
    _llk_pack_init_<PackMode::Default, false, false>(FACE_R_DIM, TILE_C_DIM, 4, 1);
    _llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE>();
#endif
#else
    binary_op_init_common(kCbIn0, kCbIn1, kCbOut);
    add_tiles_init(kCbIn0, kCbIn1);
#endif
}

inline uint32_t to_cb_addr(uint32_t l1_addr) {
    return l1_addr >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
}

inline uint32_t to_llk_addr(uint32_t l1_addr) {
    return to_cb_addr(l1_addr) - 1;
}

inline void static_add_tile(uint32_t src0_l1_addr, uint32_t src1_l1_addr, uint32_t dst_reg) {
#if BENCH_LEVEL_C_LLK_DIRECT
#ifdef TRISC_UNPACK
    _llk_unpack_AB_<BroadcastType::NONE>(to_llk_addr(src0_l1_addr), to_llk_addr(src1_l1_addr));
#endif
#ifdef TRISC_MATH
    _llk_math_eltwise_binary_<
        EltwiseBinaryType::ELWADD,
        BroadcastType::NONE,
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        MathFidelity::LoFi,
        EltwiseBinaryReuseDestType::NONE>(ckernel::DEFAULT_TENSOR_SHAPE, dst_reg, true);
#endif
#else
#ifdef TRISC_UNPACK
    get_local_cb_interface(kCbIn0).fifo_rd_ptr = to_cb_addr(src0_l1_addr);
    get_local_cb_interface(kCbIn1).fifo_rd_ptr = to_cb_addr(src1_l1_addr);
#endif
    add_tiles(kCbIn0, kCbIn1, 0, 0, dst_reg);
#endif
}

inline void static_pack_tile(uint32_t dst_reg, uint32_t dst_l1_addr) {
#if BENCH_LEVEL_C_LLK_DIRECT
#ifdef TRISC_PACK
    _llk_pack_<DST_SYNC_MODE, DST_ACCUM_MODE, PackMode::Default>(dst_reg, to_llk_addr(dst_l1_addr));
#endif
#else
#ifdef TRISC_PACK
    get_local_cb_interface(kCbOut).fifo_wr_ptr = to_cb_addr(dst_l1_addr);
    get_local_cb_interface(kCbOut).fifo_wr_tile_ptr = 0;
    pack_tile(dst_reg, kCbOut);
#endif
#endif
}
#endif

}  // namespace

void kernel_main() {
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t iterations = BENCH_ITERATIONS;
#else
    const uint32_t iterations = get_arg_val<uint32_t>(0);
#endif

#if BENCH_STATIC_PROTOCOL
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t src0_ring_addr = BENCH_SRC0_RING_ADDR;
    constexpr uint32_t src1_ring_addr = BENCH_SRC1_RING_ADDR;
    constexpr uint32_t dst_ring_addr = BENCH_DST_RING_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
    constexpr uint32_t input_ready_sem_addr = BENCH_INPUT_READY_SEM_ADDR;
    constexpr uint32_t input_consumed_sem_addr = BENCH_INPUT_CONSUMED_SEM_ADDR;
    constexpr uint32_t output_ready_sem_addr = BENCH_OUTPUT_READY_SEM_ADDR;
    constexpr uint32_t output_consumed_sem_addr = BENCH_OUTPUT_CONSUMED_SEM_ADDR;
    constexpr uint32_t protocol_start_sem_addr = BENCH_PROTOCOL_START_SEM_ADDR;
#else
    const uint32_t src0_ring_addr = get_arg_val<uint32_t>(1);
    const uint32_t src1_ring_addr = get_arg_val<uint32_t>(2);
    const uint32_t dst_ring_addr = get_arg_val<uint32_t>(3);
    const uint32_t page_size = get_arg_val<uint32_t>(4);
    const uint32_t num_pages = get_arg_val<uint32_t>(5);
    const uint32_t input_ready_sem_addr = get_arg_val<uint32_t>(6);
    const uint32_t input_consumed_sem_addr = get_arg_val<uint32_t>(7);
    const uint32_t output_ready_sem_addr = get_arg_val<uint32_t>(8);
    const uint32_t output_consumed_sem_addr = get_arg_val<uint32_t>(9);
    const uint32_t protocol_start_sem_addr = get_arg_val<uint32_t>(10);
#endif

    raw_tile_add_init(page_size);

#if BENCH_USE_STREAM_REG_SYNC
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_REG_INDEX, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_reg_ptr uint32_t* input_ready_reg = protocol_counter_ptr(kCbIn0, true);
    volatile tt_reg_ptr uint32_t* input1_ready_reg = protocol_counter_ptr(kCbIn1, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = protocol_counter_ptr(kCbIn0, false);
    volatile tt_reg_ptr uint32_t* input1_consumed_reg = protocol_counter_ptr(kCbIn1, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = protocol_counter_ptr(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = protocol_counter_ptr(kCbOut, false);

#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_REG_INDEX, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
    DEVICE_PRINT_UNPACK("rtadd compute unpack start value={} pages={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
    DEVICE_PRINT_MATH("rtadd compute math start value={} pages={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
    DEVICE_PRINT_PACK("rtadd compute pack start value={} pages={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
#endif

#if BENCH_LEVEL_C_FW_SKIP_CB_INIT
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_LEVEL_C_LLK_DIRECT_FW_SKIP_CB_INIT_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_LEVEL_C_LLK_DIRECT_FW_SKIP_CB_INIT_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_LEVEL_C_LLK_DIRECT_FW_SKIP_CB_INIT_COMPUTE_PACK");
#endif
#elif BENCH_LEVEL_C_LLK_DIRECT
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_LEVEL_C_LLK_DIRECT_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_LEVEL_C_LLK_DIRECT_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_LEVEL_C_LLK_DIRECT_COMPUTE_PACK");
#endif
#elif BENCH_LEVEL_C_GENERATED_STATIC
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_LEVEL_C_GENERATED_STATIC_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_LEVEL_C_GENERATED_STATIC_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_LEVEL_C_GENERATED_STATIC_COMPUTE_PACK");
#endif
#elif BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_ARGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_PACK");
#endif
#elif BENCH_USE_COMPILE_TIME_ARGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_STATIC_COMPILETIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_STATIC_COMPILETIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_STATIC_COMPILETIME_COMPUTE_PACK");
#endif
#elif BENCH_USE_STREAM_REG_CBREGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_CBREGS_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_CBREGS_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_CBREGS_COMPUTE_PACK");
#endif
#elif BENCH_USE_STREAM_REG_SYNC
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_STATIC_STREAMREG_COMPUTE_PACK");
#endif
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_STATIC_RUNTIME_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_STATIC_RUNTIME_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_STATIC_RUNTIME_COMPUTE_PACK");
#endif
#endif

    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t slot = i % num_pages;
        const uint32_t generation = i + 1;
        const uint32_t src0_l1_addr = src0_ring_addr + slot * page_size;
        const uint32_t src1_l1_addr = src1_ring_addr + slot * page_size;
        const uint32_t dst_l1_addr = dst_ring_addr + slot * page_size;

#if defined(TRISC_UNPACK)
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT_UNPACK("rtadd unpack wait input i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
#if BENCH_USE_STREAM_REG_SYNC
        wait_equal_stream(BENCH_STREAM_REG_INPUT_READY0_STREAM_ID, BENCH_STREAM_REG_INPUT_READY0_REG_INDEX, generation);
        wait_equal_stream(BENCH_STREAM_REG_INPUT_READY1_STREAM_ID, BENCH_STREAM_REG_INPUT_READY1_REG_INDEX, generation);
#else
        wait_equal_reg(input_ready_reg, generation);
        wait_equal_reg(input1_ready_reg, generation);
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT_UNPACK("rtadd unpack got input i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
#endif

        tile_regs_acquire();
        static_add_tile(src0_l1_addr, src1_l1_addr, kDstReg);
        tile_regs_commit();

#if defined(TRISC_PACK)
        if (i >= num_pages) {
#if BENCH_USE_STREAM_REG_SYNC
            wait_min_stream(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, BENCH_STREAM_REG_OUTPUT_CONSUMED_REG_INDEX, generation - num_pages);
#else
            wait_min_reg(output_consumed_reg, generation - num_pages);
#endif
        }
#endif
        tile_regs_wait();
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT_PACK("rtadd pack wait done i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
        static_pack_tile(kDstReg, dst_l1_addr);
        tile_regs_release();
#if defined(TRISC_PACK)
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT_PACK("rtadd pack publish i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
#if BENCH_USE_STREAM_REG_SYNC
        publish_after_pack_stream(generation);
#else
        publish_after_pack(input_consumed_reg, output_ready_reg, generation);
        input1_consumed_reg[0] = generation;
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT_PACK("rtadd pack published i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
#endif
    }
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN("RTADD_CB_COMPUTE_UNPACK");
#elif defined(TRISC_MATH)
    DeviceZoneScopedN("RTADD_CB_COMPUTE_MATH");
#elif defined(TRISC_PACK)
    DeviceZoneScopedN("RTADD_CB_COMPUTE_PACK");
#endif

    binary_op_init_common(kCbIn0, kCbIn1, kCbOut);
    add_tiles_init(kCbIn0, kCbIn1);

    for (uint32_t i = 0; i < iterations; ++i) {
        cb_wait_front(kCbIn0, kOneTile);
        cb_wait_front(kCbIn1, kOneTile);

        tile_regs_acquire();
        add_tiles(kCbIn0, kCbIn1, 0, 0, kDstReg);
        tile_regs_commit();

        cb_reserve_back(kCbOut, kOneTile);
        tile_regs_wait();
        pack_tile(kDstReg, kCbOut);
        tile_regs_release();

        cb_push_back(kCbOut, kOneTile);
        cb_pop_front(kCbIn0, kOneTile);
        cb_pop_front(kCbIn1, kOneTile);
    }
#endif
}
