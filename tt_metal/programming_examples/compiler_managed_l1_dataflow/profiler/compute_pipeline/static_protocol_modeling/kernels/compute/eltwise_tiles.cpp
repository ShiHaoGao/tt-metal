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

#ifndef BENCH_STATIC_PROTOCOL
#define BENCH_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_SERIAL_STATIC_PROTOCOL
#define BENCH_SERIAL_STATIC_PROTOCOL 0
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

#ifndef BENCH_TRACE_STATIC_PROTOCOL
#define BENCH_TRACE_STATIC_PROTOCOL 0
#endif

#ifndef BENCH_PROFILE_CASE_LABELS
#define BENCH_PROFILE_CASE_LABELS 0
#endif

#ifndef BENCH_OP_ELTWISE_CHAIN
#define BENCH_OP_ELTWISE_CHAIN 0
#endif

#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS 1
#endif

#ifndef BENCH_CHAIN_DEPTH
#define BENCH_CHAIN_DEPTH 1
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

constexpr uint32_t kCbIn0 = tt::CBIndex::c_0;
constexpr uint32_t kCbIn1 = tt::CBIndex::c_1;
constexpr uint32_t kCbOut = tt::CBIndex::c_16;
constexpr uint32_t kDstReg = 0;
constexpr uint32_t kOneTile = 1;

#define SPM_STRINGIZE_IMPL(x) #x
#define SPM_STRINGIZE(x) SPM_STRINGIZE_IMPL(x)

#if BENCH_OP_ELTWISE_CHAIN
#define SPM_OP_PREFIX "SPM_ELTWISE_CHAIN"
#else
#define SPM_OP_PREFIX "SPM_TILE_ADD"
#endif

#if BENCH_PROFILE_CASE_LABELS
#define SPM_CASE_PREFIX \
    SPM_OP_PREFIX "_T" SPM_STRINGIZE(BENCH_ITERATIONS) "_C" SPM_STRINGIZE(BENCH_CHAIN_DEPTH) "_S" \
        SPM_STRINGIZE(BENCH_NUM_PAGES)
#else
#define SPM_CASE_PREFIX SPM_OP_PREFIX
#endif

#define SPM_ZONE_STATIC_COMPILETIME_COMPUTE_UNPACK SPM_CASE_PREFIX "_STATIC_COMPILETIME_COMPUTE_UNPACK"
#define SPM_ZONE_STATIC_COMPILETIME_COMPUTE_MATH SPM_CASE_PREFIX "_STATIC_COMPILETIME_COMPUTE_MATH"
#define SPM_ZONE_STATIC_COMPILETIME_COMPUTE_PACK SPM_CASE_PREFIX "_STATIC_COMPILETIME_COMPUTE_PACK"
#define SPM_ZONE_STATIC_RUNTIME_COMPUTE_UNPACK SPM_CASE_PREFIX "_STATIC_RUNTIME_COMPUTE_UNPACK"
#define SPM_ZONE_STATIC_RUNTIME_COMPUTE_MATH SPM_CASE_PREFIX "_STATIC_RUNTIME_COMPUTE_MATH"
#define SPM_ZONE_STATIC_RUNTIME_COMPUTE_PACK SPM_CASE_PREFIX "_STATIC_RUNTIME_COMPUTE_PACK"
#define SPM_ZONE_STATIC_SERIALIZED_COMPUTE_UNPACK SPM_CASE_PREFIX "_STATIC_SERIALIZED_COMPUTE_UNPACK"
#define SPM_ZONE_STATIC_SERIALIZED_COMPUTE_MATH SPM_CASE_PREFIX "_STATIC_SERIALIZED_COMPUTE_MATH"
#define SPM_ZONE_STATIC_SERIALIZED_COMPUTE_PACK SPM_CASE_PREFIX "_STATIC_SERIALIZED_COMPUTE_PACK"
#define SPM_ZONE_STATIC_STREAMREG_COMPUTE_UNPACK SPM_CASE_PREFIX "_STATIC_STREAMREG_COMPUTE_UNPACK"
#define SPM_ZONE_STATIC_STREAMREG_COMPUTE_MATH SPM_CASE_PREFIX "_STATIC_STREAMREG_COMPUTE_MATH"
#define SPM_ZONE_STATIC_STREAMREG_COMPUTE_PACK SPM_CASE_PREFIX "_STATIC_STREAMREG_COMPUTE_PACK"
#define SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPUTE_UNPACK SPM_CASE_PREFIX "_STATIC_STREAMREG_CBREGS_COMPUTE_UNPACK"
#define SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPUTE_MATH SPM_CASE_PREFIX "_STATIC_STREAMREG_CBREGS_COMPUTE_MATH"
#define SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPUTE_PACK SPM_CASE_PREFIX "_STATIC_STREAMREG_CBREGS_COMPUTE_PACK"
#define SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_UNPACK \
    SPM_CASE_PREFIX "_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_UNPACK"
#define SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_MATH \
    SPM_CASE_PREFIX "_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_MATH"
#define SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_PACK \
    SPM_CASE_PREFIX "_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_PACK"
#define SPM_ZONE_CB_COMPUTE_UNPACK SPM_CASE_PREFIX "_CB_COMPUTE_UNPACK"
#define SPM_ZONE_CB_COMPUTE_MATH SPM_CASE_PREFIX "_CB_COMPUTE_MATH"
#define SPM_ZONE_CB_COMPUTE_PACK SPM_CASE_PREFIX "_CB_COMPUTE_PACK"

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

inline void set_reg(volatile tt_reg_ptr uint32_t* reg, uint32_t value) {
    reg[0] = value;
}

inline volatile tt_l1_ptr uint32_t* tensix_store_ptr(volatile tt_reg_ptr uint32_t* reg) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ((reinterpret_cast<uint32_t>(reg) >> 2) & 0x3ffff));
}

inline volatile tt_reg_ptr uint32_t* reg_ptr_from_cb(uint32_t cbid, bool received) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        received ? get_cb_tiles_received_ptr(cbid) : get_cb_tiles_acked_ptr(cbid));
}

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
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID, value);
    set_stream_sync(BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID, value);
    set_stream_sync(BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID, value);
#endif
}

inline volatile tt_l1_ptr uint32_t* sem_slot(uint32_t sem_base_addr, uint32_t slot) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_base_addr + slot * BENCH_SEM_SLOT_BYTES);
}

inline void raw_tile_add_init(uint32_t /*page_size_bytes*/) {
    binary_op_init_common(kCbIn0, kCbIn1, kCbOut);
    add_tiles_init(kCbIn0, kCbIn1);
}

inline uint32_t to_cb_addr(uint32_t l1_addr) {
    return l1_addr >> CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
}

inline void static_add_tile(uint32_t src0_l1_addr, uint32_t src1_l1_addr, uint32_t dst_reg) {
#ifdef TRISC_UNPACK
    get_local_cb_interface(kCbIn0).fifo_rd_ptr = to_cb_addr(src0_l1_addr);
    get_local_cb_interface(kCbIn1).fifo_rd_ptr = to_cb_addr(src1_l1_addr);
#endif
    add_tiles(kCbIn0, kCbIn1, 0, 0, dst_reg);
}

inline void static_pack_tile(uint32_t dst_reg, uint32_t dst_l1_addr) {
#ifdef TRISC_PACK
    get_local_cb_interface(kCbOut).fifo_wr_ptr = to_cb_addr(dst_l1_addr);
    get_local_cb_interface(kCbOut).fifo_wr_tile_ptr = 0;
    pack_tile(dst_reg, kCbOut);
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
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    volatile tt_l1_ptr uint32_t* protocol_start_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(protocol_start_sem_addr);
    volatile tt_reg_ptr uint32_t* input_ready_reg = reg_ptr_from_cb(kCbIn0, true);
    volatile tt_reg_ptr uint32_t* input1_ready_reg = reg_ptr_from_cb(kCbIn1, true);
    volatile tt_reg_ptr uint32_t* input_consumed_reg = reg_ptr_from_cb(kCbIn0, false);
    volatile tt_reg_ptr uint32_t* input1_consumed_reg = reg_ptr_from_cb(kCbIn1, false);
    volatile tt_reg_ptr uint32_t* output_ready_reg = reg_ptr_from_cb(kCbOut, true);
    volatile tt_reg_ptr uint32_t* output_consumed_reg = reg_ptr_from_cb(kCbOut, false);

#if BENCH_USE_STREAM_REG_CBREGS
    wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_PROTOCOL_START_VALUE);
#else
    wait_equal_local(protocol_start_sem, BENCH_PROTOCOL_START_VALUE);
#endif
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
    DEVICE_PRINT_UNPACK("spm compute unpack start value={} slots={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
    DEVICE_PRINT_MATH("spm compute math start value={} slots={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
    DEVICE_PRINT_PACK("spm compute pack start value={} slots={}\n", BENCH_PROTOCOL_START_VALUE, num_pages);
#endif

#if BENCH_USE_STREAM_REG_CBREGS && BENCH_USE_COMPILE_TIME_ARGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_UNPACK);
#elif defined(TRISC_MATH)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_MATH);
#elif defined(TRISC_PACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPILETIME_COMPUTE_PACK);
#endif
#elif BENCH_USE_COMPILE_TIME_ARGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_COMPILETIME_COMPUTE_UNPACK);
#elif defined(TRISC_MATH)
    DeviceZoneScopedN(SPM_ZONE_STATIC_COMPILETIME_COMPUTE_MATH);
#elif defined(TRISC_PACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_COMPILETIME_COMPUTE_PACK);
#endif
#elif BENCH_SERIAL_STATIC_PROTOCOL
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_SERIALIZED_COMPUTE_UNPACK);
#elif defined(TRISC_MATH)
    DeviceZoneScopedN(SPM_ZONE_STATIC_SERIALIZED_COMPUTE_MATH);
#elif defined(TRISC_PACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_SERIALIZED_COMPUTE_PACK);
#endif
#elif BENCH_USE_STREAM_REG_CBREGS
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPUTE_UNPACK);
#elif defined(TRISC_MATH)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPUTE_MATH);
#elif defined(TRISC_PACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_CBREGS_COMPUTE_PACK);
#endif
#elif BENCH_USE_STREAM_REG_SYNC
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_COMPUTE_UNPACK);
#elif defined(TRISC_MATH)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_COMPUTE_MATH);
#elif defined(TRISC_PACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_STREAMREG_COMPUTE_PACK);
#endif
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_RUNTIME_COMPUTE_UNPACK);
#elif defined(TRISC_MATH)
    DeviceZoneScopedN(SPM_ZONE_STATIC_RUNTIME_COMPUTE_MATH);
#elif defined(TRISC_PACK)
    DeviceZoneScopedN(SPM_ZONE_STATIC_RUNTIME_COMPUTE_PACK);
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
            DEVICE_PRINT_UNPACK("spm unpack wait input i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
#if BENCH_USE_STREAM_REG_SYNC
        wait_equal_stream(BENCH_STREAM_REG_INPUT_READY0_STREAM_ID, generation);
        wait_equal_stream(BENCH_STREAM_REG_INPUT_READY1_STREAM_ID, generation);
#else
        wait_equal_reg(input_ready_reg, generation);
        wait_equal_reg(input1_ready_reg, generation);
#endif
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT_UNPACK("spm unpack got input i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
#endif

        tile_regs_acquire();
        for (uint32_t chain = 0; chain < BENCH_CHAIN_DEPTH; ++chain) {
            static_add_tile(src0_l1_addr, src1_l1_addr, kDstReg);
        }
        tile_regs_commit();

#if defined(TRISC_PACK)
        if (i >= num_pages) {
#if BENCH_USE_STREAM_REG_SYNC
            wait_min_stream(BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID, generation - num_pages);
#else
            wait_min_reg(output_consumed_reg, generation - num_pages);
#endif
        }
#endif
        tile_regs_wait();
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT_PACK("spm pack wait done i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
        static_pack_tile(kDstReg, dst_l1_addr);
        tile_regs_release();
#if defined(TRISC_PACK)
#if BENCH_TRACE_STATIC_PROTOCOL
        if (i < 4) {
            DEVICE_PRINT_PACK("spm pack publish i={} slot={} gen={}\n", i, slot, generation);
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
            DEVICE_PRINT_PACK("spm pack published i={} slot={} gen={}\n", i, slot, generation);
        }
#endif
#endif
    }
#else
#if defined(TRISC_UNPACK)
    DeviceZoneScopedN(SPM_ZONE_CB_COMPUTE_UNPACK);
#elif defined(TRISC_MATH)
    DeviceZoneScopedN(SPM_ZONE_CB_COMPUTE_MATH);
#elif defined(TRISC_PACK)
    DeviceZoneScopedN(SPM_ZONE_CB_COMPUTE_PACK);
#endif

    binary_op_init_common(kCbIn0, kCbIn1, kCbOut);
    add_tiles_init(kCbIn0, kCbIn1);

    for (uint32_t i = 0; i < iterations; ++i) {
        cb_wait_front(kCbIn0, kOneTile);
        cb_wait_front(kCbIn1, kOneTile);

        tile_regs_acquire();
        for (uint32_t chain = 0; chain < BENCH_CHAIN_DEPTH; ++chain) {
            add_tiles(kCbIn0, kCbIn1, 0, 0, kDstReg);
        }
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
