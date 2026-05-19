// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef BENCH_MODE
#define BENCH_MODE 0
#endif

#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS 10000
#endif

#ifndef BENCH_ROLE
#define BENCH_ROLE 0
#endif

#ifndef BENCH_USE_COMPILE_TIME_ARGS
#define BENCH_USE_COMPILE_TIME_ARGS 0
#endif

#ifndef BENCH_WORK_BASE_ADDR
#define BENCH_WORK_BASE_ADDR 0
#endif

#ifndef BENCH_SINK_L1_ADDR
#define BENCH_SINK_L1_ADDR 0
#endif

#ifndef BENCH_SINK_DRAM_ADDR
#define BENCH_SINK_DRAM_ADDR 0
#endif

#ifndef BENCH_SINK_DRAM_OFFSET
#define BENCH_SINK_DRAM_OFFSET 0
#endif

#ifndef BENCH_INPUT_DRAM_ADDR
#define BENCH_INPUT_DRAM_ADDR 0
#endif

#ifndef BENCH_OUTPUT_DRAM_ADDR
#define BENCH_OUTPUT_DRAM_ADDR 0
#endif

#ifndef BENCH_PAGE_SIZE
#define BENCH_PAGE_SIZE 0
#endif

#ifndef BENCH_NUM_PAGES
#define BENCH_NUM_PAGES 0
#endif

#ifndef BENCH_PRODUCED_SEM_ID
#define BENCH_PRODUCED_SEM_ID 0
#endif

#ifndef BENCH_CONSUMED_SEM_ID
#define BENCH_CONSUMED_SEM_ID 0
#endif

#ifndef BENCH_STREAM_REG_PRODUCED_STREAM_ID
#define BENCH_STREAM_REG_PRODUCED_STREAM_ID 1
#endif

#ifndef BENCH_STREAM_REG_CONSUMED_STREAM_ID
#define BENCH_STREAM_REG_CONSUMED_STREAM_ID 2
#endif

#ifndef BENCH_STREAM_REG_START_STREAM_ID
#define BENCH_STREAM_REG_START_STREAM_ID 0
#endif

#ifndef BENCH_STREAM_REG_VALUE_MASK
#define BENCH_STREAM_REG_VALUE_MASK 0x00ffffffu
#endif

#ifndef BENCH_STREAM_REG_START_VALUE
#define BENCH_STREAM_REG_START_VALUE 1
#endif

#ifndef BENCH_STREAM_SYNC_REG_INDEX
#ifdef STREAM_SCRATCH32_REG_INDEX
#define BENCH_STREAM_SYNC_REG_INDEX STREAM_SCRATCH32_REG_INDEX
#else
#define BENCH_STREAM_SYNC_REG_INDEX STREAM_SCRATCH_1_REG_INDEX
#endif
#endif

namespace {

constexpr uint32_t kCbIndex = 0;
constexpr uint32_t kL1Alignment = 64;

inline void store_sink(uint32_t sink_l1_addr, uint32_t value) {
    volatile tt_l1_ptr uint32_t* sink = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sink_l1_addr);
    sink[0] = value;
}

inline volatile tt_l1_ptr uint32_t* l1_u32(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
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

inline uint32_t produced_counter_addr(uint32_t work_base_addr, uint32_t page_size, uint32_t num_pages) {
    return work_base_addr + page_size * num_pages;
}

inline uint32_t consumed_counter_addr(uint32_t work_base_addr, uint32_t page_size, uint32_t num_pages) {
    return produced_counter_addr(work_base_addr, page_size, num_pages) + kL1Alignment;
}

}  // namespace

void kernel_main() {
#if BENCH_USE_COMPILE_TIME_ARGS
    constexpr uint32_t work_base_addr = BENCH_WORK_BASE_ADDR;
    constexpr uint32_t sink_l1_addr = BENCH_SINK_L1_ADDR;
    constexpr uint32_t sink_dram_addr = BENCH_SINK_DRAM_ADDR;
    constexpr uint32_t sink_dram_offset = BENCH_SINK_DRAM_OFFSET;
    constexpr uint32_t input_dram_addr = BENCH_INPUT_DRAM_ADDR;
    constexpr uint32_t output_dram_addr = BENCH_OUTPUT_DRAM_ADDR;
    constexpr uint32_t page_size = BENCH_PAGE_SIZE;
    constexpr uint32_t num_pages = BENCH_NUM_PAGES;
#else
    const uint32_t work_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t sink_l1_addr = get_arg_val<uint32_t>(1);
    const uint32_t sink_dram_addr = get_arg_val<uint32_t>(2);
    const uint32_t sink_dram_offset = get_arg_val<uint32_t>(3);
    const uint32_t page_size = get_arg_val<uint32_t>(4);
    const uint32_t num_pages = get_arg_val<uint32_t>(5);
    const uint32_t input_dram_addr = get_arg_val<uint32_t>(8);
    const uint32_t output_dram_addr = get_arg_val<uint32_t>(9);
#endif

    uint32_t sink = 0x12345678;

#if BENCH_MODE == 0
    {
        DeviceZoneScopedN("CBP_EMPTY");
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            asm volatile("" ::: "memory");
            sink ^= i + 0x9e3779b9u;
        }
    }
#elif BENCH_MODE == 1
    {
        DeviceZoneScopedN("CBP_GET_WRITE_PTR");
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            asm volatile("" ::: "memory");
            sink ^= get_write_ptr(kCbIndex) + i;
        }
    }
#elif BENCH_MODE == 2
    {
        DeviceZoneScopedN("CBP_GET_READ_WRITE_PTR");
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            asm volatile("" ::: "memory");
            sink ^= get_write_ptr(kCbIndex);
            sink += get_read_ptr(kCbIndex) ^ i;
        }
    }
#elif BENCH_MODE == 3
    {
        DeviceZoneScopedN("CBP_GET_TILE_SIZE");
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            asm volatile("" ::: "memory");
            sink += get_tile_size(kCbIndex) ^ i;
        }
    }
#elif BENCH_MODE == 4
    {
        DeviceZoneScopedN("CBP_API_ROUNDTRIP");
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            cb_reserve_back(kCbIndex, 1);
            volatile tt_l1_ptr uint32_t* write_ptr = l1_u32(get_write_ptr(kCbIndex));
            write_ptr[0] = sink ^ i;
            cb_push_back(kCbIndex, 1);

            cb_wait_front(kCbIndex, 1);
            volatile tt_l1_ptr uint32_t* read_ptr = l1_u32(get_read_ptr(kCbIndex));
            sink += read_ptr[0] ^ i;
            cb_pop_front(kCbIndex, 1);
        }
    }
#elif BENCH_MODE == 5
    {
        DeviceZoneScopedN("CBP_STATIC_RING");
        uint32_t slot = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            const uint32_t addr = work_base_addr + slot * page_size;
            volatile tt_l1_ptr uint32_t* page = l1_u32(addr);
            page[0] = sink ^ i;
            sink += page[0] + slot;

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
    }
#elif BENCH_MODE == 6
    {
        volatile tt_l1_ptr uint32_t* produced_ptr = l1_u32(produced_counter_addr(work_base_addr, page_size, num_pages));
        volatile tt_l1_ptr uint32_t* consumed_ptr = l1_u32(consumed_counter_addr(work_base_addr, page_size, num_pages));
        produced_ptr[0] = 0;
        consumed_ptr[0] = 0;

        DeviceZoneScopedN("CBP_STATIC_COUNTER");
        uint32_t slot = 0;
        uint32_t produced = 0;
        uint32_t consumed = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            while ((produced - consumed) >= num_pages) {
                invalidate_l1_cache();
                consumed = consumed_ptr[0];
            }

            const uint32_t addr = work_base_addr + slot * page_size;
            volatile tt_l1_ptr uint32_t* page = l1_u32(addr);
            page[0] = sink ^ i;
            ++produced;
            produced_ptr[0] = produced;

            while ((produced_ptr[0] - consumed) == 0) {
                invalidate_l1_cache();
            }

            sink += page[0] + slot;
            ++consumed;
            consumed_ptr[0] = consumed;

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
    }
#elif BENCH_MODE == 7
    {
#if BENCH_ROLE == 0
        DeviceZoneScopedN("CBP_CROSS_EMPTY_PRODUCER");
#else
        DeviceZoneScopedN("CBP_CROSS_EMPTY_CONSUMER");
#endif
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            asm volatile("" ::: "memory");
            sink += i ^ static_cast<uint32_t>(BENCH_ROLE);
        }
    }
#elif BENCH_MODE == 8 || BENCH_MODE == 10
    {
#if BENCH_ROLE == 0
#if BENCH_MODE == 8
        DeviceZoneScopedN("CBP_CROSS_CB_PRODUCER");
#else
        DeviceZoneScopedN("CBP_SYSTEM_CB_PRODUCER");
#endif
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            cb_reserve_back(kCbIndex, 1);
            volatile tt_l1_ptr uint32_t* page = l1_u32(get_write_ptr(kCbIndex));
            page[0] = i;
            checksum += i;
            cb_push_back(kCbIndex, 1);
        }
        sink = checksum;
#else
#if BENCH_MODE == 8
        DeviceZoneScopedN("CBP_CROSS_CB_CONSUMER");
#else
        DeviceZoneScopedN("CBP_SYSTEM_CB_CONSUMER");
#endif
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            cb_wait_front(kCbIndex, 1);
            volatile tt_l1_ptr uint32_t* page = l1_u32(get_read_ptr(kCbIndex));
            checksum += page[0];
            cb_pop_front(kCbIndex, 1);
        }
        sink = checksum;
#endif
    }
#elif BENCH_MODE == 9 || BENCH_MODE == 11 || BENCH_MODE == 12
    {
#if BENCH_USE_COMPILE_TIME_ARGS
        volatile tt_l1_ptr uint32_t* produced_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(BENCH_PRODUCED_SEM_ID));
        volatile tt_l1_ptr uint32_t* consumed_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(BENCH_CONSUMED_SEM_ID));
#else
        volatile tt_l1_ptr uint32_t* produced_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(6)));
        volatile tt_l1_ptr uint32_t* consumed_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(7)));
#endif

#if BENCH_ROLE == 0
#if BENCH_MODE == 9
        DeviceZoneScopedN("CBP_CROSS_STATIC_PRODUCER");
#elif BENCH_MODE == 11
        DeviceZoneScopedN("CBP_SYSTEM_STATIC_RUNTIME_PRODUCER");
#else
        DeviceZoneScopedN("CBP_SYSTEM_STATIC_COMPILETIME_PRODUCER");
#endif
        uint32_t produced = 0;
        uint32_t consumed = 0;
        uint32_t slot = 0;
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            while ((produced - consumed) >= num_pages) {
                const uint32_t target_consumed = produced - num_pages + 1;
                noc_semaphore_wait_min(consumed_ptr, target_consumed);
                consumed = consumed_ptr[0];
            }

            volatile tt_l1_ptr uint32_t* page = l1_u32(work_base_addr + slot * page_size);
            page[0] = i;
            asm volatile("fence" ::: "memory");
            checksum += i;
            ++produced;
            noc_semaphore_set(produced_ptr, produced);

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
        sink = checksum;
#else
#if BENCH_MODE == 9
        DeviceZoneScopedN("CBP_CROSS_STATIC_CONSUMER");
#elif BENCH_MODE == 11
        DeviceZoneScopedN("CBP_SYSTEM_STATIC_RUNTIME_CONSUMER");
#else
        DeviceZoneScopedN("CBP_SYSTEM_STATIC_COMPILETIME_CONSUMER");
#endif
        uint32_t produced = 0;
        uint32_t consumed = 0;
        uint32_t slot = 0;
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            while ((produced - consumed) == 0) {
                const uint32_t target_produced = consumed + 1;
                noc_semaphore_wait_min(produced_ptr, target_produced);
                produced = produced_ptr[0];
            }

            volatile tt_l1_ptr uint32_t* page = l1_u32(work_base_addr + slot * page_size);
            invalidate_l1_cache();
            checksum += page[0];
            ++consumed;
            noc_semaphore_set(consumed_ptr, consumed);

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
        sink = checksum;
#endif
    }
#elif BENCH_MODE == 13
    {
        DeviceZoneScopedN("CBP_SYSTEM_STATIC_NOSYNC");
        uint32_t slot = 0;
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            volatile tt_l1_ptr uint32_t* page = l1_u32(work_base_addr + slot * page_size);
            page[0] = i;
            checksum += page[0] + slot;

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
        sink = checksum;
    }
#elif BENCH_MODE == 14
    {
        InterleavedAddrGen<true> input_addrgen = {.bank_base_address = input_dram_addr, .page_size = page_size};
        InterleavedAddrGen<true> output_addrgen = {.bank_base_address = output_dram_addr, .page_size = page_size};
#if BENCH_ROLE == 0
        DeviceZoneScopedN("CBP_DRAM_CB_PRODUCER");
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            cb_reserve_back(kCbIndex, 1);
            const uint32_t l1_write_addr = get_write_ptr(kCbIndex);
            noc_async_read(get_noc_addr(i, input_addrgen), l1_write_addr, page_size);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* page = l1_u32(l1_write_addr);
            checksum += page[0];
            cb_push_back(kCbIndex, 1);
        }
        sink = checksum;
#else
        DeviceZoneScopedN("CBP_DRAM_CB_CONSUMER");
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            cb_wait_front(kCbIndex, 1);
            const uint32_t l1_read_addr = get_read_ptr(kCbIndex);
            volatile tt_l1_ptr uint32_t* page = l1_u32(l1_read_addr);
            checksum += page[0];
            noc_async_write(l1_read_addr, get_noc_addr(i, output_addrgen), page_size);
            noc_async_write_barrier();
            cb_pop_front(kCbIndex, 1);
        }
        sink = checksum;
#endif
    }
#elif BENCH_MODE == 15 || BENCH_MODE == 16
    {
        InterleavedAddrGen<true> input_addrgen = {.bank_base_address = input_dram_addr, .page_size = page_size};
        InterleavedAddrGen<true> output_addrgen = {.bank_base_address = output_dram_addr, .page_size = page_size};
#if BENCH_USE_COMPILE_TIME_ARGS
        volatile tt_l1_ptr uint32_t* produced_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(BENCH_PRODUCED_SEM_ID));
        volatile tt_l1_ptr uint32_t* consumed_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(BENCH_CONSUMED_SEM_ID));
#else
        volatile tt_l1_ptr uint32_t* produced_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(6)));
        volatile tt_l1_ptr uint32_t* consumed_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(7)));
#endif

#if BENCH_ROLE == 0
#if BENCH_MODE == 15
        DeviceZoneScopedN("CBP_DRAM_STATIC_RUNTIME_PRODUCER");
#else
        DeviceZoneScopedN("CBP_DRAM_STATIC_COMPILETIME_PRODUCER");
#endif
        uint32_t produced = 0;
        uint32_t consumed = 0;
        uint32_t slot = 0;
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            while ((produced - consumed) >= num_pages) {
                const uint32_t target_consumed = produced - num_pages + 1;
                noc_semaphore_wait_min(consumed_ptr, target_consumed);
                consumed = consumed_ptr[0];
            }

            const uint32_t l1_write_addr = work_base_addr + slot * page_size;
            noc_async_read(get_noc_addr(i, input_addrgen), l1_write_addr, page_size);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* page = l1_u32(l1_write_addr);
            checksum += page[0];
            noc_semaphore_set(produced_ptr, produced + 1);
            ++produced;

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
        sink = checksum;
#else
#if BENCH_MODE == 15
        DeviceZoneScopedN("CBP_DRAM_STATIC_RUNTIME_CONSUMER");
#else
        DeviceZoneScopedN("CBP_DRAM_STATIC_COMPILETIME_CONSUMER");
#endif
        uint32_t produced = 0;
        uint32_t consumed = 0;
        uint32_t slot = 0;
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            while ((produced - consumed) == 0) {
                const uint32_t target_produced = consumed + 1;
                noc_semaphore_wait_min(produced_ptr, target_produced);
                produced = produced_ptr[0];
            }

            const uint32_t l1_read_addr = work_base_addr + slot * page_size;
            volatile tt_l1_ptr uint32_t* page = l1_u32(l1_read_addr);
            invalidate_l1_cache();
            checksum += page[0];
            noc_async_write(l1_read_addr, get_noc_addr(i, output_addrgen), page_size);
            noc_async_write_barrier();
            noc_semaphore_set(consumed_ptr, consumed + 1);
            ++consumed;

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
        sink = checksum;
#endif
    }
#elif BENCH_MODE == 17
    {
        DeviceZoneScopedN("CBP_DRAM_SINGLE_NOSYNC");
        InterleavedAddrGen<true> input_addrgen = {.bank_base_address = input_dram_addr, .page_size = page_size};
        InterleavedAddrGen<true> output_addrgen = {.bank_base_address = output_dram_addr, .page_size = page_size};
        uint32_t slot = 0;
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            const uint32_t l1_addr = work_base_addr + slot * page_size;
            noc_async_read(get_noc_addr(i, input_addrgen), l1_addr, page_size);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* page = l1_u32(l1_addr);
            checksum += page[0];
            noc_async_write(l1_addr, get_noc_addr(i, output_addrgen), page_size);
            noc_async_write_barrier();

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
        sink = checksum;
    }
#elif BENCH_MODE == 18 || BENCH_MODE == 19 || BENCH_MODE == 20
    {
        InterleavedAddrGen<true> input_addrgen = {.bank_base_address = input_dram_addr, .page_size = page_size};
        InterleavedAddrGen<true> output_addrgen = {.bank_base_address = output_dram_addr, .page_size = page_size};
#if BENCH_ROLE == 0
#if BENCH_MODE == 18
        DeviceZoneScopedN("CBP_CROSS_STREAMREG_PRODUCER");
#elif BENCH_MODE == 19
        DeviceZoneScopedN("CBP_SYSTEM_STREAMREG_PRODUCER");
#else
        DeviceZoneScopedN("CBP_DRAM_STREAMREG_PRODUCER");
#endif
        set_stream_sync(BENCH_STREAM_REG_PRODUCED_STREAM_ID, 0);
        set_stream_sync(BENCH_STREAM_REG_CONSUMED_STREAM_ID, 0);
        set_stream_sync(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_VALUE);
        uint32_t produced = 0;
        uint32_t consumed = 0;
        uint32_t slot = 0;
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            while ((produced - consumed) >= num_pages) {
                const uint32_t target_consumed = produced - num_pages + 1;
                wait_min_stream(BENCH_STREAM_REG_CONSUMED_STREAM_ID, target_consumed);
                consumed = read_stream_sync(BENCH_STREAM_REG_CONSUMED_STREAM_ID);
            }

            const uint32_t l1_write_addr = work_base_addr + slot * page_size;
#if BENCH_MODE == 20
            noc_async_read(get_noc_addr(i, input_addrgen), l1_write_addr, page_size);
            noc_async_read_barrier();
#endif
            volatile tt_l1_ptr uint32_t* page = l1_u32(l1_write_addr);
#if BENCH_MODE != 20
            page[0] = i;
#endif
            checksum += page[0] ^ i;
            set_stream_sync(BENCH_STREAM_REG_PRODUCED_STREAM_ID, produced + 1);
            ++produced;

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
        sink = checksum;
#else
#if BENCH_MODE == 18
        DeviceZoneScopedN("CBP_CROSS_STREAMREG_CONSUMER");
#elif BENCH_MODE == 19
        DeviceZoneScopedN("CBP_SYSTEM_STREAMREG_CONSUMER");
#else
        DeviceZoneScopedN("CBP_DRAM_STREAMREG_CONSUMER");
#endif
        wait_equal_stream(BENCH_STREAM_REG_START_STREAM_ID, BENCH_STREAM_REG_START_VALUE);
        uint32_t produced = 0;
        uint32_t consumed = 0;
        uint32_t slot = 0;
        uint32_t checksum = 0;
        for (uint32_t i = 0; i < BENCH_ITERATIONS; ++i) {
            while ((produced - consumed) == 0) {
                const uint32_t target_produced = consumed + 1;
                wait_min_stream(BENCH_STREAM_REG_PRODUCED_STREAM_ID, target_produced);
                produced = read_stream_sync(BENCH_STREAM_REG_PRODUCED_STREAM_ID);
            }

            const uint32_t l1_read_addr = work_base_addr + slot * page_size;
            volatile tt_l1_ptr uint32_t* page = l1_u32(l1_read_addr);
            invalidate_l1_cache();
            checksum += page[0];
#if BENCH_MODE == 20
            noc_async_write(l1_read_addr, get_noc_addr(i, output_addrgen), page_size);
            noc_async_write_barrier();
#endif
            set_stream_sync(BENCH_STREAM_REG_CONSUMED_STREAM_ID, consumed + 1);
            ++consumed;

            ++slot;
            if (slot == num_pages) {
                slot = 0;
            }
        }
        sink = checksum;
#endif
    }
#else
#error "Unsupported BENCH_MODE"
#endif

    store_sink(sink_l1_addr, sink);
    const uint64_t sink_noc_addr = get_noc_addr_from_bank_id<true>(0, sink_dram_addr + sink_dram_offset);
    noc_async_write(sink_l1_addr, sink_noc_addr, sizeof(uint32_t));
    noc_async_write_barrier();
}
