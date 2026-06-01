// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

void kernel_main() {
    constexpr uint32_t result_cb = get_compile_time_arg_val(0);

    const uint32_t result_addr = get_arg_val<uint32_t>(0);
    const uint32_t result_bank = get_arg_val<uint32_t>(1);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(2);
    const uint32_t expected_value = get_arg_val<uint32_t>(3);
    const uint32_t max_loops = get_arg_val<uint32_t>(4);
    const uint32_t primitive_id = get_arg_val<uint32_t>(5);

    const uint32_t semaphore = get_semaphore(semaphore_id);
    auto* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);

    uint32_t observed = 0;
    uint32_t loops = 0;
    uint32_t seen = 0;
    for (; loops < max_loops; ++loops) {
        observed = *sem_ptr;
        if (observed >= expected_value) {
            seen = 1;
            break;
        }
    }

    cb_reserve_back(result_cb, 1);
    const uint32_t result_l1 = get_write_ptr(result_cb);
    auto* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1);
    result[0] = seen;
    result[1] = observed;
    result[2] = loops;
    result[3] = primitive_id;
    result[4] = semaphore;
    result[5] = expected_value;
    result[6] = max_loops;
    result[7] = 0xface0001u;

    const uint64_t result_noc_addr = get_noc_addr_from_bank_id<true>(result_bank, result_addr);
    noc_async_write(result_l1, result_noc_addr, 32);
    noc_async_write_barrier();

    cb_push_back(result_cb, 1);
    cb_pop_front(result_cb, 1);
    noc_semaphore_set(sem_ptr, 0);
}
