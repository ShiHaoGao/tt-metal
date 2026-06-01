// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

void kernel_main() {
    const uint32_t consumer_x = get_arg_val<uint32_t>(0);
    const uint32_t consumer_y = get_arg_val<uint32_t>(1);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(2);
    const uint32_t increment = get_arg_val<uint32_t>(3);

    const uint32_t semaphore = get_semaphore(semaphore_id);
    const uint64_t remote_sem_addr = get_noc_addr(consumer_x, consumer_y, semaphore);
    noc_semaphore_inc(remote_sem_addr, increment);
    noc_async_atomic_barrier();
}
