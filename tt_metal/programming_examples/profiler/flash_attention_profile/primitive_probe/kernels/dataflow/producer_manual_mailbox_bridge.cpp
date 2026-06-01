// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

void kernel_main() {
    const uint32_t mailbox_addr = get_arg_val<uint32_t>(0);
    const uint32_t mailbox_magic = get_arg_val<uint32_t>(1);
    const uint32_t max_loops = get_arg_val<uint32_t>(2);
    const uint32_t consumer_x = get_arg_val<uint32_t>(3);
    const uint32_t consumer_y = get_arg_val<uint32_t>(4);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(5);
    const uint32_t increment = get_arg_val<uint32_t>(6);

    auto* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);

    uint32_t loops = 0;
    uint32_t ready = 0;
    for (; loops < max_loops; ++loops) {
        const uint32_t magic = mailbox[0];
        ready = mailbox[1];
        if (magic == mailbox_magic && ready == 1) {
            break;
        }
    }

    mailbox[3] = loops;
    if (ready == 1 && mailbox[0] == mailbox_magic) {
        const uint32_t semaphore = get_semaphore(semaphore_id);
        const uint64_t remote_sem_addr = get_noc_addr(consumer_x, consumer_y, semaphore);
        noc_semaphore_inc(remote_sem_addr, increment);
        noc_async_atomic_barrier();
        mailbox[4] = 0xfa310002u;
    } else {
        mailbox[4] = 0xfa31ffffu;
    }
}
