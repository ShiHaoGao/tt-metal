// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"

#include <cstdint>

void kernel_main() {
#if defined(TRISC_MATH)
    constexpr uint32_t mailbox_addr = get_compile_time_arg_val(0);
    constexpr uint32_t mailbox_magic = get_compile_time_arg_val(1);

    auto* mailbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);
    mailbox[0] = mailbox_magic;
    mailbox[2] = 0xfa310101u;
    asm volatile("fence" ::: "memory");
    mailbox[1] = 1;
#endif
}
