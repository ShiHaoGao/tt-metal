// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// This is the single SDK-owned image-loading ABI shared by firmware, native
// support producers and the host loader. Emit one record in a non-ALLOC section.
#define TT_NATIVE_IMAGE_SECTION ".tt_native_image"
#define TT_NATIVE_IMAGE_MAGIC UINT32_C(0x544e494d)

// Version 1 also fixes the startup/entry/normal-return contract, not only this
// record's framing. Kernel images are ELF32 little-endian RISC-V executables
// entered at _start; the native support invokes kernel_main and returns to the
// calling firmware after performing its declared per-role startup obligations.
#define TT_NATIVE_IMAGE_ABI_VERSION UINT16_C(1)

enum tt_native_image_kind {
    TT_NATIVE_IMAGE_FIRMWARE = 0,
    TT_NATIVE_IMAGE_KERNEL = 1,
};

enum tt_native_processor_role {
    TT_NATIVE_ROLE_BRISC = 0,
    TT_NATIVE_ROLE_NCRISC = 1,
    TT_NATIVE_ROLE_TRISC0 = 2,
    TT_NATIVE_ROLE_TRISC1 = 3,
    TT_NATIVE_ROLE_TRISC2 = 4,
    TT_NATIVE_ROLE_COUNT = 5,
};

enum tt_native_architecture {
    TT_NATIVE_ARCH_BLACKHOLE = 1,
};

enum tt_native_loading {
    TT_NATIVE_LOADING_DISCRETE = 0,
    TT_NATIVE_LOADING_CONTIGUOUS_XIP = 1,
};

enum tt_native_profile_transport {
    TT_NATIVE_PROFILE_ABSENT = 0,
    TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM = 1,
};

enum tt_native_print_transport {
    TT_NATIVE_PRINT_ABSENT = 0,
    TT_NATIVE_PRINT_TENSIX_SHARED_BUFFER = 1,
};

enum tt_native_worker_stream_owner {
    TT_NATIVE_WORKER_STREAM_SDK = 0,
    TT_NATIVE_WORKER_STREAM_PROGRAM = 1,
};

// All multibyte fields are little-endian. Firmware records describe available
// transports; kernel records describe exact requirements. A firmware capability
// may satisfy an absent kernel requirement, provided actual imported objects and
// transport state are compatible. Record equality is not firmware provenance.
// worker_stream_owner is the kernel's exact launch-state owner; on firmware it
// records the SDK default. A matching worker_stream_abi defines whether firmware
// can preserve Program-owned state under the explicit launch ownership protocol.
typedef struct tt_native_image_record {
    uint32_t magic;
    uint16_t version;
    uint8_t kind;
    uint8_t role;
    uint8_t architecture;
    uint8_t loading;
    uint8_t profile;
    uint8_t print;
    uint16_t worker_stream_abi;
    uint8_t worker_stream_owner;
    uint8_t reserved;
} tt_native_image_record;

#if defined(__cplusplus)
static_assert(sizeof(tt_native_image_record) == 16, "native image ABI size changed");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(tt_native_image_record) == 16, "native image ABI size changed");
#endif
