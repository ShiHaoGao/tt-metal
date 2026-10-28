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
// entered at _start; architecture and role select the entry calling convention.
// Wormhole NCRISC restores the firmware IRAM frame before kernel_launch, while
// other roles use the ordinary firmware call. Native support invokes kernel_main
// and returns after performing its declared per-role startup obligations.
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
    TT_NATIVE_ARCH_WORMHOLE = 2,
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

// Independent algorithm qualification, emitted only after the compiler has
// verified the actual role program. Reusable startup/observation support must
// not emit it. The complete linked ELF retains it outside every PT_LOAD.
#define TT_NATIVE_NOC_SECTION ".tt_native_noc"
#define TT_NATIVE_NOC_MAGIC UINT32_C(0x434f4e54)
#define TT_NATIVE_NOC_ABI_VERSION UINT16_C(3)
#define TT_NATIVE_NOC_HEADER_BYTES UINT16_C(32)
// Blackhole noc_parameters.h defines NOC_ADDR_NODE_ID_BITS as six. Keep
// device-only register macros out of the host image-loading ABI.
#define TT_NATIVE_NOC_COORDINATE_LIMIT UINT16_C(64)
#define TT_NATIVE_NOC_COUNT UINT16_C(2)
#define TT_NATIVE_NOC_MAX_ENDPOINTS UINT16_C(8192)
#define TT_NATIVE_NOC_WINDOW_BYTES UINT16_C(24)
#define TT_NATIVE_NOC_MAX_WINDOWS UINT16_C(1024)
typedef struct tt_native_noc_endpoint {
    uint8_t noc;
    uint8_t x;
    uint8_t y;
    uint8_t reserved;
} tt_native_noc_endpoint;

// Static algorithm classification only. Even the narrower class establishes
// neither issue/wait coverage nor live completion, payload visibility, consumer
// last-use, backing ownership, or permission to retire native resources.
enum tt_native_noc_payload_contract {
    TT_NATIVE_NOC_PAYLOAD_UNCLASSIFIED = 0,
    TT_NATIVE_NOC_PAYLOAD_WORKER_L1_NONPOSTED = 1,
};

enum tt_native_noc_window_access {
    TT_NATIVE_NOC_WINDOW_READ = 1,
    TT_NATIVE_NOC_WINDOW_WRITE = 2,
    TT_NATIVE_NOC_WINDOW_READ_WRITE = 3,
};

enum tt_native_noc_window_addressing {
    TT_NATIVE_NOC_WINDOW_ISSUER_LOCAL = 1,
    TT_NATIVE_NOC_WINDOW_RUNTIME_ENDPOINT = 2,
};

// Relative bounds are a compiler declaration, not an allocation receipt. The
// original live invocation must bind every window to its actual allocation
// and exact runtime argument tuple before it can authorize issue.
typedef struct tt_native_noc_window {
    uint32_t formal;
    // Flattened word offset of this final formal's three-word (x,y,address)
    // L1 endpoint ABI. This is distinct from the formal ordinal above.
    uint32_t abi_word_offset;
    uint32_t relative_begin;
    uint32_t relative_end;
    uint32_t base_alignment;
    uint8_t access;
    // Local accesses use the issuing Worker even when the formal's tuple
    // encodes another endpoint. Remote accesses use that tuple's endpoint.
    uint8_t addressing;
    // Zero is permitted only for issuer-local accesses. Bits 0/1 select NoCs.
    uint8_t noc_mask;
    uint8_t reserved;
} tt_native_noc_window;

typedef struct tt_native_noc_record {
    uint32_t magic;
    uint16_t version;
    uint16_t record_bytes;
    uint8_t role;
    uint8_t architecture;
    uint16_t reserved;
    uint32_t traffic_contract;
    // Conservative half-open envelope of absolute algorithm L1 accesses.
    // Zero/zero means no absolute accesses; relative windows remain separate.
    // The live SDK must exclude its actual configured loading region.
    uint32_t l1_begin;
    uint32_t l1_end;
    // record_bytes includes this header and both complete tables. Endpoints
    // precede the relative windows.
    // Entries follow in strictly increasing (noc, x, y) order. These are
    // encoded remote destinations/sources, not proof of live ownership.
    uint16_t endpoint_count;
    // Static classification never establishes live payload completion.
    uint16_t payload_contract;
    uint16_t window_count;
    // Always TT_NATIVE_NOC_WINDOW_BYTES, including an empty window table.
    // Windows are strictly lexicographically sorted by all fields, unique,
    // and carry nonempty half-open relative bounds and power-of-two alignment.
    // Earlier ABI versions are rejected rather than implicitly upgraded.
    uint16_t window_record_bytes;
} tt_native_noc_record;

#if defined(__cplusplus)
static_assert(sizeof(tt_native_image_record) == 16, "native image ABI size changed");
static_assert(sizeof(tt_native_noc_record) == TT_NATIVE_NOC_HEADER_BYTES, "native NoC ABI size changed");
static_assert(sizeof(tt_native_noc_endpoint) == 4, "native NoC endpoint ABI size changed");
static_assert(sizeof(tt_native_noc_window) == TT_NATIVE_NOC_WINDOW_BYTES, "native NoC window ABI size changed");
static_assert(
    TT_NATIVE_NOC_HEADER_BYTES + TT_NATIVE_NOC_MAX_ENDPOINTS * sizeof(tt_native_noc_endpoint) +
            TT_NATIVE_NOC_MAX_WINDOWS * TT_NATIVE_NOC_WINDOW_BYTES <=
        UINT16_MAX,
    "native NoC declaration exceeds its wire size");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(tt_native_image_record) == 16, "native image ABI size changed");
_Static_assert(sizeof(tt_native_noc_record) == TT_NATIVE_NOC_HEADER_BYTES, "native NoC ABI size changed");
_Static_assert(sizeof(tt_native_noc_endpoint) == 4, "native NoC endpoint ABI size changed");
_Static_assert(sizeof(tt_native_noc_window) == TT_NATIVE_NOC_WINDOW_BYTES, "native NoC window ABI size changed");
_Static_assert(
    TT_NATIVE_NOC_HEADER_BYTES + TT_NATIVE_NOC_MAX_ENDPOINTS * sizeof(tt_native_noc_endpoint) +
            TT_NATIVE_NOC_MAX_WINDOWS * TT_NATIVE_NOC_WINDOW_BYTES <=
        UINT16_MAX,
    "native NoC declaration exceeds its wire size");
#endif
