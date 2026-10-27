// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#define TT_NIU_ITEM_WITNESS_VERSION UINT32_C(1)
#define TT_NIU_ITEM_WITNESS_ALIGNMENT UINT32_C(32)

enum tt_niu_item_witness_state {
    TT_NIU_ITEM_WITNESS_PENDING = 0,
    TT_NIU_ITEM_WITNESS_ISSUED = 1,
    TT_NIU_ITEM_WITNESS_CLOSED = 2,
    TT_NIU_ITEM_WITNESS_REJECTED = 3,
};

enum tt_niu_item_completion_fact {
    // These are scoped physical facts only. Neither grants visibility to
    // another consumer's cache, backing release, or selector reclamation.
    TT_NIU_ITEM_READ_DESTINATION_LOCAL = 1,
    TT_NIU_ITEM_NONPOSTED_WRITE_ACKNOWLEDGED = 2,
};

// Little-endian fixed-width wire fields. The original SDK invocation owner
// initializes this immutable request in its actual retained L1 allocation
// before GO, publishes request_generation last, and never rewrites a live
// slot. Cookies correlate with that owner; bytes alone grant no authority.
// Item/generation/occurrence are nonzero and never wrap. UINT64_MAX is
// reserved for exhaustion/poison, not a reusable generation.
typedef struct tt_niu_item_witness_request {
    uint32_t abi_version;
    uint32_t request_generation;
    uint64_t boot_cookie;
    uint64_t invocation_id;
    uint64_t item;
    uint64_t generation;
    uint64_t occurrence;
    uint32_t leaf;
    uint8_t noc;
    uint8_t selector;
    uint8_t issuer;
    uint8_t completion;
    uint64_t reserved;
} tt_niu_item_witness_request;

// One actual issuing processor is the sole receipt writer. It publishes odd
// sequence, fences, stores the body, fences, then publishes the next even
// sequence. Host reads sequence/body/sequence and accepts only one stable
// even snapshot. A writer must poison rather than wrap publish_sequence.
// Issued follows actual issue; Closed follows its scoped wait, local cache
// invalidation and zero outstanding/outgoing hardware selector observation.
// Runtime cannot manufacture a receipt by echoing the request.
typedef struct tt_niu_item_witness_receipt {
    uint32_t publish_sequence;
    uint32_t state;
    tt_niu_item_witness_request request;
    uint64_t issued_occurrence;
    uint64_t closed_occurrence;
    uint64_t reserved;
} tt_niu_item_witness_receipt;

typedef struct tt_niu_item_witness_region {
    tt_niu_item_witness_request request;
    tt_niu_item_witness_receipt receipt;
} tt_niu_item_witness_region;

#ifdef __cplusplus
static_assert(sizeof(tt_niu_item_witness_request) == 64, "Item witness request layout changed");
static_assert(sizeof(tt_niu_item_witness_receipt) == 96, "Item witness receipt layout changed");
static_assert(sizeof(tt_niu_item_witness_region) == 160, "Item witness region layout changed");
static_assert(offsetof(tt_niu_item_witness_request, boot_cookie) == 8);
static_assert(offsetof(tt_niu_item_witness_request, leaf) == 48);
static_assert(offsetof(tt_niu_item_witness_request, reserved) == 56);
static_assert(offsetof(tt_niu_item_witness_region, receipt) == 64);
#endif
