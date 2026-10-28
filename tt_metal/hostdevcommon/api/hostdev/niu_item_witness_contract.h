// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#define TT_NIU_ITEM_WITNESS_VERSION UINT32_C(3)
#define TT_NIU_ITEM_WITNESS_ALIGNMENT UINT32_C(32)

enum tt_niu_item_witness_state {
    TT_NIU_ITEM_WITNESS_PENDING = 0,
    TT_NIU_ITEM_WITNESS_ISSUED = 1,
    TT_NIU_ITEM_WITNESS_CLOSED = 2,
    TT_NIU_ITEM_WITNESS_REJECTED = 3,
    TT_NIU_ITEM_WITNESS_SERIES_OPEN = 4,
    TT_NIU_ITEM_WITNESS_SERIES_END = 5,
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
// Item/generation are nonzero and never wrap; initial occurrence must be one.
// The request remains immutable for the whole invocation. Each actual source
// entry allocates the next occurrence in the device publisher, never from a
// caller-supplied ordinal. UINT64_MAX is
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
typedef struct tt_niu_item_witness_item_fact {
    uint64_t issued_occurrence;
    uint64_t closed_occurrence;
    uint64_t reserved;
} tt_niu_item_witness_item_fact;

typedef struct tt_niu_item_witness_series_fact {
    uint64_t final_count;
    uint32_t closed_sequence;
    uint32_t acknowledgement_sequence;
    uint64_t reserved;
} tt_niu_item_witness_series_fact;

// SeriesOpen and SeriesEnd use the series member and retain the original
// request (including occurrence one). Open has an all-zero series body.
// End is published only by the original issuer's normal source epilogue:
// End(0) has zero frontiers; End(N) follows exact receipt acknowledgement of
// all N source occurrences and carries their final Closed/ACK frontiers.
// Neither Stopped, a missing receipt nor the last Closed/ACK proves End.
typedef struct tt_niu_item_witness_receipt {
    uint32_t publish_sequence;
    uint32_t state;
    tt_niu_item_witness_request request;
    union {
        tt_niu_item_witness_item_fact item;
        tt_niu_item_witness_series_fact series;
    };
} tt_niu_item_witness_receipt;

// Host-owned receipt acknowledgement, published odd/body/fence/even and never
// reset or wrapped. The exact Closed sequence and its complete round identity
// acknowledge SDK receipt reception only. This does not grant payload
// visibility, last-use, Channel release, or NIU terminal clean authority.
typedef struct __attribute__((aligned(TT_NIU_ITEM_WITNESS_ALIGNMENT))) tt_niu_item_witness_acknowledgement {
    uint32_t publish_sequence;
    uint32_t closed_sequence;
    tt_niu_item_witness_request request;
    uint64_t reserved[3];
} tt_niu_item_witness_acknowledgement;

typedef struct __attribute__((aligned(TT_NIU_ITEM_WITNESS_ALIGNMENT))) tt_niu_item_witness_region {
    tt_niu_item_witness_request request;
    tt_niu_item_witness_receipt receipt;
    tt_niu_item_witness_acknowledgement acknowledgement;
} tt_niu_item_witness_region;

#ifdef __cplusplus
static_assert(sizeof(tt_niu_item_witness_request) == 64, "Item witness request layout changed");
static_assert(sizeof(tt_niu_item_witness_receipt) == 96, "Item witness receipt layout changed");
static_assert(sizeof(tt_niu_item_witness_item_fact) == 24);
static_assert(sizeof(tt_niu_item_witness_series_fact) == 24);
static_assert(offsetof(tt_niu_item_witness_receipt, item) == 72);
static_assert(offsetof(tt_niu_item_witness_receipt, series) == 72);
static_assert(offsetof(tt_niu_item_witness_series_fact, closed_sequence) == 8);
static_assert(offsetof(tt_niu_item_witness_series_fact, acknowledgement_sequence) == 12);
static_assert(offsetof(tt_niu_item_witness_series_fact, reserved) == 16);
static_assert(sizeof(tt_niu_item_witness_acknowledgement) == 96, "Item acknowledgement layout changed");
static_assert(sizeof(tt_niu_item_witness_region) == 256, "Item witness region layout changed");
static_assert(offsetof(tt_niu_item_witness_request, boot_cookie) == 8);
static_assert(offsetof(tt_niu_item_witness_request, leaf) == 48);
static_assert(offsetof(tt_niu_item_witness_request, reserved) == 56);
static_assert(offsetof(tt_niu_item_witness_region, receipt) == 64);
static_assert(offsetof(tt_niu_item_witness_acknowledgement, publish_sequence) == 0);
static_assert(offsetof(tt_niu_item_witness_acknowledgement, closed_sequence) == 4);
static_assert(offsetof(tt_niu_item_witness_acknowledgement, request) == 8);
static_assert(offsetof(tt_niu_item_witness_acknowledgement, reserved) == 72);
static_assert(offsetof(tt_niu_item_witness_region, acknowledgement) == 160);
static_assert(alignof(tt_niu_item_witness_acknowledgement) == TT_NIU_ITEM_WITNESS_ALIGNMENT);
static_assert(alignof(tt_niu_item_witness_region) == TT_NIU_ITEM_WITNESS_ALIGNMENT);

namespace tt::niu_handoff {
struct ItemWitnessSequenceAdvance {
    bool available;
    uint32_t sequence;
};
constexpr ItemWitnessSequenceAdvance advanceItemWitnessSequence(uint32_t current) {
    if ((current & 1) || current > UINT32_MAX - 2) return {false, current};
    return {true, current + 2};
}
// Reserve Pending, Issued, Closed and SeriesEnd before starting an occurrence. A reject
// replaces a remaining publication; the publisher never enters a round that
// cannot express its successful terminal state without wrapping.
constexpr bool itemWitnessRoundAvailable(uint32_t current) {
    return !(current & 1) && current <= UINT32_MAX - 9;
}
}  // namespace tt::niu_handoff
#endif
