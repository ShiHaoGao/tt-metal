// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>
#include <stddef.h>
#include "hostdev/niu_handoff_layout.h"

// Independent, non-ALLOC firmware metadata. This declares a device protocol;
// only the original successful boot and its private receiver authorize reuse.
#define TT_NIU_HANDOFF_SECTION ".tt_niu_handoff"
#define TT_NIU_HANDOFF_MAGIC UINT32_C(0x4855494e)
#define TT_NIU_HANDOFF_VERSION UINT16_C(1)
#define TT_NIU_HANDOFF_LAYOUT_VERSION UINT16_C(1)
#define TT_NIU_HANDOFF_PROTOCOL UINT16_C(1)
#define TT_NIU_HANDOFF_RECORD_BYTES UINT16_C(48)
#define TT_NIU_HANDOFF_REQUEST_OFFSET UINT16_C(0)
#define TT_NIU_HANDOFF_REQUEST_BYTES UINT16_C(64)
#define TT_NIU_HANDOFF_RECEIPT_OFFSET UINT16_C(64)
#define TT_NIU_HANDOFF_RECEIPT_BYTES UINT16_C(544)
#define TT_NIU_HANDOFF_COUNTER_CONTRACT UINT16_C(1)
#define TT_NIU_HANDOFF_PROCESSOR_MASK UINT32_C(0x1f)
#define TT_NIU_HANDOFF_NOC_COUNT 2
#define TT_NIU_HANDOFF_SELECTOR_COUNT 16
#define TT_NIU_HANDOFF_COUNTER_COUNT 5

enum tt_niu_traffic_contract {
    // The existing 2 NoCs x 2 issuing RISCs x 5 dynamic counters remain the
    // sole expected-count producers. Returns target the issuing NIU, and
    // multicast ACK accounting includes every destination. Closure does not
    // imply payload visibility for posted writes or normalized selectors.
    TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1 = 1,
};

// All multibyte fields are little-endian. The five processor records agree
// except for role. Addresses describe the actual linked firmware; the live
// boot receiver must additionally compare them with its actual HAL layout.
typedef struct tt_niu_handoff_record {
    uint32_t magic;
    uint16_t version;
    uint16_t record_bytes;
    uint8_t role;
    uint8_t architecture;
    uint8_t noc_count;
    uint8_t selectors_per_noc;
    uint16_t layout_version;
    uint16_t protocol;
    uint32_t l1_base;
    uint32_t l1_bytes;
    uint32_t dynamic_counter_base;
    uint16_t counter_bytes;
    uint16_t counter_contract;
    uint32_t processor_mask;
    uint16_t request_offset;
    uint16_t request_bytes;
    uint16_t receipt_offset;
    uint16_t receipt_bytes;
    uint32_t traffic_contract;
} tt_niu_handoff_record;

enum tt_niu_handoff_status {
    TT_NIU_HANDOFF_PENDING = 0,
    TT_NIU_HANDOFF_REJECTED = 1,
    TT_NIU_HANDOFF_TRAFFIC_CLOSED = 2,
    // Published only after the actual boot's control traffic closes. It has
    // no request identity and cannot certify a payload invocation.
    TT_NIU_HANDOFF_BOOT_READY = 3,
};

enum tt_niu_handoff_error {
    TT_NIU_HANDOFF_ERROR_NONE = 0,
    TT_NIU_HANDOFF_ERROR_REQUEST = 1,
    TT_NIU_HANDOFF_ERROR_IDENTITY = 2,
    TT_NIU_HANDOFF_ERROR_SEQUENCE = 3,
    TT_NIU_HANDOFF_ERROR_SOURCE = 4,
    TT_NIU_HANDOFF_ERROR_CONTROL = 5,
};

// The SDK installs the complete request before GO and publishes generation
// last. Cookies correlate wire traffic with the original retained host owners;
// they never constitute a boot, resource lease or completion capability.
typedef struct tt_niu_handoff_request {
    uint32_t abi_version;
    uint32_t request_generation;
    uint32_t challenge_lo;
    uint32_t challenge_hi;
    uint32_t boot_cookie_lo;
    uint32_t boot_cookie_hi;
    uint32_t invocation_id_lo;
    uint32_t invocation_id_hi;
    uint32_t group_generation;
    uint32_t final_go_count;
    uint32_t selector_mask_noc0;
    uint32_t selector_mask_noc1;
    uint32_t traffic_contract;
    uint32_t required_processor_mask;
    uint32_t first_go_sequence;
    uint32_t reserved1;
} tt_niu_handoff_request;

// An odd publish_sequence marks a write in progress. The final even sequence
// is published after all body stores, with no wrapping generations allowed.
// TrafficClosed does not normalize selectors or publish posted payload data.
typedef struct tt_niu_handoff_receipt_header {
    uint32_t publish_sequence;
    uint32_t abi_version;
    uint32_t challenge_lo;
    uint32_t challenge_hi;
    uint32_t boot_cookie_lo;
    uint32_t boot_cookie_hi;
    uint32_t invocation_id_lo;
    uint32_t invocation_id_hi;
    uint32_t group_generation;
    uint32_t completed_go_count;
    uint32_t source_valid_noc_mask;
    uint32_t completed_processor_mask;
    uint32_t status;
    uint32_t error;
    uint32_t control_valid_noc_mask;
    uint32_t completed_stage_count;
} tt_niu_handoff_receipt_header;

typedef struct tt_niu_handoff_control {
    uint32_t valid_noc_mask;
    uint32_t ready_cmd_mask_noc0;
    uint32_t ready_cmd_mask_noc1;
    uint32_t control_contract;
} tt_niu_handoff_control;

typedef struct tt_niu_handoff_selector_observation {
    uint32_t outstanding;
    uint32_t outgoing;
} tt_niu_handoff_selector_observation;

typedef struct tt_niu_handoff_receipt {
    tt_niu_handoff_receipt_header header;
    // Sum of actual per-stage deltas (modulo uint32), sampled after stage
    // initialization and before any issue. Each stage must first close its
    // original absolute counters before rebase/reload; totals alone are not
    // completion evidence. The 80-byte dynamic bank remains the sole producer.
    uint32_t source_expected[2][2][5];
    uint32_t source_observed[2][5];
    // Absolute non-wrapping count of actual GO consumption since this boot.
    // Pending snapshots expose the baseline only; it is not NIU-clean proof.
    uint32_t consumed_go_sequence;
    uint32_t reserved;
    uint32_t control_expected[2][5];
    uint32_t control_observed[2][5];
    tt_niu_handoff_control control;
    tt_niu_handoff_selector_observation selector_observed[2][16];
} tt_niu_handoff_receipt;

typedef struct tt_niu_handoff_region {
    tt_niu_handoff_request request;
    tt_niu_handoff_receipt receipt;
} tt_niu_handoff_region;

#if defined(__cplusplus)
static_assert(sizeof(tt_niu_handoff_record) == 48, "NIU handoff metadata size changed");
static_assert(sizeof(tt_niu_handoff_request) == 64, "NIU handoff request size changed");
static_assert(sizeof(tt_niu_handoff_receipt_header) == 64, "NIU handoff receipt header size changed");
static_assert(sizeof(tt_niu_handoff_receipt) == 544, "NIU handoff receipt size changed");
static_assert(sizeof(tt_niu_handoff_region) == TT_NIU_HANDOFF_L1_BYTES, "NIU handoff region size changed");
static_assert(offsetof(tt_niu_handoff_region, receipt) == 0x40, "NIU receipt offset changed");
static_assert(offsetof(tt_niu_handoff_receipt, source_expected) == 0x40, "NIU source offset changed");
static_assert(offsetof(tt_niu_handoff_receipt, control_expected) == 0xc0, "NIU control offset changed");
static_assert(offsetof(tt_niu_handoff_receipt, selector_observed) == 0x120, "NIU selector offset changed");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(tt_niu_handoff_record) == 48, "NIU handoff metadata size changed");
#endif
