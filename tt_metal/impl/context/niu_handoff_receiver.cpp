// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "niu_handoff_receiver.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <type_traits>

namespace tt::tt_metal::detail {
namespace {

constexpr uint32_t all_nocs = (1u << TT_NIU_HANDOFF_NOC_COUNT) - 1;

uint32_t decode_word(std::span<const std::byte> bytes) {
    return std::to_integer<uint32_t>(bytes[0]) | (std::to_integer<uint32_t>(bytes[1]) << 8) |
           (std::to_integer<uint32_t>(bytes[2]) << 16) | (std::to_integer<uint32_t>(bytes[3]) << 24);
}

tt_niu_handoff_region decode_region(std::span<const std::byte> bytes) {
    static_assert(std::is_trivially_copyable_v<tt_niu_handoff_region>);
    static_assert(sizeof(tt_niu_handoff_region) % sizeof(uint32_t) == 0);
    // Every field in this wire region is a uint32. Decode little-endian words
    // before copying their host representation, without aliasing struct fields
    // through an array pointer or assuming the host's byte order.
    std::array<uint32_t, sizeof(tt_niu_handoff_region) / sizeof(uint32_t)> words;
    for (size_t word = 0; word != words.size(); ++word)
        words[word] = decode_word(bytes.subspan(word * sizeof(uint32_t), sizeof(uint32_t)));
    tt_niu_handoff_region region;
    std::memcpy(&region, words.data(), sizeof(region));
    return region;
}

bool valid_request(const tt_niu_handoff_request& request) {
    return request.abi_version == TT_NIU_HANDOFF_VERSION && request.request_generation != 0 &&
           (request.challenge_lo || request.challenge_hi) && (request.boot_cookie_lo || request.boot_cookie_hi) &&
           (request.invocation_id_lo || request.invocation_id_hi) && request.group_generation != 0 &&
           request.final_go_count != 0 && request.first_go_sequence != 0 &&
           request.first_go_sequence <= std::numeric_limits<uint32_t>::max() - (request.final_go_count - 1) &&
           request.selector_mask_noc0 == 0xffff && request.selector_mask_noc1 == 0xffff &&
           request.traffic_contract == TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1 &&
           request.required_processor_mask == TT_NIU_HANDOFF_PROCESSOR_MASK && request.reserved1 == 0;
}

bool supported_header(const tt_niu_handoff_region& region) {
    const auto& header = region.receipt.header;
    if (header.abi_version != TT_NIU_HANDOFF_VERSION || region.request.reserved1 != 0 ||
        region.receipt.reserved != 0 || header.error > TT_NIU_HANDOFF_ERROR_CONTROL)
        return false;
    switch (header.status) {
        case TT_NIU_HANDOFF_PENDING:
        case TT_NIU_HANDOFF_TRAFFIC_CLOSED:
        case TT_NIU_HANDOFF_BOOT_READY: return header.error == TT_NIU_HANDOFF_ERROR_NONE;
        case TT_NIU_HANDOFF_REJECTED: return header.error != TT_NIU_HANDOFF_ERROR_NONE;
        default: return false;
    }
}

NiuHandoffReadResult unavailable(NiuHandoffReadStatus status) { return {status, std::nullopt}; }

}  // namespace

NiuHandoffReadResult NiuHandoffReceiver::read(const Read& read) {
    std::array<std::byte, sizeof(uint32_t)> sequence_bytes;
    if (read(TT_NIU_HANDOFF_RECEIPT_OFFSET, sequence_bytes) != NiuHandoffIoStatus::Complete)
        return unavailable(NiuHandoffReadStatus::IoFailure);
    const uint32_t first = decode_word(sequence_bytes);
    if (first == 0) return unavailable(NiuHandoffReadStatus::Unpublished);
    if (first & 1) return unavailable(NiuHandoffReadStatus::InProgress);

    std::array<std::byte, sizeof(tt_niu_handoff_region)> bytes;
    if (read(0, bytes) != NiuHandoffIoStatus::Complete)
        return unavailable(NiuHandoffReadStatus::IoFailure);
    if (read(TT_NIU_HANDOFF_RECEIPT_OFFSET, sequence_bytes) != NiuHandoffIoStatus::Complete)
        return unavailable(NiuHandoffReadStatus::IoFailure);
    const auto region = decode_region(bytes);
    if (decode_word(sequence_bytes) != first || region.receipt.header.publish_sequence != first)
        return unavailable(NiuHandoffReadStatus::Changed);
    if (!supported_header(region)) return unavailable(NiuHandoffReadStatus::Invalid);
    return {NiuHandoffReadStatus::Observed, NiuHandoffObservation(region)};
}

NiuHandoffCheck NiuHandoffReceiver::check(
    const NiuHandoffObservation& observation, const NiuHandoffExpectedExecution& expected) {
    const auto& region = observation.wire();
    const auto& request = expected.request;
    const auto& receipt = region.receipt;
    const auto& header = receipt.header;
    if (!valid_request(request)) return NiuHandoffCheck::InvalidRequest;
    static_assert(sizeof(tt_niu_handoff_request) == 16 * sizeof(uint32_t));
    if (std::memcmp(&region.request, &request, sizeof(request)) != 0) return NiuHandoffCheck::RequestMismatch;
    switch (header.status) {
        case TT_NIU_HANDOFF_PENDING: return NiuHandoffCheck::Pending;
        case TT_NIU_HANDOFF_REJECTED: return NiuHandoffCheck::Rejected;
        case TT_NIU_HANDOFF_BOOT_READY: return NiuHandoffCheck::NotExecution;
        case TT_NIU_HANDOFF_TRAFFIC_CLOSED: break;
        default: return NiuHandoffCheck::NotExecution;
    }
    if (header.challenge_lo != request.challenge_lo || header.challenge_hi != request.challenge_hi ||
        header.boot_cookie_lo != request.boot_cookie_lo || header.boot_cookie_hi != request.boot_cookie_hi ||
        header.invocation_id_lo != request.invocation_id_lo || header.invocation_id_hi != request.invocation_id_hi ||
        header.group_generation != request.group_generation)
        return NiuHandoffCheck::IdentityMismatch;
    if (header.completed_go_count != request.final_go_count ||
        receipt.consumed_go_sequence != request.first_go_sequence + (request.final_go_count - 1))
        return NiuHandoffCheck::GoMismatch;
    if (expected.completed_stage_count < request.final_go_count ||
        header.completed_stage_count != expected.completed_stage_count ||
        header.completed_processor_mask != request.required_processor_mask)
        return NiuHandoffCheck::StageMismatch;
    if ((expected.source_valid_noc_mask != 0 && expected.source_valid_noc_mask != all_nocs) ||
        header.source_valid_noc_mask != expected.source_valid_noc_mask)
        return NiuHandoffCheck::SourceMismatch;
    for (unsigned noc = 0; noc != TT_NIU_HANDOFF_NOC_COUNT; ++noc) {
        for (unsigned counter = 0; counter != TT_NIU_HANDOFF_COUNTER_COUNT; ++counter) {
            const auto first = receipt.source_expected[noc][0][counter];
            const auto second = receipt.source_expected[noc][1][counter];
            const auto observed = receipt.source_observed[noc][counter];
            if (expected.source_valid_noc_mask == 0) {
                if (first || second || observed) return NiuHandoffCheck::SourceMismatch;
            } else if (static_cast<uint32_t>(first + second) != observed) {
                return NiuHandoffCheck::SourceMismatch;
            }
        }
    }
    // These are the dedicated control producer's absolute counters, not a
    // second expected-count stream or a selector-normalization operation.
    if (header.control_valid_noc_mask != all_nocs || receipt.control.valid_noc_mask != all_nocs ||
        receipt.control.ready_cmd_mask_noc0 != 15 || receipt.control.ready_cmd_mask_noc1 != 15 ||
        receipt.control.control_contract != 1)
        return NiuHandoffCheck::ControlMismatch;
    for (unsigned noc = 0; noc != TT_NIU_HANDOFF_NOC_COUNT; ++noc)
        for (unsigned counter = 0; counter != TT_NIU_HANDOFF_COUNTER_COUNT; ++counter)
            if (receipt.control_expected[noc][counter] != receipt.control_observed[noc][counter])
                return NiuHandoffCheck::ControlMismatch;
    // Matching these bytes does not prove that the reader was attached to the
    // original live boot or that the unknown NIU prefix was bootstrapped.
    return NiuHandoffCheck::Match;
}

}  // namespace tt::tt_metal::detail
