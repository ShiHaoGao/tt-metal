// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "niu_handoff_receiver.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::detail {
namespace {

constexpr uint32_t all_nocs = (1u << TT_NIU_HANDOFF_NOC_COUNT) - 1;

uint32_t decode_word(std::span<const std::byte> bytes) {
    return std::to_integer<uint32_t>(bytes[0]) | (std::to_integer<uint32_t>(bytes[1]) << 8) |
           (std::to_integer<uint32_t>(bytes[2]) << 16) | (std::to_integer<uint32_t>(bytes[3]) << 24);
}

uint64_t decode_wide(std::span<const std::byte> bytes) {
    return uint64_t(decode_word(bytes.first<4>())) | (uint64_t(decode_word(bytes.subspan<4, 4>())) << 32);
}

tt_niu_item_witness_request decode_item_request(std::span<const std::byte> bytes) {
    return {decode_word(bytes.subspan<0, 4>()), decode_word(bytes.subspan<4, 4>()),
            decode_wide(bytes.subspan<8, 8>()), decode_wide(bytes.subspan<16, 8>()),
            decode_wide(bytes.subspan<24, 8>()), decode_wide(bytes.subspan<32, 8>()),
            decode_wide(bytes.subspan<40, 8>()), decode_word(bytes.subspan<48, 4>()),
            std::to_integer<uint8_t>(bytes[52]), std::to_integer<uint8_t>(bytes[53]),
            std::to_integer<uint8_t>(bytes[54]), std::to_integer<uint8_t>(bytes[55]),
            decode_wide(bytes.subspan<56, 8>())};
}

bool valid_item_request(const tt_niu_item_witness_request& request) {
    const auto valid_identity = [](uint64_t value) { return value && value != UINT64_MAX; };
    return request.abi_version == TT_NIU_ITEM_WITNESS_VERSION && request.request_generation &&
           request.request_generation != UINT32_MAX && valid_identity(request.boot_cookie) &&
           valid_identity(request.invocation_id) && valid_identity(request.item) &&
           valid_identity(request.generation) && valid_identity(request.occurrence) && request.leaf &&
           request.noc < 2 && request.selector < 16 && (request.issuer == 1 || request.issuer == 2) &&
           (request.completion == TT_NIU_ITEM_READ_DESTINATION_LOCAL ||
            request.completion == TT_NIU_ITEM_NONPOSTED_WRITE_ACKNOWLEDGED) && request.reserved == 0;
}

bool same_item_request(const tt_niu_item_witness_request& lhs, const tt_niu_item_witness_request& rhs) {
    return lhs.abi_version == rhs.abi_version && lhs.request_generation == rhs.request_generation &&
           lhs.boot_cookie == rhs.boot_cookie && lhs.invocation_id == rhs.invocation_id && lhs.item == rhs.item &&
           lhs.generation == rhs.generation && lhs.occurrence == rhs.occurrence && lhs.leaf == rhs.leaf &&
           lhs.noc == rhs.noc && lhs.selector == rhs.selector && lhs.issuer == rhs.issuer &&
           lhs.completion == rhs.completion && lhs.reserved == rhs.reserved;
}

void encode_word(uint32_t value, std::span<std::byte> bytes) {
    if (bytes.size() != sizeof(uint32_t)) throw std::invalid_argument("NIU handoff word has invalid size");
    bytes[0] = std::byte(value & 0xff);
    bytes[1] = std::byte((value >> 8) & 0xff);
    bytes[2] = std::byte((value >> 16) & 0xff);
    bytes[3] = std::byte((value >> 24) & 0xff);
}

void encode_wide(uint64_t value, std::span<std::byte, sizeof(uint64_t)> bytes) {
    encode_word(static_cast<uint32_t>(value), bytes.first<4>());
    encode_word(static_cast<uint32_t>(value >> 32), bytes.subspan<4, 4>());
}

std::array<std::byte, sizeof(tt_niu_item_witness_region)> encode_item_region(
    const tt_niu_item_witness_request& request) {
    std::array<std::byte, sizeof(tt_niu_item_witness_region)> bytes{};
    auto wire = std::span(bytes);
    encode_word(request.abi_version, wire.subspan<0, 4>());
    encode_word(request.request_generation, wire.subspan<4, 4>());
    encode_wide(request.boot_cookie, wire.subspan<8, 8>());
    encode_wide(request.invocation_id, wire.subspan<16, 8>());
    encode_wide(request.item, wire.subspan<24, 8>());
    encode_wide(request.generation, wire.subspan<32, 8>());
    encode_wide(request.occurrence, wire.subspan<40, 8>());
    encode_word(request.leaf, wire.subspan<48, 4>());
    bytes[52] = std::byte(request.noc);
    bytes[53] = std::byte(request.selector);
    bytes[54] = std::byte(request.issuer);
    bytes[55] = std::byte(request.completion);
    encode_wide(request.reserved, wire.subspan<56, 8>());
    return bytes;
}

std::array<std::byte, sizeof(tt_niu_handoff_request)> encode_request(const tt_niu_handoff_request& request) {
    static_assert(std::is_trivially_copyable_v<tt_niu_handoff_request>);
    std::array<std::byte, sizeof(tt_niu_handoff_request)> bytes{};
    std::array<uint32_t, sizeof(tt_niu_handoff_request) / sizeof(uint32_t)> words{};
    std::memcpy(words.data(), &request, sizeof(request));
    for (size_t word = 0; word != words.size(); ++word)
        encode_word(words[word], std::span(bytes).subspan(word * sizeof(uint32_t), sizeof(uint32_t)));
    return bytes;
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

struct MailboxAddress {
    tt_cxy_pair core;
    uint64_t base;
    uint64_t size;
};

MailboxAddress mailbox_address(
    const Cluster& cluster, const Hal& hal, ChipId chip, CoreCoord physical_worker) {
    const auto& soc = cluster.get_soc_desc(chip);
    if (soc.arch != hal.get_arch())
        throw std::invalid_argument("NIU handoff Cluster and HAL architectures differ");
    if (!soc.is_core_of_type(physical_worker, tt::CoreType::TENSIX, tt::CoordSystem::NOC0))
        throw std::invalid_argument("NIU handoff requires a physical Worker in the original chip");

    constexpr auto core_type = HalProgrammableCoreType::TENSIX;
    const auto local_base = hal.get_dev_addr(core_type, HalL1MemAddrType::NIU_HANDOFF);
    const auto size = hal.get_dev_size(core_type, HalL1MemAddrType::NIU_HANDOFF);
    if (size != sizeof(tt_niu_handoff_region) || local_base == 0 ||
        local_base % TT_NIU_HANDOFF_ALIGNMENT != 0 || soc.worker_l1_size <= 0 ||
        local_base > static_cast<uint64_t>(soc.worker_l1_size) ||
        size > static_cast<uint64_t>(soc.worker_l1_size) - local_base)
        throw std::invalid_argument("NIU handoff HAL mailbox does not fit the Worker L1 layout");
    const auto noc_offset = hal.get_l1_noc_offset(core_type);
    if (local_base > std::numeric_limits<uint64_t>::max() - noc_offset ||
        local_base + noc_offset > std::numeric_limits<uint64_t>::max() - size)
        throw std::invalid_argument("NIU handoff HAL mailbox NoC address overflows");
    const auto base = hal.get_dev_noc_addr(core_type, HalL1MemAddrType::NIU_HANDOFF);
    const auto translated = soc.translate_coord_to(
        physical_worker, tt::CoordSystem::NOC0, tt::CoordSystem::TRANSLATED);
    return {tt_cxy_pair(chip, translated.x, translated.y), base, size};
}

}  // namespace

NiuHandoffWriteStatus NiuHandoffReceiver::write_item_request(
    const Write& write, const Read& read, const ItemWitnessRequest& request) {
    if (!valid_item_request(request)) return NiuHandoffWriteStatus::InvalidRequest;
    const auto encoded = encode_item_region(request);
    auto pending = request;
    pending.request_generation = 0;
    const auto unpublished = encode_item_region(pending);
    std::array<std::byte, sizeof(ItemWitnessRegion)> observed{};
    if (write(0, unpublished) != NiuHandoffIoStatus::Complete ||
        read(0, observed) != NiuHandoffIoStatus::Complete) return NiuHandoffWriteStatus::IoFailure;
    if (observed != unpublished) return NiuHandoffWriteStatus::ReadbackMismatch;
    constexpr uint32_t generation_offset = offsetof(ItemWitnessRequest, request_generation);
    const auto generation = std::span(encoded).subspan<generation_offset, sizeof(uint32_t)>();
    if (write(generation_offset, generation) != NiuHandoffIoStatus::Complete ||
        read(0, observed) != NiuHandoffIoStatus::Complete) return NiuHandoffWriteStatus::IoFailure;
    return observed == encoded ? NiuHandoffWriteStatus::Published : NiuHandoffWriteStatus::ReadbackMismatch;
}

NiuItemWitnessReadResult NiuHandoffReceiver::read_item(const Read& read) {
    constexpr uint32_t sequence_offset = offsetof(ItemWitnessRegion, receipt);
    const auto unavailable = [](NiuHandoffReadStatus status) {
        return NiuItemWitnessReadResult{status, std::nullopt};
    };
    std::array<std::byte, sizeof(uint32_t)> sequence;
    if (read(sequence_offset, sequence) != NiuHandoffIoStatus::Complete)
        return unavailable(NiuHandoffReadStatus::IoFailure);
    const auto first = decode_word(sequence);
    if (!first) return unavailable(NiuHandoffReadStatus::Unpublished);
    if (first & 1) return unavailable(NiuHandoffReadStatus::InProgress);
    std::array<std::byte, sizeof(ItemWitnessRegion)> bytes;
    if (read(0, bytes) != NiuHandoffIoStatus::Complete ||
        read(sequence_offset, sequence) != NiuHandoffIoStatus::Complete)
        return unavailable(NiuHandoffReadStatus::IoFailure);
    const auto receipt = std::span<const std::byte>(bytes).subspan<64>();
    if (decode_word(sequence) != first || decode_word(receipt.first<4>()) != first)
        return unavailable(NiuHandoffReadStatus::Changed);
    const auto state = decode_word(receipt.subspan<4, 4>());
    if (state > TT_NIU_ITEM_WITNESS_REJECTED || decode_wide(receipt.subspan<88, 8>()) != 0)
        return unavailable(NiuHandoffReadStatus::Invalid);
    ItemWitnessRegion region{decode_item_request(bytes),
                             {first, state, decode_item_request(receipt.subspan<8, 64>()),
                              decode_wide(receipt.subspan<72, 8>()), decode_wide(receipt.subspan<80, 8>()), 0}};
    return {NiuHandoffReadStatus::Observed, NiuItemWitnessObservation(region)};
}

bool NiuHandoffReceiver::closed_item(
    const NiuItemWitnessObservation& observation, const ItemWitnessRequest& expected) {
    const auto& region = observation.wire();
    return valid_item_request(expected) && same_item_request(region.request, expected) &&
           same_item_request(region.receipt.request, expected) &&
           region.receipt.state == TT_NIU_ITEM_WITNESS_CLOSED &&
           region.receipt.issued_occurrence == expected.occurrence &&
           region.receipt.closed_occurrence == expected.occurrence;
}

NiuHandoffReadResult NiuHandoffReceiver::read(
    const Cluster& cluster, const Hal& hal, ChipId chip, CoreCoord physical_worker) {
    const auto [core, base, size] = mailbox_address(cluster, hal, chip, physical_worker);
    return read([&](uint32_t offset, std::span<std::byte> bytes) {
        if (offset > size || bytes.size() > size - offset)
            throw std::out_of_range("NIU handoff read exceeds the HAL mailbox");
        cluster.read_core(bytes.data(), static_cast<uint32_t>(bytes.size()), core, base + offset);
        return NiuHandoffIoStatus::Complete;
    });
}

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

NiuHandoffWriteStatus NiuHandoffReceiver::write_request(
    const Write& write, const Read& read, const tt_niu_handoff_request& request) {
    if (!valid_request(request)) return NiuHandoffWriteStatus::InvalidRequest;
    const auto encoded = encode_request(request);
    auto pending = request;
    pending.request_generation = 0;
    const auto unpublished = encode_request(pending);
    std::array<std::byte, sizeof(uint32_t)> zero{};
    std::array<std::byte, sizeof(uint32_t)> observed_generation{};
    std::array<std::byte, sizeof(uint32_t)> generation{};
    encode_word(request.request_generation, generation);
    std::array<std::byte, sizeof(tt_niu_handoff_request)> observed{};
    // Invalidate the previous request first, then verify that the invalidation
    // reached the actual mailbox before exposing any new request body.
    constexpr uint32_t generation_offset = offsetof(tt_niu_handoff_request, request_generation);
    if (write(generation_offset, zero) != NiuHandoffIoStatus::Complete ||
        read(generation_offset, observed_generation) != NiuHandoffIoStatus::Complete)
        return NiuHandoffWriteStatus::IoFailure;
    if (observed_generation != zero) return NiuHandoffWriteStatus::ReadbackMismatch;
    if (write(0, unpublished) != NiuHandoffIoStatus::Complete ||
        read(0, observed) != NiuHandoffIoStatus::Complete)
        return NiuHandoffWriteStatus::IoFailure;
    if (observed != unpublished)
        return NiuHandoffWriteStatus::ReadbackMismatch;
    // Publish the nonzero generation last, making a partially written body
    // invalid to firmware readers until its complete byte image is visible.
    if (write(generation_offset, generation) != NiuHandoffIoStatus::Complete ||
        read(0, observed) != NiuHandoffIoStatus::Complete)
        return NiuHandoffWriteStatus::IoFailure;
    if (observed != encoded)
        return NiuHandoffWriteStatus::ReadbackMismatch;
    return NiuHandoffWriteStatus::Published;
}

NiuHandoffWriteStatus NiuHandoffReceiver::write_request(
    const Cluster& cluster, const Hal& hal, ChipId chip, CoreCoord physical_worker,
    const tt_niu_handoff_request& request) {
    const auto address = mailbox_address(cluster, hal, chip, physical_worker);
    return write_request(
        [&](uint32_t offset, std::span<const std::byte> bytes) {
            if (offset > sizeof(tt_niu_handoff_request) || bytes.size() > sizeof(tt_niu_handoff_request) - offset)
                throw std::out_of_range("NIU handoff write exceeds request mailbox");
            cluster.write_core(bytes.data(), static_cast<uint32_t>(bytes.size()), address.core, address.base + offset);
            cluster.l1_barrier(chip);
            return NiuHandoffIoStatus::Complete;
        },
        [&](uint32_t offset, std::span<std::byte> bytes) {
            if (offset > sizeof(tt_niu_handoff_request) || bytes.size() > sizeof(tt_niu_handoff_request) - offset)
                throw std::out_of_range("NIU handoff read exceeds request mailbox");
            cluster.read_core(bytes.data(), static_cast<uint32_t>(bytes.size()), address.core, address.base + offset);
            return NiuHandoffIoStatus::Complete;
        },
        request);
}

std::optional<NiuHandoffExpectedExecution> NiuHandoffReceiver::initial_execution(
    const NiuHandoffObservation& observation, uint64_t boot_cookie, uint64_t invocation_id, bool has_source) {
    const auto& region = observation.wire();
    const auto& receipt = region.receipt;
    const auto& header = receipt.header;
    const tt_niu_handoff_request empty_request{};
    if (!boot_cookie || !invocation_id ||
        std::memcmp(&region.request, &empty_request, sizeof(empty_request)) != 0 ||
        header.publish_sequence != 2 || header.abi_version != TT_NIU_HANDOFF_VERSION ||
        header.status != TT_NIU_HANDOFF_BOOT_READY || header.error != TT_NIU_HANDOFF_ERROR_NONE ||
        header.challenge_lo || header.challenge_hi || header.boot_cookie_lo || header.boot_cookie_hi ||
        header.invocation_id_lo || header.invocation_id_hi || header.group_generation ||
        header.completed_go_count || header.completed_stage_count || header.source_valid_noc_mask ||
        header.completed_processor_mask != TT_NIU_HANDOFF_PROCESSOR_MASK || receipt.consumed_go_sequence ||
        header.control_valid_noc_mask != all_nocs || receipt.control.valid_noc_mask != all_nocs ||
        receipt.control.ready_cmd_mask_noc0 != 15 || receipt.control.ready_cmd_mask_noc1 != 15 ||
        receipt.control.control_contract != 1)
        return std::nullopt;
    for (unsigned noc = 0; noc != TT_NIU_HANDOFF_NOC_COUNT; ++noc) {
        for (unsigned counter = 0; counter != TT_NIU_HANDOFF_COUNTER_COUNT; ++counter) {
            if (receipt.source_expected[noc][0][counter] || receipt.source_expected[noc][1][counter] ||
                receipt.source_observed[noc][counter] ||
                receipt.control_expected[noc][counter] != receipt.control_observed[noc][counter])
                return std::nullopt;
        }
    }
    // The actual SDK issue path currently freezes exactly one non-reload GO
    // per physical Worker. Later invocations, reloads and groups require their
    // own retained journal; deriving that journal from the receipt is invalid.
    NiuHandoffExpectedExecution expected{};
    auto& request = expected.request;
    request.abi_version = TT_NIU_HANDOFF_VERSION;
    request.request_generation = 1;
    request.challenge_lo = request.invocation_id_lo = static_cast<uint32_t>(invocation_id);
    request.challenge_hi = request.invocation_id_hi = static_cast<uint32_t>(invocation_id >> 32);
    request.boot_cookie_lo = static_cast<uint32_t>(boot_cookie);
    request.boot_cookie_hi = static_cast<uint32_t>(boot_cookie >> 32);
    request.group_generation = request.final_go_count = request.first_go_sequence = 1;
    request.selector_mask_noc0 = request.selector_mask_noc1 = 0xffff;
    request.traffic_contract = TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1;
    request.required_processor_mask = TT_NIU_HANDOFF_PROCESSOR_MASK;
    expected.completed_stage_count = 1;
    expected.source_valid_noc_mask = has_source ? all_nocs : 0;
    return expected;
}

bool NiuHandoffReceiver::closed_selector(
    const NiuHandoffObservation& observation, const NiuHandoffExpectedExecution& expected,
    uint8_t noc, uint8_t selector) {
    if (noc >= TT_NIU_HANDOFF_NOC_COUNT || selector >= TT_NIU_HANDOFF_SELECTOR_COUNT ||
        check(observation, expected) != NiuHandoffCheck::Match)
        return false;
    const auto& counters = observation.wire().receipt.selector_observed[noc][selector];
    return counters.outstanding == 0 && counters.outgoing == 0;
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
