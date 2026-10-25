// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "impl/context/niu_handoff_receiver.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tt::tt_metal::detail {
namespace {

constexpr uint32_t sequence_offset = TT_NIU_HANDOFF_RECEIPT_OFFSET;

// A byte-transport fixture, not a device, LoadedFirmware, lease, or clean
// provider. These tests prove only the production decoder's observations.
class Mailbox {
public:
    tt_niu_handoff_region wire{};
    std::vector<std::pair<uint32_t, size_t>> reads;
    std::optional<size_t> fail_read;
    bool change_after_body = false;
    bool change_body_sequence_only = false;

    Mailbox() {
        auto& request = wire.request;
        request.abi_version = TT_NIU_HANDOFF_VERSION;
        request.request_generation = 11;
        request.challenge_lo = 17;
        request.challenge_hi = 3;
        request.boot_cookie_lo = 19;
        request.boot_cookie_hi = 5;
        request.invocation_id_lo = 23;
        request.invocation_id_hi = 7;
        request.group_generation = 2;
        request.final_go_count = 2;
        request.selector_mask_noc0 = request.selector_mask_noc1 = 0xffff;
        request.traffic_contract = TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1;
        request.required_processor_mask = TT_NIU_HANDOFF_PROCESSOR_MASK;
        request.first_go_sequence = 21;
        auto& receipt = wire.receipt;
        auto& header = receipt.header;
        header.publish_sequence = 14;
        header.abi_version = TT_NIU_HANDOFF_VERSION;
        header.challenge_lo = request.challenge_lo;
        header.challenge_hi = request.challenge_hi;
        header.boot_cookie_lo = request.boot_cookie_lo;
        header.boot_cookie_hi = request.boot_cookie_hi;
        header.invocation_id_lo = request.invocation_id_lo;
        header.invocation_id_hi = request.invocation_id_hi;
        header.group_generation = request.group_generation;
        header.completed_go_count = request.final_go_count;
        header.source_valid_noc_mask = 3;
        header.completed_processor_mask = TT_NIU_HANDOFF_PROCESSOR_MASK;
        header.status = TT_NIU_HANDOFF_TRAFFIC_CLOSED;
        header.control_valid_noc_mask = 3;
        header.completed_stage_count = 4;
        receipt.source_expected[0][0][0] = 5;
        receipt.source_expected[0][1][0] = 7;
        receipt.source_observed[0][0] = 12;
        receipt.consumed_go_sequence = 22;
        receipt.control_expected[1][2] = receipt.control_observed[1][2] = 31;
        receipt.control = {3, 15, 15, 1};
        // A legitimate multicast tail can retain a nonzero selector counter.
        // Observing TrafficClosed must not silently assert normalization.
        receipt.selector_observed[0][7].outstanding = 255;
    }

    NiuHandoffExpectedExecution expected() const { return {wire.request, 4, 3}; }

    NiuHandoffReadResult read() {
        return NiuHandoffReceiver::read([&](uint32_t offset, std::span<std::byte> destination) {
            const size_t index = reads.size();
            reads.emplace_back(offset, destination.size());
            if (fail_read == index || offset > sizeof(wire) || destination.size() > sizeof(wire) - offset)
                return NiuHandoffIoStatus::Failed;
            std::array<uint32_t, sizeof(wire) / sizeof(uint32_t)> words;
            std::memcpy(words.data(), &wire, sizeof(wire));
            for (size_t byte = 0; byte != destination.size(); ++byte) {
                const size_t position = offset + byte;
                destination[byte] = std::byte((words[position / 4] >> ((position % 4) * 8)) & 0xff);
            }
            if (offset == 0 && destination.size() == sizeof(wire)) {
                if (change_body_sequence_only) destination[sequence_offset] ^= std::byte{2};
                if (change_after_body) wire.receipt.header.publish_sequence += 2;
            }
            return NiuHandoffIoStatus::Complete;
        });
    }
};

TEST(NiuHandoffReceiverTest, ReadsOneStableLittleEndianMailboxWithoutCreatingAnOwner) {
    Mailbox mailbox;
    const auto result = mailbox.read();
    ASSERT_EQ(result.status, NiuHandoffReadStatus::Observed);
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(result.observation->wire().request.request_generation, 11u);
    EXPECT_EQ(result.observation->wire().receipt.header.publish_sequence, 14u);
    EXPECT_EQ(result.observation->wire().receipt.selector_observed[0][7].outstanding, 255u);
    EXPECT_EQ(mailbox.reads, (std::vector<std::pair<uint32_t, size_t>>{
                                {sequence_offset, 4}, {0, sizeof(tt_niu_handoff_region)}, {sequence_offset, 4}}));
    EXPECT_FALSE(std::is_default_constructible_v<NiuHandoffObservation>);
    EXPECT_FALSE((std::is_constructible_v<NiuHandoffObservation, tt_niu_handoff_region>));
}

TEST(NiuHandoffReceiverTest, TransportFailureNeverReturnsAnObservation) {
    for (size_t index = 0; index != 3; ++index) {
        Mailbox mailbox;
        mailbox.fail_read = index;
        const auto result = mailbox.read();
        EXPECT_EQ(result.status, NiuHandoffReadStatus::IoFailure);
        EXPECT_FALSE(result.observation);
        EXPECT_EQ(mailbox.reads.size(), index + 1);
    }
}

TEST(NiuHandoffReceiverTest, PropagatesTransportExceptionsWithoutAnObservation) {
    EXPECT_THROW(NiuHandoffReceiver::read([](uint32_t, std::span<std::byte>) -> NiuHandoffIoStatus {
                     throw std::runtime_error("actual transport failed");
                 }),
                 std::runtime_error);
}

TEST(NiuHandoffReceiverTest, UnpublishedAndOddSequencesNeverReturnAnObservation) {
    for (const uint32_t sequence : {0u, 3u, std::numeric_limits<uint32_t>::max()}) {
        Mailbox mailbox;
        mailbox.wire.receipt.header.publish_sequence = sequence;
        const auto result = mailbox.read();
        EXPECT_EQ(result.status, sequence == 0 ? NiuHandoffReadStatus::Unpublished : NiuHandoffReadStatus::InProgress);
        EXPECT_FALSE(result.observation);
        EXPECT_EQ(mailbox.reads.size(), 1u);
    }
}

TEST(NiuHandoffReceiverTest, RejectsChangedSequenceAndTornBody) {
    for (bool body_only : {false, true}) {
        Mailbox mailbox;
        mailbox.change_after_body = !body_only;
        mailbox.change_body_sequence_only = body_only;
        const auto result = mailbox.read();
        EXPECT_EQ(result.status, NiuHandoffReadStatus::Changed);
        EXPECT_FALSE(result.observation);
    }
}

TEST(NiuHandoffReceiverTest, RejectsUnknownWireAbiStatusErrorAndReservedField) {
    for (unsigned field = 0; field != 7; ++field) {
        Mailbox mailbox;
        switch (field) {
            case 0: ++mailbox.wire.receipt.header.abi_version; break;
            case 1: mailbox.wire.receipt.header.status = 4; break;
            case 2: mailbox.wire.receipt.header.error = 6; break;
            case 3: mailbox.wire.receipt.reserved = 1; break;
            case 4: mailbox.wire.request.reserved1 = 1; break;
            case 5: mailbox.wire.receipt.header.error = TT_NIU_HANDOFF_ERROR_SOURCE; break;
            case 6: mailbox.wire.receipt.header.status = TT_NIU_HANDOFF_REJECTED; break;
        }
        const auto result = mailbox.read();
        EXPECT_EQ(result.status, NiuHandoffReadStatus::Invalid);
        EXPECT_FALSE(result.observation);
    }
}

TEST(NiuHandoffReceiverTest, BootReadyPendingAndRejectedCannotMatchAnExecution) {
    for (const uint32_t status : {TT_NIU_HANDOFF_BOOT_READY, TT_NIU_HANDOFF_PENDING, TT_NIU_HANDOFF_REJECTED}) {
        Mailbox mailbox;
        const auto expected = mailbox.expected();
        mailbox.wire.receipt = {};
        auto& header = mailbox.wire.receipt.header;
        header.abi_version = TT_NIU_HANDOFF_VERSION;
        header.publish_sequence = 14;
        header.status = status;
        if (status == TT_NIU_HANDOFF_REJECTED) header.error = TT_NIU_HANDOFF_ERROR_SOURCE;
        const auto result = mailbox.read();
        ASSERT_EQ(result.status, NiuHandoffReadStatus::Observed);
        ASSERT_TRUE(result.observation);
        const auto check = NiuHandoffReceiver::check(*result.observation, expected);
        EXPECT_EQ(check, status == TT_NIU_HANDOFF_BOOT_READY ? NiuHandoffCheck::NotExecution :
                         status == TT_NIU_HANDOFF_PENDING ? NiuHandoffCheck::Pending : NiuHandoffCheck::Rejected);
    }
}

TEST(NiuHandoffReceiverTest, MatchesTheOriginalRequestWithoutRequiringSelectorNormalization) {
    Mailbox mailbox;
    const auto result = mailbox.read();
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, mailbox.expected()), NiuHandoffCheck::Match);
    EXPECT_EQ(result.observation->wire().receipt.selector_observed[0][7].outstanding, 255u);
}

TEST(NiuHandoffReceiverTest, RequiresEveryOriginalRequestWordIncludingGeneration) {
    Mailbox mailbox;
    const auto expected = mailbox.expected();
    for (size_t field = 0; field != sizeof(tt_niu_handoff_request) / 4; ++field) {
        std::array<uint32_t, sizeof(tt_niu_handoff_request) / 4> words;
        std::memcpy(words.data(), &expected.request, sizeof(expected.request));
        words[field] ^= 1;
        std::memcpy(&mailbox.wire.request, words.data(), sizeof(expected.request));
        const auto result = mailbox.read();
        if (field == offsetof(tt_niu_handoff_request, reserved1) / 4) {
            EXPECT_EQ(result.status, NiuHandoffReadStatus::Invalid);
            EXPECT_FALSE(result.observation);
            continue;
        }
        ASSERT_TRUE(result.observation);
        EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, expected), NiuHandoffCheck::RequestMismatch) << field;
    }
}

TEST(NiuHandoffReceiverTest, RequiresEveryEchoedIdentityWord) {
    for (unsigned field = 0; field != 7; ++field) {
        Mailbox mailbox;
        auto& header = mailbox.wire.receipt.header;
        std::array<uint32_t*, 7> fields{&header.challenge_lo, &header.challenge_hi, &header.boot_cookie_lo,
                                      &header.boot_cookie_hi, &header.invocation_id_lo, &header.invocation_id_hi,
                                      &header.group_generation};
        ++*fields[field];
        const auto result = mailbox.read();
        ASSERT_TRUE(result.observation);
        EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, mailbox.expected()), NiuHandoffCheck::IdentityMismatch);
    }
}

TEST(NiuHandoffReceiverTest, RequiresTheExactActualGoOrdinalAndCount) {
    for (unsigned field = 0; field != 2; ++field) {
        Mailbox mailbox;
        if (field == 0) ++mailbox.wire.receipt.consumed_go_sequence;
        else --mailbox.wire.receipt.header.completed_go_count;
        const auto result = mailbox.read();
        ASSERT_TRUE(result.observation);
        EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, mailbox.expected()), NiuHandoffCheck::GoMismatch);
    }
}

TEST(NiuHandoffReceiverTest, RejectsInvalidOrOverflowingRequestBeforeComparingReceipts) {
    for (unsigned field = 0; field != 14; ++field) {
        Mailbox mailbox;
        auto expected = mailbox.expected();
        auto& request = expected.request;
        switch (field) {
            case 0: request.request_generation = 0; break;
            case 1: request.challenge_lo = request.challenge_hi = 0; break;
            case 2: request.boot_cookie_lo = request.boot_cookie_hi = 0; break;
            case 3: request.invocation_id_lo = request.invocation_id_hi = 0; break;
            case 4: request.group_generation = 0; break;
            case 5: request.final_go_count = 0; break;
            case 6: request.first_go_sequence = std::numeric_limits<uint32_t>::max(); break;
            case 7: request.selector_mask_noc1 = 0x7fff; break;
            case 8: request.required_processor_mask = 15; break;
            case 9: request.reserved1 = 1; break;
            case 10: ++request.abi_version; break;
            case 11: ++request.traffic_contract; break;
            case 12: request.first_go_sequence = 0; break;
            case 13: request.selector_mask_noc0 = 0x7fff; break;
        }
        const auto result = mailbox.read();
        ASSERT_TRUE(result.observation);
        EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, expected), NiuHandoffCheck::InvalidRequest) << field;
    }
}

TEST(NiuHandoffReceiverTest, AcceptsTheLastNonWrappingGoOrdinalAsAnObservationOnly) {
    Mailbox mailbox;
    mailbox.wire.request.first_go_sequence = std::numeric_limits<uint32_t>::max();
    mailbox.wire.request.final_go_count = 1;
    mailbox.wire.receipt.header.completed_go_count = 1;
    mailbox.wire.receipt.consumed_go_sequence = std::numeric_limits<uint32_t>::max();
    const auto result = mailbox.read();
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, mailbox.expected()), NiuHandoffCheck::Match);
}

TEST(NiuHandoffReceiverTest, RequiresActualStageAndProcessorCoverage) {
    for (unsigned field = 0; field != 2; ++field) {
        Mailbox mailbox;
        if (field == 0) --mailbox.wire.receipt.header.completed_stage_count;
        else mailbox.wire.receipt.header.completed_processor_mask = 15;
        const auto result = mailbox.read();
        ASSERT_TRUE(result.observation);
        EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, mailbox.expected()), NiuHandoffCheck::StageMismatch);
    }
}

TEST(NiuHandoffReceiverTest, RejectsFewerStagesThanConsumedGosEvenWhenExpectedAgrees) {
    Mailbox mailbox;
    auto expected = mailbox.expected();
    expected.completed_stage_count = 1;
    mailbox.wire.receipt.header.completed_stage_count = 1;
    // The real firmware rejects a GO with no stage. Caller-supplied expected
    // bytes cannot make one stage cover the two consumed execution GOs.
    const auto result = mailbox.read();
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, expected), NiuHandoffCheck::StageMismatch);
}

TEST(NiuHandoffReceiverTest, ChecksBothSourceProducersAndBothNocsWithUnsignedCounterWrap) {
    Mailbox mailbox;
    mailbox.wire.receipt.source_expected[1][0][4] = std::numeric_limits<uint32_t>::max();
    mailbox.wire.receipt.source_expected[1][1][4] = 2;
    mailbox.wire.receipt.source_observed[1][4] = 1;
    auto result = mailbox.read();
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, mailbox.expected()), NiuHandoffCheck::Match);
    ++mailbox.wire.receipt.source_expected[1][1][4];
    result = mailbox.read();
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, mailbox.expected()), NiuHandoffCheck::SourceMismatch);
}

TEST(NiuHandoffReceiverTest, NoPayloadCannotClaimDynamicSourceCounters) {
    Mailbox mailbox;
    auto expected = mailbox.expected();
    expected.source_valid_noc_mask = 0;
    mailbox.wire.receipt.header.source_valid_noc_mask = 0;
    auto result = mailbox.read();
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, expected), NiuHandoffCheck::SourceMismatch);
    for (auto& noc : mailbox.wire.receipt.source_expected)
        for (auto& risc : noc) std::fill(std::begin(risc), std::end(risc), 0);
    for (auto& noc : mailbox.wire.receipt.source_observed) std::fill(std::begin(noc), std::end(noc), 0);
    result = mailbox.read();
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, expected), NiuHandoffCheck::Match);
}

TEST(NiuHandoffReceiverTest, RequiresCompleteControlTrafficAndDescriptorCoverage) {
    for (unsigned field = 0; field != 6; ++field) {
        Mailbox mailbox;
        auto& receipt = mailbox.wire.receipt;
        switch (field) {
            case 0: receipt.header.control_valid_noc_mask = 1; break;
            case 1: receipt.control.valid_noc_mask = 1; break;
            case 2: receipt.control.ready_cmd_mask_noc0 = 7; break;
            case 3: receipt.control.ready_cmd_mask_noc1 = 7; break;
            case 4: ++receipt.control.control_contract; break;
            case 5: ++receipt.control_observed[1][2]; break;
        }
        const auto result = mailbox.read();
        ASSERT_TRUE(result.observation);
        EXPECT_EQ(NiuHandoffReceiver::check(*result.observation, mailbox.expected()), NiuHandoffCheck::ControlMismatch);
    }
}

}  // namespace
}  // namespace tt::tt_metal::detail
