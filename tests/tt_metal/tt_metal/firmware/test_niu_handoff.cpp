// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "internal/niu_handoff.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tt::niu_handoff {
namespace {
using BeginStatus = ItemWitnessPublisher::SeriesBeginStatus;

template <class Receiver>
concept exposes_item_publisher = requires {
    typename Receiver::ItemPublisher;
    typename Receiver::ItemWitnessRequest;
    typename Receiver::ItemWitnessReceipt;
};

template <class Receiver>
void check_item_publisher_contract() {
    if constexpr (!exposes_item_publisher<Receiver>) {
        ADD_FAILURE() << "Device has no issue-before-close item witness publisher";
    } else {
        using Request = typename Receiver::ItemWitnessRequest;
        using Receipt = typename Receiver::ItemWitnessReceipt;
        struct ItemIO {
            Request request{};
            Receipt receipt{};
            tt_niu_item_witness_acknowledgement acknowledgement{};
            uint32_t outstanding = 1;
            uint32_t outgoing = 1;
            std::vector<unsigned> events;
            uint32_t requestGeneration() { return request.request_generation; }
            uint32_t receiptSequence() { return receipt.publish_sequence; }
            uint32_t acknowledgementSequence() { return acknowledgement.publish_sequence; }
            Request readRequest() { return request; }
            tt_niu_item_witness_acknowledgement readAcknowledgement() { return acknowledgement; }
            void invalidate() { events.push_back(3); }
            tt_niu_handoff_selector_observation selectorObserved(unsigned noc, unsigned selector) {
                EXPECT_EQ(noc, request.noc);
                EXPECT_EQ(selector, request.selector);
                events.push_back(4);
                return {outstanding, outgoing};
            }
            void publish(const Receipt& value) {
                receipt = value;
                events.push_back(10 + value.state);
            }
        };
        ItemIO io;
        io.request.abi_version = TT_NIU_ITEM_WITNESS_VERSION;
        io.request.request_generation = 1;
        io.request.boot_cookie = 3;
        io.request.invocation_id = 5;
        io.request.item = 7;
        io.request.generation = UINT64_C(1) << 35;
        io.request.occurrence = 1;
        io.request.leaf = 2;
        io.request.noc = 1;
        io.request.selector = 15;
        io.request.issuer = 2;
        io.request.completion = 2;
        typename Receiver::ItemPublisher publisher;
        ASSERT_TRUE(publisher.open(io));
        ASSERT_EQ(publisher.begin(io), BeginStatus::Begun);
        EXPECT_FALSE(publisher.close(io, [] {}));
        EXPECT_EQ(io.receipt.item.closed_occurrence, 0u);
        ASSERT_TRUE(publisher.issue(io, [&] { io.events.push_back(1); }));
        EXPECT_EQ(io.receipt.state, 1u);
        EXPECT_EQ(io.receipt.item.issued_occurrence, io.request.occurrence);
        EXPECT_EQ(io.receipt.item.closed_occurrence, 0u);
        EXPECT_FALSE(publisher.issue(io, [] {}));
        bool waited = false;
        ASSERT_TRUE(publisher.close(io, [&] {
            waited = true;
            io.events.push_back(2);
            io.outstanding = io.outgoing = 0;
        }));
        EXPECT_TRUE(waited);
        EXPECT_EQ(io.receipt.state, 2u);
        EXPECT_EQ(io.receipt.item.closed_occurrence, io.request.occurrence);
        EXPECT_EQ(io.receipt.publish_sequence & 1u, 0u);
        const auto issue = std::find(io.events.begin(), io.events.end(), 1u);
        const auto issued = std::find(io.events.begin(), io.events.end(), 11u);
        const auto wait = std::find(io.events.begin(), io.events.end(), 2u);
        const auto closed = std::find(io.events.begin(), io.events.end(), 12u);
        EXPECT_LT(issue, issued);
        EXPECT_LT(issued, wait);
        EXPECT_LT(wait, closed);
        EXPECT_FALSE(publisher.close(io, [] {}));
        EXPECT_FALSE(publisher.open(io));
        typename Receiver::ItemPublisher recreated;
        EXPECT_FALSE(recreated.open(io)) << "A fresh writer cannot reopen an already published SDK slot";
        // A wait callback returning without real hardware closure may not
        // publish Closed, and a changed original request poisons the writer.
        typename Receiver::ItemPublisher blocked;
        io.receipt = {};
        ASSERT_TRUE(blocked.open(io));
        ASSERT_EQ(blocked.begin(io), BeginStatus::Begun);
        ASSERT_TRUE(blocked.issue(io, [] {}));
        io.outstanding = 1;
        EXPECT_FALSE(blocked.close(io, [] {}));
        EXPECT_EQ(io.receipt.state, 3u);
        EXPECT_EQ(io.receipt.item.closed_occurrence, 0u);
        typename Receiver::ItemPublisher changed;
        io.receipt = {};
        ASSERT_TRUE(changed.open(io));
        ASSERT_EQ(changed.begin(io), BeginStatus::Begun);
        ASSERT_TRUE(changed.issue(io, [] {}));
        ++io.request.generation;
        io.outstanding = io.outgoing = 0;
        EXPECT_FALSE(changed.close(io, [] {}));
        EXPECT_EQ(io.receipt.state, 3u);
        io.request.occurrence = std::numeric_limits<uint64_t>::max();
        typename Receiver::ItemPublisher wrapped;
        EXPECT_FALSE(wrapped.open(io));
    }
}

TEST(NiuItemPublisher, RequiresActualIssueWaitAndUnchangedOriginalRequest) {
    check_item_publisher_contract<Receiver>();
}

TEST(NiuItemPublisher, PublisherOwnershipCannotBeCopiedOrMoved) {
    EXPECT_FALSE(std::is_copy_constructible_v<ItemWitnessPublisher>);
    EXPECT_FALSE(std::is_copy_assignable_v<ItemWitnessPublisher>);
    EXPECT_FALSE(std::is_move_constructible_v<ItemWitnessPublisher>);
    EXPECT_FALSE(std::is_move_assignable_v<ItemWitnessPublisher>);
}

TEST(NiuItemPublisher, IndependentPublishersCannotBothOpenOneUnpublishedSlot) {
    struct IO {
        tt_niu_item_witness_request request{TT_NIU_ITEM_WITNESS_VERSION, 1, 3, 5, 7, 11, 1, 2, 1, 15, 2, 2, 0};
        tt_niu_item_witness_receipt receipt{};
        tt_niu_item_witness_acknowledgement acknowledgement{};
        void invalidate() {}
        uint32_t requestGeneration() { return request.request_generation; }
        uint32_t receiptSequence() { return receipt.publish_sequence; }
        uint32_t acknowledgementSequence() { return acknowledgement.publish_sequence; }
        tt_niu_item_witness_request readRequest() { return request; }
        tt_niu_item_witness_acknowledgement readAcknowledgement() { return acknowledgement; }
        tt_niu_handoff_selector_observation selectorObserved(unsigned, unsigned) { return {}; }
        void publish(const tt_niu_item_witness_receipt& value) { receipt = value; }
    } io;
    ItemWitnessPublisher first;
    ItemWitnessPublisher second;
    ASSERT_TRUE(first.open(io));
    ASSERT_EQ(first.begin(io), BeginStatus::Begun);
    EXPECT_FALSE(second.open(io)) << "Single-use requires a real claim or qualified unique publisher";
    unsigned issues = 0;
    ASSERT_TRUE(first.issue(io, [&] { ++issues; }));
    EXPECT_FALSE(second.issue(io, [&] { ++issues; }));
    EXPECT_EQ(issues, 1u);
}

struct ItemMemoryPlatform {
    void invalidate() {}
    tt_niu_handoff_selector_observation selectorObserved(unsigned, unsigned) { return {}; }
};

template <class Receiver>
concept exposes_item_memory = requires { typename Receiver::template ItemMemory<ItemMemoryPlatform>; };

template <class Receiver>
void check_item_memory_contract() {
    if constexpr (!exposes_item_memory<Receiver>) {
        ADD_FAILURE() << "Device has no ordered volatile L1 item witness publication adapter";
    } else {
        using Request = typename Receiver::ItemWitnessRequest;
        using Receipt = typename Receiver::ItemWitnessReceipt;
        typename Receiver::ItemWitnessRegion region{};
        auto& request = region.request;
        request.abi_version = TT_NIU_ITEM_WITNESS_VERSION;
        request.request_generation = 1;
        request.boot_cookie = UINT64_C(0x123456789abcdef0);
        request.invocation_id = 5;
        request.item = 7;
        request.generation = UINT64_C(1) << 35;
        request.occurrence = (UINT64_C(1) << 36) + 1;
        request.leaf = 2;
        request.noc = 1;
        request.selector = 15;
        request.issuer = 2;
        request.completion = 2;
        struct Platform {
            decltype(region)& memory;
            std::vector<std::pair<uint32_t, uint32_t>> fences;
            void invalidate() { fences.emplace_back(memory.receipt.publish_sequence, memory.receipt.state); }
            tt_niu_handoff_selector_observation selectorObserved(unsigned, unsigned) { return {}; }
        } platform{region};
        typename Receiver::template ItemMemory<Platform> memory(region, platform);
        const Request copied = memory.readRequest();
        EXPECT_EQ(copied.boot_cookie, request.boot_cookie);
        EXPECT_EQ(copied.generation, request.generation);
        EXPECT_EQ(copied.occurrence, request.occurrence);
        Receipt receipt{};
        receipt.request = request;
        receipt.publish_sequence = 2;
        receipt.state = 1;
        receipt.item.issued_occurrence = request.occurrence;
        memory.publish(receipt);
        ASSERT_EQ(platform.fences.size(), 2u);
        EXPECT_EQ(platform.fences[0], (std::pair<uint32_t, uint32_t>{1, 0}));
        EXPECT_EQ(platform.fences[1], (std::pair<uint32_t, uint32_t>{1, 1}));
        EXPECT_EQ(region.receipt.publish_sequence, 2u);
        EXPECT_EQ(region.receipt.item.issued_occurrence, request.occurrence);
        EXPECT_EQ(region.receipt.item.closed_occurrence, 0u);
        EXPECT_EQ(region.request.boot_cookie, request.boot_cookie);
        EXPECT_EQ(region.request.generation, request.generation);
    }
}

TEST(NiuItemPublisher, PublishesOddBodyFenceEvenToActualVolatileSlot) {
    check_item_memory_contract<Receiver>();
}

// This fixture models only the ordered L1/MMIO boundary. An acknowledgement
// frees a receipt slot; Channel/Sync must independently place begin after the
// actual payload visibility, retention and all-consumer release boundaries.
template <class Receiver>
concept exposes_rolling_item_publisher = requires {
    typename Receiver::ItemWitnessAcknowledgement;
};

template <class Receiver>
void check_rolling_item_reuse() {
    if constexpr (!exposes_rolling_item_publisher<Receiver>) {
        ADD_FAILURE() << "Item witness cannot reuse its real slot after an exact Closed acknowledgement";
    } else {
        using Request = typename Receiver::ItemWitnessRequest;
        using Receipt = typename Receiver::ItemWitnessReceipt;
        using Acknowledgement = typename Receiver::ItemWitnessAcknowledgement;
        struct IO {
            Request request{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
            Receipt receipt{};
            Acknowledgement acknowledgement{};
            unsigned issues = 0;
            unsigned waits = 0;
            void invalidate() {}
            uint32_t requestGeneration() { return request.request_generation; }
            uint32_t receiptSequence() { return receipt.publish_sequence; }
            uint32_t acknowledgementSequence() { return acknowledgement.publish_sequence; }
            Request readRequest() { return request; }
            Acknowledgement readAcknowledgement() { return acknowledgement; }
            tt_niu_handoff_selector_observation selectorObserved(unsigned, unsigned) { return {}; }
            void publish(const Receipt& value) { receipt = value; }
        } io;
        typename Receiver::ItemPublisher publisher;
        if constexpr (!requires { publisher.begin(io); }) {
            ADD_FAILURE() << "Item publisher has no source-owned occurrence begin boundary";
        } else {
            const Request original = io.request;
            ASSERT_TRUE(publisher.open(io));
            ASSERT_EQ(publisher.begin(io), BeginStatus::Begun);
            ASSERT_TRUE(publisher.issue(io, [&] { ++io.issues; }));
            ASSERT_TRUE(publisher.close(io, [&] { ++io.waits; }));
            const Receipt first = io.receipt;
            ASSERT_EQ(first.item.closed_occurrence, original.occurrence);
            const uint64_t next = original.occurrence + 1;
            EXPECT_EQ(publisher.begin(io), BeginStatus::Pending) << "Closed alone cannot overwrite the unreceived slot";
            EXPECT_EQ(io.receipt.publish_sequence, first.publish_sequence);
            EXPECT_FALSE(publisher.issue(io, [&] { ++io.issues; }));
            io.acknowledgement.request = first.request;
            io.acknowledgement.closed_sequence = first.publish_sequence;
            io.acknowledgement.publish_sequence = 2;
            ASSERT_EQ(publisher.begin(io), BeginStatus::Begun);
            EXPECT_EQ(io.request.request_generation, original.request_generation);
            EXPECT_EQ(io.request.boot_cookie, original.boot_cookie);
            EXPECT_EQ(io.request.invocation_id, original.invocation_id);
            EXPECT_EQ(io.request.item, original.item);
            EXPECT_EQ(io.request.generation, original.generation);
            EXPECT_EQ(io.request.occurrence, original.occurrence);
            EXPECT_GT(io.receipt.publish_sequence, first.publish_sequence);
            EXPECT_EQ(io.receipt.request.occurrence, next);
            EXPECT_EQ(io.receipt.item.issued_occurrence, 0u);
            EXPECT_EQ(io.receipt.item.closed_occurrence, 0u);
            ASSERT_TRUE(publisher.issue(io, [&] { ++io.issues; }));
            ASSERT_TRUE(publisher.close(io, [&] { ++io.waits; }));
            EXPECT_EQ(io.receipt.item.issued_occurrence, next);
            EXPECT_EQ(io.receipt.item.closed_occurrence, next);
            EXPECT_EQ(io.receipt.request.selector, original.selector);
            EXPECT_EQ(io.receipt.request.noc, original.noc);
            EXPECT_EQ(io.receipt.request.issuer, original.issuer);
            EXPECT_GT(io.receipt.publish_sequence, first.publish_sequence);
            EXPECT_EQ(io.receipt.publish_sequence & 1u, 0u);
            EXPECT_EQ(publisher.begin(io), BeginStatus::Pending) << "The previous round acknowledgement is stale";
            io.acknowledgement.request = io.receipt.request;
            io.acknowledgement.closed_sequence = io.receipt.publish_sequence;
            io.acknowledgement.publish_sequence = 4;
            ASSERT_EQ(publisher.begin(io), BeginStatus::Begun);
            EXPECT_EQ(io.issues, 2u);
            EXPECT_EQ(io.waits, 2u);
            EXPECT_EQ(first.item.closed_occurrence, original.occurrence);
        }
    }
}

TEST(NiuItemPublisherRolling, ReusesSlotOnlyAfterExactClosedAcknowledgement) {
    check_rolling_item_reuse<Receiver>();
}

TEST(NiuItemPublisherRolling, ExactAcknowledgementAllowsActualSecondPublication) {
    // Keep an executable regression using the source-owned consecutive
    // occurrence protocol. Only the physical
    // transport is substituted; all open/begin/issue/close decisions are production.
    struct IO {
        tt_niu_item_witness_request request{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        tt_niu_item_witness_receipt receipt{};
        using Acknowledgement = tt_niu_item_witness_acknowledgement;
        Acknowledgement acknowledgement{};
        void invalidate() {}
        uint32_t requestGeneration() { return request.request_generation; }
        uint32_t receiptSequence() { return receipt.publish_sequence; }
        uint32_t acknowledgementSequence() { return acknowledgement.publish_sequence; }
        tt_niu_item_witness_request readRequest() { return request; }
        Acknowledgement readAcknowledgement() { return acknowledgement; }
        tt_niu_handoff_selector_observation selectorObserved(unsigned, unsigned) { return {}; }
        void publish(const tt_niu_item_witness_receipt& value) { receipt = value; }
    } io;
    ItemWitnessPublisher publisher;
    ASSERT_TRUE(publisher.open(io));
    ASSERT_EQ(publisher.begin(io), BeginStatus::Begun);
    ASSERT_TRUE(publisher.issue(io, [] {}));
    ASSERT_TRUE(publisher.close(io, [] {}));
    io.acknowledgement.request = io.receipt.request;
    io.acknowledgement.closed_sequence = io.receipt.publish_sequence;
    io.acknowledgement.publish_sequence = 2;
    ASSERT_EQ(publisher.begin(io), BeginStatus::Begun)
        << "The original publisher must continue only after the exact acknowledged Closed round";
    ASSERT_TRUE(publisher.issue(io, [] {}));
    ASSERT_TRUE(publisher.close(io, [] {}));
    EXPECT_EQ(io.receipt.item.closed_occurrence, io.request.occurrence + 1);
    EXPECT_EQ(io.request.occurrence, 1u);
}

template <class Receiver>
void check_rolling_item_rejections() {
    if constexpr (!exposes_rolling_item_publisher<Receiver>) {
        ADD_FAILURE() << "Item witness has no checked nonwrapping rolling acknowledgement protocol";
    } else {
        using Request = typename Receiver::ItemWitnessRequest;
        using Receipt = typename Receiver::ItemWitnessReceipt;
        using Acknowledgement = typename Receiver::ItemWitnessAcknowledgement;
        struct IO {
            Request request{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
            Receipt receipt{};
            Acknowledgement acknowledgement{};
            bool changing_ack = false;
            unsigned acknowledgement_reads = 0;
            void invalidate() {}
            uint32_t requestGeneration() { return request.request_generation; }
            uint32_t receiptSequence() { return receipt.publish_sequence; }
            uint32_t acknowledgementSequence() {
                return acknowledgement.publish_sequence +
                       (changing_ack && acknowledgement_reads++ != 0 ? 2 : 0);
            }
            Request readRequest() { return request; }
            Acknowledgement readAcknowledgement() { return acknowledgement; }
            tt_niu_handoff_selector_observation selectorObserved(unsigned, unsigned) { return {}; }
            void publish(const Receipt& value) { receipt = value; }
        };
        typename Receiver::ItemPublisher probe;
        IO probe_io;
        if constexpr (!requires { probe.begin(probe_io); }) {
            ADD_FAILURE() << "Item publisher lacks checked begin";
        } else {
            // Each malformed acknowledgement is exercised on its own original
            // publisher; a rejection never silently creates a new owner.
            for (unsigned mutation = 0; mutation != 9; ++mutation) {
                IO io;
                typename Receiver::ItemPublisher publisher;
                ASSERT_TRUE(publisher.open(io));
                ASSERT_EQ(publisher.begin(io), BeginStatus::Begun);
                ASSERT_TRUE(publisher.issue(io, [] {}));
                ASSERT_TRUE(publisher.close(io, [] {}));
                io.acknowledgement.request = io.receipt.request;
                io.acknowledgement.closed_sequence = io.receipt.publish_sequence;
                io.acknowledgement.publish_sequence = 2;
                switch (mutation) {
                    case 0: ++io.acknowledgement.request.boot_cookie; break;
                    case 1: ++io.acknowledgement.request.invocation_id; break;
                    case 2: --io.acknowledgement.request.selector; break;
                    case 3: ++io.acknowledgement.request.occurrence; break;
                    case 4: io.acknowledgement.closed_sequence += 2; break;
                    case 5: io.acknowledgement.publish_sequence = 1; break;
                    case 6: io.changing_ack = true; break;
                    case 7: io.acknowledgement.publish_sequence = 4; break;
                    case 8: io.acknowledgement.reserved[2] = 1; break;
                }
                EXPECT_EQ(
                    publisher.begin(io), mutation == 5 || mutation == 6 ? BeginStatus::Pending : BeginStatus::Rejected)
                    << mutation;
                unsigned issues = 0;
                EXPECT_FALSE(publisher.issue(io, [&] { ++issues; }));
                EXPECT_EQ(issues, 0u);
            }
            for (const uint64_t initial : {uint64_t{0}, uint64_t{2}, uint64_t{16}, uint64_t{17}, UINT64_MAX}) {
                IO io;
                io.request.occurrence = initial;
                typename Receiver::ItemPublisher publisher;
                EXPECT_FALSE(publisher.open(io)) << initial;
                EXPECT_EQ(publisher.begin(io), BeginStatus::Rejected);
                EXPECT_FALSE(publisher.issue(io, [] {}));
                EXPECT_EQ(io.receipt.publish_sequence, 0u);
            }
        }
    }
}

TEST(NiuItemPublisherRolling, RejectsForeignTornAndRegressingAcknowledgements) {
    check_rolling_item_rejections<Receiver>();
}

template <class Receiver>
void check_rolling_item_volatile_memory() {
    if constexpr (!exposes_rolling_item_publisher<Receiver>) {
        ADD_FAILURE() << "No versioned rolling witness acknowledgement memory";
    } else {
        using Region = typename Receiver::ItemWitnessRegion;
        Region region{};
        region.request = {TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        const auto original = region.request;
        struct Platform {
            Region& region;
            bool tear_ack = false;
            unsigned fences = 0;
            void invalidate() {
                ++fences;
                if (tear_ack) region.acknowledgement.publish_sequence += 2;
            }
            tt_niu_handoff_selector_observation selectorObserved(unsigned, unsigned) { return {}; }
        } platform{region};
        typename Receiver::template ItemMemory<Platform> memory(region, platform);
        typename Receiver::ItemPublisher publisher;
        ASSERT_TRUE(publisher.open(memory));
        ASSERT_EQ(publisher.begin(memory), BeginStatus::Begun);
        ASSERT_TRUE(publisher.issue(memory, [] {}));
        ASSERT_TRUE(publisher.close(memory, [] {}));
        const auto first = region.receipt;
        volatile Region& actual = region;
        actual.acknowledgement.publish_sequence = 1;
        region.acknowledgement.request = first.request;
        actual.acknowledgement.closed_sequence = first.publish_sequence;
        actual.acknowledgement.publish_sequence = 2;
        const auto copied = memory.readAcknowledgement();
        EXPECT_EQ(copied.request.occurrence, first.request.occurrence);
        EXPECT_EQ(copied.closed_sequence, first.publish_sequence);
        ASSERT_EQ(publisher.begin(memory), BeginStatus::Begun);
        EXPECT_EQ(region.request.occurrence, original.occurrence);
        EXPECT_EQ(region.request.request_generation, original.request_generation);
        ASSERT_TRUE(publisher.issue(memory, [] {}));
        ASSERT_TRUE(publisher.close(memory, [] {}));
        region.acknowledgement.request = region.receipt.request;
        actual.acknowledgement.closed_sequence = region.receipt.publish_sequence;
        actual.acknowledgement.publish_sequence = 4;
        const auto closed = region.receipt.publish_sequence;
        platform.tear_ack = true;
        EXPECT_EQ(publisher.begin(memory), BeginStatus::Pending);
        unsigned issues = 0;
        EXPECT_FALSE(publisher.issue(memory, [&] { ++issues; }));
        EXPECT_EQ(issues, 0u);
        EXPECT_GE(region.receipt.publish_sequence, closed);
    }
}

TEST(NiuItemPublisherRolling, ReadsActualVolatileAcknowledgementAndRejectsTearing) {
    check_rolling_item_volatile_memory<Receiver>();
}

TEST(NiuItemPublisherRolling, WireRegionsHaveExactVersionedAlignmentAndOffsets) {
    EXPECT_EQ(TT_NIU_ITEM_WITNESS_VERSION, 3u);
    EXPECT_EQ(sizeof(tt_niu_item_witness_acknowledgement), 96u);
    EXPECT_EQ(sizeof(tt_niu_item_witness_region), 256u);
    EXPECT_EQ(alignof(tt_niu_item_witness_acknowledgement), TT_NIU_ITEM_WITNESS_ALIGNMENT);
    EXPECT_EQ(alignof(tt_niu_item_witness_region), TT_NIU_ITEM_WITNESS_ALIGNMENT);
    EXPECT_EQ(offsetof(tt_niu_item_witness_acknowledgement, publish_sequence), 0u);
    EXPECT_EQ(offsetof(tt_niu_item_witness_acknowledgement, closed_sequence), 4u);
    EXPECT_EQ(offsetof(tt_niu_item_witness_acknowledgement, request), 8u);
    EXPECT_EQ(offsetof(tt_niu_item_witness_acknowledgement, reserved), 72u);
    EXPECT_EQ(offsetof(tt_niu_item_witness_region, acknowledgement), 160u);
}

TEST(NiuItemPublisherRolling, RejectsEachNonzeroReservedWireWordBeforeSecondIssue) {
    for (unsigned word = 0; word != 3; ++word) {
        tt_niu_item_witness_region region{};
        region.request = {TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        ItemMemoryPlatform platform;
        ItemWitnessMemory memory(region, platform);
        ItemWitnessPublisher publisher;
        ASSERT_TRUE(publisher.open(memory));
        ASSERT_EQ(publisher.begin(memory), BeginStatus::Begun);
        ASSERT_TRUE(publisher.issue(memory, [] {}));
        ASSERT_TRUE(publisher.close(memory, [] {}));
        region.acknowledgement.request = region.receipt.request;
        region.acknowledgement.closed_sequence = region.receipt.publish_sequence;
        region.acknowledgement.reserved[word] = 1;
        region.acknowledgement.publish_sequence = 2;
        EXPECT_EQ(publisher.begin(memory), BeginStatus::Rejected) << word;
        EXPECT_EQ(region.receipt.state, TT_NIU_ITEM_WITNESS_REJECTED);
        unsigned issues = 0;
        EXPECT_FALSE(publisher.issue(memory, [&] { ++issues; }));
        EXPECT_EQ(issues, 0u);
    }
}

TEST(NiuItemPublisherRolling, PublicationSequencesNeverWrap) {
    // Pure arithmetic permits exhaustive edge checks without mutating a live
    // publisher's private state or billions of physical publications.
    const auto check = []<class ReceiverType>() {
        if constexpr (requires { ReceiverType::advanceItemWitnessSequence(uint32_t{0}); }) {
            const auto first = ReceiverType::advanceItemWitnessSequence(0);
            EXPECT_TRUE(first.available);
            EXPECT_EQ(first.sequence, 2u);
            const auto last = ReceiverType::advanceItemWitnessSequence(UINT32_MAX - 3);
            EXPECT_TRUE(last.available);
            EXPECT_EQ(last.sequence, UINT32_MAX - 1);
            for (const uint32_t current : {uint32_t{1}, UINT32_MAX - 2, UINT32_MAX - 1, UINT32_MAX}) {
                const auto exhausted = ReceiverType::advanceItemWitnessSequence(current);
                EXPECT_FALSE(exhausted.available);
                EXPECT_EQ(exhausted.sequence, current);
            }
        } else {
            ADD_FAILURE() << "Item witness publication has no checked nonwrapping sequence boundary";
        }
    };
    check.template operator()<Receiver>();
}

TEST(NiuItemPublisherRolling, BeginBudgetsPendingIssuedClosedAndSeriesEndBeforeAnyPublication) {
    const auto check = []<class ReceiverType>() {
        if constexpr (requires { ReceiverType::itemWitnessRoundAvailable(uint32_t{0}); }) {
            EXPECT_TRUE(ReceiverType::itemWitnessRoundAvailable(0));
            EXPECT_TRUE(ReceiverType::itemWitnessRoundAvailable(UINT32_MAX - 9));
            for (const uint32_t current : {uint32_t{1}, UINT32_MAX - 7, UINT32_MAX - 5, UINT32_MAX - 3,
                                           UINT32_MAX - 1, UINT32_MAX})
                EXPECT_FALSE(ReceiverType::itemWitnessRoundAvailable(current)) << current;
        } else {
            ADD_FAILURE() << "Source begin has no complete-round and End nonwrapping publication budget";
        }
    };
    check.template operator()<Receiver>();
}

// This uses the production volatile L1 adapter and publisher. It models the
// host's ACK stores only; it does not construct a boot or managed invocation.
template <class ReceiverType>
struct ItemSeriesFixture {
    using Region = typename ReceiverType::ItemWitnessRegion;
    using Receipt = typename ReceiverType::ItemWitnessReceipt;
    Region region{};
    struct Platform {
        uint32_t outstanding = 0;
        uint32_t outgoing = 0;
        void invalidate() {}
        tt_niu_handoff_selector_observation selectorObserved(unsigned, unsigned) {
            return {outstanding, outgoing};
        }
    } platform;
    typename ReceiverType::template ItemMemory<Platform> memory{region, platform};
    unsigned issues = 0;
    unsigned waits = 0;

    ItemSeriesFixture() {
        region.request = {TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
    }
    void acknowledge(uint32_t sequence) {
        region.acknowledgement = {};
        region.acknowledgement.publish_sequence = sequence;
        region.acknowledgement.closed_sequence = region.receipt.publish_sequence;
        region.acknowledgement.request = region.receipt.request;
    }
};

template <class ReceiverType>
concept exposes_item_series_publisher = requires(
    typename ReceiverType::ItemPublisher& publisher, ItemSeriesFixture<ReceiverType>& fixture) {
    typename ReceiverType::ItemPublisher::SeriesFinishStatus;
    publisher.open(fixture.memory);
    publisher.begin(fixture.memory);
    publisher.finish(fixture.memory);
};

template <class ReceiverType, class Check>
void check_item_series_publisher(Check check) {
    if constexpr (!exposes_item_series_publisher<ReceiverType>) {
        ADD_FAILURE() << "Item publisher lacks source-owned open/begin/SeriesEnd reception";
    } else {
        ItemSeriesFixture<ReceiverType> fixture;
        typename ReceiverType::ItemPublisher publisher;
        check(publisher, fixture);
    }
}

// Check the fixed little-endian wire offsets without requiring the proposed
// tagged-union C++ field names before the shared ABI is integrated.
template <class Receipt>
uint64_t item_series_wire(const Receipt& receipt, size_t offset, unsigned bytes) {
    const auto* data = reinterpret_cast<const unsigned char*>(&receipt);
    uint64_t value = 0;
    for (unsigned byte = 0; byte != bytes; ++byte) value |= uint64_t(data[offset + byte]) << (8 * byte);
    return value;
}

TEST(NiuItemPublisherSeries, ZeroOccurrencesRequireExplicitSourceSeriesEnd) {
    check_item_series_publisher<Receiver>([](auto& publisher, auto& fixture) {
        using Finish = typename std::remove_reference_t<decltype(publisher)>::SeriesFinishStatus;
        ASSERT_TRUE(publisher.open(fixture.memory));
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 2u);
        EXPECT_EQ(fixture.region.receipt.state, 4u);  // SeriesOpen, never an issued item.
        ASSERT_EQ(publisher.finish(fixture.memory), Finish::NotExecuted);
        const auto& end = fixture.region.receipt;
        EXPECT_EQ(end.publish_sequence, 4u);
        EXPECT_EQ(end.state, 5u);  // SeriesEnd.
        EXPECT_EQ(item_series_wire(end, 72, 8), 0u);  // final_count.
        EXPECT_EQ(item_series_wire(end, 80, 4), 0u);  // Closed frontier.
        EXPECT_EQ(item_series_wire(end, 84, 4), 0u);  // ACK frontier.
        EXPECT_EQ(item_series_wire(end, 88, 8), 0u);
        EXPECT_EQ(fixture.issues, 0u);
        EXPECT_EQ(fixture.waits, 0u);
        EXPECT_EQ(end.request.occurrence, 1u);
        EXPECT_EQ(fixture.region.request.occurrence, 1u);
    });
}

TEST(NiuItemPublisherSeries, FinalExactAcknowledgementEndsWithoutStartingAPhantomOccurrence) {
    check_item_series_publisher<Receiver>([](auto& publisher, auto& fixture) {
        using Finish = typename std::remove_reference_t<decltype(publisher)>::SeriesFinishStatus;
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 4u);
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_PENDING);
        ASSERT_TRUE(publisher.issue(fixture.memory, [&] {
            EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_PENDING);
            ++fixture.issues;
            fixture.platform.outstanding = fixture.platform.outgoing = 1;
        }));
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 6u);
        ASSERT_TRUE(publisher.close(fixture.memory, [&] {
            EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_ISSUED);
            ++fixture.waits;
            fixture.platform.outstanding = fixture.platform.outgoing = 0;
        }));
        const auto closed = fixture.region.receipt;
        EXPECT_EQ(closed.publish_sequence, 8u);
        EXPECT_EQ(closed.request.occurrence, 1u);
        fixture.acknowledge(2);
        ASSERT_EQ(publisher.finish(fixture.memory), Finish::Finished);
        const auto& end = fixture.region.receipt;
        EXPECT_EQ(end.state, 5u);
        EXPECT_EQ(end.publish_sequence, 10u);
        EXPECT_EQ(item_series_wire(end, 72, 8), 1u);
        EXPECT_EQ(item_series_wire(end, 80, 4), 8u);
        EXPECT_EQ(item_series_wire(end, 84, 4), 2u);
        EXPECT_EQ(fixture.issues, 1u);
        EXPECT_EQ(fixture.waits, 1u);
        EXPECT_EQ(closed.state, TT_NIU_ITEM_WITNESS_CLOSED);
        EXPECT_EQ(closed.publish_sequence, 8u);
    });
}

TEST(NiuItemPublisherSeries, MissingOrInProgressFinalAcknowledgementPreservesClosed) {
    check_item_series_publisher<Receiver>([](auto& publisher, auto& fixture) {
        using Finish = typename std::remove_reference_t<decltype(publisher)>::SeriesFinishStatus;
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
        ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
        ASSERT_TRUE(publisher.close(fixture.memory, [] {}));
        const auto closed = fixture.region.receipt;
        EXPECT_EQ(publisher.finish(fixture.memory), Finish::Pending);
        EXPECT_EQ(std::memcmp(&fixture.region.receipt, &closed, sizeof(closed)), 0);
        fixture.acknowledge(1);  // The odd host publication is not an ACK.
        EXPECT_EQ(publisher.finish(fixture.memory), Finish::Pending);
        EXPECT_EQ(std::memcmp(&fixture.region.receipt, &closed, sizeof(closed)), 0);
        fixture.acknowledge(2);
        EXPECT_EQ(publisher.finish(fixture.memory), Finish::Finished);
        EXPECT_EQ(item_series_wire(fixture.region.receipt, 72, 8), 1u);
    });
}

TEST(NiuItemPublisherSeries, BeginAllocatesConsecutiveSourceOccurrencesAfterExactAcknowledgement) {
    check_item_series_publisher<Receiver>([](auto& publisher, auto& fixture) {
        using Finish = typename std::remove_reference_t<decltype(publisher)>::SeriesFinishStatus;
        ASSERT_TRUE(publisher.open(fixture.memory));
        for (uint64_t occurrence = 1; occurrence != 3; ++occurrence) {
            ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
            EXPECT_EQ(fixture.region.receipt.request.occurrence, occurrence);
            EXPECT_EQ(fixture.region.receipt.publish_sequence, 6 * occurrence - 2);
            ASSERT_TRUE(publisher.issue(fixture.memory, [&] { ++fixture.issues; }));
            ASSERT_TRUE(publisher.close(fixture.memory, [&] { ++fixture.waits; }));
            const auto closed = fixture.region.receipt;
            EXPECT_EQ(closed.publish_sequence, 6 * occurrence + 2);
            EXPECT_EQ(publisher.begin(fixture.memory), BeginStatus::Pending);
            EXPECT_EQ(std::memcmp(&fixture.region.receipt, &closed, sizeof(closed)), 0);
            fixture.acknowledge(static_cast<uint32_t>(2 * occurrence));
        }
        ASSERT_EQ(publisher.finish(fixture.memory), Finish::Finished);
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 16u);
        EXPECT_EQ(item_series_wire(fixture.region.receipt, 72, 8), 2u);
        EXPECT_EQ(item_series_wire(fixture.region.receipt, 80, 4), 14u);
        EXPECT_EQ(item_series_wire(fixture.region.receipt, 84, 4), 4u);
        EXPECT_EQ(fixture.region.request.occurrence, 1u);
        EXPECT_EQ(fixture.issues, 2u);
        EXPECT_EQ(fixture.waits, 2u);
    });
}

TEST(NiuItemPublisherSeries, SourceSeriesEndPreventsAnyLaterBeginIssueOrReopen) {
    for (const bool executed : {false, true}) {
        check_item_series_publisher<Receiver>([&](auto& publisher, auto& fixture) {
            using Finish = typename std::remove_reference_t<decltype(publisher)>::SeriesFinishStatus;
            ASSERT_TRUE(publisher.open(fixture.memory));
            if (executed) {
                ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
                ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
                ASSERT_TRUE(publisher.close(fixture.memory, [] {}));
                fixture.acknowledge(2);
            }
            ASSERT_EQ(publisher.finish(fixture.memory), executed ? Finish::Finished : Finish::NotExecuted);
            const auto end = fixture.region.receipt;
            EXPECT_EQ(publisher.begin(fixture.memory), BeginStatus::Rejected);
            EXPECT_FALSE(publisher.issue(fixture.memory, [&] { ++fixture.issues; }));
            EXPECT_FALSE(publisher.close(fixture.memory, [&] { ++fixture.waits; }));
            EXPECT_FALSE(publisher.open(fixture.memory));
            EXPECT_EQ(std::memcmp(&fixture.region.receipt, &end, sizeof(end)), 0);
            EXPECT_EQ(fixture.issues, 0u);
            EXPECT_EQ(fixture.waits, 0u);
        });
    }
}

TEST(NiuItemPublisherSeries, AnotherPublisherCannotAdoptOrFinishAnOpenedSeries) {
    check_item_series_publisher<Receiver>([](auto& publisher, auto& fixture) {
        using Publisher = std::remove_reference_t<decltype(publisher)>;
        using Finish = typename Publisher::SeriesFinishStatus;
        ASSERT_TRUE(publisher.open(fixture.memory));
        const auto opened = fixture.region.receipt;
        Publisher foreign;
        EXPECT_FALSE(foreign.open(fixture.memory));
        EXPECT_EQ(foreign.finish(fixture.memory), Finish::Rejected);
        EXPECT_EQ(std::memcmp(&fixture.region.receipt, &opened, sizeof(opened)), 0);
        EXPECT_EQ(publisher.finish(fixture.memory), Finish::NotExecuted);
    });
}

TEST(NiuItemPublisherSeries, IncompleteOrForeignRoundCannotPublishSeriesEnd) {
    for (unsigned mutation = 0; mutation != 4; ++mutation) {
        check_item_series_publisher<Receiver>([&](auto& publisher, auto& fixture) {
            using Finish = typename std::remove_reference_t<decltype(publisher)>::SeriesFinishStatus;
            ASSERT_TRUE(publisher.open(fixture.memory));
            ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
            if (mutation != 0) {
                ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
                if (mutation != 1) {
                    ASSERT_TRUE(publisher.close(fixture.memory, [] {}));
                    fixture.acknowledge(2);
                    if (mutation == 2) ++fixture.region.acknowledgement.request.invocation_id;
                    if (mutation == 3) fixture.region.acknowledgement.closed_sequence -= 2;
                }
            }
            EXPECT_EQ(publisher.finish(fixture.memory), Finish::Rejected) << mutation;
            EXPECT_NE(fixture.region.receipt.state, 5u) << mutation;
        });
    }
}

TEST(NiuItemPublisherSeries, SourceOccurrenceCannotComeFromCallerIntegerOrRetiredArmRearmApi) {
    check_item_series_publisher<Receiver>([](auto& publisher, auto& fixture) {
        using Publisher = std::remove_reference_t<decltype(publisher)>;
        using Memory = std::remove_reference_t<decltype(fixture.memory)>;
        using Finish = typename Publisher::SeriesFinishStatus;
        EXPECT_TRUE((std::is_same_v<decltype(publisher.open(fixture.memory)), bool>));
        EXPECT_TRUE((std::is_same_v<decltype(publisher.begin(fixture.memory)), BeginStatus>));
        EXPECT_TRUE((std::is_same_v<decltype(publisher.finish(fixture.memory)), Finish>));
        EXPECT_FALSE((requires(Publisher& value, Memory& io) { value.begin(io, uint64_t{1}); }));
        EXPECT_FALSE((requires(Publisher& value, Memory& io) { value.arm(io); }));
        EXPECT_FALSE((requires(Publisher& value, Memory& io) { value.rearm(io, uint64_t{1}); }));
        EXPECT_NE(Finish::Pending, Finish::NotExecuted);
        EXPECT_NE(Finish::Pending, Finish::Finished);
        EXPECT_NE(Finish::Pending, Finish::Rejected);
        EXPECT_EQ(TT_NIU_ITEM_WITNESS_VERSION, 3u);
    });
}

template <class ReceiverType, class Check>
void check_typed_item_begin(Check check) {
    using Publisher = typename ReceiverType::ItemPublisher;
    if constexpr (requires { typename Publisher::SeriesBeginStatus; }) {
        ItemSeriesFixture<ReceiverType> fixture;
        Publisher publisher;
        check(publisher, fixture);
    } else {
        ADD_FAILURE() << "Source begin conflates ACK backpressure with terminal rejection in bool false";
    }
}

TEST(NiuItemPublisherBegin, TypedBeginCannotBeUsedAsAnAmbiguousBoolean) {
    check_typed_item_begin<Receiver>([](auto& publisher, auto& fixture) {
        using Begin = typename std::remove_reference_t<decltype(publisher)>::SeriesBeginStatus;
        EXPECT_TRUE((std::is_same_v<decltype(publisher.begin(fixture.memory)), Begin>));
        EXPECT_FALSE((std::is_convertible_v<Begin, bool>));
        EXPECT_NE(Begin::Pending, Begin::Begun);
        EXPECT_NE(Begin::Pending, Begin::Rejected);
        EXPECT_NE(Begin::Begun, Begin::Rejected);
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), Begin::Begun);
        EXPECT_EQ(fixture.region.receipt.request.occurrence, 1u);
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 4u);
    });
}

TEST(NiuItemPublisherBegin, PendingAcknowledgementPreservesClosedAndDoesNotConsumeAnOccurrence) {
    check_typed_item_begin<Receiver>([](auto& publisher, auto& fixture) {
        using Begin = typename std::remove_reference_t<decltype(publisher)>::SeriesBeginStatus;
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), Begin::Begun);
        ASSERT_TRUE(publisher.issue(fixture.memory, [&] { ++fixture.issues; }));
        ASSERT_TRUE(publisher.close(fixture.memory, [&] { ++fixture.waits; }));
        const auto closed = fixture.region.receipt;
        for (unsigned retry = 0; retry != 2; ++retry) {
            EXPECT_EQ(publisher.begin(fixture.memory), Begin::Pending);
            EXPECT_EQ(std::memcmp(&fixture.region.receipt, &closed, sizeof(closed)), 0);
        }
        fixture.acknowledge(1);
        EXPECT_EQ(publisher.begin(fixture.memory), Begin::Pending);
        EXPECT_EQ(std::memcmp(&fixture.region.receipt, &closed, sizeof(closed)), 0);
        fixture.acknowledge(2);
        ASSERT_EQ(publisher.begin(fixture.memory), Begin::Begun);
        EXPECT_EQ(fixture.region.receipt.request.occurrence, 2u);
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 10u);
        EXPECT_EQ(fixture.issues, 1u);
        EXPECT_EQ(fixture.waits, 1u);
        ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
        ASSERT_TRUE(publisher.close(fixture.memory, [] {}));
        // The exact previous ACK may remain visible while the next is pending.
        EXPECT_EQ(publisher.begin(fixture.memory), Begin::Pending);
        EXPECT_EQ(fixture.region.receipt.request.occurrence, 2u);
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 14u);
    });
}

TEST(NiuItemPublisherBegin, ForeignAcknowledgementIsRejectedAndCannotBeRetriedAsPending) {
    check_typed_item_begin<Receiver>([](auto& publisher, auto& fixture) {
        using Begin = typename std::remove_reference_t<decltype(publisher)>::SeriesBeginStatus;
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), Begin::Begun);
        ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
        ASSERT_TRUE(publisher.close(fixture.memory, [] {}));
        fixture.acknowledge(2);
        ++fixture.region.acknowledgement.request.invocation_id;
        EXPECT_EQ(publisher.begin(fixture.memory), Begin::Rejected);
        const auto rejected = fixture.region.receipt;
        EXPECT_EQ(rejected.state, TT_NIU_ITEM_WITNESS_REJECTED);
        fixture.acknowledge(2);
        EXPECT_EQ(publisher.begin(fixture.memory), Begin::Rejected);
        EXPECT_FALSE(publisher.issue(fixture.memory, [&] { ++fixture.issues; }));
        EXPECT_EQ(fixture.issues, 0u);
        EXPECT_EQ(std::memcmp(&fixture.region.receipt, &rejected, sizeof(rejected)), 0);
    });
}

TEST(NiuItemPublisherBegin, SourceEndAndChangedOriginalRequestAreRejectedWithoutAnotherOccurrence) {
    for (const bool change_request : {false, true}) {
        check_typed_item_begin<Receiver>([&](auto& publisher, auto& fixture) {
            using Publisher = std::remove_reference_t<decltype(publisher)>;
            using Begin = typename Publisher::SeriesBeginStatus;
            using Finish = typename Publisher::SeriesFinishStatus;
            EXPECT_EQ(publisher.begin(fixture.memory), Begin::Rejected);
            ASSERT_TRUE(publisher.open(fixture.memory));
            if (change_request) {
                ++fixture.region.request.invocation_id;
            } else {
                ASSERT_EQ(publisher.finish(fixture.memory), Finish::NotExecuted);
            }
            EXPECT_EQ(publisher.begin(fixture.memory), Begin::Rejected);
            EXPECT_FALSE(publisher.issue(fixture.memory, [&] { ++fixture.issues; }));
            EXPECT_EQ(fixture.issues, 0u);
            EXPECT_EQ(fixture.region.receipt.request.occurrence, 1u);
        });
    }
}

template <class Publisher, class IO>
concept exposes_item_split_boundaries = requires(
    Publisher& publisher, IO& io, typename Publisher::IssueTicket&& issue, typename Publisher::CloseTicket&& close) {
    publisher.startIssue(io);
    publisher.completeIssue(io, std::move(issue));
    publisher.startClose(io);
    publisher.completeClose(io, std::move(close));
};

template <class Check>
void check_item_split_boundaries(Check check) {
    using Fixture = ItemSeriesFixture<Receiver>;
    using Publisher = Receiver::ItemPublisher;
    if constexpr (exposes_item_split_boundaries<Publisher, decltype(Fixture::memory)>) {
        Fixture fixture;
        Publisher publisher;
        check(publisher, fixture);
    } else {
        ADD_FAILURE() << "Native support lacks typed split issue and close boundaries";
    }
}

TEST(NiuItemPublisherSplit, TicketsAreOpaqueMoveOnlyAndDistinguishIssueFromClose) {
    check_item_split_boundaries([](auto& publisher, auto& fixture) {
        using Publisher = std::remove_reference_t<decltype(publisher)>;
        using Issue = typename Publisher::IssueTicket;
        using Close = typename Publisher::CloseTicket;
        EXPECT_FALSE((std::is_same_v<Issue, Close>));
        EXPECT_FALSE((std::is_default_constructible_v<Issue>));
        EXPECT_FALSE((std::is_default_constructible_v<Close>));
        EXPECT_FALSE((std::is_copy_constructible_v<Issue>));
        EXPECT_FALSE((std::is_copy_constructible_v<Close>));
        EXPECT_FALSE((std::is_copy_assignable_v<Issue>));
        EXPECT_FALSE((std::is_copy_assignable_v<Close>));
        EXPECT_TRUE((std::is_move_constructible_v<Issue>));
        EXPECT_TRUE((std::is_move_constructible_v<Close>));
        EXPECT_FALSE((std::is_convertible_v<Issue, bool>));
        EXPECT_FALSE((std::is_convertible_v<Close, bool>));
        EXPECT_TRUE((std::is_same_v<decltype(publisher.startIssue(fixture.memory)), Issue>));
        EXPECT_TRUE((std::is_same_v<decltype(publisher.startClose(fixture.memory)), Close>));
    });
}

TEST(NiuItemPublisherSplit, SeparatesActualIssueAndScopedWaitFromTheirPublication) {
    check_item_split_boundaries([](auto& publisher, auto& fixture) {
        using Finish = typename std::remove_reference_t<decltype(publisher)>::SeriesFinishStatus;
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
        auto issue = publisher.startIssue(fixture.memory);
        ASSERT_TRUE(issue);
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_PENDING);
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 4u);
        ++fixture.issues;
        fixture.platform.outstanding = fixture.platform.outgoing = 1;
        ASSERT_TRUE(publisher.completeIssue(fixture.memory, std::move(issue)));
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_ISSUED);
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 6u);
        EXPECT_EQ(fixture.region.receipt.item.issued_occurrence, 1u);
        auto close = publisher.startClose(fixture.memory);
        ASSERT_TRUE(close);
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_ISSUED);
        EXPECT_EQ(fixture.region.receipt.item.closed_occurrence, 0u);
        ++fixture.waits;
        fixture.platform.outstanding = fixture.platform.outgoing = 0;
        ASSERT_TRUE(publisher.completeClose(fixture.memory, std::move(close)));
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_CLOSED);
        EXPECT_EQ(fixture.region.receipt.publish_sequence, 8u);
        EXPECT_EQ(fixture.region.receipt.item.closed_occurrence, 1u);
        fixture.acknowledge(2);
        ASSERT_EQ(publisher.finish(fixture.memory), Finish::Finished);
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_SERIES_END);
        EXPECT_EQ(fixture.region.receipt.series.final_count, 1u);
        EXPECT_EQ(fixture.issues, 1u);
        EXPECT_EQ(fixture.waits, 1u);
    });
}

TEST(NiuItemPublisherSplit, MovingAndCompletingEachTicketConsumesItsOnlyAuthority) {
    check_item_split_boundaries([](auto& publisher, auto& fixture) {
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
        auto original_issue = publisher.startIssue(fixture.memory);
        ASSERT_TRUE(original_issue);
        auto issue = std::move(original_issue);
        EXPECT_FALSE(original_issue);
        EXPECT_FALSE(publisher.completeIssue(fixture.memory, std::move(original_issue)));
        ASSERT_TRUE(publisher.completeIssue(fixture.memory, std::move(issue)));
        EXPECT_FALSE(issue);
        const auto issued = fixture.region.receipt;
        EXPECT_FALSE(publisher.completeIssue(fixture.memory, std::move(issue)));
        EXPECT_EQ(std::memcmp(&fixture.region.receipt, &issued, sizeof(issued)), 0);
        auto original_close = publisher.startClose(fixture.memory);
        auto close = std::move(original_close);
        EXPECT_FALSE(original_close);
        EXPECT_FALSE(publisher.completeClose(fixture.memory, std::move(original_close)));
        ASSERT_TRUE(publisher.completeClose(fixture.memory, std::move(close)));
        EXPECT_FALSE(close);
        const auto closed = fixture.region.receipt;
        EXPECT_FALSE(publisher.completeClose(fixture.memory, std::move(close)));
        EXPECT_EQ(std::memcmp(&fixture.region.receipt, &closed, sizeof(closed)), 0);
    });
}

TEST(NiuItemPublisherSplit, ForeignPublisherCannotConsumeAnotherIssuersIssueOrClose) {
    for (const bool closing : {false, true}) {
        check_item_split_boundaries([&](auto& publisher, auto& fixture) {
            using Publisher = std::remove_reference_t<decltype(publisher)>;
            ItemSeriesFixture<Receiver> other_fixture;
            Publisher other;
            ASSERT_TRUE(publisher.open(fixture.memory));
            ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
            ASSERT_TRUE(other.open(other_fixture.memory));
            ASSERT_EQ(other.begin(other_fixture.memory), BeginStatus::Begun);
            if (closing) {
                ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
                ASSERT_TRUE(other.issue(other_fixture.memory, [] {}));
                auto foreign = publisher.startClose(fixture.memory);
                auto own = other.startClose(other_fixture.memory);
                ASSERT_TRUE(foreign);
                ASSERT_TRUE(own);
                const auto before = other_fixture.region.receipt;
                EXPECT_FALSE(other.completeClose(other_fixture.memory, std::move(foreign)));
                EXPECT_EQ(std::memcmp(&other_fixture.region.receipt, &before, sizeof(before)), 0);
                ASSERT_TRUE(other.completeClose(other_fixture.memory, std::move(own)));
            } else {
                auto foreign = publisher.startIssue(fixture.memory);
                auto own = other.startIssue(other_fixture.memory);
                ASSERT_TRUE(foreign);
                ASSERT_TRUE(own);
                const auto before = other_fixture.region.receipt;
                EXPECT_FALSE(other.completeIssue(other_fixture.memory, std::move(foreign)));
                EXPECT_EQ(std::memcmp(&other_fixture.region.receipt, &before, sizeof(before)), 0);
                ASSERT_TRUE(other.completeIssue(other_fixture.memory, std::move(own)));
            }
        });
    }
}

TEST(NiuItemPublisherSplit, PendingSplitPreventsDuplicateStartsAndCallbackReentry) {
    check_item_split_boundaries([](auto& publisher, auto& fixture) {
        EXPECT_FALSE(publisher.startIssue(fixture.memory));
        EXPECT_FALSE(publisher.startClose(fixture.memory));
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
        EXPECT_FALSE(publisher.startClose(fixture.memory));
        auto issue = publisher.startIssue(fixture.memory);
        ASSERT_TRUE(issue);
        EXPECT_FALSE(publisher.startIssue(fixture.memory));
        EXPECT_FALSE(publisher.startClose(fixture.memory));
        EXPECT_FALSE(publisher.issue(fixture.memory, [&] { ++fixture.issues; }));
        EXPECT_EQ(fixture.issues, 0u);
        ASSERT_TRUE(publisher.completeIssue(fixture.memory, std::move(issue)));
        auto close = publisher.startClose(fixture.memory);
        ASSERT_TRUE(close);
        EXPECT_FALSE(publisher.startIssue(fixture.memory));
        EXPECT_FALSE(publisher.startClose(fixture.memory));
        EXPECT_FALSE(publisher.close(fixture.memory, [&] { ++fixture.waits; }));
        EXPECT_EQ(fixture.waits, 0u);
        ASSERT_TRUE(publisher.completeClose(fixture.memory, std::move(close)));
    });
}

TEST(NiuItemPublisherSplit, ChangedOriginalRequestBetweenStartAndCompletionRejectsPublication) {
    for (const bool closing : {false, true}) {
        check_item_split_boundaries([&](auto& publisher, auto& fixture) {
            ASSERT_TRUE(publisher.open(fixture.memory));
            ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
            if (closing) {
                ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
                auto ticket = publisher.startClose(fixture.memory);
                ASSERT_TRUE(ticket);
                ++fixture.region.request.invocation_id;
                EXPECT_FALSE(publisher.completeClose(fixture.memory, std::move(ticket)));
            } else {
                auto ticket = publisher.startIssue(fixture.memory);
                ASSERT_TRUE(ticket);
                ++fixture.region.request.invocation_id;
                EXPECT_FALSE(publisher.completeIssue(fixture.memory, std::move(ticket)));
            }
            EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_REJECTED);
            EXPECT_EQ(fixture.region.receipt.item.closed_occurrence, 0u);
        });
    }
}

TEST(NiuItemPublisherSplit, ForeignReceiptSequenceCannotBeOverwrittenByCompletion) {
    for (const bool closing : {false, true}) {
        check_item_split_boundaries([&](auto& publisher, auto& fixture) {
            ASSERT_TRUE(publisher.open(fixture.memory));
            ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
            if (closing) {
                ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
                auto ticket = publisher.startClose(fixture.memory);
                ASSERT_TRUE(ticket);
                fixture.region.receipt.publish_sequence += 2;
                const auto foreign = fixture.region.receipt;
                EXPECT_FALSE(publisher.completeClose(fixture.memory, std::move(ticket)));
                EXPECT_EQ(std::memcmp(&fixture.region.receipt, &foreign, sizeof(foreign)), 0);
            } else {
                auto ticket = publisher.startIssue(fixture.memory);
                ASSERT_TRUE(ticket);
                fixture.region.receipt.publish_sequence += 2;
                const auto foreign = fixture.region.receipt;
                EXPECT_FALSE(publisher.completeIssue(fixture.memory, std::move(ticket)));
                EXPECT_EQ(std::memcmp(&fixture.region.receipt, &foreign, sizeof(foreign)), 0);
            }
        });
    }
}

TEST(NiuItemPublisherSplit, CloseCompletionReadsBothActualHardwareSelectorBanks) {
    for (const bool outgoing : {false, true}) {
        check_item_split_boundaries([&](auto& publisher, auto& fixture) {
            ASSERT_TRUE(publisher.open(fixture.memory));
            ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
            ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
            auto close = publisher.startClose(fixture.memory);
            ASSERT_TRUE(close);
            (outgoing ? fixture.platform.outgoing : fixture.platform.outstanding) = 1;
            EXPECT_FALSE(publisher.completeClose(fixture.memory, std::move(close)));
            EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_REJECTED);
            EXPECT_EQ(fixture.region.receipt.item.closed_occurrence, 0u);
        });
    }
}

TEST(NiuItemPublisherSplit, SourceFinishRejectsEachUncompletedBoundaryAndItsOutstandingTicket) {
    for (const bool closing : {false, true}) {
        check_item_split_boundaries([&](auto& publisher, auto& fixture) {
            using Finish = typename std::remove_reference_t<decltype(publisher)>::SeriesFinishStatus;
            ASSERT_TRUE(publisher.open(fixture.memory));
            ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
            if (closing) {
                ASSERT_TRUE(publisher.issue(fixture.memory, [] {}));
                auto ticket = publisher.startClose(fixture.memory);
                ASSERT_TRUE(ticket);
                EXPECT_EQ(publisher.finish(fixture.memory), Finish::Rejected);
                EXPECT_FALSE(publisher.completeClose(fixture.memory, std::move(ticket)));
            } else {
                auto ticket = publisher.startIssue(fixture.memory);
                ASSERT_TRUE(ticket);
                EXPECT_EQ(publisher.finish(fixture.memory), Finish::Rejected);
                EXPECT_FALSE(publisher.completeIssue(fixture.memory, std::move(ticket)));
            }
            EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_REJECTED);
            EXPECT_EQ(fixture.region.receipt.item.closed_occurrence, 0u);
        });
    }
}

TEST(NiuItemPublisherSplit, ConsumedTicketsCannotPublishForTheNextSourceOccurrence) {
    check_item_split_boundaries([](auto& publisher, auto& fixture) {
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
        auto old_issue = publisher.startIssue(fixture.memory);
        ASSERT_TRUE(publisher.completeIssue(fixture.memory, std::move(old_issue)));
        auto old_close = publisher.startClose(fixture.memory);
        ASSERT_TRUE(publisher.completeClose(fixture.memory, std::move(old_close)));
        fixture.acknowledge(2);
        ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
        auto issue = publisher.startIssue(fixture.memory);
        ASSERT_TRUE(issue);
        EXPECT_FALSE(publisher.completeIssue(fixture.memory, std::move(old_issue)));
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_PENDING);
        EXPECT_EQ(fixture.region.receipt.request.occurrence, 2u);
        ASSERT_TRUE(publisher.completeIssue(fixture.memory, std::move(issue)));
        auto close = publisher.startClose(fixture.memory);
        ASSERT_TRUE(close);
        EXPECT_FALSE(publisher.completeClose(fixture.memory, std::move(old_close)));
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_ISSUED);
        ASSERT_TRUE(publisher.completeClose(fixture.memory, std::move(close)));
        EXPECT_EQ(fixture.region.receipt.item.closed_occurrence, 2u);
    });
}

TEST(NiuItemPublisherSplit, IssueTicketCannotCrossPublisherLifetimeOnTheSameStorage) {
    for (const bool change_invocation : {false, true}) {
        check_item_split_boundaries([&](auto& unused_publisher, auto& fixture) {
            using Publisher = std::remove_reference_t<decltype(unused_publisher)>;
            alignas(Publisher) unsigned char storage[sizeof(Publisher)];
            auto* original = ::new (static_cast<void*>(storage)) Publisher;
            ASSERT_TRUE(original->open(fixture.memory));
            ASSERT_EQ(original->begin(fixture.memory), BeginStatus::Begun);
            auto old_ticket = original->startIssue(fixture.memory);
            ASSERT_TRUE(old_ticket);
            original->~Publisher();

            ItemSeriesFixture<Receiver> next;
            if (change_invocation)
                ++next.region.request.invocation_id;
            else
                ++next.region.request.generation;
            auto* replacement = ::new (static_cast<void*>(storage)) Publisher;
            ASSERT_TRUE(replacement->open(next.memory));
            ASSERT_EQ(replacement->begin(next.memory), BeginStatus::Begun);
            auto own_ticket = replacement->startIssue(next.memory);
            ASSERT_TRUE(own_ticket);
            const auto before = next.region.receipt;
            EXPECT_EQ(before.request.occurrence, 1u);
            EXPECT_EQ(before.publish_sequence, 4u);
            EXPECT_FALSE(replacement->completeIssue(next.memory, std::move(old_ticket)));
            EXPECT_EQ(std::memcmp(&next.region.receipt, &before, sizeof(before)), 0);
            EXPECT_FALSE(old_ticket);
            ASSERT_TRUE(replacement->completeIssue(next.memory, std::move(own_ticket)));
            EXPECT_EQ(next.region.receipt.state, TT_NIU_ITEM_WITNESS_ISSUED);
            replacement->~Publisher();
        });
    }
}

TEST(NiuItemPublisherSplit, CloseTicketCannotCrossPublisherLifetimeOnTheSameStorage) {
    for (const bool change_invocation : {false, true}) {
        check_item_split_boundaries([&](auto& unused_publisher, auto& fixture) {
            using Publisher = std::remove_reference_t<decltype(unused_publisher)>;
            alignas(Publisher) unsigned char storage[sizeof(Publisher)];
            auto* original = ::new (static_cast<void*>(storage)) Publisher;
            ASSERT_TRUE(original->open(fixture.memory));
            ASSERT_EQ(original->begin(fixture.memory), BeginStatus::Begun);
            ASSERT_TRUE(original->issue(fixture.memory, [] {}));
            auto old_ticket = original->startClose(fixture.memory);
            ASSERT_TRUE(old_ticket);
            original->~Publisher();

            ItemSeriesFixture<Receiver> next;
            if (change_invocation)
                ++next.region.request.invocation_id;
            else
                ++next.region.request.generation;
            auto* replacement = ::new (static_cast<void*>(storage)) Publisher;
            ASSERT_TRUE(replacement->open(next.memory));
            ASSERT_EQ(replacement->begin(next.memory), BeginStatus::Begun);
            ASSERT_TRUE(replacement->issue(next.memory, [] {}));
            auto own_ticket = replacement->startClose(next.memory);
            ASSERT_TRUE(own_ticket);
            const auto before = next.region.receipt;
            EXPECT_EQ(before.request.occurrence, 1u);
            EXPECT_EQ(before.publish_sequence, 6u);
            EXPECT_FALSE(replacement->completeClose(next.memory, std::move(old_ticket)));
            EXPECT_EQ(std::memcmp(&next.region.receipt, &before, sizeof(before)), 0);
            EXPECT_FALSE(old_ticket);
            ASSERT_TRUE(replacement->completeClose(next.memory, std::move(own_ticket)));
            EXPECT_EQ(next.region.receipt.state, TT_NIU_ITEM_WITNESS_CLOSED);
            replacement->~Publisher();
        });
    }
}

TEST(NiuItemPublisherSplit, CallbackConveniencesUseTheSameExclusiveBoundaries) {
    check_item_split_boundaries([](auto& publisher, auto& fixture) {
        ASSERT_TRUE(publisher.open(fixture.memory));
        ASSERT_EQ(publisher.begin(fixture.memory), BeginStatus::Begun);
        ASSERT_TRUE(publisher.issue(fixture.memory, [&] {
            ++fixture.issues;
            EXPECT_FALSE(publisher.startIssue(fixture.memory));
            EXPECT_FALSE(publisher.startClose(fixture.memory));
            EXPECT_FALSE(publisher.issue(fixture.memory, [&] { ++fixture.issues; }));
            EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_PENDING);
        }));
        ASSERT_TRUE(publisher.close(fixture.memory, [&] {
            ++fixture.waits;
            EXPECT_FALSE(publisher.startIssue(fixture.memory));
            EXPECT_FALSE(publisher.startClose(fixture.memory));
            EXPECT_FALSE(publisher.close(fixture.memory, [&] { ++fixture.waits; }));
            EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_ISSUED);
        }));
        EXPECT_EQ(fixture.issues, 1u);
        EXPECT_EQ(fixture.waits, 1u);
        EXPECT_EQ(fixture.region.receipt.state, TT_NIU_ITEM_WITNESS_CLOSED);
    });
}

enum class Event { SourceRead, Poll, Rebase, ProfileExit, Notify, DescriptorReadback, BootReady, Pending, Publish };

// Only the physical MMIO and L1 boundary is modeled. Admission, accounting,
// stage closure and publication ordering are owned by the production receiver.
struct DeviceIO {
    tt_niu_handoff_request request{
        1, 1, 17, 0, 23, 0, 29, 0, 1, 1, 0xffff, 0xffff, 1, 0x1f, 1, 0};
    uint32_t expected[2][2][5]{};
    uint32_t observed[2][5]{};
    uint32_t control[2][5]{};
    tt_niu_handoff_receipt receipt{};
    std::vector<Event> events;
    unsigned source_reads = 0;
    unsigned publications = 0;
    unsigned pending_publications = 0;
    unsigned boot_publications = 0;
    bool block = false;
    bool torn_request = false;
    unsigned generation_reads = 0;
    uint8_t subordinate_signals[4]{};

    DeviceIO() {
        // Nonzero inherited hardware baselines must never be counted as
        // traffic issued by this invocation.
        for (unsigned noc = 0; noc != 2; ++noc) {
            for (unsigned counter = 0; counter != 5; ++counter) {
                expected[noc][0][counter] = 13 + 2 * noc;
                expected[noc][1][counter] = 17 + 2 * noc;
                observed[noc][counter] = 30 + 4 * noc;
                control[noc][counter] = observed[noc][counter];
            }
        }
    }
    uint32_t requestGeneration() {
        return request.request_generation + (torn_request && generation_reads++ != 0);
    }
    tt_niu_handoff_request readRequest() { return request; }
    uint8_t subordinateSignal(unsigned role) { return subordinate_signals[role]; }
    static constexpr uint8_t doneSignal() { return 0; }
    void invalidate() {}
    void invalidateReceipt() { receipt.header.publish_sequence = UINT32_MAX; }
    uint32_t sourceExpected(unsigned noc, unsigned risc, unsigned counter) {
        ++source_reads;
        events.push_back(Event::SourceRead);
        return expected[noc][risc][counter];
    }
    uint32_t observedCounter(unsigned noc, unsigned counter) { return observed[noc][counter]; }
    uint32_t controlExpected(unsigned noc, unsigned counter) { return control[noc][counter]; }
    void poll() {
        events.push_back(Event::Poll);
        if (block) throw std::runtime_error("device traffic remains outstanding");
        for (unsigned noc = 0; noc != 2; ++noc)
            for (unsigned counter = 0; counter != 5; ++counter)
                observed[noc][counter] = control[noc][counter];
    }
    void rebaseControl() {
        events.push_back(Event::Rebase);
        for (unsigned noc = 0; noc != 2; ++noc)
            for (unsigned counter = 0; counter != 5; ++counter)
                control[noc][counter] = observed[noc][counter];
    }
    void profilerExit() {
        events.push_back(Event::ProfileExit);
        ++control[1][4];
        observed[1][4] = control[1][4];
    }
    void notify() {
        events.push_back(Event::Notify);
        ++control[0][4];
        // The final posted request deliberately remains unsent until poll.
    }
    bool descriptorReady(unsigned, unsigned) { return true; }
    void descriptorReadback(unsigned, unsigned) { events.push_back(Event::DescriptorReadback); }
    tt_niu_handoff_selector_observation selectorObserved(unsigned noc, unsigned selector) {
        // Multicast can leave a nonzero terminal outstanding count. Nothing
        // in the receiver may turn these observations into an all-bank reset.
        return {0x80u + noc * 16u + selector, 0};
    }
    void publish(const tt_niu_handoff_receipt& value) {
        receipt = value;
        if (value.header.status == TT_NIU_HANDOFF_TRAFFIC_CLOSED) {
            events.push_back(Event::Publish);
            ++publications;
        } else if (value.header.status == TT_NIU_HANDOFF_BOOT_READY) {
            events.push_back(Event::BootReady);
            ++boot_publications;
        } else {
            events.push_back(Event::Pending);
            ++pending_publications;
        }
    }
};

struct ProfileScope {
    DeviceIO& io;
    ~ProfileScope() { io.profilerExit(); }
};

void issuePackets(DeviceIO& io) {
    for (unsigned noc = 0; noc != 2; ++noc) {
        for (unsigned counter = 0; counter != 5; ++counter) {
            io.expected[noc][0][counter] += 3 + 2 * noc;
            io.expected[noc][1][counter] += 7 + 2 * noc;
            io.observed[noc][counter] += 10 + 4 * noc;
        }
    }
}

void sourceStage(Receiver& receiver, DeviceIO& io) {
    receiver.beginStage(io, Source::BlackholeDynamic, 0x1f);
    issuePackets(io);
    receiver.closeStage(io);
}

TEST(NiuHandoff, ActualTwoRiscCountsPrecedeControlRebaseAndPublication) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    ASSERT_EQ(receiver.runGo(io, [&] { sourceStage(receiver, io); }), Admission::Accepted);
    ASSERT_EQ(io.publications, 1u);
    EXPECT_EQ(io.receipt.header.status, TT_NIU_HANDOFF_TRAFFIC_CLOSED);
    EXPECT_EQ(io.receipt.source_expected[0][0][2], 3u);
    EXPECT_EQ(io.receipt.source_expected[0][1][2], 7u);
    EXPECT_EQ(io.receipt.source_observed[0][2], 10u);
    EXPECT_EQ(io.receipt.source_observed[1][2], 14u);
    EXPECT_EQ(io.receipt.header.completed_stage_count, 1u);
    EXPECT_LT(std::find(io.events.begin(), io.events.end(), Event::SourceRead),
              std::find(io.events.begin(), io.events.end(), Event::Rebase));
    EXPECT_EQ(io.receipt.selector_observed[1][15].outstanding, 0x9fu);
}

TEST(NiuHandoff, PendingSourceCannotBeHiddenByRebaseOrFinalStage) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    io.observed[1][2] = 13;
    io.block = true;
    bool second_stage_entered = false;
    EXPECT_THROW(receiver.runGo(io, [&] {
        sourceStage(receiver, io);
        second_stage_entered = true;
        sourceStage(receiver, io);
    }), std::runtime_error);
    EXPECT_FALSE(second_stage_entered);
    EXPECT_EQ(io.publications, 0u);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Rebase), 0);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Notify), 0);
}

TEST(NiuHandoff, LaterStageFailureCannotPublishEarlierStageReceipt) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    EXPECT_THROW(receiver.runGo(io, [&] {
        sourceStage(receiver, io);
        receiver.beginStage(io, Source::BlackholeDynamic, 0x1f);
        io.expected[0][1][0] += 2;
        io.block = true;
        receiver.closeStage(io);
    }), std::runtime_error);
    EXPECT_EQ(io.publications, 0u);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Rebase), 1);
}

TEST(NiuHandoff, EveryStageContributesItsActualIssueDelta) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    ASSERT_EQ(receiver.runGo(io, [&] {
        sourceStage(receiver, io);
        sourceStage(receiver, io);
    }), Admission::Accepted);
    ASSERT_EQ(io.publications, 1u);
    EXPECT_EQ(io.receipt.header.completed_stage_count, 2u);
    EXPECT_EQ(io.receipt.source_expected[0][0][2], 6u);
    EXPECT_EQ(io.receipt.source_expected[0][1][2], 14u);
    EXPECT_EQ(io.receipt.source_observed[0][2], 20u);
    EXPECT_EQ(io.receipt.source_observed[1][2], 28u);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Rebase), 2);
}

TEST(NiuHandoff, StageProfilerTrafficCannotEscapeThroughReload) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    bool reloaded = false;
    EXPECT_THROW(receiver.runGo(io, [&] {
        receiver.beginStage(io, Source::BlackholeDynamic, 0x1f);
        issuePackets(io);
        receiver.closeStage(io, [&] {
            // A real stage perf-counter push issues a posted request after
            // source closure but before reload. It has not left the NIU yet.
            ++io.control[0][4];
            io.block = true;
        });
        reloaded = true;
        sourceStage(receiver, io);
    }), std::runtime_error);
    EXPECT_FALSE(reloaded);
    EXPECT_EQ(io.publications, 0u);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Rebase), 1);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Notify), 0);
}

TEST(NiuHandoff, CounterWrapPreservesActualModuloDelta) {
    DeviceIO io;
    for (unsigned noc = 0; noc != 2; ++noc) {
        for (unsigned counter = 0; counter != 5; ++counter) {
            io.expected[noc][0][counter] = 0xfffffffeu;
            io.expected[noc][1][counter] = 0xfffffffdu;
            io.observed[noc][counter] = 0xfffffffbu;
            io.control[noc][counter] = io.observed[noc][counter];
        }
    }
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    ASSERT_EQ(receiver.runGo(io, [&] { sourceStage(receiver, io); }), Admission::Accepted);
    EXPECT_EQ(io.receipt.source_expected[0][0][0], 3u);
    EXPECT_EQ(io.receipt.source_expected[0][1][0], 7u);
    EXPECT_EQ(io.receipt.source_observed[0][0], 10u);
    EXPECT_EQ(io.receipt.source_observed[1][0], 14u);
}

TEST(NiuHandoff, IdleGoDoesNotRelabelOldDynamicCountsAsPayloadProof) {
    DeviceIO io;
    io.expected[0][1][0] = 0xfeed;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    ASSERT_EQ(receiver.runGo(io, [&] {
        receiver.beginStage(io, Source::None, 0);
        receiver.closeStage(io);
    }), Admission::Accepted);
    ASSERT_EQ(io.publications, 1u);
    EXPECT_EQ(io.source_reads, 0u);
    EXPECT_EQ(io.receipt.header.source_valid_noc_mask, 0u);
    EXPECT_EQ(io.receipt.source_expected[0][1][0], 0u);
    EXPECT_EQ(io.receipt.header.control_valid_noc_mask, 3u);
    EXPECT_EQ(io.receipt.header.completed_processor_mask, 0x1fu);
}

TEST(NiuHandoff, ActiveSubordinateCannotBeCertifiedBySourceCounters) {
    for (unsigned role = 0; role != 4; ++role) {
        DeviceIO io;
        Receiver receiver;
        ASSERT_TRUE(receiver.initializeBoot(io));
        io.events.clear();
        EXPECT_THROW(receiver.runGo(io, [&] {
            receiver.beginStage(io, Source::BlackholeDynamic, 0x1f);
            issuePackets(io);
            io.subordinate_signals[role] = 0x80;
            io.block = true;
            receiver.closeStage(io);
        }), std::runtime_error);
        EXPECT_EQ(io.publications, 0u);
        EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Rebase), 0);
        EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Notify), 0);
    }
}

TEST(NiuHandoff, IdleGoStillRequiresRealSubordinateDoneSignals) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    EXPECT_THROW(receiver.runGo(io, [&] {
        receiver.beginStage(io, Source::None, 0);
        io.subordinate_signals[3] = 0x40;
        io.block = true;
        receiver.closeStage(io);
    }), std::runtime_error);
    EXPECT_EQ(io.publications, 0u);
    EXPECT_EQ(io.source_reads, 0u);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Rebase), 0);
}

TEST(NiuHandoff, ProfileDestructorAndActualNotifyCompleteBeforeReceipt) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    ASSERT_EQ(receiver.runGo(io, [&] {
        ProfileScope profile{io};
        sourceStage(receiver, io);
    }), Admission::Accepted);
    ASSERT_EQ(io.publications, 1u);
    auto profile = std::find(io.events.begin(), io.events.end(), Event::ProfileExit);
    auto notify = std::find(io.events.begin(), io.events.end(), Event::Notify);
    auto readback = std::find(io.events.begin(), io.events.end(), Event::DescriptorReadback);
    auto publish = std::find(io.events.begin(), io.events.end(), Event::Publish);
    EXPECT_LT(profile, notify);
    EXPECT_LT(notify, readback);
    EXPECT_LT(readback, publish);
    EXPECT_EQ(io.receipt.control_expected[0][4], 41u);
    EXPECT_EQ(io.receipt.control_observed[0][4], 41u);
    EXPECT_EQ(io.receipt.control_expected[1][4], 49u);
    EXPECT_EQ(io.receipt.control_observed[1][4], 49u);
}

TEST(NiuHandoff, PendingNotifyNeverPublishesTrafficClosed) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    EXPECT_THROW(receiver.runGo(io, [&] {
        sourceStage(receiver, io);
        io.block = true;
    }), std::runtime_error);
    EXPECT_EQ(io.publications, 0u);
}

TEST(NiuHandoff, MalformedOrTornRequestCannotStartSourceOrNotify) {
    for (unsigned mutation = 0; mutation != 7; ++mutation) {
        DeviceIO io;
        switch (mutation) {
            case 0: io.request.abi_version = 2; break;
            case 1: io.request.reserved1 = 1; break;
            case 2: io.request.final_go_count = 0; break;
            case 3: io.request.selector_mask_noc1 = 0x10000; break;
            case 4: io.request.traffic_contract = 0; break;
            case 5: io.request.boot_cookie_lo = 0; break;
            case 6: io.torn_request = true; break;
        }
        Receiver receiver;
        ASSERT_TRUE(receiver.initializeBoot(io));
        io.events.clear();
        bool entered = false;
        EXPECT_EQ(receiver.runGo(io, [&] { entered = true; }), Admission::Rejected);
        EXPECT_FALSE(entered);
        EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Notify), 0);
    }
}

TEST(NiuHandoff, RequestMutationInsideWindowDoesNotAuthorizeAnotherGo) {
    DeviceIO io;
    io.request.final_go_count = 2;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    ASSERT_EQ(receiver.runGo(io, [&] { sourceStage(receiver, io); }), Admission::Accepted);
    EXPECT_EQ(io.receipt.header.completed_go_count, 1u);
    ++io.request.invocation_id_lo;
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; }), Admission::Rejected);
    EXPECT_FALSE(entered);
    EXPECT_EQ(io.publications, 1u);
}

TEST(NiuHandoff, FinalGoCountAndOriginalIdentityTravelWithReceipt) {
    DeviceIO io;
    io.request.final_go_count = 2;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    for (unsigned go = 0; go != 2; ++go) {
        ASSERT_EQ(receiver.runGo(io, [&] {
            receiver.beginStage(io, Source::None, 0);
            receiver.closeStage(io);
        }), Admission::Accepted);
        EXPECT_EQ(io.receipt.header.completed_go_count, go + 1);
        EXPECT_EQ(io.receipt.header.challenge_lo, 17u);
        EXPECT_EQ(io.receipt.header.boot_cookie_lo, 23u);
        EXPECT_EQ(io.receipt.header.invocation_id_lo, 29u);
    }
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; }), Admission::Rejected);
    EXPECT_FALSE(entered);
    EXPECT_EQ(io.publications, 2u);
}

TEST(NiuHandoff, IdleGoCannotHidePendingControlByRebase) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    ++io.control[1][4];
    io.block = true;
    EXPECT_THROW(receiver.runGo(io, [&] {
        receiver.beginStage(io, Source::None, 0);
        receiver.closeStage(io);
    }), std::runtime_error);
    EXPECT_EQ(io.publications, 0u);
    EXPECT_EQ(io.source_reads, 0u);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Rebase), 0);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Notify), 0);
}

TEST(NiuHandoff, UnmanagedGoAdvancesOrdinalButCannotSupplyManagedPredecessor) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    io.request.request_generation = 0;
    ASSERT_EQ(receiver.runGo(io, [] {}), Admission::Disabled);
    EXPECT_EQ(io.receipt.header.status, TT_NIU_HANDOFF_PENDING);
    EXPECT_EQ(io.receipt.consumed_go_sequence, 1u);
    io.request.request_generation = 1;
    io.request.first_go_sequence = 2;
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; sourceStage(receiver, io); }), Admission::Rejected);
    EXPECT_FALSE(entered);
    EXPECT_EQ(io.receipt.consumed_go_sequence, 2u);
    EXPECT_EQ(io.receipt.header.status, TT_NIU_HANDOFF_PENDING);
    EXPECT_EQ(io.pending_publications, 2u);
}

TEST(NiuHandoff, StaleFirstGoCannotClaimAnUnmanagedPrefix) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    io.request.request_generation = 0;
    ASSERT_EQ(receiver.runGo(io, [] {}), Admission::Disabled);
    io.request.request_generation = 1;
    // The request expects GO 1, but an actual unmanaged GO already consumed it.
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; sourceStage(receiver, io); }), Admission::Rejected);
    EXPECT_FALSE(entered);
    EXPECT_EQ(io.receipt.consumed_go_sequence, 2u);
    EXPECT_EQ(io.receipt.header.status, TT_NIU_HANDOFF_PENDING);
    EXPECT_EQ(io.publications, 0u);
}

TEST(NiuHandoff, NewGoClearsOldCleanReceiptBeforeEnteringBody) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    ASSERT_EQ(receiver.runGo(io, [&] { sourceStage(receiver, io); }), Admission::Accepted);
    const auto previous_sequence = io.receipt.header.publish_sequence;
    io.request.request_generation = 0;
    ASSERT_EQ(receiver.runGo(io, [&] {
        EXPECT_EQ(io.receipt.header.status, TT_NIU_HANDOFF_PENDING);
        EXPECT_EQ(io.receipt.header.challenge_lo, 0u);
        EXPECT_EQ(io.receipt.header.invocation_id_lo, 0u);
        EXPECT_EQ(io.receipt.header.completed_processor_mask, 0u);
        EXPECT_EQ(io.receipt.header.source_valid_noc_mask, 0u);
        EXPECT_EQ(io.receipt.header.control_valid_noc_mask, 0u);
        EXPECT_EQ(io.receipt.consumed_go_sequence, 2u);
        EXPECT_GT(io.receipt.header.publish_sequence, previous_sequence);
        EXPECT_EQ(io.receipt.header.publish_sequence & 1, 0u);
    }), Admission::Disabled);
}

TEST(NiuHandoff, EightSlotWrapDoesNotWrapConsumedGoOrdinal) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    io.request.request_generation = 0;
    // More than two turns through the physical eight-slot launch ring.
    for (unsigned go = 0; go != 17; ++go)
        ASSERT_EQ(receiver.runGo(io, [] {}), Admission::Disabled);
    io.request.request_generation = 1;
    io.request.first_go_sequence = 18;
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; sourceStage(receiver, io); }), Admission::Rejected);
    EXPECT_FALSE(entered);
    EXPECT_EQ(io.receipt.consumed_go_sequence, 18u);
    EXPECT_EQ(io.receipt.header.completed_go_count, 0u);
}

TEST(NiuHandoff, ClosedManagedGroupsReceiveTheNextActualOrdinal) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    ASSERT_EQ(receiver.runGo(io, [&] { sourceStage(receiver, io); }), Admission::Accepted);
    ++io.request.request_generation;
    ++io.request.challenge_lo;
    ++io.request.group_generation;
    io.request.first_go_sequence = 2;
    ASSERT_EQ(receiver.runGo(io, [&] {
        // Real firmware initializes the shared dynamic bank for the new
        // stage only after the previous managed control tail has closed.
        for (unsigned noc = 0; noc != 2; ++noc) {
            for (unsigned counter = 0; counter != 5; ++counter) {
                io.expected[noc][0][counter] = io.observed[noc][counter];
                io.expected[noc][1][counter] = 0;
            }
        }
        sourceStage(receiver, io);
    }), Admission::Accepted);
    EXPECT_EQ(io.receipt.consumed_go_sequence, 2u);
    EXPECT_EQ(io.receipt.header.group_generation, 2u);
    EXPECT_EQ(io.receipt.header.completed_go_count, 1u);
}

TEST(NiuHandoff, BootCannotBeReusedToClearAnUnmanagedPredecessor) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.request.request_generation = 0;
    ASSERT_EQ(receiver.runGo(io, [] {}), Admission::Disabled);
    EXPECT_FALSE(receiver.initializeBoot(io));
    io.request.request_generation = 1;
    io.request.first_go_sequence = 2;
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; sourceStage(receiver, io); }), Admission::Rejected);
    EXPECT_FALSE(entered);
}

TEST(NiuHandoff, UnmanagedResetCannotReuseBootReadyOrConsumeAGoOrdinal) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    const auto boot_sequence = io.receipt.header.publish_sequence;
    io.request.request_generation = 0;
    bool reset_executed = false;
    ASSERT_EQ(receiver.runUnmanagedControl(io, [&] {
        reset_executed = true;
        EXPECT_EQ(io.receipt.header.status, TT_NIU_HANDOFF_PENDING);
        EXPECT_EQ(io.receipt.consumed_go_sequence, 0u);
        EXPECT_GT(io.receipt.header.publish_sequence, boot_sequence);
        io.notify();
    }), Admission::Disabled);
    EXPECT_TRUE(reset_executed);
    io.request.request_generation = 1;
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; }), Admission::Rejected);
    EXPECT_FALSE(entered);
    EXPECT_EQ(io.receipt.consumed_go_sequence, 1u);
}

TEST(NiuHandoff, UnmanagedReplayCannotReuseClosedReceiptOrConsumeAGoOrdinal) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    ASSERT_EQ(receiver.runGo(io, [&] { sourceStage(receiver, io); }), Admission::Accepted);
    const auto closed_sequence = io.receipt.header.publish_sequence;
    io.request.request_generation = 0;
    bool replay_executed = false;
    ASSERT_EQ(receiver.runUnmanagedControl(io, [&] {
        replay_executed = true;
        EXPECT_EQ(io.receipt.header.status, TT_NIU_HANDOFF_PENDING);
        EXPECT_EQ(io.receipt.header.challenge_lo, 0u);
        EXPECT_EQ(io.receipt.header.control_valid_noc_mask, 0u);
        EXPECT_EQ(io.receipt.consumed_go_sequence, 1u);
        EXPECT_GT(io.receipt.header.publish_sequence, closed_sequence);
        io.notify();
    }), Admission::Disabled);
    EXPECT_TRUE(replay_executed);
    io.request.request_generation = 2;
    ++io.request.challenge_lo;
    io.request.first_go_sequence = 2;
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; }), Admission::Rejected);
    EXPECT_FALSE(entered);
    EXPECT_EQ(io.receipt.consumed_go_sequence, 2u);
}

TEST(NiuHandoff, BootReadyPublishesActualControlWithoutPayloadIdentity) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    EXPECT_EQ(io.boot_publications, 1u);
    EXPECT_EQ(io.receipt.header.status, TT_NIU_HANDOFF_BOOT_READY);
    EXPECT_EQ(io.receipt.header.abi_version, TT_NIU_HANDOFF_VERSION);
    EXPECT_GT(io.receipt.header.publish_sequence, 0u);
    EXPECT_EQ(io.receipt.header.publish_sequence & 1, 0u);
    EXPECT_EQ(io.receipt.consumed_go_sequence, 0u);
    EXPECT_EQ(io.receipt.header.challenge_lo, 0u);
    EXPECT_EQ(io.receipt.header.invocation_id_lo, 0u);
    EXPECT_EQ(io.receipt.header.source_valid_noc_mask, 0u);
    EXPECT_EQ(io.receipt.header.completed_processor_mask, 0x1fu);
    EXPECT_EQ(io.receipt.header.control_valid_noc_mask, 3u);
    EXPECT_EQ(io.receipt.control_expected[1][4], 34u);
    EXPECT_EQ(io.receipt.control_observed[1][4], 34u);
    EXPECT_EQ(io.receipt.control.ready_cmd_mask_noc0, 15u);
    EXPECT_EQ(io.receipt.control.ready_cmd_mask_noc1, 15u);
}

TEST(NiuHandoff, BootMustCloseActualControlBeforeAdmittingManagedWork) {
    DeviceIO io;
    Receiver receiver;
    ++io.control[0][4];
    io.block = true;
    EXPECT_THROW(receiver.initializeBoot(io), std::runtime_error);
    EXPECT_EQ(io.publications, 0u);
    EXPECT_EQ(std::count(io.events.begin(), io.events.end(), Event::Rebase), 0);
}

TEST(NiuHandoff, ConsumedGoOrdinalCannotWrapOrSaturateSuccessfully) {
    auto first = advanceGoSequence(0);
    EXPECT_EQ(first.status, GoSequenceStatus::Consumed);
    EXPECT_EQ(first.sequence, 1u);
    auto last = advanceGoSequence(UINT32_MAX - 1);
    EXPECT_EQ(last.status, GoSequenceStatus::Consumed);
    EXPECT_EQ(last.sequence, UINT32_MAX);
    auto exhausted = advanceGoSequence(UINT32_MAX);
    EXPECT_EQ(exhausted.status, GoSequenceStatus::Exhausted);
    EXPECT_EQ(exhausted.sequence, UINT32_MAX);
}

TEST(NiuHandoff, RequestGoRangeCannotOverflowOrdinal) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    io.request.first_go_sequence = UINT32_MAX;
    io.request.final_go_count = 2;
    bool entered = false;
    EXPECT_EQ(receiver.runGo(io, [&] { entered = true; sourceStage(receiver, io); }), Admission::Rejected);
    EXPECT_FALSE(entered);
    EXPECT_EQ(io.publications, 0u);
}

TEST(NiuHandoff, OpenStageCannotBeReportedAsClosed) {
    DeviceIO io;
    Receiver receiver;
    ASSERT_TRUE(receiver.initializeBoot(io));
    io.events.clear();
    EXPECT_EQ(receiver.runGo(io, [&] {
        receiver.beginStage(io, Source::BlackholeDynamic, 0x1f);
    }), Admission::Rejected);
    EXPECT_EQ(io.publications, 0u);
}
}  // namespace
}  // namespace tt::niu_handoff
