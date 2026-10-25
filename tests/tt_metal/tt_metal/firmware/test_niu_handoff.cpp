// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "internal/niu_handoff.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace tt::niu_handoff {
namespace {
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
