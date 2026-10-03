// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "hostdev/worker_stream_state_contract.h"
#include "gtest/gtest.h"
#include <stdexcept>
#include <vector>

namespace tt::worker_stream_state {
namespace {
enum class Event { Request, Poll, Acknowledge, Publish };
struct PendingReset final {
    std::vector<Event> events;
    bool done = false;
    bool block = false;
    unsigned polls = 0;
    void request() { events.push_back(Event::Request); done = false; }
    bool complete() const { return done; }
    void poll() {
        events.push_back(Event::Poll);
        if (block) throw std::runtime_error("reset remains pending");
        if (++polls == 3) {
            done = true;
            events.push_back(Event::Acknowledge);
        }
    }
};
TEST(WorkerStreamStateResetOrder, SdkBoundaryCannotPublishBeforeActualAck) {
    for (Boundary boundary : {Boundary::LaunchEntry, Boundary::LaunchExit}) {
        PendingReset reset;
        auto decision = receive(kVersion, static_cast<uint8_t>(Owner::SdkCircularBuffers), boundary, false);
        ASSERT_TRUE(completeReset(decision, reset));
        reset.events.push_back(Event::Publish);
        EXPECT_EQ(reset.events, (std::vector<Event>{Event::Request, Event::Poll, Event::Poll,
                                                   Event::Poll, Event::Acknowledge, Event::Publish}));
    }
}
TEST(WorkerStreamStateResetOrder, PendingResetDoesNotAuthorizeLaunchOrDone) {
    PendingReset reset;
    reset.block = true;
    bool published = false;
    EXPECT_THROW({
        if (completeReset(Decision::ClearSdkCounters, reset)) published = true;
    }, std::runtime_error);
    EXPECT_FALSE(published);
    EXPECT_EQ(reset.events, (std::vector<Event>{Event::Request, Event::Poll}));
}
TEST(WorkerStreamStateResetOrder, ProgramBoundaryNeverIssuesResetOrWaits) {
    PendingReset reset;
    reset.block = true;
    EXPECT_TRUE(completeReset(Decision::Preserve, reset));
    EXPECT_TRUE(reset.events.empty());
}
TEST(WorkerStreamStateResetOrder, RejectedContractCannotPublishOrReset) {
    for (Decision invalid : {Decision::RejectVersion, Decision::RejectOwner, Decision::RejectSdkDescriptor}) {
        PendingReset reset;
        EXPECT_FALSE(completeReset(invalid, reset));
        EXPECT_TRUE(reset.events.empty());
    }
}
TEST(WorkerStreamStateResetOrder, BootUsesSameAcknowledgedClearWithoutLaunchOwner) {
    PendingReset reset;
    ASSERT_TRUE(completeReset(Decision::ClearSdkCounters, reset));
    EXPECT_TRUE(reset.done);
    EXPECT_EQ(reset.events.back(), Event::Acknowledge);
}
} // namespace
} // namespace tt::worker_stream_state
