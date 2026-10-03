// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "hostdev/worker_stream_state_contract.h"
#include "gtest/gtest.h"
#include <array>
#include <cstdint>
#include <vector>

namespace tt::worker_stream_state {
namespace {
struct Write {
    uint32_t bank;
    Counter counter;
    uint32_t value;
};
struct Registers {
    std::array<uint32_t, 64> received;
    std::array<uint32_t, 64> acknowledged;
    std::vector<Write> writes;
    Registers() {
        for (uint32_t bank = 0; bank != received.size(); ++bank) {
            received[bank] = 0x10001u + bank * 3;
            acknowledged[bank] = 0x90001u + bank * 7;
        }
    }
    void operator()(uint32_t bank, Counter counter, uint32_t value) {
        ASSERT_LT(bank, received.size());
        writes.push_back({bank, counter, value});
        (counter == Counter::Received ? received : acknowledged)[bank] = value;
    }
};

TEST(WorkerStreamStateContract, ProgramOwnerPreservesEveryCounterAtEntryAndExit) {
    Registers registers;
    const auto received = registers.received;
    const auto acknowledged = registers.acknowledged;
    for (Boundary boundary : {Boundary::LaunchEntry, Boundary::LaunchExit}) {
        auto decision = receive(kVersion, static_cast<uint8_t>(Owner::Program), boundary, false);
        ASSERT_EQ(decision, Decision::Preserve);
        apply(decision, 64, registers);
    }
    EXPECT_EQ(registers.received, received);
    EXPECT_EQ(registers.acknowledged, acknowledged);
    EXPECT_TRUE(registers.writes.empty());
}

TEST(WorkerStreamStateContract, SdkOwnsItsFullMappedBankEvenWithoutDescriptors) {
    for (bool descriptors : {false, true}) {
        for (Boundary boundary : {Boundary::LaunchEntry, Boundary::LaunchExit}) {
            Registers registers;
            auto decision = receive(kVersion, static_cast<uint8_t>(Owner::SdkCircularBuffers), boundary, descriptors);
            ASSERT_EQ(decision, Decision::ClearSdkCounters);
            apply(decision, 64, registers);
            ASSERT_EQ(registers.writes.size(), 128u);
            for (unsigned bank = 0; bank != 64; ++bank) {
                EXPECT_EQ(registers.received[bank], 0u);
                EXPECT_EQ(registers.acknowledged[bank], 0u);
                const auto &received = registers.writes[bank * 2];
                const auto &acked = registers.writes[bank * 2 + 1];
                EXPECT_EQ(received.bank, bank);
                EXPECT_EQ(received.counter, Counter::Received);
                EXPECT_EQ(received.value, 0u);
                EXPECT_EQ(acked.bank, bank);
                EXPECT_EQ(acked.counter, Counter::Acknowledged);
                EXPECT_EQ(acked.value, 0u);
            }
        }
    }
}

TEST(WorkerStreamStateContract, OldMissingAndUnknownVersionNeverSelectReset) {
    for (uint8_t version : {uint8_t{0}, uint8_t{2}, uint8_t{255}}) {
        for (Owner owner : {Owner::SdkCircularBuffers, Owner::Program}) {
            Registers registers;
            const auto before = registers.received;
            auto decision = receive(version, static_cast<uint8_t>(owner), Boundary::LaunchEntry, false);
            EXPECT_EQ(decision, Decision::RejectVersion);
            apply(decision, 64, registers);
            EXPECT_EQ(registers.received, before);
            EXPECT_TRUE(registers.writes.empty());
        }
    }
}

TEST(WorkerStreamStateContract, UnknownOwnerNeverDefaultsToSdkOrProgram) {
    for (uint8_t owner : {uint8_t{0}, uint8_t{3}, uint8_t{255}}) {
        Registers registers;
        auto decision = receive(kVersion, owner, Boundary::LaunchExit, false);
        EXPECT_EQ(decision, Decision::RejectOwner);
        apply(decision, 64, registers);
        EXPECT_TRUE(registers.writes.empty());
    }
}

TEST(WorkerStreamStateContract, ProgramOwnerRejectsAnySdkDescriptorBeforeWrites) {
    Registers registers;
    auto decision = receive(kVersion, static_cast<uint8_t>(Owner::Program), Boundary::LaunchEntry, true);
    EXPECT_EQ(decision, Decision::RejectSdkDescriptor);
    apply(decision, 64, registers);
    EXPECT_TRUE(registers.writes.empty());
}

TEST(WorkerStreamStateContract, ProgramToSdkEntryClearsPriorProgramValues) {
    Registers registers;
    apply(receive(kVersion, static_cast<uint8_t>(Owner::Program), Boundary::LaunchExit, false), 64, registers);
    ASSERT_NE(registers.received[17], 0u);
    apply(receive(kVersion, static_cast<uint8_t>(Owner::SdkCircularBuffers), Boundary::LaunchEntry, false), 64, registers);
    EXPECT_EQ(registers.received[17], 0u);
    EXPECT_EQ(registers.acknowledged[17], 0u);
    EXPECT_EQ(registers.writes.size(), 128u);
}

TEST(WorkerStreamStateContract, SdkToProgramPreservesNewInvocationSentinels) {
    Registers registers;
    apply(receive(kVersion, static_cast<uint8_t>(Owner::SdkCircularBuffers), Boundary::LaunchExit, true), 64, registers);
    // This unit test models initialization only after the reset action returns.
    // Actual BRISC/TRISC0 acknowledgment order needs firmware integration tests.
    registers.received[17] = 0x123456;
    registers.acknowledged[17] = 0x654321;
    registers.writes.clear();
    apply(receive(kVersion, static_cast<uint8_t>(Owner::Program), Boundary::LaunchEntry, false), 64, registers);
    apply(receive(kVersion, static_cast<uint8_t>(Owner::Program), Boundary::LaunchExit, false), 64, registers);
    EXPECT_EQ(registers.received[17], 0x123456u);
    EXPECT_EQ(registers.acknowledged[17], 0x654321u);
    EXPECT_TRUE(registers.writes.empty());
}

TEST(WorkerStreamStateContract, ShorterArchitectureUsesOnlyItsMappedCounters) {
    Registers registers;
    auto before = registers.received;
    apply(receive(kVersion, static_cast<uint8_t>(Owner::SdkCircularBuffers), Boundary::LaunchEntry, true), 32, registers);
    EXPECT_EQ(registers.writes.size(), 64u);
    for (unsigned bank = 32; bank != 64; ++bank)
        EXPECT_EQ(registers.received[bank], before[bank]);
}
}  // namespace
}  // namespace tt::worker_stream_state
