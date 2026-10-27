// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "impl/context/niu_admission.hpp"
#include "impl/device/firmware/risc_firmware_initializer.hpp"

#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <thread>

namespace tt::tt_metal::detail {
namespace {

// These instances test the allocation algorithm only. No test constructs a
// LoadedFirmware, publishes a boot, or grants an executable admission.
class NiuLedgerTest : public ::testing::Test {
protected:
    std::optional<NiuAllocation> acquire(NiuKey key, uint8_t count) {
        const std::array requests{NiuRequest{key, count}};
        return ledger.acquire_managed(requests, error);
    }
    std::optional<NiuToken> ordinary(NiuKey key) {
        const std::array keys{key};
        return ledger.reserve_ordinary(keys, error);
    }
    static constexpr NiuNoc noc0 = NiuNoc::Noc0;
    static constexpr NiuNoc noc1 = NiuNoc::Noc1;
    const NiuKey a{0, {1, 2}, noc0};
    const NiuKey b{0, {1, 2}, noc1};
    NiuLedger ledger;
    std::string error;
};

TEST_F(NiuLedgerTest, ManagedSelectorsAreUniqueWithinOnePhysicalNiu) {
    auto first = acquire(a, 5);
    ASSERT_TRUE(first) << error;
    auto second = acquire(a, 11);
    ASSERT_TRUE(second) << error;
    ASSERT_EQ(first->selections.size(), 1u);
    ASSERT_EQ(second->selections.size(), 1u);
    EXPECT_EQ(first->selections[0].selectors.count(), 5u);
    EXPECT_EQ(second->selections[0].selectors.count(), 11u);
    EXPECT_FALSE((first->selections[0].selectors & second->selections[0].selectors).any());
    EXPECT_FALSE(acquire(a, 1));
    EXPECT_FALSE(ordinary(a));
}

TEST_F(NiuLedgerTest, SelectorSubleasesUseOnlyTheirActualAllocationMask) {
    auto occupied = acquire(a, 5);
    auto allocation = acquire(a, 4);
    ASSERT_TRUE(occupied) << error;
    ASSERT_TRUE(allocation) << error;
    NiuSelectorLeaseBook selectors(allocation->selections);
    EXPECT_FALSE(selectors.acquire(b));
    NiuSelectorMask observed;
    for (unsigned item = 0; item != 4; ++item) {
        auto selector = selectors.acquire(a);
        ASSERT_TRUE(selector);
        ASSERT_LT(*selector, niu_selector_count);
        EXPECT_TRUE(allocation->selections[0].selectors.test(*selector));
        EXPECT_FALSE(occupied->selections[0].selectors.test(*selector));
        EXPECT_FALSE(observed.test(*selector));
        observed.set(*selector);
    }
    EXPECT_EQ(observed, allocation->selections[0].selectors);
    EXPECT_FALSE(selectors.acquire(a));
}

TEST_F(NiuLedgerTest, SelectorSubleasesRemainDistinctAcrossConcurrentClaimers) {
    auto allocation = acquire(a, 16);
    ASSERT_TRUE(allocation) << error;
    NiuSelectorLeaseBook selectors(allocation->selections);
    std::array<std::optional<uint8_t>, 32> claims;
    std::vector<std::thread> threads;
    for (size_t item = 0; item != claims.size(); ++item) {
        threads.emplace_back([&, item] { claims[item] = selectors.acquire(a); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    NiuSelectorMask observed;
    size_t accepted = 0;
    for (const auto& selector : claims) {
        if (!selector) {
            continue;
        }
        ++accepted;
        ASSERT_LT(*selector, niu_selector_count);
        EXPECT_FALSE(observed.test(*selector));
        observed.set(*selector);
    }
    EXPECT_EQ(accepted, niu_selector_count);
    EXPECT_EQ(observed, allocation->selections[0].selectors);
    EXPECT_FALSE(selectors.acquire(a));
}

TEST_F(NiuLedgerTest, SelectorSubleaseExhaustionDoesNotReleaseThePhysicalAllocation) {
    auto allocation = acquire(a, 1);
    ASSERT_TRUE(allocation) << error;
    NiuSelectorLeaseBook selectors(allocation->selections);
    ASSERT_TRUE(selectors.acquire(a));
    EXPECT_FALSE(selectors.acquire(a));
    // A consumed/dropped selector is not an NIU-clean receipt. The original
    // physical allocation still excludes ordinary traffic and remains live.
    EXPECT_FALSE(ordinary(a));
    auto remainder = acquire(a, 15);
    ASSERT_TRUE(remainder) << error;
    EXPECT_FALSE(acquire(a, 1));
}

TEST_F(NiuLedgerTest, OrdinaryProgramsShareButEveryReservationExcludesManaged) {
    auto first = ordinary(a);
    auto second = ordinary(a);
    ASSERT_TRUE(first) << error;
    ASSERT_TRUE(second) << error;
    EXPECT_FALSE(acquire(a, 1));
    ASSERT_TRUE(ledger.cancel_unissued(*first, error)) << error;
    EXPECT_FALSE(acquire(a, 1));
    ASSERT_TRUE(ledger.cancel_unissued(*second, error)) << error;
    EXPECT_TRUE(acquire(a, 16)) << error;
}

TEST_F(NiuLedgerTest, OrdinaryStopAllowsAnotherLaunchButRequiresCleanForManagedReuse) {
    auto first = ordinary(a);
    ASSERT_TRUE(first) << error;
    ASSERT_TRUE(ledger.mark_issued(*first, error)) << error;
    ASSERT_TRUE(ledger.record_stop(*first, error)) << error;
    EXPECT_FALSE(acquire(a, 1));
    auto next = ordinary(a);
    ASSERT_TRUE(next) << error;
    ASSERT_TRUE(ledger.mark_issued(*next, error)) << error;
    ASSERT_TRUE(ledger.record_clean(*first, error)) << error;
    EXPECT_FALSE(acquire(a, 1));
    ASSERT_TRUE(ledger.record_stop(*next, error)) << error;
    EXPECT_FALSE(acquire(a, 1));
    ASSERT_TRUE(ledger.record_clean(*next, error)) << error;
    EXPECT_TRUE(acquire(a, 16)) << error;
}

TEST_F(NiuLedgerTest, ManagedStopAlsoRequiresSeparateCleanProof) {
    auto allocation = acquire(a, 16);
    ASSERT_TRUE(allocation) << error;
    EXPECT_FALSE(ledger.record_clean(allocation->token, error));
    ASSERT_TRUE(ledger.mark_issued(allocation->token, error)) << error;
    EXPECT_FALSE(ledger.record_clean(allocation->token, error));
    ASSERT_TRUE(ledger.record_stop(allocation->token, error)) << error;
    EXPECT_FALSE(acquire(a, 1));
    EXPECT_FALSE(ordinary(a));
    ASSERT_TRUE(ledger.record_clean(allocation->token, error)) << error;
    EXPECT_TRUE(ordinary(a)) << error;
}

TEST_F(NiuLedgerTest, FailedManagedBatchDoesNotOccupyItsEarlierNiu) {
    auto occupied = ordinary(b);
    ASSERT_TRUE(occupied) << error;
    const std::array requests{NiuRequest{a, 16}, NiuRequest{b, 1}};
    EXPECT_FALSE(ledger.acquire_managed(requests, error));
    EXPECT_TRUE(acquire(a, 16)) << error;
}

TEST_F(NiuLedgerTest, FailedOrdinaryBatchDoesNotOccupyItsEarlierNiu) {
    auto occupied = acquire(b, 1);
    ASSERT_TRUE(occupied) << error;
    const std::array keys{a, b};
    EXPECT_FALSE(ledger.reserve_ordinary(keys, error));
    EXPECT_TRUE(acquire(a, 16)) << error;
}

TEST_F(NiuLedgerTest, CapacityFailureDoesNotConsumeAnyOtherSelector) {
    auto occupied = acquire(b, 15);
    ASSERT_TRUE(occupied) << error;
    const std::array requests{NiuRequest{a, 16}, NiuRequest{b, 2}};
    EXPECT_FALSE(ledger.acquire_managed(requests, error));
    EXPECT_TRUE(acquire(a, 16)) << error;
    EXPECT_TRUE(acquire(b, 1)) << error;
}

TEST_F(NiuLedgerTest, NocWorkerAndChipHaveIndependentSelectorSpaces) {
    ASSERT_TRUE(acquire(a, 16)) << error;
    EXPECT_TRUE(acquire(b, 16)) << error;
    EXPECT_TRUE(acquire({0, {2, 2}, noc0}, 16)) << error;
    EXPECT_TRUE(acquire({1, {1, 2}, noc0}, 16)) << error;
    EXPECT_FALSE(acquire(a, 1));
}

TEST_F(NiuLedgerTest, IssuedTokenDestructionAndCancellationCannotReleaseSelectors) {
    {
        auto token = ordinary(a);
        ASSERT_TRUE(token) << error;
        ASSERT_TRUE(ledger.mark_issued(*token, error)) << error;
        EXPECT_FALSE(ledger.cancel_unissued(*token, error));
    }
    EXPECT_FALSE(acquire(a, 1));
    EXPECT_FALSE(ordinary(a));
    {
        auto token = acquire(b, 16);
        ASSERT_TRUE(token) << error;
        ASSERT_TRUE(ledger.mark_issued(token->token, error)) << error;
    }
    EXPECT_FALSE(acquire(b, 1));
    EXPECT_FALSE(ordinary(b));
}

TEST_F(NiuLedgerTest, StoppedOrdinaryTokenDestructionPreservesOrdinaryCompatibility) {
    {
        auto token = ordinary(a);
        ASSERT_TRUE(token) << error;
        ASSERT_TRUE(ledger.mark_issued(*token, error)) << error;
        ASSERT_TRUE(ledger.record_stop(*token, error)) << error;
    }
    EXPECT_FALSE(acquire(a, 1));
    auto next = ordinary(a);
    ASSERT_TRUE(next) << error;
    ASSERT_TRUE(ledger.mark_issued(*next, error)) << error;
    ASSERT_TRUE(ledger.record_stop(*next, error)) << error;
    ASSERT_TRUE(ledger.record_clean(*next, error)) << error;
    // The lost stopped owner still has no NIU-clean proof.
    EXPECT_FALSE(acquire(a, 1));
    EXPECT_TRUE(ordinary(a)) << error;
}

TEST_F(NiuLedgerTest, QuarantineRejectsBothKindsAndCannotBeCancelledOrCleaned) {
    auto token = ordinary(a);
    ASSERT_TRUE(token) << error;
    ASSERT_TRUE(ledger.mark_issued(*token, error)) << error;
    ASSERT_TRUE(ledger.quarantine(*token, error)) << error;
    EXPECT_FALSE(ordinary(a));
    EXPECT_FALSE(acquire(a, 1));
    EXPECT_FALSE(ledger.cancel_unissued(*token, error));
    EXPECT_FALSE(ledger.record_stop(*token, error));
    EXPECT_FALSE(ledger.record_clean(*token, error));
    EXPECT_TRUE(acquire(b, 16)) << error;
}

TEST_F(NiuLedgerTest, QuarantinePreventsIssuingAnAlreadyReservedOrdinaryBatch) {
    auto first = ordinary(a);
    const std::array keys{b, a};
    auto pending = ledger.reserve_ordinary(keys, error);
    ASSERT_TRUE(first) << error;
    ASSERT_TRUE(pending) << error;
    ASSERT_TRUE(ledger.quarantine(*first, error)) << error;

    EXPECT_FALSE(ledger.mark_issued(*pending, error));
    ASSERT_TRUE(ledger.cancel_unissued(*pending, error)) << error;
    auto healthy = ordinary(b);
    ASSERT_TRUE(healthy) << error;
    EXPECT_TRUE(ledger.mark_issued(*healthy, error)) << error;
}

TEST_F(NiuLedgerTest, LostIssuedOwnerPreventsIssuingAnAlreadyReservedOrdinaryOwner) {
    auto first = ordinary(a);
    auto pending = ordinary(a);
    auto healthy = ordinary(b);
    ASSERT_TRUE(first) << error;
    ASSERT_TRUE(pending) << error;
    ASSERT_TRUE(healthy) << error;
    ASSERT_TRUE(ledger.mark_issued(*first, error)) << error;
    first.reset();

    // No intervening reserve/acquire may be needed to discover the lost owner.
    EXPECT_FALSE(ledger.mark_issued(*pending, error));
    ASSERT_TRUE(ledger.cancel_unissued(*pending, error)) << error;
    EXPECT_TRUE(ledger.mark_issued(*healthy, error)) << error;
}

TEST_F(NiuLedgerTest, QuarantinePreventsIssuingOtherReservedSelectorsOnTheSameNiu) {
    auto first = acquire(a, 1);
    auto pending = acquire(a, 15);
    ASSERT_TRUE(first) << error;
    ASSERT_TRUE(pending) << error;
    ASSERT_FALSE((first->selections[0].selectors & pending->selections[0].selectors).any());
    ASSERT_TRUE(ledger.quarantine(first->token, error)) << error;

    EXPECT_FALSE(ledger.mark_issued(pending->token, error));
    ASSERT_TRUE(ledger.cancel_unissued(pending->token, error)) << error;
    auto healthy = acquire(b, 16);
    ASSERT_TRUE(healthy) << error;
    EXPECT_TRUE(ledger.mark_issued(healthy->token, error)) << error;
}

TEST_F(NiuLedgerTest, ForeignAndRetiredTokensCannotChangeAnotherReservation) {
    auto token = ordinary(a);
    ASSERT_TRUE(token) << error;
    NiuLedger foreign;
    EXPECT_FALSE(foreign.cancel_unissued(*token, error));
    EXPECT_FALSE(ledger.cancel_unissued(NiuToken{}, error));
    ASSERT_TRUE(ledger.cancel_unissued(*token, error)) << error;
    auto replacement = ordinary(a);
    ASSERT_TRUE(replacement) << error;
    EXPECT_FALSE(ledger.cancel_unissued(*token, error));
    EXPECT_FALSE(acquire(a, 1));
}

TEST_F(NiuLedgerTest, InvalidBatchesHaveNoAllocationSideEffects) {
    EXPECT_FALSE(ledger.acquire_managed({}, error));
    EXPECT_FALSE(ledger.reserve_ordinary({}, error));
    EXPECT_FALSE(acquire(a, 0));
    EXPECT_FALSE(acquire(a, 17));
    EXPECT_FALSE(acquire({0, {1, 2}, static_cast<NiuNoc>(2)}, 1));
    const std::array requests{NiuRequest{a, 1}, NiuRequest{a, 1}};
    EXPECT_FALSE(ledger.acquire_managed(requests, error));
    const std::array keys{a, a};
    EXPECT_FALSE(ledger.reserve_ordinary(keys, error));
    EXPECT_TRUE(acquire(a, 16)) << error;
}

TEST_F(NiuLedgerTest, WithdrawRejectsNewAdmissionAndCannotRecycleOldTokens) {
    auto token = ordinary(a);
    ASSERT_TRUE(token) << error;
    ASSERT_TRUE(ledger.mark_issued(*token, error)) << error;
    ledger.withdraw();
    EXPECT_FALSE(acquire(b, 1));
    EXPECT_FALSE(ordinary(b));
    EXPECT_FALSE(ledger.cancel_unissued(*token, error));
    EXPECT_FALSE(ledger.record_stop(*token, error));
    EXPECT_FALSE(ledger.record_clean(*token, error));
    NiuLedger next;
    EXPECT_FALSE(next.record_clean(*token, error));
    const std::array requests{NiuRequest{a, 16}};
    EXPECT_TRUE(next.acquire_managed(requests, error)) << error;
}

}  // namespace
}  // namespace tt::tt_metal::detail

namespace tt::tt_metal {
namespace {

template <typename Initializer>
concept ExposesNativeFirmwareLookup = requires(const Initializer& initializer, ChipId chip) {
    initializer.native_firmware(chip);
};

class NiuAdmissionTest : public ::testing::Test {
protected:
    std::optional<NiuAdmission::Token> ordinary(detail::NiuKey key) {
        const std::array keys{key};
        return admission.reserve_ordinary(keys, error);
    }
    const detail::NiuKey a{0, {1, 2}, detail::NiuNoc::Noc0};
    const detail::NiuKey b{0, {1, 2}, detail::NiuNoc::Noc1};
    NiuAdmission admission;
    std::string error;
};

TEST_F(NiuAdmissionTest, InitializerDoesNotExposeUnpublishedNativeFirmwareLookup) {
    EXPECT_FALSE(ExposesNativeFirmwareLookup<RiscFirmwareInitializer>);
}

TEST_F(NiuAdmissionTest, FirmwareLookupRequiresAnActualPublishedBoot) {
    const NiuAdmission& published = admission;
    EXPECT_FALSE(published.native_firmware(a.chip));
    EXPECT_FALSE(published.native_firmware(1));

    auto pending = ordinary(a);
    ASSERT_TRUE(pending) << error;
    EXPECT_FALSE(published.native_firmware(a.chip));
    ASSERT_TRUE(admission.cancel_unissued(*pending, error)) << error;

    auto issued = ordinary(a);
    ASSERT_TRUE(issued) << error;
    ASSERT_TRUE(admission.mark_issued(*issued, error)) << error;
    EXPECT_FALSE(published.native_firmware(a.chip));
    ASSERT_TRUE(admission.quarantine(*issued, error)) << error;
    EXPECT_FALSE(published.native_firmware(a.chip));
}

TEST_F(NiuAdmissionTest, OrdinaryIntentsCanShareBeforeAnyNativeBoot) {
    auto first = ordinary(a);
    auto second = ordinary(a);
    ASSERT_TRUE(first) << error;
    ASSERT_TRUE(second) << error;
    ASSERT_TRUE(admission.cancel_unissued(*first, error)) << error;
    ASSERT_TRUE(admission.mark_issued(*second, error)) << error;
    EXPECT_FALSE(admission.cancel_unissued(*second, error));
    EXPECT_TRUE(ordinary(a)) << error;
}

TEST_F(NiuAdmissionTest, ManagedAdmissionRejectsAbsentAndNullBoots) {
    const std::array requests{detail::NiuRequest{a, 1}};
    EXPECT_FALSE(admission.acquire_managed({}, requests, error));
    const NiuAdmission::Boots absent{{0, nullptr}};
    EXPECT_FALSE(admission.acquire_managed(absent, requests, error));
    EXPECT_TRUE(ordinary(a)) << error;
}

TEST_F(NiuAdmissionTest, LostIssuedIntentQuarantinesOnlyItsPhysicalNiuBeforeBoot) {
    {
        auto token = ordinary(a);
        ASSERT_TRUE(token) << error;
        ASSERT_TRUE(admission.mark_issued(*token, error)) << error;
    }
    EXPECT_FALSE(ordinary(a));
    EXPECT_TRUE(ordinary(b)) << error;
}

TEST_F(NiuAdmissionTest, ForeignAndRetiredTokensCannotCancelAnotherIntent) {
    auto token = ordinary(a);
    ASSERT_TRUE(token) << error;
    NiuAdmission foreign;
    EXPECT_FALSE(foreign.cancel_unissued(*token, error));
    EXPECT_FALSE(admission.cancel_unissued(NiuAdmission::Token{}, error));
    ASSERT_TRUE(admission.cancel_unissued(*token, error)) << error;
    auto replacement = ordinary(a);
    ASSERT_TRUE(replacement) << error;
    EXPECT_FALSE(admission.cancel_unissued(*token, error));
    ASSERT_TRUE(admission.mark_issued(*replacement, error)) << error;
}

TEST_F(NiuAdmissionTest, ExplicitQuarantineRejectsFutureOrdinaryIntents) {
    auto token = ordinary(a);
    ASSERT_TRUE(token) << error;
    ASSERT_TRUE(admission.quarantine(*token, error)) << error;
    EXPECT_FALSE(admission.cancel_unissued(*token, error));
    EXPECT_FALSE(admission.mark_issued(*token, error));
    EXPECT_FALSE(ordinary(a));
    EXPECT_TRUE(ordinary(b)) << error;
}

TEST_F(NiuAdmissionTest, QuarantinePreventsIssuingAnAlreadyReservedOrdinaryBatch) {
    auto first = ordinary(a);
    const std::array keys{b, a};
    auto pending = admission.reserve_ordinary(keys, error);
    ASSERT_TRUE(first) << error;
    ASSERT_TRUE(pending) << error;
    ASSERT_TRUE(admission.quarantine(*first, error)) << error;

    EXPECT_FALSE(admission.mark_issued(*pending, error));
    ASSERT_TRUE(admission.cancel_unissued(*pending, error)) << error;
    auto healthy = ordinary(b);
    ASSERT_TRUE(healthy) << error;
    EXPECT_TRUE(admission.mark_issued(*healthy, error)) << error;
}

TEST_F(NiuAdmissionTest, LostIssuedOwnerPreventsIssuingAnAlreadyReservedOrdinaryOwner) {
    auto first = ordinary(a);
    auto pending = ordinary(a);
    auto healthy = ordinary(b);
    ASSERT_TRUE(first) << error;
    ASSERT_TRUE(pending) << error;
    ASSERT_TRUE(healthy) << error;
    ASSERT_TRUE(admission.mark_issued(*first, error)) << error;
    first.reset();

    // The issue boundary itself must discover the abandoned issued intent.
    EXPECT_FALSE(admission.mark_issued(*pending, error));
    ASSERT_TRUE(admission.cancel_unissued(*pending, error)) << error;
    EXPECT_TRUE(admission.mark_issued(*healthy, error)) << error;
}

}  // namespace
}  // namespace tt::tt_metal
