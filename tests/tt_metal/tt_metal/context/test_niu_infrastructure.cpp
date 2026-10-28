// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "impl/context/niu_admission.hpp"
#include "impl/context/niu_handoff_receiver.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <limits>
#include <future>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/sub_device.hpp>
#include "distributed/mesh_device_impl.hpp"
#include "distributed/fd_mesh_command_queue.hpp"
#include "distributed/pinned_memory_impl.hpp"
#include "../api/metal2_host_api/test_helpers.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/buffer_impl.hpp"
#include "impl/experimental/native_kernel/native_image.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/dispatch/system_memory_manager.hpp"
#include "impl/internal/service/service_core_manager_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/program_execution.hpp"
#include "impl/threading/thread_pool.hpp"
#include "llrt/tt_cluster.hpp"

namespace {
thread_local size_t unexpected_handoff_reads = 0;
}

namespace tt::tt_metal::detail {
// The real Mock fixtures cannot create a successful firmware boot. Reject any
// attempt to reach mailbox transport rather than manufacture observation bytes.
__attribute__((visibility("default")))
NiuHandoffReadResult NiuHandoffReceiver::read(const Cluster&, const Hal&, ChipId, CoreCoord) {
    ++unexpected_handoff_reads;
    throw std::runtime_error("Unexpected mailbox transport without an actual managed boot");
}
}

namespace tt::tt_metal {
namespace {

template <class Allocation>
constexpr bool mutable_managed_token = requires(Allocation& allocation) {
    allocation.token = NiuAdmission::Token{};
};
template <class Allocation>
constexpr bool mutable_managed_selections = requires(Allocation& allocation) {
    allocation.selections.clear();
};

TEST(NiuManagedInvocationContractTest, AllocationProjectionCannotBeForgedOrMutated) {
    EXPECT_FALSE(std::is_default_constructible_v<NiuAdmission::Allocation>);
    EXPECT_FALSE(std::is_aggregate_v<NiuAdmission::Allocation>);
    EXPECT_FALSE(mutable_managed_token<NiuAdmission::Allocation>);
    EXPECT_FALSE(mutable_managed_selections<NiuAdmission::Allocation>);
    EXPECT_FALSE(std::is_copy_assignable_v<NiuAdmission::Allocation>);
    EXPECT_FALSE(std::is_move_assignable_v<NiuAdmission::Allocation>);
    using SelectionView = decltype(std::declval<const NiuAdmission::Allocation&>().selections());
    EXPECT_TRUE((std::is_same_v<SelectionView, std::span<const detail::NiuSelection>>));
}

template <class Binding>
concept exposes_completion_transaction_identity = requires(const Binding& binding) {
    binding.niuId;
    binding.trid;
};

TEST(NiuManagedInvocationContractTest, CompletionBindingCarriesOnlyActualAllocationAndBootCorrelation) {
    using Binding = detail::ProgramExecution::ManagedInvocation::CompletionBinding;
    EXPECT_FALSE(exposes_completion_transaction_identity<Binding>);
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Binding&>().niu), detail::NiuKey>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Binding&>().selectors), detail::NiuSelectorMask>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Binding&>().boot_cookie), uint64_t>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Binding&>().invocation_id), uint64_t>));
    // The SDK deliberately has no transaction-ID provider or per-item
    // completion observation yet. A runtime adapter must therefore reject a
    // nonempty package inventory instead of deriving NIU/TRID from a scope or
    // address. Aggregate mailbox observations remain separately typed.
    using Observation = detail::ProgramExecution::ManagedInvocation::CompletionObservation;
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Observation&>().binding), Binding>));
    EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Observation&>().handoff),
                                detail::ProgramExecution::HandoffObservation>));
}

template <class Allocation>
concept exposes_selector_sublease = requires(const Allocation& allocation, const detail::NiuKey& key) {
    allocation.acquire_selector(key);
};

template <class Invocation>
concept exposes_selector_observation = requires(
    const Invocation& invocation, const typename Invocation::SelectorLease& selector) {
    typename Invocation::SelectorObservation;
    invocation.observe_selector(selector);
    invocation.retire_selectors();
};

template <class Invocation>
void check_selector_observation_contract() {
    if constexpr (!exposes_selector_observation<Invocation>) {
        ADD_FAILURE() << "Managed invocation has no original-owner selector observation";
    } else {
        using Observation = typename Invocation::SelectorObservation;
        EXPECT_FALSE(std::is_default_constructible_v<Observation>);
        EXPECT_FALSE(std::is_aggregate_v<Observation>);
        EXPECT_TRUE((std::is_same_v<
            decltype(std::declval<const Invocation&>().observe_selector(
                std::declval<const typename Invocation::SelectorLease&>())),
            std::optional<Observation>>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Observation&>().niu()), detail::NiuKey>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Observation&>().selector()), uint8_t>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Observation&>().boot_cookie()), uint64_t>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Observation&>().invocation_id()), uint64_t>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Invocation&>().retire_selectors()), bool>));
    }
}

TEST(NiuManagedInvocationContractTest, SelectorObservationCannotBeMintedByTheRuntimeAdapter) {
    check_selector_observation_contract<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
concept exposes_item_witness_owner = requires(
    const Invocation& invocation, const typename Invocation::SelectorLease& selector,
    const typename Invocation::ItemWitnessLease& lease,
    const typename Invocation::ItemWitnessRequirement& requirement) {
    typename Invocation::ItemWitnessObservation;
    invocation.acquire_item_witness(selector, requirement);
    invocation.poll_item(lease);
};

template <class Invocation>
void check_item_witness_owner_contract() {
    if constexpr (!exposes_item_witness_owner<Invocation>) {
        ADD_FAILURE() << "Managed invocation has no owned item witness allocation/observer";
    } else {
        using Lease = typename Invocation::ItemWitnessLease;
        using Observation = typename Invocation::ItemWitnessObservation;
        EXPECT_FALSE(std::is_default_constructible_v<Lease>);
        EXPECT_FALSE(std::is_aggregate_v<Lease>);
        EXPECT_FALSE(std::is_copy_constructible_v<Lease>);
        EXPECT_TRUE(std::is_move_constructible_v<Lease>);
        EXPECT_FALSE(std::is_default_constructible_v<Observation>);
        EXPECT_FALSE(std::is_aggregate_v<Observation>);
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Lease&>().address()), uint32_t>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Lease&>().request()),
                                   const tt_niu_item_witness_request&>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Observation&>().request()),
                                   const tt_niu_item_witness_request&>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Invocation&>().poll_item(
                                       std::declval<const Lease&>())), typename Invocation::ItemWitnessResult>));
    }
}

TEST(NiuManagedInvocationContractTest, ItemWitnessCanOnlyComeFromOriginalSdkAllocationAndObserver) {
    check_item_witness_owner_contract<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
concept exposes_item_witness_poll = requires(
    const Invocation& invocation, const typename Invocation::ItemWitnessLease& lease) {
    typename Invocation::ItemWitnessStatus;
    typename Invocation::ItemWitnessResult;
    invocation.poll_item(lease);
};

template <class Invocation>
void check_item_witness_poll_contract() {
    if constexpr (!exposes_item_witness_poll<Invocation>) {
        ADD_FAILURE() << "Managed invocation cannot distinguish pending item progress from terminal failure";
    } else {
        using Lease = typename Invocation::ItemWitnessLease;
        using Observation = typename Invocation::ItemWitnessObservation;
        using Status = typename Invocation::ItemWitnessStatus;
        using Result = typename Invocation::ItemWitnessResult;
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Invocation&>().poll_item(
                                       std::declval<const Lease&>())), Result>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Result&>().status), Status>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Result&>().observation), std::optional<Observation>>));
        EXPECT_NE(Status::Pending, Status::Complete);
        EXPECT_NE(Status::Pending, Status::Rejected);
        EXPECT_NE(Status::Pending, Status::Invalidated);
        EXPECT_NE(Status::Pending, Status::IoFailure);
    }
}

TEST(NiuManagedInvocationContractTest, ItemWitnessPollSeparatesPendingFromTerminalFailure) {
    check_item_witness_poll_contract<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
concept exposes_item_acknowledgement_owner = requires(
    const Invocation& invocation, const typename Invocation::ItemWitnessObservation& observation) {
    typename Invocation::ItemAcknowledgementStatus;
    invocation.acknowledge_item(observation);
};

template <class Invocation, class Candidate>
concept accepts_item_acknowledgement = requires(const Invocation& invocation, const Candidate& candidate) {
    invocation.acknowledge_item(candidate);
};

template <class Invocation>
void check_item_acknowledgement_owner_contract() {
    if constexpr (!exposes_item_acknowledgement_owner<Invocation>) {
        ADD_FAILURE() << "Managed invocation lacks an original-owner opaque item acknowledgement receiver";
    } else {
        using Observation = typename Invocation::ItemWitnessObservation;
        using Status = typename Invocation::ItemAcknowledgementStatus;
        EXPECT_FALSE(std::is_default_constructible_v<Observation>);
        EXPECT_FALSE(std::is_aggregate_v<Observation>);
        EXPECT_FALSE((std::is_constructible_v<Observation, tt_niu_item_witness_receipt>));
        EXPECT_FALSE((std::is_constructible_v<Observation, tt_niu_item_witness_acknowledgement>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Invocation&>().acknowledge_item(
                                       std::declval<const Observation&>())), Status>));
        EXPECT_FALSE((accepts_item_acknowledgement<Invocation, tt_niu_item_witness_receipt>));
        EXPECT_FALSE((accepts_item_acknowledgement<Invocation, tt_niu_item_witness_acknowledgement>));
        EXPECT_FALSE((accepts_item_acknowledgement<Invocation, tt_niu_item_witness_request>));
        EXPECT_FALSE((accepts_item_acknowledgement<Invocation, bool>));
        EXPECT_NE(Status::Acknowledged, Status::Pending);
        EXPECT_NE(Status::Acknowledged, Status::Invalidated);
        EXPECT_NE(Status::Acknowledged, Status::Rejected);
        EXPECT_NE(Status::Acknowledged, Status::IoFailure);
    }
}

TEST(NiuManagedInvocationContractTest, ItemAcknowledgementRequiresOriginalOpaqueObservation) {
    check_item_acknowledgement_owner_contract<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
void check_item_series_terminal_contract() {
    if constexpr (requires(const typename Invocation::ItemWitnessResult& result) {
                      typename Invocation::ItemSeriesObservation;
                      Invocation::ItemWitnessStatus::NotExecuted;
                      Invocation::ItemWitnessStatus::Finished;
                      result.series;
                  }) {
        using Series = typename Invocation::ItemSeriesObservation;
        using Round = typename Invocation::ItemWitnessObservation;
        using Status = typename Invocation::ItemWitnessStatus;
        using Result = typename Invocation::ItemWitnessResult;
        EXPECT_FALSE(std::is_default_constructible_v<Series>);
        EXPECT_FALSE(std::is_aggregate_v<Series>);
        EXPECT_FALSE((std::is_constructible_v<Series, uint64_t>));
        EXPECT_FALSE((std::is_constructible_v<Series, tt_niu_item_witness_receipt>));
        EXPECT_FALSE((std::is_constructible_v<Series, Round>));
        EXPECT_FALSE((std::is_convertible_v<Series, Round>));
        EXPECT_FALSE((accepts_item_acknowledgement<Invocation, Series>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Result&>().series), std::optional<Series>>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Series&>().final_count()), uint64_t>));
        EXPECT_TRUE(
            (std::is_same_v<decltype(std::declval<const Series&>().request()), const tt_niu_item_witness_request&>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Series&>().niu()), detail::NiuKey>));
        EXPECT_NE(Status::NotExecuted, Status::Pending);
        EXPECT_NE(Status::NotExecuted, Status::Complete);
        EXPECT_NE(Status::Finished, Status::Complete);
        EXPECT_NE(Status::Finished, Status::NotExecuted);
        const Result pending{Status::Pending, std::nullopt};
        EXPECT_FALSE(pending.observation);
        EXPECT_FALSE(pending.series) << "No item or source End may be fabricated by a status-only result";
    } else {
        ADD_FAILURE() << "Managed invocation loses source-series End or aliases it to item completion";
    }
}

TEST(NiuManagedInvocationContractTest, SeriesEndIsAnOriginalOwnedObservationDistinctFromItemCompletion) {
    check_item_series_terminal_contract<detail::ProgramExecution::ManagedInvocation>();
}

template <class Status>
concept exposes_retired_resource_status = requires { Status::Retired; };

template <class Invocation>
void check_resource_retirement_contract() {
    if constexpr (requires {
                      typename Invocation::ResourceRetirementReception;
                      typename Invocation::ResourceRetirementReception::Status;
                  }) {
        using Status = typename Invocation::ResourceRetirementReception::Status;
        EXPECT_FALSE(exposes_retired_resource_status<Status>);
        EXPECT_FALSE((std::is_convertible_v<Status, bool>));
        EXPECT_NE(Status::Pending, Status::Ready);
        EXPECT_NE(Status::Rejected, Status::Ready);
        EXPECT_NE(Status::Invalidated, Status::Ready);
    } else {
        ADD_FAILURE() << "Managed invocation has no read-only native resource retirement prerequisites";
    }
}

TEST(NiuManagedInvocationContractTest, ResourceRetirementReceptionCannotPublishNativeResourceTerminal) {
    check_resource_retirement_contract<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
concept exposes_native_resource_retirement = requires(const Invocation& invocation) {
    invocation.retire_native_resources();
};

template <class Invocation>
concept accepts_native_retirement_claim = requires(const Invocation& invocation) {
    invocation.retire_native_resources(true);
};

template <class Invocation, class Argument>
concept accepts_native_retirement_argument = requires(const Invocation& invocation, Argument argument) {
    invocation.retire_native_resources(argument);
};

template <class Invocation>
concept exposes_native_retirement_status = requires {
    typename Invocation::NativeResourceRetirementStatus;
};

template <class Invocation>
concept accepts_native_retirement_observation = requires(
    const Invocation& invocation,
    const typename Invocation::ItemWitnessObservation& item,
    const typename Invocation::ItemSeriesObservation& series) {
    invocation.retire_native_resources(item);
    invocation.retire_native_resources(series);
};

template <class Invocation>
void check_native_resource_retirement_contract() {
    if constexpr (!exposes_native_resource_retirement<Invocation> ||
                  !exposes_native_retirement_status<Invocation>) {
        ADD_FAILURE() << "Managed invocation has no normal native-resource retirement receiver";
    } else {
        using ExpectedStatus = typename Invocation::NativeResourceRetirementStatus;
        using Status = decltype(std::declval<Invocation&>().retire_native_resources());
        // Retirement is an original-owner action, not caller-supplied proof.
        // A bool cannot distinguish pending, invalidated, and completed work.
        EXPECT_TRUE((std::is_same_v<Status, ExpectedStatus>));
        EXPECT_TRUE(std::is_enum_v<Status>);
        EXPECT_FALSE((std::is_same_v<Status, bool>));
        EXPECT_FALSE((std::is_convertible_v<Status, bool>));
        EXPECT_NE(Status::Pending, Status::Retired);
        EXPECT_NE(Status::Rejected, Status::Retired);
        EXPECT_NE(Status::Invalidated, Status::Retired);
        EXPECT_FALSE(accepts_native_retirement_claim<Invocation>);
        EXPECT_FALSE(accepts_native_retirement_observation<Invocation>);
        EXPECT_FALSE((accepts_native_retirement_argument<Invocation, uint64_t>));
        EXPECT_FALSE((accepts_native_retirement_argument<Invocation, NiuAdmission::Token>));
        EXPECT_FALSE((accepts_native_retirement_argument<Invocation, NiuAdmission::Boots>));
    }
}

TEST(NiuManagedInvocationContractTest, NativeResourceRetirementIsTypedAndHasNoCallerProofInput) {
    check_native_resource_retirement_contract<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
concept exposes_native_backing_terminal = requires {
    typename Invocation::NativeBackingTerminal;
};

TEST(NiuManagedInvocationContractTest, NativeBackingTerminalCannotBeMintedFromItemOrEndEvidence) {
    // The sealed terminal is intentionally private to ProgramExecution. Its
    // public consumer is the no-argument typed retirement method above; item
    // completion and source End observations cannot be passed as a substitute.
    EXPECT_FALSE((exposes_native_backing_terminal<detail::ProgramExecution::ManagedInvocation>));
    EXPECT_FALSE((exposes_native_backing_terminal<detail::ProgramExecution>));
}

template <class Admission>
concept exposes_clean_terminal = requires { typename Admission::CleanTerminal; };

TEST(NiuManagedInvocationContractTest, NativeCleanTerminalCannotBeMintedFromPublicLedgerOrToken) {
    EXPECT_FALSE((exposes_clean_terminal<NiuAdmission>));
}

template <class Invocation>
concept exposes_resource_retirement_reception = requires(
    const detail::ProgramPreparation& preparation,
    std::span<const detail::NiuHandoffReceiver::ItemReception* const> series) {
    typename Invocation::ResourceRetirementReception;
    typename Invocation::ResourceRetirementReception::Status;
    typename Invocation::ResourceRetirementReception::Identity;
    typename Invocation::ResourceRetirementReception::Inventory;
    Invocation::ResourceRetirementReception::receive(
        Invocation::ResourceRetirementReception::Identity::Current,
        preparation,
        series,
        Invocation::ResourceRetirementReception::Inventory::RequireSeries);
};

// Bytes exercise the actual SDK item decoder and ACK transport/readback. This
// fixture cannot create a firmware boot, a managed invocation, a live Device
// epoch, or resource-release authority. Ready below is reception only.
struct ResourceRetirementMailbox {
    using Receiver = detail::NiuHandoffReceiver;
    using Reception = Receiver::ItemReception;
    Receiver::ItemWitnessRegion region{};
    const Receiver::ItemWitnessRequest original;

    explicit ResourceRetirementMailbox(uint64_t item = 11) :
        original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, item, 13, 1, 2, 1, 15, 2, 2, 0} {
        region.request = region.receipt.request = original;
    }
    detail::NiuHandoffIoStatus read(uint32_t offset, std::span<std::byte> bytes) const {
        if (offset > sizeof(region) || bytes.size() > sizeof(region) - offset) {
            return detail::NiuHandoffIoStatus::Failed;
        }
        std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
        return detail::NiuHandoffIoStatus::Complete;
    }
    detail::NiuItemWitnessReadResult observe() const {
        return Receiver::read_item([&](uint32_t offset, std::span<std::byte> bytes) { return read(offset, bytes); });
    }
    void closed() {
        region.receipt = {};
        region.receipt.request = original;
        region.receipt.publish_sequence = 8;
        region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
        region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = 1;
    }
    void end(uint64_t count) {
        region.receipt = {};
        region.receipt.request = original;
        region.receipt.publish_sequence = count ? 10 : 4;
        region.receipt.state = TT_NIU_ITEM_WITNESS_SERIES_END;
        region.receipt.series.final_count = count;
        region.receipt.series.closed_sequence = count ? 8 : 0;
        region.receipt.series.acknowledgement_sequence = count ? 2 : 0;
    }
    bool acknowledge(Reception& reception) {
        const auto ack = reception.acknowledgement();
        if (!ack) {
            return false;
        }
        const auto status = Receiver::write_item_acknowledgement(
            [&](uint32_t offset, std::span<const std::byte> bytes) {
                if (offset > sizeof(region) || bytes.size() > sizeof(region) - offset) {
                    return detail::NiuHandoffIoStatus::Failed;
                }
                std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
                return detail::NiuHandoffIoStatus::Complete;
            },
            [&](uint32_t offset, std::span<std::byte> bytes) { return read(offset, bytes); },
            *ack);
        return status == detail::NiuHandoffWriteStatus::Published && reception.acknowledge(*ack);
    }
};

template <class Invocation>
void check_resource_retirement_inventory() {
    if constexpr (!exposes_resource_retirement_reception<Invocation>) {
        ADD_FAILURE() << "Native resource retirement has no actual SDK reception seam";
    } else {
        using Receiver = typename Invocation::ResourceRetirementReception;
        using Status = typename Receiver::Status;
        using Identity = typename Receiver::Identity;
        using Inventory = typename Receiver::Inventory;
        using Reception = detail::NiuHandoffReceiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        ResourceRetirementMailbox mailbox;
        Reception reception(mailbox.original);
        mailbox.end(0);
        ASSERT_EQ(
            reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Reception::Status::NotExecuted);
        detail::ProgramPreparation preparation;
        const std::array<const Reception*, 1> null_series{nullptr};
        const std::array<const Reception*, 2> duplicated{&reception, &reception};
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, {}, Inventory::RequireSeries), Status::Rejected);
        // Selector cleaning already supports invocations with no registered
        // source series. This profile waives only the nonempty obligation.
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, {}, Inventory::AllowEmpty), Status::Ready);
        ASSERT_TRUE(preparation.acquire());
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, {}, Inventory::AllowEmpty), Status::Pending);
        EXPECT_TRUE(preparation.owned());
        preparation.release();
        EXPECT_EQ(
            Receiver::receive(Identity::Invalidated, preparation, {}, Inventory::AllowEmpty), Status::Invalidated);
        EXPECT_EQ(
            Receiver::receive(Identity::Current, preparation, null_series, Inventory::RequireSeries), Status::Rejected);
        EXPECT_EQ(
            Receiver::receive(Identity::Current, preparation, null_series, Inventory::AllowEmpty), Status::Rejected);
        EXPECT_EQ(
            Receiver::receive(Identity::Current, preparation, duplicated, Inventory::RequireSeries), Status::Rejected);
        EXPECT_EQ(
            Receiver::receive(Identity::Current, preparation, duplicated, Inventory::AllowEmpty), Status::Rejected);
        EXPECT_FALSE(preparation.occupied());
        EXPECT_FALSE(preparation.cancelled());
        ASSERT_TRUE(reception.terminal());
        EXPECT_EQ(reception.terminal()->final_count(), 0u);
        EXPECT_FALSE(reception.accepted());
        EXPECT_EQ(reception.closed_prefix(), 0u);
        EXPECT_EQ(reception.acknowledged_prefix(), 0u);
    }
}

TEST(NiuManagedInvocationContractTest, ResourceRetirementRejectsEmptyNullAndDuplicateSeriesWithoutMutation) {
    check_resource_retirement_inventory<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
void check_resource_retirement_requires_end() {
    if constexpr (!exposes_resource_retirement_reception<Invocation>) {
        ADD_FAILURE() << "Native resource retirement has no actual SDK reception seam";
    } else {
        using Receiver = typename Invocation::ResourceRetirementReception;
        using Status = typename Receiver::Status;
        using Identity = typename Receiver::Identity;
        using Inventory = typename Receiver::Inventory;
        using Reception = detail::NiuHandoffReceiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        ResourceRetirementMailbox mailbox;
        Reception reception(mailbox.original);
        const std::array<const Reception*, 1> series{&reception};
        detail::ProgramPreparation preparation;
        mailbox.closed();
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Reception::Status::Complete);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Pending);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Pending);
        ASSERT_TRUE(reception.accepted());
        EXPECT_EQ(reception.closed_prefix(), 1u);
        EXPECT_EQ(reception.acknowledged_prefix(), 0u);
        ASSERT_TRUE(mailbox.acknowledge(reception));
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Pending);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Pending);
        EXPECT_FALSE(reception.accepted());
        EXPECT_FALSE(reception.terminal());
        EXPECT_EQ(reception.closed_prefix(), 1u);
        EXPECT_EQ(reception.acknowledged_prefix(), 1u);
        EXPECT_FALSE(preparation.occupied());
        EXPECT_FALSE(preparation.cancelled());
        mailbox.end(1);
        ASSERT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Reception::Status::Finished);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Ready);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Ready);
    }
}

TEST(NiuManagedInvocationContractTest, ResourceRetirementCannotReplaceSourceEndWithClosedAndAcknowledgement) {
    check_resource_retirement_requires_end<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
void check_resource_retirement_requires_all_series() {
    if constexpr (!exposes_resource_retirement_reception<Invocation>) {
        ADD_FAILURE() << "Native resource retirement has no actual SDK reception seam";
    } else {
        using Receiver = typename Invocation::ResourceRetirementReception;
        using Status = typename Receiver::Status;
        using Identity = typename Receiver::Identity;
        using Inventory = typename Receiver::Inventory;
        using Reception = detail::NiuHandoffReceiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        ResourceRetirementMailbox first(11), second(12);
        Reception first_reception(first.original), second_reception(second.original);
        const std::array<const Reception*, 2> series{&first_reception, &second_reception};
        detail::ProgramPreparation preparation;
        first.end(0);
        ASSERT_EQ(
            first_reception.receive(Issuer::Stopped, Issuer::Stopped, first.observe()), Reception::Status::NotExecuted);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Pending);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Pending);
        second.end(0);
        ASSERT_EQ(
            second_reception.receive(Issuer::Issued, Issuer::Stopped, second.observe()), Reception::Status::Pending);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Pending);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Pending);
        EXPECT_FALSE(second_reception.terminal());
        ASSERT_TRUE(first_reception.terminal());
        EXPECT_FALSE(preparation.occupied());
        ASSERT_EQ(
            second_reception.receive(Issuer::Stopped, Issuer::Stopped, second.observe()),
            Reception::Status::NotExecuted);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Ready);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Ready);
        EXPECT_EQ(first_reception.terminal()->final_count(), 0u);
        EXPECT_EQ(second_reception.terminal()->final_count(), 0u);
    }
}

TEST(NiuManagedInvocationContractTest, ResourceRetirementRequiresFreshStoppedEndForEveryRegisteredSeries) {
    check_resource_retirement_requires_all_series<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
void check_resource_retirement_preparation() {
    if constexpr (!exposes_resource_retirement_reception<Invocation>) {
        ADD_FAILURE() << "Native resource retirement has no actual SDK reception seam";
    } else {
        using Receiver = typename Invocation::ResourceRetirementReception;
        using Status = typename Receiver::Status;
        using Identity = typename Receiver::Identity;
        using Inventory = typename Receiver::Inventory;
        using Reception = detail::NiuHandoffReceiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        ResourceRetirementMailbox mailbox;
        Reception reception(mailbox.original);
        mailbox.end(0);
        ASSERT_EQ(
            reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Reception::Status::NotExecuted);
        const std::array<const Reception*, 1> series{&reception};
        detail::ProgramPreparation preparation;
        ASSERT_TRUE(preparation.acquire());
        ASSERT_TRUE(preparation.acquire());
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Pending);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Pending);
        EXPECT_TRUE(preparation.owned());
        EXPECT_FALSE(preparation.cancelled());
        preparation.release();
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Pending);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Pending);
        EXPECT_TRUE(preparation.owned());
        preparation.release();
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Ready);
        EXPECT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Ready);
        EXPECT_FALSE(preparation.occupied());
        ASSERT_TRUE(reception.terminal());
        EXPECT_EQ(reception.terminal()->final_count(), 0u);
        preparation.cancel();
        EXPECT_EQ(
            Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Invalidated);
        EXPECT_EQ(
            Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Invalidated);
        EXPECT_TRUE(preparation.ready_to_cancel());
    }
}

TEST(NiuManagedInvocationContractTest, ResourceRetirementDefersUntilLastPreparationBorrowWithoutCancellingIt) {
    check_resource_retirement_preparation<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
void check_resource_retirement_identity_revalidation() {
    if constexpr (!exposes_resource_retirement_reception<Invocation>) {
        ADD_FAILURE() << "Native resource retirement has no actual SDK reception seam";
    } else {
        using Receiver = typename Invocation::ResourceRetirementReception;
        using Status = typename Receiver::Status;
        using Identity = typename Receiver::Identity;
        using Inventory = typename Receiver::Inventory;
        using Reception = detail::NiuHandoffReceiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        ResourceRetirementMailbox mailbox;
        Reception reception(mailbox.original);
        mailbox.end(0);
        ASSERT_EQ(
            reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Reception::Status::NotExecuted);
        const std::array<const Reception*, 1> series{&reception};
        detail::ProgramPreparation preparation;
        ASSERT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::RequireSeries), Status::Ready);
        ASSERT_EQ(Receiver::receive(Identity::Current, preparation, series, Inventory::AllowEmpty), Status::Ready);
        // The actual invocation must freshly derive this input from original
        // owner/boot/epoch validation. Cached End cannot overrule its failure;
        // these CPU observations do not prove a real boot or native release.
        EXPECT_EQ(
            Receiver::receive(Identity::Invalidated, preparation, series, Inventory::RequireSeries),
            Status::Invalidated);
        EXPECT_EQ(
            Receiver::receive(Identity::Invalidated, preparation, series, Inventory::AllowEmpty), Status::Invalidated);
        EXPECT_FALSE(preparation.occupied());
        EXPECT_FALSE(preparation.cancelled());
        ASSERT_TRUE(reception.terminal());
        EXPECT_EQ(reception.terminal()->final_count(), 0u);
        EXPECT_EQ(reception.closed_prefix(), 0u);
        EXPECT_EQ(reception.acknowledged_prefix(), 0u);
    }
}

TEST(NiuManagedInvocationContractTest, ResourceRetirementReceptionRejectsInvalidatedIdentityDespiteCachedEnd) {
    check_resource_retirement_identity_revalidation<detail::ProgramExecution::ManagedInvocation>();
}

template <class Invocation>
concept exposes_invocation_selector_sublease = requires(const Invocation& invocation, const detail::NiuKey& key) {
    invocation.acquire_selector(key);
};

template <class Allocation>
void check_selector_sublease_contract() {
    EXPECT_TRUE(exposes_selector_sublease<Allocation>);
    if constexpr (exposes_selector_sublease<Allocation>) {
        using Lease = typename decltype(std::declval<const Allocation&>().acquire_selector(
            std::declval<const detail::NiuKey&>()))::value_type;
        EXPECT_FALSE(std::is_copy_constructible_v<Lease>);
        EXPECT_TRUE(std::is_move_constructible_v<Lease>);
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Lease&>().niu()), detail::NiuKey>));
        EXPECT_TRUE((std::is_same_v<decltype(std::declval<const Lease&>().selector()), uint8_t>));
    }
}

TEST(NiuManagedInvocationContractTest, ManagedInvocationIsTheOnlyPublicSelectorLeaseOwner) {
    // A retained allocation is an immutable inspection view. Only the live
    // invocation may mint a selector, after checking Reserved, cancellation,
    // the admission token, and the original firmware boot. An Allocation copy
    // must not bypass those checks after issue or retirement.
    EXPECT_FALSE(exposes_selector_sublease<NiuAdmission::Allocation>);
    check_selector_sublease_contract<detail::ProgramExecution::ManagedInvocation>();
    EXPECT_TRUE(exposes_invocation_selector_sublease<detail::ProgramExecution::ManagedInvocation>);
}

TEST(NiuManagedInvocationContractTest, PreparationHasOneHostOwnerAndSupportsNestedBorrowing) {
    detail::ProgramPreparation preparation;
    ASSERT_TRUE(preparation.acquire());
    ASSERT_TRUE(preparation.acquire());
    EXPECT_TRUE(preparation.owned());
    auto competing_acquire = [&] {
        bool acquired = false;
        std::thread contender([&] {
            acquired = preparation.acquire();
            if (acquired) preparation.release();
        });
        contender.join();
        return acquired;
    };
    EXPECT_FALSE(competing_acquire());
    EXPECT_TRUE(preparation.owned());
    preparation.release();
    EXPECT_FALSE(competing_acquire());
    preparation.release();
    EXPECT_FALSE(preparation.occupied());
    EXPECT_TRUE(competing_acquire());
    EXPECT_FALSE(preparation.occupied());
}

TEST(NiuManagedInvocationContractTest, PreparationCancellationWaitsForLastBorrow) {
    detail::ProgramPreparation preparation;
    ASSERT_TRUE(preparation.acquire());
    ASSERT_TRUE(preparation.acquire());
    preparation.cancel();
    EXPECT_TRUE(preparation.cancelled());
    EXPECT_FALSE(preparation.ready_to_cancel());
    EXPECT_FALSE(preparation.acquire());
    preparation.release();
    EXPECT_FALSE(preparation.ready_to_cancel());
    preparation.release();
    EXPECT_TRUE(preparation.ready_to_cancel());
    EXPECT_FALSE(preparation.acquire());
}

TEST(NiuManagedInvocationContractTest, HandoffCorrelationIdentitiesNeverWrapOrRepeat) {
    using experimental::native_detail::next_handoff_identity;
    std::atomic<uint64_t> next{1};
    EXPECT_EQ(next_handoff_identity(next), 1u);
    EXPECT_EQ(next_handoff_identity(next), 2u);
    next.store(std::numeric_limits<uint64_t>::max() - 1);
    EXPECT_EQ(next_handoff_identity(next), std::numeric_limits<uint64_t>::max() - 1);
    EXPECT_THROW((void)next_handoff_identity(next), std::overflow_error);
    EXPECT_THROW((void)next_handoff_identity(next), std::overflow_error);
    EXPECT_EQ(next.load(), std::numeric_limits<uint64_t>::max());
    next.store(0);
    EXPECT_THROW((void)next_handoff_identity(next), std::overflow_error);
    EXPECT_FALSE(std::is_default_constructible_v<experimental::native_detail::LoadedFirmware>);
    EXPECT_FALSE(std::is_copy_constructible_v<experimental::native_detail::LoadedFirmware>);
}

TEST(NiuManagedInvocationContractTest, ConcurrentHandoffCorrelationIdentitiesStayDistinct) {
    std::atomic<uint64_t> next{1};
    std::array<uint64_t, 32> identities{};
    std::vector<std::thread> threads;
    for (auto& identity : identities)
        threads.emplace_back([&next, &identity] {
            identity = experimental::native_detail::next_handoff_identity(next);
        });
    for (auto& thread : threads) thread.join();
    std::sort(identities.begin(), identities.end());
    for (size_t index = 0; index != identities.size(); ++index) EXPECT_EQ(identities[index], index + 1);
}

template <class Execution, class... Inputs>
void expect_unbooted_managed_rejection(Inputs&&... inputs) {
    if constexpr (requires { Execution::acquire_managed(std::forward<Inputs>(inputs)...); }) {
        try {
            auto invocation = Execution::acquire_managed(std::forward<Inputs>(inputs)...);
            FAIL() << "Managed invocation must require a successful actual SDK firmware boot";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("boot"), std::string::npos) << error.what();
        }
    } else {
        ADD_FAILURE() << "SDK has no token-bound managed invocation entrypoint";
    }
}

void expect_handoff_rejection(detail::ProgramExecution& execution) {
    const auto before = unexpected_handoff_reads;
    EXPECT_THROW(execution.observe_handoff(), std::runtime_error);
    EXPECT_EQ(unexpected_handoff_reads, before);
}

template <class Execution>
void expect_ordered_managed_rejection(
    distributed::MeshDevice& mesh,
    std::span<distributed::MeshWorkload* const> workloads,
    std::string_view expected_error) {
    if constexpr (requires { Execution::acquire_managed(mesh, workloads); }) {
        try {
            auto invocation = Execution::acquire_managed(mesh, workloads);
            FAIL() << "Managed ordered invocation must reject before publishing any member";
        } catch (const std::runtime_error& error) {
            const std::string diagnostic = error.what();
            EXPECT_NE(diagnostic.find(expected_error), std::string::npos) << diagnostic;
            if (expected_error != "boot") {
                EXPECT_EQ(diagnostic.find("boot"), std::string::npos)
                    << "Every workload must be structurally checked before looking up any firmware boot: "
                    << diagnostic;
            }
        }
    } else {
        ADD_FAILURE() << "SDK has no managed invocation entrypoint for an ordered workload collection";
    }
}

// Exercise the real SDK managers and SOC mapping. Minimal context/device setup
// stops before firmware compilation or launch and cannot publish LoadedFirmware.
class NiuInfrastructureTest : public ::testing::Test {
protected:
    void SetUp() override {
        MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1).value());
        descriptor.set_device_profiler_mode(DeviceProfilerMode::Disabled);
        environment = std::make_unique<MetalEnv>(std::move(descriptor));
        context_id = MetalContext::create_instance(*environment);
        context = &MetalContext::instance(*context_id);
        context->rtoptions().set_fast_dispatch(true);
        context->initialize(dispatch_config, 1, {}, DEFAULT_WORKER_L1_SIZE, true);
        const auto& cluster = MetalEnvAccessor(*environment).impl().get_cluster();
        ASSERT_EQ(cluster.get_target_device_type(), tt::TargetDevice::Mock);
        ASSERT_EQ(cluster.all_chip_ids().size(), 1u);
        chip = *cluster.all_chip_ids().begin();
        device = std::make_unique<Device>(
            environment.get(), context, chip, 1, 0, 0, ttsl::Span<const std::uint32_t>{}, true);
        ASSERT_FALSE(context->native_firmware(chip));
        free_cores = dispatch().get_available_dispatch_cores(chip);
        ASSERT_GE(free_cores.size(), 2u);
    }

    void TearDown() override {
        device.reset();
        if (context_id) {
            MetalContext::destroy_instance(false, *context_id);
        }
        environment.reset();
    }

    dispatch_core_manager& dispatch() { return context->get_dispatch_core_manager(); }
    internal::ServiceCoreManager& services() { return context->get_service_core_manager(); }

    void quarantine(CoreCoord logical_core, detail::NiuNoc noc = detail::NiuNoc::Noc0) {
        const auto& cluster = MetalEnvAccessor(*environment).impl().get_cluster();
        const auto physical = cluster.get_soc_desc(chip).get_physical_tensix_core_from_logical(logical_core);
        const std::array keys{detail::NiuKey{chip, physical, noc}};
        std::string error;
        auto token = context->niu_admission().reserve_ordinary(keys, error);
        ASSERT_TRUE(token) << error;
        ASSERT_TRUE(context->niu_admission().quarantine(*token, error)) << error;
    }

    template <typename Action>
    void expect_quarantined_program_rejection(Action&& action) {
        const CoreCoord worker{0, 0};
        ASSERT_NO_FATAL_FAILURE(quarantine(worker, detail::NiuNoc::Noc1));
        Program program;
        CreateKernelFromString(
            program,
            "void kernel_main() {}",
            worker,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
        ASSERT_FALSE(program.impl().is_compiled());
        ASSERT_FALSE(program.impl().is_finalized());
        try {
            action(program);
            FAIL() << "Quarantined physical NIU must reject before Program preparation or device configuration";
        } catch (const std::runtime_error& error) {
            const std::string diagnostic = error.what();
            EXPECT_NE(diagnostic.find("NIU"), std::string::npos) << diagnostic;
            EXPECT_NE(diagnostic.find("quarantin"), std::string::npos) << diagnostic;
        }
        EXPECT_FALSE(program.impl().is_compiled());
        EXPECT_FALSE(program.impl().is_finalized());
        EXPECT_FALSE(context->native_firmware(chip));
    }

    Program worker_program(CoreCoord core = {0, 0}) {
        Program program(std::make_shared<detail::ProgramImpl>(*context_id));
        EXPECT_EQ(program.impl().get_context_id(), *context_id);
        CreateKernelFromString(program, "void kernel_main() {}", core,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
        return program;
    }

    void open_device() {
        ASSERT_TRUE(device->initialize(1, 0, 0, DEFAULT_WORKER_L1_SIZE, {}, false));
        ASSERT_TRUE(device->is_initialized());
        ASSERT_FALSE(context->native_firmware(chip));
    }

    const DispatchCoreConfig dispatch_config{DispatchCoreType::WORKER, DispatchCoreAxis::COL};
    std::unique_ptr<MetalEnv> environment;
    std::optional<ContextId> context_id;
    MetalContext* context = nullptr;
    std::unique_ptr<Device> device;
    ChipId chip = -1;
    std::vector<CoreCoord> free_cores;
};

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsLaunchBeforeCompilation) {
    expect_quarantined_program_rejection(
        [&](Program& program) { detail::LaunchProgram(device.get(), program, false, true); });
}

TEST_F(NiuInfrastructureTest, ManagedDirectInvocationRequiresActualBootBeforePreparation) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    const std::array<IDevice*, 1> devices{device.get()};
    expect_unbooted_managed_rejection<detail::ProgramExecution>(program, std::span<IDevice* const>(devices));
    // A rejected acquisition must not publish a stale Program binding or keep
    // any ordinary/full-NIU ownership behind.
    EXPECT_NO_THROW(detail::ProgramExecution::reserve(program, devices));
    expect_unbooted_managed_rejection<detail::ProgramExecution>(program, std::span<IDevice* const>(devices));
    EXPECT_FALSE(program.impl().is_compiled());
    EXPECT_FALSE(program.impl().is_finalized());
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, ManagedReloadWithoutOwnedStageJournalRejectsBeforeBootOrIo) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    program.impl().set_reload_table(0x10000, CoreRangeSet(CoreRange({0, 0}, {0, 0})));
    const std::array<IDevice*, 1> devices{device.get()};
    try {
        auto invocation = detail::ProgramExecution::acquire_managed(program, devices);
        FAIL() << "Unjournaled reload must reject before admission";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("reload"), std::string::npos) << error.what();
    }
    EXPECT_FALSE(program.impl().is_compiled());
    EXPECT_FALSE(program.impl().is_finalized());
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, OrdinaryExecutionRejectsHandoffObservationBeforeMailboxIo) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    auto execution = detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()});
    ASSERT_TRUE(execution);
    expect_handoff_rejection(*execution);
    EXPECT_FALSE(context->native_firmware(chip));
    EXPECT_FALSE(program.impl().is_compiled());
}

TEST_F(NiuInfrastructureTest, ClosedDeviceExecutionRejectsHandoffObservationBeforeMailboxIo) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    auto execution = detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()});
    ASSERT_TRUE(execution);
    ASSERT_TRUE(device->close());
    expect_handoff_rejection(*execution);
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, ExpiredContextExecutionRejectsHandoffObservationBeforeMailboxIo) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    std::shared_ptr<detail::ProgramExecution> execution;
    {
        auto program = worker_program();
        execution = detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()});
    }
    device.reset();
    MetalContext::destroy_instance(false, *context_id);
    context_id.reset();
    environment.reset();
    expect_handoff_rejection(*execution);
    EXPECT_NO_THROW(execution.reset());
}

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsConfigureOnlyBeforeCompilation) {
    expect_quarantined_program_rejection(
        [&](Program& program) { experimental::ConfigureProgramWithoutLaunch(device.get(), program); });
}

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsDirectDispatchBeforePreparedProgramChecks) {
    expect_quarantined_program_rejection(
        [&](Program& program) { experimental::DispatchCompiledProgramToDevice(device.get(), program); });
}

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsDeviceConfigurationBeforeAllocation) {
    expect_quarantined_program_rejection(
        [&](Program& program) { detail::ConfigureDeviceWithProgram(device.get(), program, true); });
}

TEST_F(NiuInfrastructureTest, QuarantinedProgramRejectsRuntimeArgumentsBeforeDeviceWrites) {
    expect_quarantined_program_rejection(
        [&](Program& program) { detail::WriteRuntimeArgsToDevice(device.get(), program, true); });
}

TEST_F(NiuInfrastructureTest, ActualDeviceCloseRetiresOrdinaryTokensWithoutNativeBoot) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    const auto physical = context->get_cluster().get_soc_desc(chip).get_physical_tensix_core_from_logical({0, 0});
    const std::array keys{detail::NiuKey{chip, physical, detail::NiuNoc::Noc0}};
    std::string error;
    auto token = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(token) << error;
    ASSERT_TRUE(device->close());
    EXPECT_FALSE(context->niu_admission().mark_issued(*token, error));
    EXPECT_FALSE(context->niu_admission().cancel_unissued(*token, error));
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, ClosedActualDeviceRejectsNewExecutionReservation) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    ASSERT_TRUE(device->close());
    EXPECT_THROW(detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()}), std::runtime_error);
    EXPECT_THROW(detail::WriteRuntimeArgsToDevice(device.get(), program, true), std::runtime_error);
    EXPECT_FALSE(program.impl().is_compiled());
}

TEST_F(NiuInfrastructureTest, ExecutionReservationCanOutliveItsContextWithoutFirmware) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    std::shared_ptr<detail::ProgramExecution> execution;
    {
        auto program = worker_program();
        execution = detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()});
    }
    device.reset();
    MetalContext::destroy_instance(false, *context_id);
    context_id.reset();
    environment.reset();
    EXPECT_NO_THROW(execution.reset());
}

TEST_F(NiuInfrastructureTest, DirectPreparationFailureDoesNotPoisonEitherPhysicalNiu) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    auto program = worker_program();
    EXPECT_THROW(experimental::DispatchCompiledProgramToDevice(device.get(), program), std::runtime_error);
    const auto physical = context->get_cluster().get_soc_desc(chip).get_physical_tensix_core_from_logical({0, 0});
    const std::array keys{detail::NiuKey{chip, physical, detail::NiuNoc::Noc0},
                          detail::NiuKey{chip, physical, detail::NiuNoc::Noc1}};
    std::string error;
    auto next = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(next) << error;
    EXPECT_TRUE(context->niu_admission().cancel_unissued(*next, error)) << error;
    EXPECT_FALSE(program.impl().is_compiled());
}

TEST_F(NiuInfrastructureTest, EthernetProgramDoesNotClaimWorkerNius) {
    ASSERT_NO_FATAL_FAILURE(open_device());
    ASSERT_NO_FATAL_FAILURE(quarantine({0, 0}, detail::NiuNoc::Noc0));
    ASSERT_NO_FATAL_FAILURE(quarantine({1, 0}, detail::NiuNoc::Noc1));
    const auto ethernet = device->get_inactive_ethernet_cores();
    ASSERT_FALSE(ethernet.empty());
    Program program;
    CreateKernelFromString(program, "void kernel_main() {}", *ethernet.begin(), EthernetConfig{.eth_mode = Eth::IDLE});
    EXPECT_NO_THROW(detail::ProgramExecution::reserve(program, std::array<IDevice*, 1>{device.get()}));
    EXPECT_FALSE(program.impl().is_compiled());
}

class NiuExecutionMeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value());
        descriptor.set_device_profiler_mode(DeviceProfilerMode::Disabled);
        environment = std::make_unique<MetalEnv>(std::move(descriptor));
        context_id = MetalContext::create_instance(*environment);
        context = &MetalContext::instance(*context_id);
        const DispatchCoreConfig dispatch_config{DispatchCoreType::WORKER, DispatchCoreAxis::COL};
        // Use the existing minimal context seam before creating actual Mesh,
        // Device and command-queue objects. Matching initialization parameters
        // preserve that context and never enter the firmware JIT/build phase.
        context->initialize(dispatch_config, 1, {}, DEFAULT_WORKER_L1_SIZE, true);
        mesh = distributed::MeshDeviceImpl::create(
            *context_id,
            distributed::MeshDeviceConfig(distributed::MeshShape{1, 2}),
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            1,
            dispatch_config,
            {},
            DEFAULT_WORKER_L1_SIZE);
        ASSERT_EQ(context->get_cluster().get_target_device_type(), tt::TargetDevice::Mock);
        ASSERT_EQ(mesh->get_devices().size(), 2u);
        for (auto* device : mesh->get_devices()) {
            ASSERT_FALSE(context->native_firmware(device->id()));
        }
    }
    void TearDown() override {
        if (mesh) {
            const auto devices = mesh->get_devices();
            // Actual-chip withdrawal tests deliberately close one member in
            // the body. Complete physical close before releasing the uniform
            // mesh facade; its state queries require matching member states.
            if (std::any_of(devices.begin(), devices.end(), [](auto* device) { return !device->is_initialized(); })) {
                for (auto* device : devices) {
                    if (device->is_initialized()) {
                        EXPECT_TRUE(device->close());
                    }
                }
            }
            mesh->close();
            mesh.reset();
        }
        if (context_id) {
            MetalContext::destroy_instance(false, *context_id);
        }
        environment.reset();
    }
    void quarantine(IDevice* device, CoreCoord logical) {
        const auto core =
            context->get_cluster().get_soc_desc(device->id()).get_physical_tensix_core_from_logical(logical);
        const std::array keys{detail::NiuKey{device->id(), core, detail::NiuNoc::Noc1}};
        std::string error;
        auto token = context->niu_admission().reserve_ordinary(keys, error);
        ASSERT_TRUE(token) << error;
        ASSERT_TRUE(context->niu_admission().quarantine(*token, error)) << error;
    }
    distributed::MeshWorkload workload() {
        Program program(std::make_shared<detail::ProgramImpl>(*context_id));
        EXPECT_EQ(program.impl().get_context_id(), *context_id);
        CreateKernelFromString(
            program,
            "void kernel_main() {}",
            CoreCoord{0, 0},
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
        distributed::MeshWorkload result;
        result.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 0}), std::move(program));
        return result;
    }
    void expect_rejected_before_compile(distributed::MeshWorkload& work) {
        try {
            distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), work, false);
            FAIL() << "Quarantined FD GO recipient must reject before compilation";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("NIU"), std::string::npos) << error.what();
            EXPECT_NE(std::string(error.what()).find("quarantin"), std::string::npos) << error.what();
        }
        for (const auto& [range, program] : work.get_programs()) {
            EXPECT_FALSE(program.impl().is_compiled());
        }
    }
    void expect_ordered_rejection_before_preparation(
        std::span<distributed::MeshWorkload* const> workloads,
        std::string_view expected_error,
        std::span<distributed::MeshWorkload* const> intact_workloads) {
        for (unsigned attempt = 0; attempt != 2; ++attempt) {
            SCOPED_TRACE(attempt);
            expect_ordered_managed_rejection<detail::ProgramExecution>(*mesh, workloads, expected_error);
            for (auto* work : intact_workloads) {
                EXPECT_EQ(detail::ProgramExecution::validate_workload(*mesh, *work), nullptr);
                EXPECT_NO_THROW(
                    detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{0}}, *work));
                for (const auto& [range, program] : work->get_programs()) {
                    EXPECT_FALSE(program.impl().is_compiled());
                    EXPECT_FALSE(program.impl().is_finalized());
                }
            }
            EXPECT_NO_THROW(detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{0}}));
            for (auto* device : mesh->get_devices()) EXPECT_FALSE(context->native_firmware(device->id()));
        }
    }
    std::shared_ptr<distributed::MeshDevice> mesh;
    std::unique_ptr<MetalEnv> environment;
    std::optional<ContextId> context_id;
    MetalContext* context = nullptr;
};

// These are real SDK allocation/lifetime checks on the existing CPU-only
// MeshDevice fixture. No firmware boot, handoff transport or GO is fabricated.
class ScopedHybridAllocatorMode {
public:
    ScopedHybridAllocatorMode() {
        if (const auto* value = std::getenv("TT_METAL_ALLOCATOR_MODE_HYBRID")) {
            previous = value;
        }
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", 1);
    }
    ~ScopedHybridAllocatorMode() {
        if (previous) {
            setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", previous->c_str(), 1);
        } else {
            unsetenv("TT_METAL_ALLOCATOR_MODE_HYBRID");
        }
    }

private:
    std::optional<std::string> previous;
};

BufferShardingArgs l1_pin_sharding() {
    BufferShardingArgs args(
        ShardSpecBuffer(
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0))),
            {1, 1},
            ShardOrientation::ROW_MAJOR,
            {1, 1},
            {2, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(args, true);
    return args;
}

class L1AllocationPinTest : public NiuExecutionMeshTest {
protected:
    void SetUp() override {
        ScopedHybridAllocatorMode option;
        NiuExecutionMeshTest::SetUp();
    }

    BufferShardingArgs sharding() { return l1_pin_sharding(); }

    std::shared_ptr<Buffer> allocate(IDevice* device) {
        return BufferImpl::create(device, 2048, 1024, BufferType::L1, sharding());
    }
};

// Closing a physical Device is tested without a parent MeshDevice owner, whose
// separate close would intentionally try to close that physical Device again.
class L1AllocationPinWithdrawalTest : public NiuInfrastructureTest {
protected:
    void SetUp() override {
        ScopedHybridAllocatorMode option;
        NiuInfrastructureTest::SetUp();
        ASSERT_NO_FATAL_FAILURE(open_device());
    }
    std::shared_ptr<Buffer> allocate(IDevice* device) {
        return BufferImpl::create(device, 2048, 1024, BufferType::L1, l1_pin_sharding());
    }
};

TEST_F(L1AllocationPinTest, PinnedAllocationRejectsExplicitAndGlobalReleaseWithoutPartialEffects) {
    auto* device = mesh->get_devices().front();
    auto buffer = allocate(device);
    auto unrelated = allocate(device);
    auto& allocator = *device->allocator_impl();
    auto pin = allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{64, 128});
    const auto before = allocator.get_allocated_buffers();
    EXPECT_THROW(buffer->impl().deallocate(*buffer), std::runtime_error);
    EXPECT_THROW(allocator.deallocate_buffer(buffer.get()), std::runtime_error);
    EXPECT_THROW(allocator.deallocate_buffers(), std::runtime_error);
    EXPECT_THROW(allocator.clear(), std::runtime_error);
    EXPECT_EQ(allocator.get_allocated_buffers(), before);
    EXPECT_TRUE(buffer->is_allocated());
    EXPECT_TRUE(unrelated->is_allocated());
    EXPECT_NO_THROW(unrelated->impl().deallocate(*unrelated));
    EXPECT_FALSE(unrelated->is_allocated());
    EXPECT_TRUE(pin.live());
}

TEST_F(L1AllocationPinTest, ExactAllocationCoreAndSubrangeSurviveMovesUntilLastPin) {
    auto* device = mesh->get_devices().front();
    auto buffer = allocate(device);
    const auto base = experimental::per_core_allocation::get_per_core_address(*buffer, {1, 0});
    auto& allocator = *device->allocator_impl();
    using Pin = decltype(allocator.pin_l1(buffer, CoreCoord{1, 0}, BufferRegion{64, 128}));
    EXPECT_FALSE(std::is_default_constructible_v<Pin>);
    EXPECT_FALSE(std::is_copy_constructible_v<Pin>);
    EXPECT_TRUE(std::is_nothrow_move_constructible_v<Pin>);
    EXPECT_TRUE(std::is_nothrow_destructible_v<Pin>);
    std::optional<Pin> first(allocator.pin_l1(buffer, CoreCoord{1, 0}, BufferRegion{64, 128}));
    std::optional<Pin> second(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 32}));
    EXPECT_EQ(&first->buffer(), buffer.get());
    EXPECT_EQ(first->allocation_id(), buffer->unique_id());
    EXPECT_EQ(first->device(), device);
    EXPECT_EQ(first->core(), CoreCoord(1, 0));
    EXPECT_EQ(first->begin(), base + 64);
    EXPECT_EQ(first->end(), base + 192);
    std::optional<Pin> moved(std::move(*first));
    first.reset();
    second.reset();
    EXPECT_THROW(buffer->impl().deallocate(*buffer), std::runtime_error);
    EXPECT_TRUE(moved->live());
    EXPECT_NO_THROW(moved.reset());
    EXPECT_NO_THROW(buffer->impl().deallocate(*buffer));
    EXPECT_FALSE(buffer->is_allocated());
}

TEST_F(L1AllocationPinTest, LastPinRetainsAndDestroysOriginalBufferOutsideAllocatorLock) {
    auto* device = mesh->get_devices().front();
    auto buffer = allocate(device);
    const auto before = device->allocator_impl()->get_num_allocated_buffers();
    std::weak_ptr<Buffer> retained = buffer;
    auto& allocator = *device->allocator_impl();
    using Pin = decltype(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 32}));
    std::optional<Pin> pin(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 32}));
    buffer.reset();
    EXPECT_FALSE(retained.expired());
    EXPECT_TRUE(retained.lock()->is_allocated());
    EXPECT_EQ(allocator.get_num_allocated_buffers(), before);
    EXPECT_NO_THROW(pin.reset());
    EXPECT_TRUE(retained.expired());
    EXPECT_EQ(allocator.get_num_allocated_buffers(), before - 1);
}

TEST_F(L1AllocationPinTest, PinRetainsTheSdkOwnerWhenTheCallerSuppliesAnotherControlBlock) {
    auto* device = mesh->get_devices().front();
    auto buffer = allocate(device);
    const auto original_owners = buffer.use_count();
    // Keep the same actual registered object, but do not grant this caller's
    // unrelated control block authority over the SDK allocation's lifetime.
    std::shared_ptr<Buffer> caller(buffer.get(), [](Buffer*) {});
    auto pin = device->allocator_impl()->pin_l1(caller, CoreCoord{0, 0}, BufferRegion{0, 32});
    EXPECT_EQ(&pin.buffer(), buffer.get());
    EXPECT_EQ(pin.allocation_id(), buffer->unique_id());
    EXPECT_EQ(buffer.use_count(), original_owners + 1);
    caller.reset();
    EXPECT_TRUE(pin.live());
}

TEST_F(L1AllocationPinWithdrawalTest, AllocatorWithdrawalWaitsForTheLastPinsOriginalBufferDeallocation) {
    IDevice* device = this->device.get();
    auto buffer = allocate(device);
    auto* original = buffer.get();
    std::weak_ptr<Buffer> retained = buffer;
    auto& allocator = *device->allocator_impl();
    using Pin = decltype(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 32}));
    std::optional<Pin> pin(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 32}));

    struct FinalDeallocationGate final : IGraphProcessor, IGraphHooks {
        explicit FinalDeallocationGate(Buffer* original) : original(original), resume(resumed.get_future().share()) {}
        bool is_capture_processor() const override { return false; }
        void track_deallocate(Buffer* buffer) override {
            if (buffer == original) {
                entered.set_value();
                timed_out.store(
                    resume.wait_for(std::chrono::seconds(5)) != std::future_status::ready,
                    std::memory_order_release);
            }
        }
        bool hook_allocate(const Buffer*) override { return false; }
        bool hook_program(Program*) override { return false; }
        bool hook_write_to_device(const Buffer*) override { return false; }
        bool hook_read_from_device(Buffer*) override { return false; }
        bool hook_read_from_device(const distributed::MeshBuffer*) override { return false; }
        bool hook_write_to_device(const distributed::MeshBuffer*) override { return false; }
        bool hook_deallocate(Buffer* buffer) override {
            // Only the failing candidate can finish withdrawal while the real
            // Buffer destructor is paused. Protect that RED run from calling
            // its already-destroyed allocator. A correct implementation takes
            // the ordinary, actual allocator deallocation path below.
            const bool unsafe = withdrawal_passed.load(std::memory_order_acquire) ||
                                timed_out.load(std::memory_order_acquire);
            if (buffer == original && unsafe) {
                bypass_used.store(true, std::memory_order_release);
                return true;
            }
            return false;
        }
        Buffer* const original;
        std::promise<void> entered;
        std::promise<void> resumed;
        const std::shared_future<void> resume;
        std::atomic<bool> withdrawal_passed{false};
        std::atomic<bool> timed_out{false};
        std::atomic<bool> bypass_used{false};
    };
    auto gate = std::make_shared<FinalDeallocationGate>(original);
    auto entered = gate->entered.get_future();
    buffer.reset();
    auto released = std::async(std::launch::async, [pin = std::move(pin), gate]() mutable {
        auto& tracker = GraphTracker::instance();
        tracker.push_processor(gate);
        EXPECT_TRUE(tracker.add_hook(gate));
        pin.reset();
        tracker.clear_hook();
        tracker.pop_processor();
    });
    const auto entered_status = entered.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(entered_status, std::future_status::ready);
    if (entered_status != std::future_status::ready) {
        gate->resumed.set_value();
        EXPECT_EQ(released.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        return;
    }
    auto withdrawn = std::async(std::launch::async, [&] { return device->close(); });
    const auto withdrawal_status = withdrawn.wait_for(std::chrono::milliseconds(100));
    gate->withdrawal_passed.store(withdrawal_status == std::future_status::ready, std::memory_order_release);
    EXPECT_EQ(withdrawal_status, std::future_status::timeout);
    gate->resumed.set_value();
    EXPECT_EQ(released.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_NO_THROW(released.get());
    EXPECT_EQ(withdrawn.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(withdrawn.get());
    EXPECT_FALSE(gate->timed_out.load(std::memory_order_acquire));
    EXPECT_FALSE(gate->bypass_used.load(std::memory_order_acquire));
    EXPECT_TRUE(retained.expired());
}

TEST_F(L1AllocationPinTest, RejectsViewsForeignAllocationsMissingCoresAndInvalidSubranges) {
    auto* device = mesh->get_devices().front();
    auto buffer = allocate(device);
    auto foreign = allocate(mesh->get_devices().back());
    auto dram = BufferImpl::create(device, 1024, 1024, BufferType::DRAM);
    auto view = BufferImpl::create(device, buffer->address(), 1024, 1024, BufferType::L1);
    auto& allocator = *device->allocator_impl();
    const auto before = allocator.get_allocated_buffers();
    EXPECT_THROW(allocator.pin_l1(view, CoreCoord{0, 0}, BufferRegion{0, 32}), std::runtime_error);
    EXPECT_THROW(allocator.pin_l1(foreign, CoreCoord{0, 0}, BufferRegion{0, 32}), std::runtime_error);
    EXPECT_THROW(allocator.pin_l1(dram, CoreCoord{0, 0}, BufferRegion{0, 32}), std::runtime_error);
    EXPECT_THROW(allocator.pin_l1(buffer, CoreCoord{2, 0}, BufferRegion{0, 32}), std::runtime_error);
    EXPECT_THROW(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 0}), std::runtime_error);
    EXPECT_THROW(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{1024, 1}), std::runtime_error);
    EXPECT_THROW(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{UINT64_MAX, 2}), std::runtime_error);
    EXPECT_THROW(allocator.pin_l1(std::shared_ptr<Buffer>{}, CoreCoord{0, 0}, BufferRegion{0, 32}), std::runtime_error);
    EXPECT_EQ(allocator.get_allocated_buffers(), before);
    EXPECT_NO_THROW(buffer->impl().deallocate(*buffer));
}

TEST_F(L1AllocationPinTest, RejectsMeshDeviceProxyBeforePinningItsRegisteredAllocation) {
    // A MeshDevice's allocator owns proxy bookkeeping, rather than one
    // physical device reservation. Its manager teardown clears that allocator
    // before the allocator withdrawal boundary, so it cannot issue a physical
    // allocation pin even when this is an original registered Buffer.
    auto buffer = allocate(mesh.get());
    auto& allocator = *mesh->allocator_impl();
    const auto before = allocator.get_allocated_buffers();
    EXPECT_THROW(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 32}), std::runtime_error);
    EXPECT_EQ(allocator.get_allocated_buffers(), before);
    EXPECT_TRUE(buffer->is_allocated());
    EXPECT_NO_THROW(buffer->impl().deallocate(*buffer));
}

TEST_F(L1AllocationPinTest, MeshPinRetainsExactDeviceWithoutSharedBackingAndRejectsDestructiveMutation) {
    distributed::DeviceLocalBufferConfig local;
    local.page_size = 1024;
    local.buffer_type = BufferType::L1;
    local.sharding_args = sharding();
    auto allocation = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{2048}, local, mesh.get());
    ASSERT_EQ(allocation->get_backing_buffer(), nullptr);
    const distributed::MeshCoordinate coordinate{0, 1};
    auto* device = mesh->impl().get_device(coordinate);
    auto* child = allocation->get_device_buffer(coordinate);
    auto* other = allocation->get_device_buffer(distributed::MeshCoordinate{0, 0});
    const auto base = experimental::per_core_allocation::get_per_core_address(*child, {0, 0});
    std::weak_ptr<distributed::MeshBuffer> retained = allocation;
    auto& allocator = *device->allocator_impl();
    using Pin = decltype(allocator.pin_l1(allocation, coordinate, CoreCoord{0, 0}, BufferRegion{32, 64}));
    std::optional<Pin> pin(allocator.pin_l1(allocation, coordinate, CoreCoord{0, 0}, BufferRegion{32, 64}));
    EXPECT_EQ(&pin->buffer(), child);
    EXPECT_EQ(pin->device(), device);
    EXPECT_EQ(pin->begin(), base + 32);
    EXPECT_EQ(pin->end(), base + 96);
    EXPECT_THROW(allocation->deallocate(), std::runtime_error);
    EXPECT_THROW({ distributed::MeshBuffer moved(std::move(*allocation)); }, std::runtime_error);
    EXPECT_TRUE(allocation->is_allocated());
    EXPECT_TRUE(child->is_allocated());
    EXPECT_TRUE(other->is_allocated());
    allocation.reset();
    EXPECT_FALSE(retained.expired());
    EXPECT_TRUE(pin->live());
    EXPECT_NO_THROW(pin.reset());
    EXPECT_TRUE(retained.expired());
}

TEST_F(L1AllocationPinTest, MeshPinRetainsTheSdkOwnerFromBothFactoriesWithAnotherCallerControlBlock) {
    const distributed::MeshCoordinate coordinate{0, 1};
    auto* device = mesh->impl().get_device(coordinate);
    distributed::DeviceLocalBufferConfig local;
    local.page_size = 1024;
    local.buffer_type = BufferType::L1;
    local.sharding_args = sharding();
    for (const bool single_device : {false, true}) {
        SCOPED_TRACE(single_device);
        auto allocation = single_device
                              ? experimental::per_core_allocation::create_on_single_device(
                                    distributed::ReplicatedBufferConfig{2048}, local, mesh.get(), coordinate)
                              : distributed::MeshBuffer::create(
                                    distributed::ReplicatedBufferConfig{2048}, local, mesh.get());
        auto* original_child = allocation->get_device_buffer(coordinate);
        const auto original_owners = allocation.use_count();
        std::shared_ptr<distributed::MeshBuffer> caller(allocation.get(), [](distributed::MeshBuffer*) {});
        auto pin = device->allocator_impl()->pin_l1(caller, coordinate, CoreCoord{0, 0}, BufferRegion{0, 32});
        EXPECT_EQ(&pin.buffer(), original_child);
        EXPECT_EQ(pin.allocation_id(), original_child->unique_id());
        EXPECT_EQ(allocation.use_count(), original_owners + 1);
        caller.reset();
        EXPECT_TRUE(pin.live());
    }
}

TEST_F(L1AllocationPinTest, MeshPinRejectsAnUnregisteredMovedObjectWithoutChangingItsAllocation) {
    distributed::DeviceLocalBufferConfig local;
    local.page_size = 1024;
    local.buffer_type = BufferType::L1;
    local.sharding_args = sharding();
    auto allocation = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{2048}, local, mesh.get());
    const distributed::MeshCoordinate coordinate{0, 1};
    auto* child = allocation->get_device_buffer(coordinate);
    auto moved = std::make_shared<distributed::MeshBuffer>(std::move(*allocation));
    auto& allocator = *mesh->impl().get_device(coordinate)->allocator_impl();
    const auto before = allocator.get_allocated_buffers();
    // Moving an allocation is supported, but a caller-created control block
    // does not become an SDK factory lifetime owner capable of issuing pins.
    EXPECT_FALSE(allocation->is_allocated());
    EXPECT_TRUE(moved->is_allocated());
    EXPECT_THROW(allocator.pin_l1(moved, coordinate, CoreCoord{0, 0}, BufferRegion{0, 32}), std::runtime_error);
    EXPECT_EQ(allocator.get_allocated_buffers(), before);
    EXPECT_EQ(moved->get_device_buffer(coordinate), child);
    EXPECT_TRUE(child->is_allocated());
    EXPECT_NO_THROW(moved->deallocate());
}

TEST_F(L1AllocationPinTest, MeshMoveAssignmentRetainsTheFactoryDestinationLifetimeOwner) {
    distributed::DeviceLocalBufferConfig local;
    local.page_size = 1024;
    local.buffer_type = BufferType::L1;
    local.sharding_args = sharding();
    auto destination = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{2048}, local, mesh.get());
    auto source = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{2048}, local, mesh.get());
    const distributed::MeshCoordinate coordinate{0, 1};
    auto* child = source->get_device_buffer(coordinate);
    *destination = std::move(*source);
    const auto destination_owners = destination.use_count();
    EXPECT_FALSE(source->is_allocated());
    EXPECT_EQ(destination->get_device_buffer(coordinate), child);
    auto pin = mesh->impl().get_device(coordinate)->allocator_impl()->pin_l1(
        destination, coordinate, CoreCoord{0, 0}, BufferRegion{0, 32});
    EXPECT_EQ(&pin.buffer(), child);
    EXPECT_EQ(destination.use_count(), destination_owners + 1);
    EXPECT_TRUE(pin.live());
}

TEST_F(L1AllocationPinWithdrawalTest, OriginalAllocatorWithdrawalInvalidatesPinAndLeavesNoThrowingLateDestructor) {
    IDevice* device = this->device.get();
    auto buffer = allocate(device);
    auto& allocator = *device->allocator_impl();
    using Pin = decltype(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 32}));
    std::optional<Pin> pin(allocator.pin_l1(buffer, CoreCoord{0, 0}, BufferRegion{0, 32}));
    ASSERT_TRUE(pin->live());
    EXPECT_TRUE(device->close());
    EXPECT_FALSE(pin->live());
    EXPECT_FALSE(buffer->is_allocated());
    EXPECT_NO_THROW(pin.reset());
    EXPECT_NO_THROW(buffer.reset());
}

TEST_F(L1AllocationPinTest, DirectChildPinRejectsWholeMeshReleaseBeforeAnyOtherDeviceAllocationIsDropped) {
    distributed::DeviceLocalBufferConfig local;
    local.page_size = 1024;
    local.buffer_type = BufferType::L1;
    local.sharding_args = sharding();
    auto allocation = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{2048}, local, mesh.get());
    const distributed::MeshCoordinate coordinate{0, 1};
    auto* device = mesh->impl().get_device(coordinate);
    auto child = allocation->get_device_buffer(coordinate)->shared_from_this();
    auto* other = allocation->get_device_buffer(distributed::MeshCoordinate{0, 0});
    auto& allocator = *device->allocator_impl();
    auto pin = allocator.pin_l1(child, CoreCoord{0, 0}, BufferRegion{0, 32});
    EXPECT_THROW(allocation->deallocate(), std::runtime_error);
    EXPECT_TRUE(allocation->is_allocated());
    EXPECT_TRUE(child->is_allocated());
    EXPECT_TRUE(other->is_allocated());
    EXPECT_TRUE(pin.live());
}

TEST_F(NiuExecutionMeshTest, EventAllocationRejectsExhaustionWithoutReusingZero) {
    auto& manager = mesh->get_devices().front()->sysmem_manager();
    constexpr auto maximum = std::numeric_limits<uint32_t>::max();
    manager.set_current_and_last_completed_event(0, maximum - 2, maximum - 2);
    EXPECT_EQ(manager.get_next_event(0), maximum - 1);
    for (unsigned attempt = 0; attempt != 2; ++attempt) {
        EXPECT_THROW(manager.get_next_event(0), std::runtime_error);
        EXPECT_EQ(manager.get_current_event(0), maximum - 1);
        EXPECT_EQ(manager.get_last_completed_event(0), maximum - 2);
    }
}

TEST_F(NiuExecutionMeshTest, EventIncrementRejectsOverflowWithoutChangingTheNextEvent) {
    auto& manager = mesh->get_devices().front()->sysmem_manager();
    constexpr auto maximum = std::numeric_limits<uint32_t>::max();
    manager.set_current_and_last_completed_event(0, maximum - 2, maximum - 2);
    EXPECT_THROW(manager.increment_event_id(0, 3), std::runtime_error);
    EXPECT_EQ(manager.get_current_event(0), maximum - 2);
    EXPECT_EQ(manager.get_next_event(0), maximum - 1);
    EXPECT_THROW(manager.increment_event_id(0, 1), std::runtime_error);
    EXPECT_EQ(manager.get_current_event(0), maximum - 1);
}

TEST_F(NiuExecutionMeshTest, QuiescenceWaitSerializesWithTheQueueApiBeforeInspectingPendingWork) {
    // The production constructor accepts its real API lock owner. Use that
    // public seam, with actual SDK queue/pools, without changing device state.
    std::mutex api_mutex;
    auto pool = create_passthrough_thread_pool(*context_id);
    auto shared_state = std::make_shared<CQSharedState>();
    distributed::FDMeshCommandQueue queue(
        mesh.get(), 0, pool, pool, shared_state,
        [&] { return std::lock_guard<std::mutex>(api_mutex); }, nullptr);
    std::promise<void> entered;
    auto started = entered.get_future();
    std::future<void> wait;
    {
        std::lock_guard api_lock(api_mutex);
        wait = std::async(std::launch::async, [&] {
            entered.set_value();
            queue.wait_for_completion(false);
        });
        started.wait();
        EXPECT_EQ(wait.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    }
    EXPECT_NO_THROW(wait.get());
}

TEST_F(NiuExecutionMeshTest, NumericEventProgressCannotPrunePinnedMemoryBeforeSynchronization) {
    context->rtoptions().set_fast_dispatch(true);
    std::array<uint32_t, 16> host_data{};
    experimental::PinnedMemoryImpl pinned(
        MetalEnvAccessor(*environment).impl(), mesh->get_devices(), host_data.data(), sizeof(host_data));
    const distributed::MeshEvent event(
        7, mesh->mesh_command_queue(), distributed::MeshCoordinateRange(mesh->shape()));
    // This updates ordinary host bookkeeping only. No device response or
    // firmware completion has been observed by the private event receiver.
    for (auto* device : mesh->get_devices())
        device->sysmem_manager().set_last_completed_event(0, 7);
    pinned.add_barrier_event(event);
    EXPECT_TRUE(pinned.lock_may_block());
    EXPECT_EQ(pinned.lock(), host_data.data());
    EXPECT_FALSE(pinned.lock_may_block());
    pinned.unlock();
    for (auto* device : mesh->get_devices()) EXPECT_FALSE(context->native_firmware(device->id()));
}

TEST_F(NiuExecutionMeshTest, EventEpochResetSerializesWithTheQueueApiBeforeInspectingPendingWork) {
    std::mutex api_mutex;
    auto pool = create_passthrough_thread_pool(*context_id);
    auto shared_state = std::make_shared<CQSharedState>();
    distributed::FDMeshCommandQueue queue(
        mesh.get(), 0, pool, pool, shared_state,
        [&] { return std::lock_guard<std::mutex>(api_mutex); }, nullptr);
    std::promise<void> entered;
    auto started = entered.get_future();
    std::future<void> reset;
    {
        std::lock_guard api_lock(api_mutex);
        reset = std::async(std::launch::async, [&] {
            entered.set_value();
            queue.finish_and_reset_in_use();
        });
        started.wait();
        EXPECT_EQ(reset.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    }
    EXPECT_NO_THROW(reset.get());
}

TEST_F(NiuExecutionMeshTest, FastDispatchAdmitsIdleWorkerGoRecipientsBeforeCompilation) {
    auto work = workload();
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().front(), {1, 0}));
    expect_rejected_before_compile(work);
}

TEST_F(NiuExecutionMeshTest, ManagedMeshInvocationRequiresActualBootBeforePreparation) {
    auto work = workload();
    Program second(std::make_shared<detail::ProgramImpl>(*context_id));
    ASSERT_EQ(second.impl().get_context_id(), *context_id);
    CreateKernelFromString(second, "void kernel_main() {}", CoreCoord{1, 0},
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
    work.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 1}), std::move(second));
    const std::array workloads{&work};
    expect_unbooted_managed_rejection<detail::ProgramExecution>(*mesh, workloads);
    EXPECT_EQ(detail::ProgramExecution::validate_workload(*mesh, work), nullptr);
    EXPECT_NO_THROW(detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{0}}, work));
    expect_unbooted_managed_rejection<detail::ProgramExecution>(*mesh, workloads);
    for (const auto& [range, program] : work.get_programs()) {
        EXPECT_FALSE(program.impl().is_compiled());
        EXPECT_FALSE(program.impl().is_finalized());
    }
    for (auto* device : mesh->get_devices()) EXPECT_FALSE(context->native_firmware(device->id()));
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRequireActualBootBeforePreparingEitherGroup) {
    auto first = workload();
    auto second = workload();
    const std::array ordered{&first, &second};
    expect_ordered_rejection_before_preparation(ordered, "boot", ordered);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsValidateTheLastGroupBeforeBootLookup) {
    auto first = workload();
    distributed::MeshWorkload empty;
    const std::array ordered{&first, &empty};
    expect_ordered_rejection_before_preparation(ordered, "empty workload", ordered);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectAnEmptyCollection) {
    expect_ordered_rejection_before_preparation({}, "nonempty workload collection", {});
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectALaterNullWorkload) {
    auto first = workload();
    const std::array ordered{&first, static_cast<distributed::MeshWorkload*>(nullptr)};
    const std::array intact{&first};
    expect_ordered_rejection_before_preparation(ordered, "null workload", intact);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectALaterMovedFromWorkload) {
    auto first = workload();
    auto source = workload();
    auto moved = std::move(source);
    const std::array ordered{&first, &source};
    const std::array intact{&first, &moved};
    expect_ordered_rejection_before_preparation(ordered, "moved-from workload", intact);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectARepeatedWorkloadOwner) {
    auto first = workload();
    const std::array ordered{&first, &first};
    const std::array intact{&first};
    expect_ordered_rejection_before_preparation(ordered, "duplicate workload", intact);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectTheSameProgramOwnerInDifferentGroups) {
    auto first = workload();
    auto& program = first.get_programs().begin()->second;
    Program alias(program.impl().shared_from_this());
    distributed::MeshWorkload second;
    second.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 1}), std::move(alias));
    const std::array ordered{&first, &second};
    expect_ordered_rejection_before_preparation(ordered, "distinct", ordered);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectAForeignProgramContextBeforeBootLookup) {
    auto first = workload();
    Program foreign;
    ASSERT_NE(foreign.impl().get_context_id(), *context_id);
    CreateKernelFromString(
        foreign, "void kernel_main() {}", CoreCoord{0, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
    distributed::MeshWorkload second;
    second.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 1}), std::move(foreign));
    const std::array ordered{&first, &second};
    expect_ordered_rejection_before_preparation(ordered, "another context", ordered);
}

TEST_F(NiuExecutionMeshTest, ManagedOrderedWorkloadsRejectOverlappingRangesInALaterGroupBeforeBootLookup) {
    auto first = workload();
    auto second = workload();
    Program overlapping(std::make_shared<detail::ProgramImpl>(*context_id));
    CreateKernelFromString(
        overlapping, "void kernel_main() {}", CoreCoord{1, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
    // The public mutable map can bypass add_program's range check. Acquisition
    // must revalidate the actual complete workload, including late groups.
    second.get_programs().emplace(distributed::MeshCoordinateRange(mesh->shape()), std::move(overlapping));
    const std::array ordered{&first, &second};
    expect_ordered_rejection_before_preparation(ordered, "overlap", ordered);
}

TEST_F(NiuExecutionMeshTest, FastDispatchAdmitsLocalChipWithoutAProgramBeforeCompilation) {
    auto work = workload();
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().back(), {0, 0}));
    expect_rejected_before_compile(work);
}

TEST_F(NiuExecutionMeshTest, DispatchReservationExcludesUnrelatedSubdevice) {
    SubDevice first(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    SubDevice second(std::array{CoreRangeSet(CoreRange({1, 0}, {1, 0}))});
    auto manager = mesh->create_sub_device_manager({first, second}, 0);
    mesh->load_sub_device_manager(manager);
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().back(), {1, 0}));
    EXPECT_NO_THROW(detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{0}}));
    EXPECT_THROW(detail::ProgramExecution::reserve_dispatch(*mesh, std::array{SubDeviceId{1}}), std::runtime_error);
}

TEST_F(NiuExecutionMeshTest, SubmeshDispatchDoesNotAdmitUnrelatedLocalChip) {
    auto local = mesh->create_submesh(distributed::MeshShape{1, 1}, distributed::MeshCoordinate{0, 0});
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().back(), {0, 0}));
    EXPECT_NO_THROW(detail::ProgramExecution::reserve_dispatch(*local, std::array{SubDeviceId{0}}));
    local->close();
}

TEST_F(NiuExecutionMeshTest, ManagerResetAdmitsOldWorkerAbsentFromNewManager) {
    SubDevice first(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    SubDevice second(std::array{CoreRangeSet(CoreRange({1, 0}, {1, 0}))});
    const auto old_manager = mesh->create_sub_device_manager({first, second}, 0);
    const auto new_manager = mesh->create_sub_device_manager({first}, 0);
    mesh->load_sub_device_manager(old_manager);
    ASSERT_NO_FATAL_FAILURE(quarantine(mesh->get_devices().back(), {1, 0}));
    try {
        mesh->load_sub_device_manager(new_manager);
        FAIL() << "Reset GO must admit old-manager workers before remapping";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("quarantin"), std::string::npos) << error.what();
    }
    EXPECT_EQ(mesh->num_sub_devices(), 2u);
}

TEST_F(NiuExecutionMeshTest, MockManagerReductionDoesNotMintIssueOrStopOnClose) {
    SubDevice first(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    SubDevice second(std::array{CoreRangeSet(CoreRange({1, 0}, {1, 0}))});
    const auto old_manager = mesh->create_sub_device_manager({first, second}, 0);
    const auto new_manager = mesh->create_sub_device_manager({first}, 0);
    mesh->load_sub_device_manager(old_manager);
    mesh->load_sub_device_manager(new_manager);
    mesh->mesh_command_queue().finish();
    auto* device = mesh->get_devices().front();
    const auto physical =
        context->get_cluster().get_soc_desc(device->id()).get_physical_tensix_core_from_logical({1, 0});
    const std::array keys{
        detail::NiuKey{device->id(), physical, detail::NiuNoc::Noc0},
        detail::NiuKey{device->id(), physical, detail::NiuNoc::Noc1}};
    ASSERT_TRUE(device->close());
    std::string error;
    auto next = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(next) << error;
    EXPECT_TRUE(context->niu_admission().cancel_unissued(*next, error)) << error;
}

TEST_F(NiuExecutionMeshTest, ServiceManagerRejectsMeshIdentityBeforeChipLookup) {
    auto& services = context->get_service_core_manager();
    try {
        services.get_claimable_cores(mesh.get());
        FAIL() << "Mesh identity must not be interpreted as ChipId";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("actual local Device"), std::string::npos) << error.what();
    }
    EXPECT_THROW(services.claim(mesh.get(), {{0, 0}}), std::runtime_error);
    EXPECT_THROW(services.release(mesh.get(), {{0, 0}}), std::runtime_error);
    EXPECT_THROW(services.allocate_l1(mesh.get(), {0, 0}, 32), std::runtime_error);
}

TEST_F(NiuExecutionMeshTest, ActualChipCloseQuarantinesIssuedCrossChipOwnerOnly) {
    const auto devices = mesh->get_devices();
    auto key = [&](IDevice* device, CoreCoord logical) {
        return detail::NiuKey{
            device->id(),
            context->get_cluster().get_soc_desc(device->id()).get_physical_tensix_core_from_logical(logical),
            detail::NiuNoc::Noc0};
    };
    const std::array cross{key(devices.front(), {0, 0}), key(devices.back(), {0, 0})};
    const std::array unrelated{key(devices.back(), {1, 0})};
    std::string error;
    auto issued = context->niu_admission().reserve_ordinary(cross, error);
    auto separate = context->niu_admission().reserve_ordinary(unrelated, error);
    ASSERT_TRUE(issued) << error;
    ASSERT_TRUE(separate) << error;
    // Coordinator bookkeeping deliberately models unknown issue; no native
    // firmware or successful hardware completion is manufactured.
    ASSERT_TRUE(context->niu_admission().mark_issued(*issued, error)) << error;
    ASSERT_TRUE(devices.front()->close());
    EXPECT_FALSE(context->niu_admission().mark_issued(*issued, error));
    EXPECT_FALSE(context->niu_admission().cancel_unissued(*issued, error));
    EXPECT_FALSE(context->niu_admission().reserve_ordinary(std::span(cross).last(1), error));
    EXPECT_TRUE(context->niu_admission().cancel_unissued(*separate, error)) << error;
}

class NiuExecutionSlowMeshTest : public NiuExecutionMeshTest {
protected:
    void SetUp() override {
        slow.emplace();
        NiuExecutionMeshTest::SetUp();
    }
    void TearDown() override {
        NiuExecutionMeshTest::TearDown();
        slow.reset();
    }
    std::optional<experimental::test_helpers::ScopedSlowDispatchOverride> slow;
};

TEST_F(NiuExecutionSlowMeshTest, MockQueueReturnDoesNotCreateIssuedQuarantineOnClose) {
    auto work = workload();
    ASSERT_NO_THROW(mesh->mesh_command_queue().enqueue_mesh_workload(work, false));
    auto* device = mesh->get_devices().front();
    const auto physical =
        context->get_cluster().get_soc_desc(device->id()).get_physical_tensix_core_from_logical({0, 0});
    const std::array keys{
        detail::NiuKey{device->id(), physical, detail::NiuNoc::Noc0},
        detail::NiuKey{device->id(), physical, detail::NiuNoc::Noc1}};
    ASSERT_TRUE(device->close());
    std::string error;
    auto next = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(next) << error;
    EXPECT_TRUE(context->niu_admission().cancel_unissued(*next, error)) << error;
}

TEST_F(NiuInfrastructureTest, QuarantinedPhysicalNiuRejectsWholeServiceClaim) {
    ASSERT_NO_FATAL_FAILURE(quarantine(free_cores[1], detail::NiuNoc::Noc1));
    EXPECT_THROW(services().claim(device.get(), {free_cores[0], free_cores[1]}), std::runtime_error);
    EXPECT_TRUE(services().claimed_cores(chip).empty());
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
    ASSERT_NO_THROW(services().claim(device.get(), {free_cores[0]}));
    EXPECT_EQ(services().claimed_cores(chip), std::unordered_set<CoreCoord>{free_cores[0]});
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, RejectedDuplicateClaimDoesNotPublishEarlierCore) {
    ASSERT_NO_THROW(services().claim(device.get(), {free_cores[1]}));
    EXPECT_THROW(services().claim(device.get(), {free_cores[0], free_cores[1]}), std::runtime_error);
    EXPECT_EQ(services().claimed_cores(chip), std::unordered_set<CoreCoord>{free_cores[1]});
    ASSERT_NO_THROW(services().claim(device.get(), {free_cores[0]}));
}

TEST_F(NiuInfrastructureTest, ServiceClaimRevalidatesDispatchAllocationAfterQuery) {
    const auto claimable = services().get_claimable_cores(device.get());
    ASSERT_FALSE(claimable.empty());
    const auto assigned = dispatch().prefetcher_core(chip, 0, 0);
    ASSERT_EQ(CoreCoord(assigned.x, assigned.y), claimable.front());
    EXPECT_THROW(services().claim(device.get(), {claimable.front()}), std::runtime_error);
    EXPECT_TRUE(services().claimed_cores(chip).empty());
}

TEST_F(NiuInfrastructureTest, QuarantinedPhysicalNiuRejectsDispatchWithoutConsumingPool) {
    ASSERT_NO_FATAL_FAILURE(quarantine(free_cores.front()));
    EXPECT_THROW(dispatch().prefetcher_core(chip, 0, 0), std::runtime_error);
    EXPECT_FALSE(dispatch().is_prefetcher_core_allocated(chip, 0, 0));
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
}

TEST_F(NiuInfrastructureTest, QuarantinedPhysicalNiuRejectsFabricMuxWithoutPublishingAssignment) {
    ASSERT_NO_FATAL_FAILURE(quarantine(free_cores.front(), detail::NiuNoc::Noc1));
    EXPECT_THROW(dispatch().fabric_mux_core(chip, 0, 0, 0), std::runtime_error);
    EXPECT_FALSE(dispatch().is_fabric_mux_core_allocated(chip, 0, 0, 0));
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
}

TEST_F(NiuInfrastructureTest, CachedDispatchAssignmentCannotBypassLaterQuarantine) {
    const auto assigned = dispatch().prefetcher_core(chip, 0, 0);
    ASSERT_NO_FATAL_FAILURE(quarantine({assigned.x, assigned.y}));
    EXPECT_THROW(dispatch().prefetcher_core(chip, 0, 0), std::runtime_error);
}

TEST_F(NiuInfrastructureTest, QuarantinedProfilerCoreRejectsDispatchManagerConstruction) {
    const auto profiler = dispatch().get_reserved_realtime_profiler_core(chip);
    ASSERT_TRUE(profiler);
    ASSERT_NO_FATAL_FAILURE(quarantine({profiler->x, profiler->y}));
    auto& env_impl = MetalEnvAccessor(*environment).impl();
    EXPECT_THROW((dispatch_core_manager{dispatch_config, 1, env_impl, *context}), std::runtime_error);
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, UnlaunchedServiceReleaseRestoresDispatchAvailability) {
    const CoreCoord core = free_cores.front();
    ASSERT_NO_THROW(services().claim(device.get(), {core}));
    const auto available = dispatch().get_available_dispatch_cores(chip);
    EXPECT_EQ(std::find(available.begin(), available.end(), core), available.end());
    ASSERT_NO_THROW(services().release(device.get(), {core}));
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
    const auto assigned = dispatch().prefetcher_core(chip, 0, 0);
    EXPECT_EQ(CoreCoord(assigned.x, assigned.y), core);
}

TEST_F(NiuInfrastructureTest, ServiceClaimKeepsFastDispatchGridAfterSlowDispatchSwitch) {
    const auto fd_grid = device->compute_with_storage_grid_size();
    ASSERT_NO_THROW(services().claim(device.get(), {free_cores.front()}));
    context->rtoptions().set_fast_dispatch(false);
    EXPECT_EQ(services().impl().get_safe_compute_grid(chip), fd_grid);
    EXPECT_EQ(device->compute_with_storage_grid_size(), fd_grid);
    ASSERT_NO_THROW(services().release(device.get(), {free_cores.front()}));
    EXPECT_FALSE(services().impl().get_safe_compute_grid(chip));
}

TEST_F(NiuInfrastructureTest, ServiceClaimsSurviveDispatchManagerReconstruction) {
    const CoreCoord core = free_cores.front();
    const auto fd_grid = device->compute_with_storage_grid_size();
    ASSERT_NO_THROW(services().claim(device.get(), {core}));
    ASSERT_NO_THROW(context->set_fast_dispatch_mode(false));
    EXPECT_EQ(services().impl().get_safe_compute_grid(chip), fd_grid);
    EXPECT_EQ(device->compute_with_storage_grid_size(), fd_grid);
    ASSERT_NO_THROW(context->set_fast_dispatch_mode(true));
    EXPECT_EQ(services().claimed_cores(chip), std::unordered_set<CoreCoord>{core});
    const auto available = dispatch().get_available_dispatch_cores(chip);
    EXPECT_EQ(std::find(available.begin(), available.end(), core), available.end());
    ASSERT_NO_THROW(services().release(device.get(), {core}));
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, RejectedProfilerReconstructionPreservesPublishedManager) {
    auto* published = &dispatch();
    const auto profiler = dispatch().get_reserved_realtime_profiler_core(chip);
    ASSERT_TRUE(profiler);
    ASSERT_NO_FATAL_FAILURE(quarantine({profiler->x, profiler->y}));
    EXPECT_THROW(context->set_fast_dispatch_mode(true), std::runtime_error);
    EXPECT_EQ(&dispatch(), published);
    EXPECT_EQ(dispatch().get_available_dispatch_cores(chip), free_cores);
    EXPECT_THROW(dispatch().get_reserved_realtime_profiler_core(chip), std::runtime_error);
    EXPECT_FALSE(context->native_firmware(chip));
}

TEST_F(NiuInfrastructureTest, LaunchedServiceReleaseDoesNotCertifyNiuClean) {
    const CoreCoord core = free_cores.front();
    ASSERT_NO_THROW(services().claim(device.get(), {core}));
    // A real coordinator bookkeeping owner is lost after issue. This is not a
    // device launch receipt; releasing an unrelated planning claim must retain
    // its unknown NIU exclusion.
    const auto physical = context->get_cluster().get_soc_desc(chip).get_physical_tensix_core_from_logical(core);
    const std::array keys{detail::NiuKey{chip, physical, detail::NiuNoc::Noc0}};
    std::string error;
    auto issued = context->niu_admission().reserve_ordinary(keys, error);
    ASSERT_TRUE(issued) << error;
    ASSERT_TRUE(context->niu_admission().mark_issued(*issued, error)) << error;
    issued.reset();
    ASSERT_NO_THROW(services().release(device.get(), {core}));
    EXPECT_TRUE(services().claimed_cores(chip).empty());
    EXPECT_THROW(services().claim(device.get(), {core}), std::runtime_error);
    EXPECT_THROW(dispatch().prefetcher_core(chip, 0, 0), std::runtime_error);
}

TEST_F(NiuInfrastructureTest, EthernetDispatchDoesNotReserveWorkerNius) {
    auto& env_impl = MetalEnvAccessor(*environment).impl();
    dispatch_core_manager ethernet(
        DispatchCoreConfig{DispatchCoreType::ETH, DispatchCoreAxis::COL}, 1, env_impl, *context);
    ASSERT_EQ(ethernet.get_dispatch_core_type(), CoreType::ETH);
    const auto available = ethernet.get_available_dispatch_cores(chip);
    ASSERT_FALSE(available.empty());
    // Poison the worker mapping at the same logical coordinate. It is a
    // different physical resource and must not govern an ethernet assignment.
    ASSERT_NO_FATAL_FAILURE(quarantine(available.front()));
    EXPECT_NO_THROW(ethernet.prefetcher_core(chip, 0, 0));
    EXPECT_FALSE(ethernet.get_reserved_realtime_profiler_core(chip));
}

}  // namespace
}  // namespace tt::tt_metal
