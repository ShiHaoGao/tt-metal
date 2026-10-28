// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "impl/context/niu_handoff_receiver.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace {
// Interpose only the actual Cluster read boundary in this CPU executable.
// The real Cluster, SOC mapping, HAL and production mailbox adapter remain.
// An unexpected call fails instead of opening a device or inventing bytes.
using ClusterRead = std::function<void(const tt::Cluster&, void*, uint32_t, tt_cxy_pair, uint64_t)>;
thread_local ClusterRead cluster_read;
using ClusterWrite = std::function<void(const tt::Cluster&, const void*, uint32_t, tt_cxy_pair, uint64_t)>;
thread_local ClusterWrite cluster_write;
thread_local std::function<void(const tt::Cluster&, tt::ChipId)> cluster_barrier;
}

namespace tt {
__attribute__((visibility("default")))
void Cluster::read_core(void* destination, uint32_t bytes, tt_cxy_pair core, uint64_t address) const {
    if (!cluster_read) throw std::runtime_error("unregistered CPU Cluster read");
    cluster_read(*this, destination, bytes, core, address);
}
__attribute__((visibility("default")))
void Cluster::write_core(const void* source, uint32_t bytes, tt_cxy_pair core, uint64_t address) const {
    if (!cluster_write) throw std::runtime_error("unregistered CPU Cluster write");
    cluster_write(*this, source, bytes, core, address);
}
__attribute__((visibility("default")))
void Cluster::l1_barrier(ChipId chip) const {
    if (!cluster_barrier) throw std::runtime_error("unregistered CPU Cluster barrier");
    cluster_barrier(*this, chip);
}
}

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

template <class Receiver>
concept exposes_closed_selector = requires(
    const NiuHandoffObservation& observation, const NiuHandoffExpectedExecution& expected) {
    Receiver::closed_selector(observation, expected, uint8_t{0}, uint8_t{0});
};

template <class Receiver>
void check_closed_selector_contract() {
    if constexpr (!exposes_closed_selector<Receiver>) {
        ADD_FAILURE() << "SDK decoder has no request-correlated selector closure observer";
    } else {
        Mailbox mailbox;
        auto observed = mailbox.read();
        ASSERT_TRUE(observed.observation);
        // TrafficClosed may retain a nonzero multicast selector counter.
        EXPECT_FALSE(Receiver::closed_selector(*observed.observation, mailbox.expected(), 0, 7));
        EXPECT_TRUE(Receiver::closed_selector(*observed.observation, mailbox.expected(), 1, 7));
        EXPECT_FALSE(Receiver::closed_selector(*observed.observation, mailbox.expected(), 2, 0));
        EXPECT_FALSE(Receiver::closed_selector(*observed.observation, mailbox.expected(), 0, 16));
        auto foreign = mailbox.expected();
        ++foreign.request.invocation_id_lo;
        EXPECT_FALSE(Receiver::closed_selector(*observed.observation, foreign, 1, 7));
        mailbox.wire.receipt.selector_observed[1][7].outgoing = 1;
        observed = mailbox.read();
        ASSERT_TRUE(observed.observation);
        EXPECT_FALSE(Receiver::closed_selector(*observed.observation, mailbox.expected(), 1, 7));
        mailbox.wire.receipt.selector_observed[1][7].outgoing = 0;
        mailbox.wire.receipt.header.status = TT_NIU_HANDOFF_PENDING;
        observed = mailbox.read();
        ASSERT_TRUE(observed.observation);
        EXPECT_FALSE(Receiver::closed_selector(*observed.observation, mailbox.expected(), 1, 7));
    }
}

TEST(NiuHandoffSelectorTest, RequiresExactRequestAndBothActualSelectorCountersClosed) {
    check_closed_selector_contract<NiuHandoffReceiver>();
}

template <class Receiver>
concept exposes_item_witness = requires {
    typename Receiver::ItemWitnessRequest;
    typename Receiver::ItemWitnessRegion;
};

template <class Receiver>
void check_item_witness_contract() {
    if constexpr (!exposes_item_witness<Receiver>) {
        ADD_FAILURE() << "SDK decoder has no stable invocation/item/occurrence witness receiver";
    } else {
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        Region region{};
        auto& request = region.request;
        request.abi_version = TT_NIU_ITEM_WITNESS_VERSION;
        request.request_generation = 7;
        request.boot_cookie = UINT64_C(0x123456789abcdef0);
        request.invocation_id = UINT64_C(0x223456789abcdef0);
        request.item = UINT64_C(0x323456789abcdef0);
        request.generation = UINT64_C(0x423456789abcdef0);
        request.occurrence = 1;
        request.leaf = 3;
        request.noc = 1;
        request.selector = 15;
        request.issuer = 2;
        request.completion = 2;
        region.receipt.request = request;
        region.receipt.publish_sequence = 4;
        region.receipt.state = 2;
        region.receipt.item.issued_occurrence = request.occurrence;
        region.receipt.item.closed_occurrence = request.occurrence;
        bool change_after_body = false;
        auto read = [&] {
            return Receiver::read_item([&](uint32_t offset, std::span<std::byte> destination) {
                if (offset > sizeof(region) || destination.size() > sizeof(region) - offset) {
                    return NiuHandoffIoStatus::Failed;
                }
                std::memcpy(
                    destination.data(), reinterpret_cast<const std::byte*>(&region) + offset, destination.size());
                if (change_after_body && destination.size() == sizeof(region)) {
                    region.receipt.publish_sequence += 2;
                }
                return NiuHandoffIoStatus::Complete;
            });
        };
        const Request expected = request;
        auto observed = read();
        ASSERT_TRUE(observed.observation);
        EXPECT_TRUE(Receiver::closed_item(*observed.observation, expected));
        // Completion before actual issue, zero counters, or copied identity
        // must never mint the original owner's online completion fact.
        region.receipt.item.issued_occurrence = 0;
        observed = read();
        ASSERT_TRUE(observed.observation);
        EXPECT_FALSE(Receiver::closed_item(*observed.observation, expected));
        region.receipt.item.issued_occurrence = expected.occurrence;
        --region.receipt.item.closed_occurrence;
        observed = read();
        ASSERT_TRUE(observed.observation);
        EXPECT_FALSE(Receiver::closed_item(*observed.observation, expected));
        region.receipt.item.closed_occurrence = expected.occurrence;
        for (auto member :
             {&Request::boot_cookie,
              &Request::invocation_id,
              &Request::item,
              &Request::generation,
              &Request::occurrence}) {
            Request foreign = expected;
            ++(foreign.*member);
            EXPECT_FALSE(Receiver::closed_item(*read().observation, foreign));
        }
        for (unsigned field = 0; field != 6; ++field) {
            Request foreign = expected;
            switch (field) {
                case 0: ++foreign.request_generation; break;
                case 1: ++foreign.leaf; break;
                case 2: foreign.noc = 0; break;
                case 3: foreign.selector = 0; break;
                case 4: foreign.issuer = 1; break;
                case 5: foreign.completion = 1; break;
            }
            EXPECT_FALSE(Receiver::closed_item(*read().observation, foreign));
        }
        region.receipt.state = 0;
        EXPECT_FALSE(Receiver::closed_item(*read().observation, expected));
        region.receipt.state = 1;
        EXPECT_FALSE(Receiver::closed_item(*read().observation, expected));
        region.receipt.state = 3;
        EXPECT_FALSE(Receiver::closed_item(*read().observation, expected));
        region.receipt.state = 2;
        region.receipt.request.selector = 14;
        EXPECT_FALSE(Receiver::closed_item(*read().observation, expected));
        region.receipt.request = expected;
        ++region.request.generation;
        EXPECT_FALSE(Receiver::closed_item(*read().observation, expected));
        region.request = expected;
        region.receipt.request.reserved = 1;
        EXPECT_FALSE(Receiver::closed_item(*read().observation, expected));
        region.receipt.request = expected;
        Request invalid = expected;
        invalid.occurrence = 0;
        EXPECT_FALSE(Receiver::closed_item(*read().observation, invalid));
        invalid = expected;
        invalid.occurrence = std::numeric_limits<uint64_t>::max();
        EXPECT_FALSE(Receiver::closed_item(*read().observation, invalid));
        region.receipt.publish_sequence = 0;
        EXPECT_EQ(read().status, NiuHandoffReadStatus::Unpublished);
        region.receipt.publish_sequence = 3;
        EXPECT_EQ(read().status, NiuHandoffReadStatus::InProgress);
        region.receipt.publish_sequence = 6;
        change_after_body = true;
        EXPECT_EQ(read().status, NiuHandoffReadStatus::Changed);
        EXPECT_EQ(
            Receiver::read_item([](uint32_t, std::span<std::byte>) { return NiuHandoffIoStatus::Failed; }).status,
            NiuHandoffReadStatus::IoFailure);
    }
}

TEST(NiuItemWitnessTest, RequiresStableOriginalItemAndPostIssueClosure) {
    check_item_witness_contract<NiuHandoffReceiver>();
}

template <class Receiver>
concept exposes_item_request_writer = requires(
    const typename Receiver::Write& write,
    const typename Receiver::Read& read,
    const typename Receiver::ItemWitnessRequest& request) { Receiver::write_item_request(write, read, request); };

template <class Receiver>
void check_item_request_writer_contract() {
    if constexpr (!exposes_item_request_writer<Receiver>) {
        ADD_FAILURE() << "SDK has no ordered item witness initialization/readback receiver";
    } else {
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        Request request{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, UINT64_C(1) << 35, 1, 2, 1, 15, 2, 2, 0};
        std::array<std::byte, sizeof(Region)> bytes;
        bytes.fill(std::byte{0xa5});
        std::vector<std::pair<uint32_t, size_t>> writes;
        bool fail_body = false;
        bool corrupt_readback = false;
        auto write = [&](uint32_t offset, std::span<const std::byte> source) {
            writes.emplace_back(offset, source.size());
            if (fail_body || offset > bytes.size() || source.size() > bytes.size() - offset) {
                return NiuHandoffIoStatus::Failed;
            }
            std::copy(source.begin(), source.end(), bytes.begin() + offset);
            return NiuHandoffIoStatus::Complete;
        };
        auto read = [&](uint32_t offset, std::span<std::byte> destination) {
            if (offset > bytes.size() || destination.size() > bytes.size() - offset) {
                return NiuHandoffIoStatus::Failed;
            }
            std::copy_n(bytes.begin() + offset, destination.size(), destination.begin());
            if (corrupt_readback && destination.size() == bytes.size()) {
                destination[24] ^= std::byte{1};
            }
            return NiuHandoffIoStatus::Complete;
        };
        EXPECT_EQ(Receiver::write_item_request(write, read, request), NiuHandoffWriteStatus::Published);
        ASSERT_EQ(writes.size(), 2u);
        EXPECT_EQ(writes[0], (std::pair<uint32_t, size_t>{0, sizeof(Region)}));
        EXPECT_EQ(writes[1], (std::pair<uint32_t, size_t>{4, 4}));
        Region region;
        std::memcpy(&region, bytes.data(), bytes.size());
        EXPECT_EQ(region.request.request_generation, request.request_generation);
        EXPECT_EQ(region.request.generation, request.generation);
        EXPECT_EQ(region.request.occurrence, request.occurrence);
        EXPECT_EQ(region.receipt.publish_sequence, 0u);
        EXPECT_EQ(region.receipt.item.issued_occurrence, 0u);
        EXPECT_EQ(region.receipt.item.closed_occurrence, 0u);
        writes.clear();
        corrupt_readback = true;
        EXPECT_EQ(Receiver::write_item_request(write, read, request), NiuHandoffWriteStatus::ReadbackMismatch);
        ASSERT_EQ(writes.size(), 1u);
        std::memcpy(&region, bytes.data(), bytes.size());
        EXPECT_EQ(region.request.request_generation, 0u);
        writes.clear();
        corrupt_readback = false;
        fail_body = true;
        EXPECT_EQ(Receiver::write_item_request(write, read, request), NiuHandoffWriteStatus::IoFailure);
        ASSERT_EQ(writes.size(), 1u);
        writes.clear();
        request.occurrence = UINT64_MAX;
        EXPECT_EQ(Receiver::write_item_request(write, read, request), NiuHandoffWriteStatus::InvalidRequest);
        EXPECT_TRUE(writes.empty());
    }
}

TEST(NiuItemWitnessTest, InitializesOnlyWithOrderedGenerationAndExactReadback) {
    check_item_request_writer_contract<NiuHandoffReceiver>();
}

template <class Receiver>
concept exposes_item_witness_check =
    requires(const NiuItemWitnessObservation& observation, const typename Receiver::ItemWitnessRequest& request) {
        typename Receiver::ItemWitnessCheck;
        Receiver::check_item(observation, request);
    };

template <class Receiver>
void check_item_witness_outcome_contract() {
    if constexpr (!exposes_item_witness_check<Receiver>) {
        ADD_FAILURE() << "Item witness decoder conflates pending, rejection and invalid identity";
    } else {
        using Check = typename Receiver::ItemWitnessCheck;
        typename Receiver::ItemWitnessRegion region{};
        const typename Receiver::ItemWitnessRequest expected{
            TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, UINT64_C(1) << 35, 1, 2, 1, 15, 2, 2, 0};
        region.request = region.receipt.request = expected;
        region.receipt.publish_sequence = 4;
        const auto check = [&](const auto& original) {
            const auto observed = Receiver::read_item([&](uint32_t offset, std::span<std::byte> bytes) {
                if (offset > sizeof(region) || bytes.size() > sizeof(region) - offset) {
                    return NiuHandoffIoStatus::Failed;
                }
                std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
                return NiuHandoffIoStatus::Complete;
            });
            EXPECT_TRUE(observed.observation);
            return Receiver::check_item(*observed.observation, original);
        };
        region.receipt.state = TT_NIU_ITEM_WITNESS_PENDING;
        EXPECT_EQ(check(expected), Check::Pending);
        region.receipt.state = TT_NIU_ITEM_WITNESS_ISSUED;
        region.receipt.item.issued_occurrence = expected.occurrence;
        EXPECT_EQ(check(expected), Check::Pending);
        region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
        EXPECT_EQ(check(expected), Check::OccurrenceMismatch);
        region.receipt.item.closed_occurrence = expected.occurrence;
        EXPECT_EQ(check(expected), Check::Match);
        region.receipt.state = TT_NIU_ITEM_WITNESS_REJECTED;
        region.receipt.item.closed_occurrence = 0;
        EXPECT_EQ(check(expected), Check::Rejected);
        auto foreign = expected;
        ++foreign.invocation_id;
        EXPECT_EQ(check(foreign), Check::RequestMismatch);
        ++region.request.generation;
        EXPECT_EQ(check(expected), Check::RequestMismatch);
        region.request = expected;
        region.receipt.request.selector = 14;
        EXPECT_EQ(check(expected), Check::RequestMismatch);
        region.receipt.request = expected;
        foreign = expected;
        foreign.occurrence = UINT64_MAX;
        EXPECT_EQ(check(foreign), Check::InvalidRequest);
        region.receipt.state = TT_NIU_ITEM_WITNESS_ISSUED;
        region.receipt.item.issued_occurrence = 0;
        EXPECT_EQ(check(expected), Check::OccurrenceMismatch);
        region.receipt.item.issued_occurrence = expected.occurrence;
        region.receipt.item.closed_occurrence = expected.occurrence;
        EXPECT_EQ(check(expected), Check::OccurrenceMismatch);
        region.receipt.state = TT_NIU_ITEM_WITNESS_PENDING;
        EXPECT_EQ(check(expected), Check::OccurrenceMismatch);
    }
}

TEST(NiuItemWitnessTest, SeparatesPendingFromTerminalRejectAndIdentityMismatch) {
    check_item_witness_outcome_contract<NiuHandoffReceiver>();
}

// These test the production reception algorithm, not LoadedFirmware or a
// managed lease. ProgramExecution validates the actual owner before reception.
void check_item_reception_stopped_contract() {
    using Receiver = NiuHandoffReceiver;
    using Reception = typename Receiver::ItemReception;
    using Issuer = typename Reception::IssuerState;
    using Status = typename Reception::Status;
    const typename Receiver::ItemWitnessRequest expected{
        TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, UINT64_C(1) << 35, 1, 2, 1, 15, 2, 2, 0};
    typename Receiver::ItemWitnessRegion region{};
    region.request = region.receipt.request = expected;
    const auto read = [&] {
        return Receiver::read_item([&](uint32_t offset, std::span<std::byte> bytes) {
            std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
            return NiuHandoffIoStatus::Complete;
        });
    };
    for (const auto sequence : {0u, 1u, 4u}) {
        Reception reception(expected);
        region.receipt.publish_sequence = sequence;
        region.receipt.state = TT_NIU_ITEM_WITNESS_ISSUED;
        region.receipt.item.issued_occurrence = expected.occurrence;
        region.receipt.item.closed_occurrence = 0;
        EXPECT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, read()), Status::Rejected)
            << "A stopped original issuer cannot advance an incomplete receipt";
        EXPECT_FALSE(reception.accepted());
        region.receipt.publish_sequence = 8;
        region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
        region.receipt.item.closed_occurrence = expected.occurrence;
        EXPECT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, read()), Status::Rejected)
            << "A terminal rejection cannot recover from later bytes";
    }
    Reception issued(expected);
    region.receipt.publish_sequence = 0;
    EXPECT_EQ(issued.receive(Issuer::Reserved, Issuer::Reserved, read()), Status::Pending);
    EXPECT_EQ(issued.receive(Issuer::Issued, Issuer::Issued, read()), Status::Pending);
    EXPECT_FALSE(issued.accepted());
    region.receipt.publish_sequence = 8;
    region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
    region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = expected.occurrence;
    EXPECT_EQ(issued.receive(Issuer::Reserved, Issuer::Reserved, read()), Status::Pending);
    EXPECT_FALSE(issued.accepted());
    EXPECT_EQ(issued.receive(Issuer::Stopped, Issuer::Stopped, read()), Status::Rejected);
    EXPECT_FALSE(issued.accepted());
    EXPECT_FALSE(issued.terminal());
}

TEST(NiuItemWitnessReceptionTest, StoppedIssuerCannotLeaveIncompletePublicationPending) {
    check_item_reception_stopped_contract();
}

void check_item_reception_retirement_contract() {
    using Receiver = NiuHandoffReceiver;
    using Reception = typename Receiver::ItemReception;
    using Issuer = typename Reception::IssuerState;
    using Status = typename Reception::Status;
    const typename Receiver::ItemWitnessRequest expected{
        TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, UINT64_C(1) << 35, 1, 2, 1, 15, 2, 2, 0};
    typename Receiver::ItemWitnessRegion region{};
    region.request = region.receipt.request = expected;
    region.receipt.publish_sequence = 6;
    region.receipt.state = TT_NIU_ITEM_WITNESS_ISSUED;
    region.receipt.item.issued_occurrence = expected.occurrence;
    const auto read = [&] {
        return Receiver::read_item([&](uint32_t offset, std::span<std::byte> bytes) {
            std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
            return NiuHandoffIoStatus::Complete;
        });
    };
    Reception first(expected);
    auto second_request = expected;
    ++second_request.item;
    Reception second(second_request);
    const auto all_accepted = [&] { return first.accepted() && second.accepted(); };
    EXPECT_FALSE(all_accepted());
    EXPECT_EQ(first.receive(Issuer::Issued, Issuer::Issued, read()), Status::Pending);
    EXPECT_FALSE(all_accepted());
    region.receipt.publish_sequence = 8;
    region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
    region.receipt.item.closed_occurrence = expected.occurrence;
    ASSERT_EQ(first.receive(Issuer::Issued, Issuer::Issued, read()), Status::Complete);
    EXPECT_FALSE(all_accepted()) << "One accepted item cannot retire all items";
    region.receipt.publish_sequence = 0;
    EXPECT_EQ(first.receive(Issuer::Issued, Issuer::Issued, read()), Status::Complete);
    ASSERT_TRUE(first.accepted());
    EXPECT_EQ(first.accepted()->wire().receipt.publish_sequence, 8u);
    EXPECT_EQ(first.accepted()->wire().receipt.request.item, expected.item);
    region.request = region.receipt.request = second_request;
    region.receipt.publish_sequence = 8;
    region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
    region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = second_request.occurrence;
    ASSERT_EQ(second.receive(Issuer::Issued, Issuer::Issued, read()), Status::Complete);
    EXPECT_TRUE(all_accepted());
    EXPECT_EQ(first.request().item, expected.item);
    EXPECT_EQ(second.request().item, second_request.item);
}

TEST(NiuItemWitnessReceptionTest, RetainsExactAcceptedReceiptBeforeAllItemRetirement) {
    check_item_reception_retirement_contract();
}

void check_sampled_item_reception_contract() {
    using Reception = NiuHandoffReceiver::ItemReception;
    using Issuer = typename Reception::IssuerState;
    using Status = typename Reception::Status;
    const NiuHandoffReceiver::ItemWitnessRequest expected{
        TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
    NiuHandoffReceiver::ItemWitnessRegion region{};
    region.request = region.receipt.request = expected;
    const auto read = [&] {
        return NiuHandoffReceiver::read_item([&](uint32_t offset, std::span<std::byte> bytes) {
            std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
            return NiuHandoffIoStatus::Complete;
        });
    };
    Reception rejected(expected);
    const auto before_stop = read();
    EXPECT_EQ(rejected.receive(Issuer::Issued, Issuer::Stopped, before_stop), Status::Pending);
    EXPECT_FALSE(rejected.accepted());
    EXPECT_EQ(rejected.receive(Issuer::Stopped, Issuer::Stopped, read()), Status::Rejected);
    Reception completed(expected);
    EXPECT_EQ(completed.receive(Issuer::Issued, Issuer::Stopped, before_stop), Status::Pending);
    region.receipt.publish_sequence = 8;
    region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
    region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = expected.occurrence;
    EXPECT_EQ(completed.receive(Issuer::Stopped, Issuer::Stopped, read()), Status::Rejected);
    EXPECT_FALSE(completed.accepted());
    EXPECT_FALSE(completed.terminal());
    Reception stopped_before_io(expected);
    region.receipt.publish_sequence = 0;
    EXPECT_EQ(stopped_before_io.receive(Issuer::Stopped, Issuer::Stopped, read()), Status::Rejected);
}

TEST(NiuItemWitnessReceptionTest, DiscardsPreStopSampleAndRequiresFreshStoppedPublication) {
    check_sampled_item_reception_contract();
}

template <class Receiver>
concept exposes_rolling_item_reception = requires(typename Receiver::ItemReception& reception) {
    typename Receiver::ItemAcknowledgement;
    typename Receiver::ItemWitnessAcknowledgement;
    reception.acknowledgement();
};

// Exercise production byte transport and reception. No fixture constructs a
// LoadedFirmware, ManagedInvocation lease or payload-reclamation capability.
template <class Receiver>
void check_rolling_item_reception() {
    if constexpr (!exposes_rolling_item_reception<Receiver>) {
        ADD_FAILURE() << "Item reception cannot acknowledge and receive a later device-origin occurrence";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        using Acknowledgement = typename Receiver::ItemAcknowledgement;
        using Region = typename Receiver::ItemWitnessRegion;
        using Request = typename Receiver::ItemWitnessRequest;
        const Request original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        Region region{};
        region.request = region.receipt.request = original;
        const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
            if (offset > sizeof(region) || bytes.size() > sizeof(region) - offset) {
                return NiuHandoffIoStatus::Failed;
            }
            std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
            return NiuHandoffIoStatus::Complete;
        };
        std::vector<std::pair<uint32_t, size_t>> writes;
        const auto write = [&](uint32_t offset, std::span<const std::byte> bytes) {
            if (offset > sizeof(region) || bytes.size() > sizeof(region) - offset) {
                return NiuHandoffIoStatus::Failed;
            }
            writes.emplace_back(offset, bytes.size());
            std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
            return NiuHandoffIoStatus::Complete;
        };
        Reception reception(original);
        EXPECT_FALSE(std::is_default_constructible_v<Acknowledgement>);
        EXPECT_FALSE(std::is_aggregate_v<Acknowledgement>);
        EXPECT_FALSE(reception.acknowledgement());
        region.receipt.publish_sequence = 6;
        region.receipt.state = TT_NIU_ITEM_WITNESS_ISSUED;
        region.receipt.item.issued_occurrence = original.occurrence;
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Pending);
        EXPECT_FALSE(reception.acknowledgement());
        region.receipt.publish_sequence = 8;
        region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
        region.receipt.item.closed_occurrence = original.occurrence;
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
        ASSERT_TRUE(reception.accepted());
        const auto first_observation = *reception.accepted();
        const auto first_ack = reception.acknowledgement();
        ASSERT_TRUE(first_ack);
        EXPECT_EQ(first_ack->wire().closed_sequence, 8u);
        EXPECT_EQ(first_ack->wire().request.occurrence, original.occurrence);
        EXPECT_EQ(first_ack->wire().request.boot_cookie, original.boot_cookie);
        EXPECT_EQ(first_ack->wire().request.invocation_id, original.invocation_id);
        EXPECT_FALSE(reception.acknowledge(*first_ack)) << "An untransported acknowledgement grants no reuse";
        Reception same_shaped(original);
        ASSERT_EQ(same_shaped.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
        EXPECT_FALSE(same_shaped.acknowledge(*first_ack)) << "Identical bytes do not transfer reception ownership";
        ASSERT_EQ(Receiver::write_item_acknowledgement(write, read, *first_ack), NiuHandoffWriteStatus::Published);
        constexpr uint32_t ack_offset = offsetof(Region, acknowledgement);
        ASSERT_EQ(writes.size(), 3u);
        EXPECT_EQ(writes[0], (std::pair<uint32_t, size_t>{ack_offset, sizeof(uint32_t)}));
        EXPECT_EQ(
            writes[1],
            (std::pair<uint32_t, size_t>{
                ack_offset + sizeof(uint32_t),
                sizeof(typename Receiver::ItemWitnessAcknowledgement) - sizeof(uint32_t)}));
        EXPECT_EQ(writes[2], (std::pair<uint32_t, size_t>{ack_offset, sizeof(uint32_t)}));
        EXPECT_EQ(region.request.occurrence, original.occurrence);
        EXPECT_EQ(region.receipt.publish_sequence, 8u);
        EXPECT_EQ(region.receipt.item.closed_occurrence, original.occurrence);
        ASSERT_TRUE(reception.acknowledge(*first_ack));
        EXPECT_FALSE(reception.accepted());
        EXPECT_FALSE(reception.acknowledgement());
        // The still-visible, already acknowledged Closed snapshot is ignored;
        // it cannot report the first completion a second time.
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Pending);
        EXPECT_FALSE(reception.accepted());
        EXPECT_FALSE(reception.acknowledge(*first_ack));
        const uint64_t next = original.occurrence + 1;
        region.receipt.request.occurrence = next;
        region.receipt.publish_sequence = 10;
        region.receipt.state = TT_NIU_ITEM_WITNESS_PENDING;
        region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = 0;
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Pending);
        region.receipt.publish_sequence = 12;
        region.receipt.state = TT_NIU_ITEM_WITNESS_ISSUED;
        region.receipt.item.issued_occurrence = next;
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Pending);
        region.receipt.publish_sequence = 14;
        region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
        region.receipt.item.closed_occurrence = next;
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
        ASSERT_TRUE(reception.accepted());
        EXPECT_EQ(reception.accepted()->wire().receipt.item.closed_occurrence, next);
        const auto second_ack = reception.acknowledgement();
        ASSERT_TRUE(second_ack);
        EXPECT_GT(second_ack->wire().publish_sequence, first_ack->wire().publish_sequence);
        EXPECT_EQ(second_ack->wire().closed_sequence, 14u);
        EXPECT_EQ(second_ack->wire().request.occurrence, next);
        EXPECT_EQ(reception.request().occurrence, original.occurrence);
        EXPECT_EQ(first_observation.wire().receipt.item.closed_occurrence, original.occurrence);
        EXPECT_EQ(first_ack->wire().request.occurrence, original.occurrence);
        EXPECT_FALSE(reception.acknowledge(*first_ack));
        writes.clear();
        ASSERT_EQ(Receiver::write_item_acknowledgement(write, read, *second_ack), NiuHandoffWriteStatus::Published);
        ASSERT_TRUE(reception.acknowledge(*second_ack));
    }
}

TEST(NiuItemWitnessRollingTest, AcknowledgesOnlyAcceptedClosedAndRetainsEachRoundSnapshot) {
    check_rolling_item_reception<NiuHandoffReceiver>();
}

template <class Receiver>
void check_rolling_item_identity_rejections() {
    if constexpr (!exposes_rolling_item_reception<Receiver>) {
        ADD_FAILURE() << "Rolling item reception has no acknowledgement identity and monotonicity checks";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        const Request original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        const auto closed = [&](Region& region, const Request& request, uint32_t sequence) {
            region.request = original;
            region.receipt.request = request;
            region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
            region.receipt.publish_sequence = sequence;
            region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = request.occurrence;
        };
        for (unsigned mutation = 0; mutation != 7; ++mutation) {
            Region region{};
            closed(region, original, 8);
            const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
                std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
                return NiuHandoffIoStatus::Complete;
            };
            Reception reception(original);
            ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
            const auto ack = reception.acknowledgement();
            ASSERT_TRUE(ack);
            const auto write = [&](uint32_t offset, std::span<const std::byte> bytes) {
                std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
                return NiuHandoffIoStatus::Complete;
            };
            ASSERT_EQ(Receiver::write_item_acknowledgement(write, read, *ack), NiuHandoffWriteStatus::Published);
            ASSERT_TRUE(reception.acknowledge(*ack));
            auto next = original;
            ++next.occurrence;
            closed(region, next, 14);
            switch (mutation) {
                case 0: ++region.receipt.request.boot_cookie; break;
                case 1: ++region.receipt.request.invocation_id; break;
                case 2: --region.receipt.request.selector; break;
                case 3:
                    region.receipt.request.occurrence = original.occurrence - 1;
                    region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence =
                        original.occurrence - 1;
                    break;
                case 4: region.receipt.publish_sequence = 4; break;
                case 5: region.receipt.request.occurrence = UINT64_MAX; break;
                case 6: ++region.request.request_generation; break;
            }
            const auto status = reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read));
            EXPECT_TRUE(status == Status::Rejected || status == Status::Invalidated) << mutation;
            EXPECT_FALSE(reception.accepted());
            EXPECT_FALSE(reception.acknowledgement());
            closed(region, next, 14);
            EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), status)
                << "A poisoned transport cannot recover from later valid bytes";
        }
        // A later publication cannot erase an accepted but unacknowledged round.
        Region region{};
        closed(region, original, 8);
        const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
            std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
            return NiuHandoffIoStatus::Complete;
        };
        Reception reception(original);
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
        auto next = original;
        ++next.occurrence;
        closed(region, next, 14);
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
        ASSERT_TRUE(reception.accepted());
        EXPECT_EQ(reception.accepted()->wire().receipt.item.closed_occurrence, original.occurrence);
        EXPECT_EQ(reception.acknowledgement()->wire().closed_sequence, 8u);
        Request foreign = original;
        ++foreign.invocation_id;
        Reception other(foreign);
        EXPECT_FALSE(other.acknowledge(*reception.acknowledgement()));
    }
}

TEST(NiuItemWitnessRollingTest, RejectsForeignRegressingAndExhaustedRoundsWithoutLosingAcceptedItem) {
    check_rolling_item_identity_rejections<NiuHandoffReceiver>();
}

template <class Receiver>
void check_rolling_acknowledgement_transport_failures() {
    if constexpr (!exposes_rolling_item_reception<Receiver>) {
        ADD_FAILURE() << "Rolling acknowledgement requires original reception and successful IO";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        const Request original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        constexpr uint32_t ack_offset = offsetof(Region, acknowledgement);
        for (unsigned failure = 0; failure != 9; ++failure) {
            Region region{};
            region.request = region.receipt.request = original;
            region.receipt.publish_sequence = 8;
            region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
            region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = original.occurrence;
            unsigned writes = 0;
            bool inject = false;
            const unsigned failing_write = failure / 3 + 1;
            const unsigned failure_kind = failure % 3;
            const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
                if (inject && offset == ack_offset && writes == failing_write && failure_kind == 1) {
                    return NiuHandoffIoStatus::Failed;
                }
                std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
                if (inject && offset == ack_offset && writes == failing_write && failure_kind == 2) {
                    bytes.back() ^= std::byte{1};
                }
                return NiuHandoffIoStatus::Complete;
            };
            const auto write = [&](uint32_t offset, std::span<const std::byte> bytes) {
                ++writes;
                if (writes == failing_write && failure_kind == 0) {
                    return NiuHandoffIoStatus::Failed;
                }
                std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
                return NiuHandoffIoStatus::Complete;
            };
            Reception reception(original);
            ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
            const auto ack = reception.acknowledgement();
            ASSERT_TRUE(ack);
            inject = true;
            const auto result = Receiver::write_item_acknowledgement(write, read, *ack);
            EXPECT_EQ(
                result, failure_kind == 2 ? NiuHandoffWriteStatus::ReadbackMismatch : NiuHandoffWriteStatus::IoFailure)
                << failure;
            EXPECT_FALSE(reception.acknowledge(*ack)) << failure;
            ASSERT_TRUE(reception.accepted());
            EXPECT_EQ(reception.accepted()->wire().receipt.item.closed_occurrence, original.occurrence);
            EXPECT_EQ(ack->wire().closed_sequence, 8u);
        }
    }
}

TEST(NiuItemWitnessRollingTest, FailedAcknowledgementIoCannotOpenNextReception) {
    check_rolling_acknowledgement_transport_failures<NiuHandoffReceiver>();
}

template <class Receiver>
void check_rolling_acknowledgement_lifetime() {
    if constexpr (!exposes_rolling_item_reception<Receiver>) {
        ADD_FAILURE() << "Rolling acknowledgement lacks lifetime validation";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        const Request original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        const auto closed = [&] {
            Region region{};
            region.request = region.receipt.request = original;
            region.receipt.publish_sequence = 8;
            region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
            region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = original.occurrence;
            return region;
        };
        Region region = closed();
        unsigned writes = 0;
        const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
            std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
            return NiuHandoffIoStatus::Complete;
        };
        const auto write = [&](uint32_t offset, std::span<const std::byte> bytes) {
            ++writes;
            std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
            return NiuHandoffIoStatus::Complete;
        };
        auto reception = std::make_unique<Reception>(original);
        ASSERT_EQ(reception->receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
        const auto abandoned = reception->acknowledgement();
        ASSERT_TRUE(abandoned);
        reception.reset();
        EXPECT_EQ(Receiver::write_item_acknowledgement(write, read, *abandoned), NiuHandoffWriteStatus::InvalidRequest);
        EXPECT_EQ(writes, 0u);
        // Keeping the immutable old wire alive does not keep the abandoned
        // reception writable or authorize a same-shaped replacement.
        EXPECT_EQ(abandoned->wire().request.occurrence, original.occurrence);
        Reception replacement(original);
        ASSERT_EQ(replacement.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
        EXPECT_FALSE(replacement.acknowledge(*abandoned));
        const auto ack = replacement.acknowledgement();
        ASSERT_TRUE(ack);
        ++region.receipt.publish_sequence;
        EXPECT_EQ(Receiver::write_item_acknowledgement(write, read, *ack), NiuHandoffWriteStatus::ReadbackMismatch);
        EXPECT_EQ(writes, 0u);
        EXPECT_FALSE(replacement.acknowledge(*ack));
        region = closed();
        ASSERT_EQ(Receiver::write_item_acknowledgement(write, read, *ack), NiuHandoffWriteStatus::Published);
        ASSERT_TRUE(replacement.acknowledge(*ack));
        writes = 0;
        EXPECT_EQ(Receiver::write_item_acknowledgement(write, read, *ack), NiuHandoffWriteStatus::InvalidRequest);
        EXPECT_EQ(writes, 0u);
    }
}

TEST(NiuItemWitnessRollingTest, ExpiredChangedAndReplayedAcknowledgementsCannotWriteTheSlot) {
    check_rolling_acknowledgement_lifetime<NiuHandoffReceiver>();
}

template <class Receiver>
void check_rolling_previous_receipt_after_progress() {
    if constexpr (!exposes_rolling_item_reception<Receiver>) {
        ADD_FAILURE() << "Rolling reception has no progress regression checks";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        const Request original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        Region region{};
        region.request = region.receipt.request = original;
        region.receipt.publish_sequence = 8;
        region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
        region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = original.occurrence;
        const auto first = region.receipt;
        const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
            std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
            return NiuHandoffIoStatus::Complete;
        };
        const auto write = [&](uint32_t offset, std::span<const std::byte> bytes) {
            std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
            return NiuHandoffIoStatus::Complete;
        };
        Reception reception(original);
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
        const auto ack = reception.acknowledgement();
        ASSERT_TRUE(ack);
        ASSERT_EQ(Receiver::write_item_acknowledgement(write, read, *ack), NiuHandoffWriteStatus::Published);
        ASSERT_TRUE(reception.acknowledge(*ack));
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Pending);
        region.receipt.publish_sequence = 10;
        region.receipt.state = TT_NIU_ITEM_WITNESS_PENDING;
        ++region.receipt.request.occurrence;
        region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = 0;
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Pending);
        region.receipt = first;
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Rejected);
        EXPECT_FALSE(reception.accepted());
        EXPECT_FALSE(reception.acknowledgement());
    }
}

TEST(NiuItemWitnessRollingTest, RejectsAcknowledgedReceiptAfterNewOccurrenceProgress) {
    check_rolling_previous_receipt_after_progress<NiuHandoffReceiver>();
}

template <class Receiver>
void check_rolling_exact_round_sequences() {
    if constexpr (!exposes_rolling_item_reception<Receiver>) {
        ADD_FAILURE() << "Rolling reception lacks exact round publication sequence checks";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        const Request original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        for (const auto [state, sequence] :
             {std::pair{TT_NIU_ITEM_WITNESS_PENDING, 6u},
              std::pair{TT_NIU_ITEM_WITNESS_ISSUED, 4u},
              std::pair{TT_NIU_ITEM_WITNESS_CLOSED, 4u},
              std::pair{TT_NIU_ITEM_WITNESS_CLOSED, 6u},
              std::pair{TT_NIU_ITEM_WITNESS_CLOSED, 10u},
              std::pair{TT_NIU_ITEM_WITNESS_CLOSED, UINT32_MAX - 1}}) {
            Region region{};
            region.request = region.receipt.request = original;
            region.receipt.publish_sequence = sequence;
            region.receipt.state = state;
            if (state >= TT_NIU_ITEM_WITNESS_ISSUED) {
                region.receipt.item.issued_occurrence = original.occurrence;
            }
            if (state == TT_NIU_ITEM_WITNESS_CLOSED) {
                region.receipt.item.closed_occurrence = original.occurrence;
            }
            const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
                std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
                return NiuHandoffIoStatus::Complete;
            };
            Reception reception(original);
            EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Rejected)
                << state << '/' << sequence;
            EXPECT_FALSE(reception.accepted());
            EXPECT_FALSE(reception.acknowledgement());
        }
        // Host may miss intermediate publications, but the stable state still
        // has the exact sequence allocated by the real arm/issue/close writer.
        for (const auto next_sequence : {10u, 12u, 14u, 16u}) {
            Region region{};
            region.request = region.receipt.request = original;
            region.receipt.publish_sequence = 8;
            region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
            region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = original.occurrence;
            const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
                std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
                return NiuHandoffIoStatus::Complete;
            };
            const auto write = [&](uint32_t offset, std::span<const std::byte> bytes) {
                std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
                return NiuHandoffIoStatus::Complete;
            };
            Reception reception(original);
            ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
            const auto ack = reception.acknowledgement();
            ASSERT_TRUE(ack);
            ASSERT_EQ(Receiver::write_item_acknowledgement(write, read, *ack), NiuHandoffWriteStatus::Published);
            ASSERT_TRUE(reception.acknowledge(*ack));
            ++region.receipt.request.occurrence;
            region.receipt.publish_sequence = next_sequence;
            region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence =
                region.receipt.request.occurrence;
            EXPECT_EQ(
                reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)),
                next_sequence == 14 ? Status::Complete : Status::Rejected)
                << next_sequence;
        }
    }
}

TEST(NiuItemWitnessRollingTest, RejectsImpossiblePublicationSequencesAndAcceptsMissedIntermediateStates) {
    check_rolling_exact_round_sequences<NiuHandoffReceiver>();
}

template <class Receiver>
void check_rolling_acknowledgement_publication_order() {
    if constexpr (!exposes_rolling_item_reception<Receiver>) {
        ADD_FAILURE() << "Rolling acknowledgement lacks odd/body/even publication";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        const Request original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
        constexpr uint32_t ack_offset = offsetof(Region, acknowledgement);
        Region region{};
        region.request = region.receipt.request = original;
        region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
        std::vector<std::tuple<bool, uint32_t, size_t>> events;
        bool exposed_mixed_stable_ack = false;
        const auto read = [&](uint32_t offset, std::span<std::byte> bytes) {
            if (offset >= ack_offset) {
                events.emplace_back(false, offset, bytes.size());
            }
            std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
            return NiuHandoffIoStatus::Complete;
        };
        const auto write = [&](uint32_t offset, std::span<const std::byte> bytes) {
            events.emplace_back(true, offset, bytes.size());
            // A transport need not expose fields inside one write in address
            // order. Expose its body first, as a real fragmented write may.
            if (offset == ack_offset && bytes.size() > sizeof(uint32_t)) {
                const auto old_sequence = region.acknowledgement.publish_sequence;
                std::memcpy(
                    reinterpret_cast<std::byte*>(&region) + offset + sizeof(uint32_t),
                    bytes.data() + sizeof(uint32_t),
                    bytes.size() - sizeof(uint32_t));
                if (old_sequence && !(old_sequence & 1)) {
                    exposed_mixed_stable_ack = true;
                }
                std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), sizeof(uint32_t));
            } else {
                std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
            }
            return NiuHandoffIoStatus::Complete;
        };
        Reception reception(original);
        for (unsigned round = 0; round != 2; ++round) {
            region.receipt.publish_sequence = 6 * (round + 1) + 2;
            region.receipt.request.occurrence = original.occurrence + round;
            region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence =
                region.receipt.request.occurrence;
            ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, Receiver::read_item(read)), Status::Complete);
            const auto ack = reception.acknowledgement();
            ASSERT_TRUE(ack);
            events.clear();
            ASSERT_EQ(Receiver::write_item_acknowledgement(write, read, *ack), NiuHandoffWriteStatus::Published);
            const std::vector<std::tuple<bool, uint32_t, size_t>> expected{
                {true, ack_offset, sizeof(uint32_t)},
                {false, ack_offset, sizeof(uint32_t)},
                {true,
                 ack_offset + sizeof(uint32_t),
                 sizeof(typename Receiver::ItemWitnessAcknowledgement) - sizeof(uint32_t)},
                {false, ack_offset, sizeof(typename Receiver::ItemWitnessAcknowledgement)},
                {true, ack_offset, sizeof(uint32_t)},
                {false, ack_offset, sizeof(typename Receiver::ItemWitnessAcknowledgement)}};
            EXPECT_EQ(events, expected) << round;
            ASSERT_TRUE(reception.acknowledge(*ack));
        }
        EXPECT_FALSE(exposed_mixed_stable_ack) << "A stable old ACK must be invalidated before its body changes";
    }
}

TEST(NiuItemWitnessRollingTest, PublishesOddBeforeAnyAcknowledgementBodyAndEvenAfterVerifiedBody) {
    check_rolling_acknowledgement_publication_order<NiuHandoffReceiver>();
}

template <class Receiver>
concept exposes_item_series_reception = requires(const typename Receiver::ItemReception& reception) {
    typename Receiver::ItemReception::SeriesTerminal;
    reception.terminal()->final_count();
    reception.closed_prefix();
    reception.acknowledged_prefix();
};

// This fixture supplies transport bytes only. Production must prove source
// termination and contiguous reception independently from real ACK transport.
template <class Receiver>
struct ItemSeriesMailbox {
    using Region = typename Receiver::ItemWitnessRegion;
    using Request = typename Receiver::ItemWitnessRequest;
    Region region{};
    const Request original{TT_NIU_ITEM_WITNESS_VERSION, 7, 3, 5, 11, 13, 1, 2, 1, 15, 2, 2, 0};
    ItemSeriesMailbox() { region.request = region.receipt.request = original; }
    NiuHandoffIoStatus read(uint32_t offset, std::span<std::byte> bytes) const {
        if (offset > sizeof(region) || bytes.size() > sizeof(region) - offset) {
            return NiuHandoffIoStatus::Failed;
        }
        std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&region) + offset, bytes.size());
        return NiuHandoffIoStatus::Complete;
    }
    NiuItemWitnessReadResult observe() const {
        return Receiver::read_item([&](uint32_t offset, std::span<std::byte> bytes) { return read(offset, bytes); });
    }
    void open() {
        region.receipt = {};
        region.receipt.request = original;
        region.receipt.publish_sequence = 2;
        region.receipt.state = 4;
    }
    void closed(uint64_t occurrence) {
        region.receipt = {};
        region.receipt.request = original;
        region.receipt.request.occurrence = occurrence;
        region.receipt.publish_sequence = 6 * occurrence + 2;
        region.receipt.state = TT_NIU_ITEM_WITNESS_CLOSED;
        region.receipt.item.issued_occurrence = region.receipt.item.closed_occurrence = occurrence;
    }
    void pending() {
        closed(1);
        region.receipt.publish_sequence = 4;
        region.receipt.state = TT_NIU_ITEM_WITNESS_PENDING;
        region.receipt.item = {};
    }
    void end(uint64_t count) {
        region.receipt = {};
        region.receipt.request = original;
        region.receipt.publish_sequence = 6 * count + 4;
        region.receipt.state = 5;
        region.receipt.series.final_count = count;
        region.receipt.series.closed_sequence = count ? 6 * count + 2 : 0;
        region.receipt.series.acknowledgement_sequence = 2 * count;
    }
    bool acknowledge(typename Receiver::ItemReception& reception) {
        const auto ack = reception.acknowledgement();
        if (!ack) {
            return false;
        }
        const auto status = Receiver::write_item_acknowledgement(
            [&](uint32_t offset, std::span<const std::byte> bytes) {
                if (offset > sizeof(region) || bytes.size() > sizeof(region) - offset) {
                    return NiuHandoffIoStatus::Failed;
                }
                std::memcpy(reinterpret_cast<std::byte*>(&region) + offset, bytes.data(), bytes.size());
                return NiuHandoffIoStatus::Complete;
            },
            [&](uint32_t offset, std::span<std::byte> bytes) { return read(offset, bytes); },
            *ack);
        return status == NiuHandoffWriteStatus::Published && reception.acknowledge(*ack);
    }
};

template <class Receiver>
concept exposes_item_terminal_reception =
    requires(const typename Receiver::ItemReception& reception) { reception.terminal(); };

// These exercise final-round reception only. The original ManagedInvocation
// still owns real stop, boot, token, allocation and device-epoch validation.
template <class Receiver>
void check_item_terminal_freshness() {
    using Reception = typename Receiver::ItemReception;
    using Issuer = typename Reception::IssuerState;
    using Status = typename Reception::Status;
    ItemSeriesMailbox<Receiver> mailbox;
    Reception reception(mailbox.original);
    mailbox.closed(1);
    ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Complete);
    const auto item = *reception.accepted();
    EXPECT_FALSE(reception.terminal());
    ASSERT_TRUE(mailbox.acknowledge(reception));
    mailbox.end(1);
    const auto before_stop = mailbox.observe();
    EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Stopped, before_stop), Status::Pending);
    EXPECT_FALSE(reception.terminal()) << "A pre-stop End is not a fresh stopped-writer publication";
    ASSERT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::Finished);
    ASSERT_TRUE(reception.terminal());
    EXPECT_EQ(reception.terminal()->observation().wire().receipt.publish_sequence, 10u);
    EXPECT_EQ(reception.terminal()->final_count(), 1u);
    EXPECT_EQ(item.wire().receipt.item.closed_occurrence, 1u);

    ItemSeriesMailbox<Receiver> missing_end;
    Reception incomplete(missing_end.original);
    missing_end.closed(1);
    ASSERT_EQ(incomplete.receive(Issuer::Issued, Issuer::Issued, missing_end.observe()), Status::Complete);
    const auto old = *incomplete.accepted();
    EXPECT_EQ(incomplete.receive(Issuer::Stopped, Issuer::Stopped, missing_end.observe()), Status::Rejected);
    EXPECT_FALSE(incomplete.terminal()) << "Cached physical completion cannot hide the missing source End";
    EXPECT_EQ(old.wire().receipt.item.closed_occurrence, 1u);
}

TEST(NiuItemWitnessTerminalTest, RequiresFreshStoppedObservationOfTheActualFinalRound) {
    check_item_terminal_freshness<NiuHandoffReceiver>();
}

template <class Receiver>
void check_item_terminal_acknowledgement_order() {
    using Reception = typename Receiver::ItemReception;
    using Issuer = typename Reception::IssuerState;
    using Status = typename Reception::Status;
    for (const bool observe_end_online : {false, true}) {
        ItemSeriesMailbox<Receiver> mailbox;
        Reception reception(mailbox.original);
        mailbox.closed(1);
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Complete);
        const auto accepted = *reception.accepted();
        const auto ack = reception.acknowledgement();
        ASSERT_TRUE(ack);
        ASSERT_TRUE(mailbox.acknowledge(reception));
        EXPECT_FALSE(reception.accepted());
        EXPECT_FALSE(reception.terminal()) << "ACK alone grants no terminal or stopped proof";
        mailbox.end(1);
        if (observe_end_online) {
            EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Pending);
            EXPECT_FALSE(reception.terminal());
        }
        ASSERT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::Finished);
        ASSERT_TRUE(reception.terminal());
        EXPECT_EQ(reception.terminal()->final_count(), 1u);
        EXPECT_EQ(reception.terminal()->observation().wire().receipt.publish_sequence, 10u);
        EXPECT_FALSE(reception.accepted()) << "The source End cannot redeliver an ACKed item";
        EXPECT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::Finished);
        EXPECT_FALSE(reception.acknowledge(*ack));
        EXPECT_EQ(accepted.wire().receipt.item.closed_occurrence, 1u);
        EXPECT_EQ(ack->wire().request.occurrence, 1u);
    }
}

TEST(NiuItemWitnessTerminalTest, PreservesFinalRoundAcrossAcknowledgementWithoutDuplicateCompletion) {
    check_item_terminal_acknowledgement_order<NiuHandoffReceiver>();
}

template <class Receiver>
void check_item_terminal_latest_round() {
    using Reception = typename Receiver::ItemReception;
    using Issuer = typename Reception::IssuerState;
    using Status = typename Reception::Status;
    for (unsigned outcome = 0; outcome != 6; ++outcome) {
        ItemSeriesMailbox<Receiver> mailbox;
        Reception reception(mailbox.original);
        mailbox.closed(1);
        const auto first = mailbox.region.receipt;
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Complete);
        const auto old = *reception.accepted();
        const auto ack = reception.acknowledgement();
        ASSERT_TRUE(ack);
        ASSERT_TRUE(mailbox.acknowledge(reception));
        mailbox.closed(2);
        mailbox.region.receipt.publish_sequence = 10;
        mailbox.region.receipt.state = TT_NIU_ITEM_WITNESS_PENDING;
        mailbox.region.receipt.item = {};
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Pending);
        switch (outcome) {
            case 0: break;
            case 1:
                mailbox.region.receipt.publish_sequence = 12;
                mailbox.region.receipt.state = TT_NIU_ITEM_WITNESS_ISSUED;
                mailbox.region.receipt.item.issued_occurrence = 2;
                break;
            case 2: mailbox.region.receipt = first; break;
            default:
                mailbox.closed(2);
                if (outcome == 3) {
                    ++mailbox.region.receipt.request.invocation_id;
                }
                if (outcome == 4) {
                    mailbox.region.receipt.item.issued_occurrence = 0;
                }
                break;
        }
        const auto status = reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe());
        EXPECT_TRUE(status == Status::Rejected || status == Status::Invalidated) << outcome;
        EXPECT_FALSE(reception.terminal()) << "Even final Closed(2) must not stand in for source End(2)";
        EXPECT_EQ(old.wire().receipt.item.closed_occurrence, 1u);
        EXPECT_EQ(ack->wire().request.occurrence, 1u);
    }
}

TEST(NiuItemWitnessTerminalTest, RejectsIncompleteForeignAndRegressingFinalRoundsAfterEarlierAcknowledgement) {
    check_item_terminal_latest_round<NiuHandoffReceiver>();
}

template <class Receiver>
void check_zero_source_series() {
    if constexpr (!exposes_item_series_reception<Receiver>) {
        ADD_FAILURE() << "Receiver cannot distinguish explicit End(0) from an absent item";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        ItemSeriesMailbox<Receiver> mailbox;
        Reception reception(mailbox.original);
        mailbox.open();
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Pending);
        mailbox.end(0);
        EXPECT_EQ(reception.receive(Issuer::Reserved, Issuer::Reserved, mailbox.observe()), Status::Pending);
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Pending);
        EXPECT_FALSE(reception.terminal());
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Stopped, mailbox.observe()), Status::Pending);
        EXPECT_FALSE(reception.terminal());
        ASSERT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::NotExecuted);
        ASSERT_TRUE(reception.terminal());
        EXPECT_EQ(reception.terminal()->final_count(), 0u);
        EXPECT_FALSE(reception.accepted());
        EXPECT_FALSE(reception.acknowledgement());
        EXPECT_EQ(reception.closed_prefix(), 0u);
        EXPECT_EQ(reception.acknowledged_prefix(), 0u);
    }
}
TEST(NiuItemWitnessSeriesTest, ZeroExecutionRequiresExplicitEndAndFreshStoppedRead) {
    check_zero_source_series<NiuHandoffReceiver>();
}

template <class Receiver>
void check_source_series_missing_end() {
    if constexpr (!exposes_item_series_reception<Receiver>) {
        ADD_FAILURE() << "Receiver still infers series termination from stopped Closed";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        for (unsigned sample = 0; sample != 5; ++sample) {
            ItemSeriesMailbox<Receiver> mailbox;
            Reception reception(mailbox.original);
            if (sample == 1) {
                mailbox.open();
            }
            if (sample == 2) {
                mailbox.pending();
            }
            if (sample >= 3) {
                mailbox.closed(1);
                if (sample == 3) {
                    mailbox.region.receipt.publish_sequence = 6;
                    mailbox.region.receipt.state = TT_NIU_ITEM_WITNESS_ISSUED;
                    mailbox.region.receipt.item.closed_occurrence = 0;
                }
            }
            EXPECT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::Rejected)
                << "Stopped sample " << sample << " cannot manufacture End";
            EXPECT_FALSE(reception.terminal());
        }
        ItemSeriesMailbox<Receiver> mailbox;
        Reception cached(mailbox.original);
        mailbox.closed(1);
        ASSERT_EQ(cached.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Complete);
        EXPECT_EQ(cached.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::Rejected);
        EXPECT_FALSE(cached.terminal());
    }
}
TEST(NiuItemWitnessSeriesTest, StoppedAbsentOpenPendingIssuedAndClosedDoNotProveEnd) {
    check_source_series_missing_end<NiuHandoffReceiver>();
}

template <class Receiver>
void check_source_series_ack_frontier() {
    if constexpr (!exposes_item_series_reception<Receiver>) {
        ADD_FAILURE() << "Receiver lacks independent Closed and transported ACK frontiers";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        ItemSeriesMailbox<Receiver> mailbox;
        Reception reception(mailbox.original);
        mailbox.closed(1);
        ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Complete);
        const auto snapshot = *reception.accepted();
        EXPECT_EQ(reception.closed_prefix(), 1u);
        EXPECT_EQ(reception.acknowledged_prefix(), 0u);
        ASSERT_TRUE(mailbox.acknowledge(reception));
        EXPECT_EQ(reception.acknowledged_prefix(), 1u);
        mailbox.end(1);
        EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Pending);
        EXPECT_FALSE(reception.terminal());
        ASSERT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::Finished);
        ASSERT_TRUE(reception.terminal());
        EXPECT_EQ(reception.terminal()->final_count(), 1u);
        EXPECT_EQ(snapshot.wire().receipt.item.closed_occurrence, 1u);
        EXPECT_FALSE(reception.accepted());
        EXPECT_FALSE(reception.acknowledgement());
        ItemSeriesMailbox<Receiver> unacknowledged;
        Reception missing(unacknowledged.original);
        unacknowledged.closed(1);
        ASSERT_EQ(missing.receive(Issuer::Issued, Issuer::Issued, unacknowledged.observe()), Status::Complete);
        const auto ack = missing.acknowledgement();
        ASSERT_TRUE(ack);
        EXPECT_FALSE(missing.acknowledge(*ack));
        unacknowledged.end(1);
        EXPECT_EQ(missing.receive(Issuer::Stopped, Issuer::Stopped, unacknowledged.observe()), Status::Rejected);
        EXPECT_FALSE(missing.terminal());
    }
}
TEST(NiuItemWitnessSeriesTest, EndRequiresReceivedAndTransportedAcknowledgedPrefix) {
    check_source_series_ack_frontier<NiuHandoffReceiver>();
}

template <class Receiver>
void check_contiguous_source_series() {
    if constexpr (!exposes_item_series_reception<Receiver>) {
        ADD_FAILURE() << "Receiver does not require source-generated contiguous ordinals";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        ItemSeriesMailbox<Receiver> mailbox;
        Reception reception(mailbox.original);
        for (uint64_t occurrence = 1; occurrence != 3; ++occurrence) {
            mailbox.closed(occurrence);
            ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Complete);
            ASSERT_TRUE(mailbox.acknowledge(reception));
        }
        mailbox.end(2);
        ASSERT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::Finished);
        ASSERT_TRUE(reception.terminal());
        EXPECT_EQ(reception.terminal()->final_count(), 2u);
        EXPECT_EQ(reception.closed_prefix(), 2u);
        EXPECT_EQ(reception.acknowledged_prefix(), 2u);
        ItemSeriesMailbox<Receiver> skipped;
        Reception gap(skipped.original);
        skipped.closed(1);
        ASSERT_EQ(gap.receive(Issuer::Issued, Issuer::Issued, skipped.observe()), Status::Complete);
        ASSERT_TRUE(skipped.acknowledge(gap));
        skipped.closed(3);
        EXPECT_EQ(gap.receive(Issuer::Issued, Issuer::Issued, skipped.observe()), Status::Rejected);
        EXPECT_FALSE(gap.terminal());
    }
}
TEST(NiuItemWitnessSeriesTest, AcceptsTwoAcknowledgedRoundsAndRejectsSkippedOccurrence) {
    check_contiguous_source_series<NiuHandoffReceiver>();
}

template <class Receiver>
void check_source_series_end_identity() {
    if constexpr (!exposes_item_series_reception<Receiver>) {
        ADD_FAILURE() << "Receiver has no explicit End identity and frontier validation";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        for (unsigned field = 0; field != 10; ++field) {
            ItemSeriesMailbox<Receiver> mailbox;
            Reception reception(mailbox.original);
            mailbox.closed(1);
            ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Complete);
            ASSERT_TRUE(mailbox.acknowledge(reception));
            mailbox.end(1);
            switch (field) {
                case 0: mailbox.region.receipt.series.final_count = 0; break;
                case 1: mailbox.region.receipt.series.final_count = 2; break;
                case 2: mailbox.region.receipt.series.closed_sequence = 6; break;
                case 3: mailbox.region.receipt.series.acknowledgement_sequence = 0; break;
                case 4: mailbox.region.receipt.series.reserved = 1; break;
                case 5: mailbox.region.receipt.publish_sequence = 12; break;
                case 6: ++mailbox.region.receipt.request.invocation_id; break;
                case 7: ++mailbox.region.request.boot_cookie; break;
                case 8: mailbox.region.receipt.request.occurrence = 2; break;
                case 9: mailbox.region.receipt.series.final_count = UINT64_MAX; break;
            }
            const auto status = reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe());
            EXPECT_TRUE(status == Status::Rejected || status == Status::Invalidated) << "field " << field;
            EXPECT_FALSE(reception.terminal());
        }
    }
}
TEST(NiuItemWitnessSeriesTest, RejectsForeignMalformedEndAndUnprovedFrontier) {
    check_source_series_end_identity<NiuHandoffReceiver>();
}

template <class Receiver>
void check_source_series_no_progress_after_end() {
    if constexpr (!exposes_item_series_reception<Receiver>) {
        ADD_FAILURE() << "Receiver cannot reject same-sequence End/Pending substitution";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        for (bool end_first : {false, true}) {
            ItemSeriesMailbox<Receiver> mailbox;
            Reception reception(mailbox.original);
            if (end_first) {
                mailbox.end(0);
            } else {
                mailbox.pending();
            }
            ASSERT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Pending);
            if (end_first) {
                mailbox.pending();
            } else {
                mailbox.end(0);
            }
            EXPECT_EQ(reception.receive(Issuer::Issued, Issuer::Issued, mailbox.observe()), Status::Rejected);
            EXPECT_FALSE(reception.terminal());
        }
    }
}
TEST(NiuItemWitnessSeriesTest, EndAndNextPendingCannotReplaceEachOtherAtTheSameSequence) {
    check_source_series_no_progress_after_end<NiuHandoffReceiver>();
}

template <class Receiver>
void check_source_series_zero_frontier() {
    if constexpr (!exposes_item_series_reception<Receiver>) {
        ADD_FAILURE() << "Receiver does not validate empty source-series and ACK frontiers";
    } else {
        using Reception = typename Receiver::ItemReception;
        using Issuer = typename Reception::IssuerState;
        using Status = typename Reception::Status;
        for (unsigned field = 0; field != 5; ++field) {
            ItemSeriesMailbox<Receiver> mailbox;
            Reception reception(mailbox.original);
            mailbox.end(0);
            switch (field) {
                case 0: mailbox.region.receipt.series.closed_sequence = 8; break;
                case 1: mailbox.region.receipt.series.acknowledgement_sequence = 2; break;
                case 2: mailbox.region.receipt.series.reserved = 1; break;
                case 3: mailbox.region.receipt.publish_sequence = 2; break;
                case 4: mailbox.region.acknowledgement.publish_sequence = 2; break;
            }
            EXPECT_EQ(reception.receive(Issuer::Stopped, Issuer::Stopped, mailbox.observe()), Status::Rejected);
            EXPECT_FALSE(reception.terminal());
        }
    }
}
TEST(NiuItemWitnessSeriesTest, EndZeroRequiresBothWireAndReceivedFrontiersToBeEmpty) {
    check_source_series_zero_frontier<NiuHandoffReceiver>();
}

class NiuHandoffClusterTransportTest : public ::testing::Test {
protected:
    void SetUp() override {
        MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1).value());
        descriptor.set_device_profiler_mode(DeviceProfilerMode::Disabled);
        environment = std::make_unique<MetalEnv>(std::move(descriptor));
        auto& implementation = MetalEnvAccessor(*environment).impl();
        cluster = &implementation.get_cluster();
        hal = &implementation.get_hal();
        ASSERT_EQ(cluster->get_target_device_type(), tt::TargetDevice::Mock);
        ASSERT_EQ(cluster->all_chip_ids().size(), 1u);
        chip = *cluster->all_chip_ids().begin();
        const auto& soc = cluster->get_soc_desc(chip);
        physical = soc.get_physical_tensix_core_from_logical({0, 0});
        const auto translated_coord = soc.translate_coord_to(
            physical, tt::CoordSystem::NOC0, tt::CoordSystem::TRANSLATED);
        translated = {translated_coord.x, translated_coord.y};
        base = hal->get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::NIU_HANDOFF);
        ASSERT_NE(base, 0u);
        ASSERT_EQ(hal->get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::NIU_HANDOFF),
                  sizeof(tt_niu_handoff_region));
        cluster_read = [&](const Cluster& actual, void* destination, uint32_t bytes,
                           tt_cxy_pair core, uint64_t address) {
            EXPECT_EQ(&actual, cluster);
            calls.emplace_back(core.chip, core.x, core.y, address, bytes);
            if (fail_read == calls.size() - 1) throw std::runtime_error("mailbox transport failed");
            if (address < base || address - base > sizeof(mailbox.wire) ||
                bytes > sizeof(mailbox.wire) - (address - base))
                throw std::runtime_error("mailbox read outside the actual HAL region");
            std::array<uint32_t, sizeof(mailbox.wire) / sizeof(uint32_t)> words;
            std::memcpy(words.data(), &mailbox.wire, sizeof(mailbox.wire));
            auto* output = static_cast<std::byte*>(destination);
            for (uint32_t byte = 0; byte != bytes; ++byte) {
                const size_t position = address - base + byte;
                output[byte] = std::byte((words[position / 4] >> ((position % 4) * 8)) & 0xff);
            }
        };
        cluster_write = [&](const Cluster& actual, const void* source, uint32_t bytes,
                            tt_cxy_pair core, uint64_t address) {
            EXPECT_EQ(&actual, cluster);
            EXPECT_EQ(core.chip, chip);
            EXPECT_EQ(core.x, translated.x);
            EXPECT_EQ(core.y, translated.y);
            writes.emplace_back(address - base, bytes);
            if (address < base || address - base > sizeof(mailbox.wire.request) ||
                bytes > sizeof(mailbox.wire.request) - (address - base))
                throw std::runtime_error("mailbox write outside request region");
            std::memcpy(reinterpret_cast<std::byte*>(&mailbox.wire.request) + address - base, source, bytes);
        };
        cluster_barrier = [&](const Cluster& actual, ChipId actual_chip) {
            EXPECT_EQ(&actual, cluster);
            EXPECT_EQ(actual_chip, chip);
            ++barriers;
        };
    }

    void TearDown() override {
        cluster_read = {};
        cluster_write = {};
        cluster_barrier = {};
        environment.reset();
    }

    NiuHandoffReadResult read() { return NiuHandoffReceiver::read(*cluster, *hal, chip, physical); }
    std::unique_ptr<MetalEnv> environment;
    const Cluster* cluster = nullptr;
    const Hal* hal = nullptr;
    ChipId chip = 0;
    CoreCoord physical;
    CoreCoord translated;
    uint64_t base = 0;
    Mailbox mailbox;
    using ReadCall = std::tuple<ChipId, size_t, size_t, uint64_t, uint32_t>;
    std::vector<ReadCall> calls;
    std::optional<size_t> fail_read;
    std::vector<std::pair<uint32_t, size_t>> writes;
    size_t barriers = 0;
};

TEST_F(NiuHandoffClusterTransportTest, PublishesRequestToOriginalTranslatedWorkerWithOrderedWrites) {
    auto request = mailbox.wire.request;
    ++request.request_generation;
    ++request.invocation_id_lo;
    const auto receipt = mailbox.wire.receipt;
    EXPECT_EQ(NiuHandoffReceiver::write_request(*cluster, *hal, chip, physical, request),
              NiuHandoffWriteStatus::Published);
    EXPECT_EQ(std::memcmp(&mailbox.wire.request, &request, sizeof(request)), 0);
    EXPECT_EQ(std::memcmp(&mailbox.wire.receipt, &receipt, sizeof(receipt)), 0);
    EXPECT_EQ(writes, (std::vector<std::pair<uint32_t, size_t>>{{4, 4}, {0, 64}, {4, 4}}));
    EXPECT_EQ(barriers, 3u);
}

class RequestTransport {
public:
    std::array<std::byte, sizeof(tt_niu_handoff_request)> bytes{};
    std::vector<std::pair<bool, std::pair<uint32_t, size_t>>> operations;
    std::optional<size_t> fail;
    std::optional<size_t> corrupt;

    NiuHandoffWriteStatus publish(const tt_niu_handoff_request& request) {
        return NiuHandoffReceiver::write_request(
            [&](uint32_t offset, std::span<const std::byte> source) {
                operations.push_back({true, {offset, source.size()}});
                if (fail == operations.size() - 1) return NiuHandoffIoStatus::Failed;
                EXPECT_LE(offset + source.size(), bytes.size());
                std::copy(source.begin(), source.end(), bytes.begin() + offset);
                return NiuHandoffIoStatus::Complete;
            },
            [&](uint32_t offset, std::span<std::byte> destination) {
                operations.push_back({false, {offset, destination.size()}});
                if (fail == operations.size() - 1) return NiuHandoffIoStatus::Failed;
                EXPECT_LE(offset + destination.size(), bytes.size());
                std::copy_n(bytes.begin() + offset, destination.size(), destination.begin());
                if (corrupt == operations.size() - 1) destination.front() ^= std::byte{1};
                return NiuHandoffIoStatus::Complete;
            }, request);
    }
};

Mailbox initial_mailbox() {
    Mailbox mailbox;
    mailbox.wire = {};
    auto& header = mailbox.wire.receipt.header;
    header.publish_sequence = 2;
    header.abi_version = TT_NIU_HANDOFF_VERSION;
    header.status = TT_NIU_HANDOFF_BOOT_READY;
    header.control_valid_noc_mask = 3;
    header.completed_processor_mask = TT_NIU_HANDOFF_PROCESSOR_MASK;
    mailbox.wire.receipt.control = {3, 15, 15, 1};
    return mailbox;
}

TEST(NiuHandoffInitialRequestTest, KeepsHostIdentityAndActualSingleGoPayloadCoverage) {
    auto mailbox = initial_mailbox();
    const auto observed = mailbox.read();
    ASSERT_TRUE(observed.observation);
    for (const bool payload : {false, true}) {
        const auto expected = NiuHandoffReceiver::initial_execution(
            *observed.observation, UINT64_C(0x123456789), UINT64_C(0xabcdef012), payload);
        ASSERT_TRUE(expected);
        EXPECT_EQ(expected->request.boot_cookie_hi, 1u);
        EXPECT_EQ(expected->request.boot_cookie_lo, UINT32_C(0x23456789));
        EXPECT_EQ(expected->request.invocation_id_hi, 10u);
        EXPECT_EQ(expected->request.invocation_id_lo, UINT32_C(0xbcdef012));
        EXPECT_EQ(expected->request.challenge_hi, expected->request.invocation_id_hi);
        EXPECT_EQ(expected->request.challenge_lo, expected->request.invocation_id_lo);
        EXPECT_EQ(expected->request.first_go_sequence, 1u);
        EXPECT_EQ(expected->request.final_go_count, 1u);
        EXPECT_EQ(expected->request.group_generation, 1u);
        EXPECT_EQ(expected->request.request_generation, 1u);
        EXPECT_EQ(expected->completed_stage_count, 1u);
        EXPECT_EQ(expected->source_valid_noc_mask, payload ? 3u : 0u);
        EXPECT_EQ(NiuHandoffReceiver::check(*observed.observation, *expected), NiuHandoffCheck::RequestMismatch);
        RequestTransport transport;
        EXPECT_EQ(transport.publish(expected->request), NiuHandoffWriteStatus::Published);
    }
}

TEST(NiuHandoffInitialRequestTest, RejectsZeroHostIdentitiesAndAnyPreviousRequestOrGo) {
    auto mailbox = initial_mailbox();
    auto observed = mailbox.read();
    ASSERT_TRUE(observed.observation);
    EXPECT_FALSE(NiuHandoffReceiver::initial_execution(*observed.observation, 0, 3, true));
    EXPECT_FALSE(NiuHandoffReceiver::initial_execution(*observed.observation, 2, 0, true));
    for (unsigned field = 0; field != 11; ++field) {
        mailbox = initial_mailbox();
        switch (field) {
            case 0: mailbox.wire.request.request_generation = 1; break;
            case 1: mailbox.wire.request.invocation_id_lo = 1; break;
            case 2: mailbox.wire.receipt.consumed_go_sequence = 1; break;
            case 3: mailbox.wire.receipt.header.completed_stage_count = 1; break;
            case 4: mailbox.wire.receipt.header.completed_go_count = 1; break;
            case 5: mailbox.wire.receipt.header.challenge_lo = 1; break;
            case 6: mailbox.wire.receipt.header.status = TT_NIU_HANDOFF_PENDING; break;
            case 7: mailbox.wire.receipt.header.source_valid_noc_mask = 3; break;
            case 8: mailbox.wire.receipt.header.control_valid_noc_mask = 1; break;
            case 9: mailbox.wire.receipt.control_expected[0][0] = 1; break;
            case 10: mailbox.wire.receipt.header.completed_processor_mask = 0; break;
        }
        observed = mailbox.read();
        ASSERT_TRUE(observed.observation);
        EXPECT_FALSE(NiuHandoffReceiver::initial_execution(*observed.observation, 2, 3, true)) << field;
    }
}

TEST(NiuHandoffRequestTest, InvalidatesOldGenerationAndVerifiesBodyBeforePublishingNewGeneration) {
    RequestTransport transport;
    transport.bytes.fill(std::byte{0xff});
    const auto request = Mailbox().wire.request;
    ASSERT_EQ(transport.publish(request), NiuHandoffWriteStatus::Published);
    EXPECT_EQ(transport.operations, (decltype(transport.operations){
        {true, {4, 4}}, {false, {4, 4}}, {true, {0, 64}}, {false, {0, 64}},
        {true, {4, 4}}, {false, {0, 64}}}));
    std::array<uint32_t, sizeof(request) / 4> words;
    std::memcpy(words.data(), &request, sizeof(request));
    for (size_t index = 0; index != transport.bytes.size(); ++index)
        EXPECT_EQ(transport.bytes[index], std::byte((words[index / 4] >> (8 * (index % 4))) & 0xff));
}

TEST(NiuHandoffRequestTest, EveryIoFailureStopsPublicationWithoutRetry) {
    for (size_t operation = 0; operation != 6; ++operation) {
        RequestTransport transport;
        transport.fail = operation;
        EXPECT_EQ(transport.publish(Mailbox().wire.request), NiuHandoffWriteStatus::IoFailure);
        EXPECT_EQ(transport.operations.size(), operation + 1);
        if (operation < 4) EXPECT_EQ(transport.bytes[4], std::byte{0});
    }
}

TEST(NiuHandoffRequestTest, EveryReadbackMismatchStopsPublicationWithoutRetry) {
    for (size_t operation : {1u, 3u, 5u}) {
        RequestTransport transport;
        transport.corrupt = operation;
        EXPECT_EQ(transport.publish(Mailbox().wire.request), NiuHandoffWriteStatus::ReadbackMismatch);
        EXPECT_EQ(transport.operations.size(), operation + 1);
    }
}

TEST(NiuHandoffRequestTest, InvalidRequestDoesNotModifyMailbox) {
    RequestTransport transport;
    auto request = Mailbox().wire.request;
    request.request_generation = 0;
    EXPECT_EQ(transport.publish(request), NiuHandoffWriteStatus::InvalidRequest);
    EXPECT_TRUE(transport.operations.empty());
}

TEST_F(NiuHandoffClusterTransportTest, ReadsOriginalChipTranslatedWorkerAndHalMailbox) {
    const auto result = read();
    ASSERT_EQ(result.status, NiuHandoffReadStatus::Observed);
    ASSERT_TRUE(result.observation);
    EXPECT_EQ(result.observation->wire().request.invocation_id_lo, 23u);
    EXPECT_EQ(calls, (std::vector<ReadCall>{
                         {chip, translated.x, translated.y, base + TT_NIU_HANDOFF_RECEIPT_OFFSET, 4},
                         {chip, translated.x, translated.y, base, sizeof(tt_niu_handoff_region)},
                         {chip, translated.x, translated.y, base + TT_NIU_HANDOFF_RECEIPT_OFFSET, 4}}));
}

TEST_F(NiuHandoffClusterTransportTest, ActualReadFailureCannotReturnAnObservationOrContinueIo) {
    for (size_t index = 0; index != 3; ++index) {
        SCOPED_TRACE(index);
        calls.clear();
        fail_read = index;
        EXPECT_THROW((void)read(), std::runtime_error);
        EXPECT_EQ(calls.size(), index + 1);
    }
}

TEST_F(NiuHandoffClusterTransportTest, RejectsNonWorkerBeforeClusterIo) {
    const auto pcie = cluster->get_soc_desc(chip).get_cores(tt::CoreType::PCIE, tt::CoordSystem::NOC0);
    ASSERT_FALSE(pcie.empty());
    physical = {pcie.front().x, pcie.front().y};
    EXPECT_THROW((void)read(), std::exception);
    EXPECT_TRUE(calls.empty());
}

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
