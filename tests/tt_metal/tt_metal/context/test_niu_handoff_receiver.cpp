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
        request.abi_version = 1;
        request.request_generation = 7;
        request.boot_cookie = UINT64_C(0x123456789abcdef0);
        request.invocation_id = UINT64_C(0x223456789abcdef0);
        request.item = UINT64_C(0x323456789abcdef0);
        request.generation = UINT64_C(0x423456789abcdef0);
        request.occurrence = UINT64_C(0x523456789abcdef0);
        request.leaf = 3;
        request.noc = 1;
        request.selector = 15;
        request.issuer = 2;
        request.completion = 2;
        region.receipt.request = request;
        region.receipt.publish_sequence = 2;
        region.receipt.state = 2;
        region.receipt.issued_occurrence = request.occurrence;
        region.receipt.closed_occurrence = request.occurrence;
        bool change_after_body = false;
        auto read = [&] {
            return Receiver::read_item([&](uint32_t offset, std::span<std::byte> destination) {
                if (offset > sizeof(region) || destination.size() > sizeof(region) - offset)
                    return NiuHandoffIoStatus::Failed;
                std::memcpy(destination.data(), reinterpret_cast<const std::byte*>(&region) + offset,
                            destination.size());
                if (change_after_body && destination.size() == sizeof(region))
                    region.receipt.publish_sequence += 2;
                return NiuHandoffIoStatus::Complete;
            });
        };
        const Request expected = request;
        auto observed = read();
        ASSERT_TRUE(observed.observation);
        EXPECT_TRUE(Receiver::closed_item(*observed.observation, expected));
        // Completion before actual issue, zero counters, or copied identity
        // must never mint the original owner's online completion fact.
        region.receipt.issued_occurrence = 0;
        observed = read();
        ASSERT_TRUE(observed.observation);
        EXPECT_FALSE(Receiver::closed_item(*observed.observation, expected));
        region.receipt.issued_occurrence = expected.occurrence;
        --region.receipt.closed_occurrence;
        observed = read();
        ASSERT_TRUE(observed.observation);
        EXPECT_FALSE(Receiver::closed_item(*observed.observation, expected));
        region.receipt.closed_occurrence = expected.occurrence;
        for (auto member : {&Request::boot_cookie, &Request::invocation_id, &Request::item,
                            &Request::generation, &Request::occurrence}) {
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
        region.receipt.publish_sequence = 4;
        change_after_body = true;
        EXPECT_EQ(read().status, NiuHandoffReadStatus::Changed);
        EXPECT_EQ(Receiver::read_item([](uint32_t, std::span<std::byte>) {
                      return NiuHandoffIoStatus::Failed;
                  }).status, NiuHandoffReadStatus::IoFailure);
    }
}

TEST(NiuItemWitnessTest, RequiresStableOriginalItemAndPostIssueClosure) {
    check_item_witness_contract<NiuHandoffReceiver>();
}

template <class Receiver>
concept exposes_item_request_writer = requires(
    const typename Receiver::Write& write, const typename Receiver::Read& read,
    const typename Receiver::ItemWitnessRequest& request) {
    Receiver::write_item_request(write, read, request);
};

template <class Receiver>
void check_item_request_writer_contract() {
    if constexpr (!exposes_item_request_writer<Receiver>) {
        ADD_FAILURE() << "SDK has no ordered item witness initialization/readback receiver";
    } else {
        using Request = typename Receiver::ItemWitnessRequest;
        using Region = typename Receiver::ItemWitnessRegion;
        Request request{1, 7, 3, 5, 11, UINT64_C(1) << 35, (UINT64_C(1) << 36) + 1, 2, 1, 15, 2, 2, 0};
        std::array<std::byte, sizeof(Region)> bytes;
        bytes.fill(std::byte{0xa5});
        std::vector<std::pair<uint32_t, size_t>> writes;
        bool fail_body = false;
        bool corrupt_readback = false;
        auto write = [&](uint32_t offset, std::span<const std::byte> source) {
            writes.emplace_back(offset, source.size());
            if (fail_body || offset > bytes.size() || source.size() > bytes.size() - offset)
                return NiuHandoffIoStatus::Failed;
            std::copy(source.begin(), source.end(), bytes.begin() + offset);
            return NiuHandoffIoStatus::Complete;
        };
        auto read = [&](uint32_t offset, std::span<std::byte> destination) {
            if (offset > bytes.size() || destination.size() > bytes.size() - offset)
                return NiuHandoffIoStatus::Failed;
            std::copy_n(bytes.begin() + offset, destination.size(), destination.begin());
            if (corrupt_readback && destination.size() == bytes.size()) destination[24] ^= std::byte{1};
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
        EXPECT_EQ(region.receipt.issued_occurrence, 0u);
        EXPECT_EQ(region.receipt.closed_occurrence, 0u);
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
