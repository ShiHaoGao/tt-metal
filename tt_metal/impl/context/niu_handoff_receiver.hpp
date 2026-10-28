// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>

#include "hostdev/niu_handoff_contract.h"
#include "hostdev/niu_item_witness_contract.h"

namespace tt { class Cluster; }
namespace tt::tt_metal { class Hal; }

namespace tt::tt_metal::detail {

// This internal transport/decoder observes bytes only. Neither an observation
// nor a successful request comparison is a boot, lease, clean-state receipt,
// selector normalization, or payload-visibility capability.
enum class NiuHandoffIoStatus { Complete, Failed };
enum class NiuHandoffReadStatus { Observed, Unpublished, InProgress, Changed, Invalid, IoFailure };
enum class NiuHandoffWriteStatus { Published, InvalidRequest, IoFailure, ReadbackMismatch };
enum class NiuHandoffCheck {
    Match,
    Pending,
    Rejected,
    NotExecution,
    InvalidRequest,
    RequestMismatch,
    IdentityMismatch,
    GoMismatch,
    StageMismatch,
    SourceMismatch,
    ControlMismatch,
};

class NiuHandoffReceiver;
class NiuHandoffObservation final {
public:
    const tt_niu_handoff_region& wire() const { return wire_; }

private:
    friend class NiuHandoffReceiver;
    explicit NiuHandoffObservation(tt_niu_handoff_region wire) : wire_(wire) {}
    tt_niu_handoff_region wire_;
};

struct NiuHandoffReadResult {
    NiuHandoffReadStatus status;
    std::optional<NiuHandoffObservation> observation;
};

struct NiuHandoffExpectedExecution {
    tt_niu_handoff_request request;
    uint32_t completed_stage_count;
    uint32_t source_valid_noc_mask;
};

class NiuItemWitnessObservation final {
public:
    const tt_niu_item_witness_region& wire() const { return wire_; }

private:
    friend class NiuHandoffReceiver;
    explicit NiuItemWitnessObservation(tt_niu_item_witness_region wire) : wire_(wire) {}
    tt_niu_item_witness_region wire_;
};

struct NiuItemWitnessReadResult {
    NiuHandoffReadStatus status;
    std::optional<NiuItemWitnessObservation> observation;
};

class NiuHandoffReceiver final {
    struct ItemAcknowledgementState;

public:
    using ItemWitnessRequest = tt_niu_item_witness_request;
    using ItemWitnessRegion = tt_niu_item_witness_region;
    using ItemWitnessAcknowledgement = tt_niu_item_witness_acknowledgement;
    class ItemReception;
    // An immutable snapshot bound to one reception and one accepted round.
    // Only successful transport/readback authorizes the local acknowledge.
    class ItemAcknowledgement final {
    public:
        const ItemWitnessAcknowledgement& wire() const;

    private:
        friend class NiuHandoffReceiver;
        friend class ItemReception;
        explicit ItemAcknowledgement(std::shared_ptr<ItemAcknowledgementState> state) : state_(std::move(state)) {}
        std::shared_ptr<ItemAcknowledgementState> state_;
    };
    // Serialized by the original invocation's admission guard. This owns
    // reception state only; it cannot mint a boot, managed lease or visibility
    // proof. The caller must validate the original issuer before each receive.
    class ItemReception final {
    public:
        enum class IssuerState { Reserved, Issued, Stopped };
        enum class Status { Pending, Complete, NotExecuted, Finished, Rejected, Invalidated, IoFailure };
        // Distinct from a physical item completion. Only an explicit source
        // End, received after its original writer stopped, creates this proof.
        class SeriesTerminal final {
        public:
            uint64_t final_count() const { return observation_.wire().receipt.series.final_count; }
            const NiuItemWitnessObservation& observation() const { return observation_; }

        private:
            friend class ItemReception;
            explicit SeriesTerminal(NiuItemWitnessObservation observation) : observation_(std::move(observation)) {}
            NiuItemWitnessObservation observation_;
        };
        explicit ItemReception(ItemWitnessRequest request);
        ~ItemReception();
        ItemReception(const ItemReception&) = delete;
        ItemReception& operator=(const ItemReception&) = delete;
        const ItemWitnessRequest& request() const { return request_; }
        const std::optional<NiuItemWitnessObservation>& accepted() const { return accepted_; }
        // Established only by a fresh End read bracketed by the original SDK
        // leaf's stopped state. Closed and receipt ACK never imply source End.
        const std::optional<SeriesTerminal>& terminal() const { return terminal_; }
        uint64_t closed_prefix() const { return closed_prefix_; }
        uint64_t acknowledged_prefix() const { return acknowledged_prefix_; }
        std::optional<ItemAcknowledgement> acknowledgement() const;
        bool acknowledge(const ItemAcknowledgement& acknowledgement);
        Status receive(IssuerState sampled, IssuerState current, const NiuItemWitnessReadResult& observed);

    private:
        const ItemWitnessRequest request_;
        Status status_ = Status::Pending;
        std::optional<NiuItemWitnessObservation> accepted_;
        std::optional<NiuItemWitnessObservation> observed_;
        std::optional<NiuItemWitnessObservation> acknowledged_;
        std::optional<SeriesTerminal> terminal_;
        uint64_t closed_prefix_ = 0;
        uint64_t acknowledged_prefix_ = 0;
        ItemWitnessAcknowledgement last_acknowledgement_{};
        std::shared_ptr<ItemAcknowledgementState> acknowledgement_;
    };
    enum class ItemWitnessCheck { Match, Pending, Rejected, InvalidRequest, RequestMismatch, OccurrenceMismatch };
    // Offsets are relative to the mailbox. This byte decoder neither chooses
    // the device endpoint nor establishes owner/boot lifetime.
    using Read = std::function<NiuHandoffIoStatus(uint32_t, std::span<std::byte>)>;
    using Write = std::function<NiuHandoffIoStatus(uint32_t, std::span<const std::byte>)>;
    // Complete writes are ordered before the following read. The owner must
    // exclude GO and retain/revalidate the original boot throughout publication.
    static NiuHandoffWriteStatus write_request(const Write&, const Read&, const tt_niu_handoff_request&);
    static NiuHandoffWriteStatus write_request(
        const Cluster&, const Hal&, ChipId, CoreCoord, const tt_niu_handoff_request&);
    // Projects the initial single-GO journal only. BootReady bytes do not grant
    // bootstrap or clean state; ProgramExecution owns the live owner checks.
    static std::optional<NiuHandoffExpectedExecution> initial_execution(
        const NiuHandoffObservation&, uint64_t boot_cookie, uint64_t invocation_id, bool has_source);
    static NiuHandoffReadResult read(const Read& read);
    // physical_worker is in the original chip's NOC0 coordinates. The caller
    // keeps Cluster/Hal alive and performs IO outside the admission lock;
    // its owner layer must revalidate boot and execution state afterward.
    // Transport exceptions propagate without producing an observation.
    static NiuHandoffReadResult read(
        const Cluster& cluster, const Hal& hal, ChipId chip, CoreCoord physical_worker);
    static NiuHandoffCheck check(
        const NiuHandoffObservation& observation, const NiuHandoffExpectedExecution& expected);
    // Pure observation only. The invocation receiver must additionally retain
    // and validate the original token, stopped issuers, boot and Device epoch.
    // This does not establish posted-payload or consumer-cache visibility.
    static bool closed_selector(
        const NiuHandoffObservation& observation, const NiuHandoffExpectedExecution& expected,
        uint8_t noc, uint8_t selector);
    // Byte observation only. The ManagedInvocation must own the original
    // allocation, request, selector lease, live boot and issuing processor.
    // The expected request must come from that retained owner, never receipt
    // echo. Physical completion facts do not establish consumer visibility.
    // Initialization requires a new, exclusively owned slot with GO excluded.
    // The complete gen0 region is verified before the generation is exposed.
    static NiuHandoffWriteStatus write_item_request(
        const Write&, const Read&, const ItemWitnessRequest&);
    // The original invocation owner supplies real slot IO under its admission
    // guard. ACK releases only the receipt slot, never payload backing.
    static NiuHandoffWriteStatus write_item_acknowledgement(
        const Write&, const Read&, const ItemAcknowledgement&);
    static NiuItemWitnessReadResult read_item(const Read& read);
    static ItemWitnessCheck check_item(
        const NiuItemWitnessObservation& observation, const ItemWitnessRequest& expected);
    static bool closed_item(
        const NiuItemWitnessObservation& observation, const ItemWitnessRequest& expected);
};

}  // namespace tt::tt_metal::detail
