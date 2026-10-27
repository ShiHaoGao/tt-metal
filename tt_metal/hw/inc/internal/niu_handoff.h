// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "hostdev/niu_handoff_contract.h"
#include "hostdev/niu_item_witness_contract.h"

// The firmware compiler otherwise completely peels the two-NoC loops during
// LTO, even with the per-loop unroll directive. Preserve the receiver's loops
// and its bounded instruction footprint across boot and execution callers.
#if defined(__riscv) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-loop-optimize")
#endif

namespace tt::niu_handoff {
enum class Source { None, BlackholeDynamic };
enum class Admission { Disabled, Accepted, Rejected };
enum class GoSequenceStatus { Consumed, Exhausted };
struct GoSequenceAdvance {
    GoSequenceStatus status;
    uint32_t sequence;
};

// Pure arithmetic boundary; no receiver completion or clean-state setter.
constexpr GoSequenceAdvance advanceGoSequence(uint32_t current) {
    if (current == UINT32_MAX) return {GoSequenceStatus::Exhausted, current};
    return {GoSequenceStatus::Consumed, current + 1};
}

// Actual L1 adapter shared by the generated device path and CPU ordering
// tests. Platform owns the architectural cache fence and selector MMIO. The
// adapter owns every volatile load/store, including split RV32 uint64 words.
template <typename Platform>
class ItemWitnessMemory final {
public:
    ItemWitnessMemory(volatile tt_niu_item_witness_region& region, Platform& platform) :
        region_(region), platform_(platform) {}
    void invalidate() { platform_.invalidate(); }
    uint32_t requestGeneration() { return region_.request.request_generation; }
    uint32_t receiptSequence() { return region_.receipt.publish_sequence; }
    tt_niu_item_witness_request readRequest() {
        tt_niu_item_witness_request request;
        const auto* input = reinterpret_cast<volatile const unsigned char*>(&region_.request);
        auto* output = reinterpret_cast<unsigned char*>(&request);
#ifdef __clang__
#pragma clang loop unroll(disable) interleave(disable) vectorize(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
        for (unsigned byte = 0; byte != sizeof(request); ++byte) output[byte] = input[byte];
        return request;
    }
    tt_niu_handoff_selector_observation selectorObserved(unsigned noc, unsigned selector) {
        return platform_.selectorObserved(noc, selector);
    }
    void publish(const tt_niu_item_witness_receipt& receipt) {
        // One writer only, and Publisher supplies nonwrapping even sequences.
        // The first fence makes odd visible before any body store; the second
        // makes all body stores visible before the final even publication.
        region_.receipt.publish_sequence = receipt.publish_sequence - 1;
        platform_.invalidate();
        auto* output = reinterpret_cast<volatile unsigned char*>(&region_.receipt);
        const auto* input = reinterpret_cast<const unsigned char*>(&receipt);
#ifdef __clang__
#pragma clang loop unroll(disable) interleave(disable) vectorize(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
        for (unsigned byte = sizeof(uint32_t); byte != sizeof(receipt); ++byte) output[byte] = input[byte];
        platform_.invalidate();
        region_.receipt.publish_sequence = receipt.publish_sequence;
    }

private:
    volatile tt_niu_item_witness_region& region_;
    Platform& platform_;
};

// One instance owns one immutable, SDK-initialized slot. It is deliberately
// single-use: safe rolling-slot reuse requires a separate acknowledged
// generation frontier and cannot be inferred from the previous Closed state.
class ItemWitnessPublisher final {
public:
    template <typename IO>
    bool arm(IO& io) {
        if (phase_ != Phase::Cold) return false;
        io.invalidate();
        const auto sequence = io.receiptSequence();
        const auto generation = io.requestGeneration();
        request_ = io.readRequest();
        io.invalidate();
        if (sequence || io.receiptSequence() || generation != io.requestGeneration() ||
            generation != request_.request_generation || !valid(request_)) {
            phase_ = Phase::Poisoned;
            return false;
        }
        receipt_.request = request_;
        phase_ = Phase::Armed;
        return true;
    }

    template <typename IO, typename Issue>
    bool issue(IO& io, Issue&& issue_body) {
        if (phase_ != Phase::Armed) return false;
        if (!unchanged(io)) return reject(io);
        issue_body();  // The generated algorithm owns actual scoped issue.
        io.invalidate();
        if (!unchanged(io)) return reject(io);
        receipt_.issued_occurrence = request_.occurrence;
        phase_ = Phase::Issued;
        return publish(io, TT_NIU_ITEM_WITNESS_ISSUED);
    }

    template <typename IO, typename Wait>
    bool close(IO& io, Wait&& scoped_wait) {
        if (phase_ != Phase::Issued) return false;
        if (!unchanged(io)) return reject(io);
        scoped_wait();  // Exact read/nonposted-write TRID barrier, not queue wait.
        io.invalidate();
        if (!unchanged(io)) return reject(io);
        const auto selector = io.selectorObserved(request_.noc, request_.selector);
        if (selector.outstanding || selector.outgoing) return reject(io);
        receipt_.closed_occurrence = request_.occurrence;
        phase_ = Phase::Closed;
        return publish(io, TT_NIU_ITEM_WITNESS_CLOSED);
    }

private:
    enum class Phase { Cold, Armed, Issued, Closed, Poisoned };
    static bool valid(const tt_niu_item_witness_request& request) {
        return request.abi_version == TT_NIU_ITEM_WITNESS_VERSION && request.request_generation &&
               request.request_generation != UINT32_MAX && request.boot_cookie && request.boot_cookie != UINT64_MAX &&
               request.invocation_id && request.invocation_id != UINT64_MAX && request.item && request.item != UINT64_MAX &&
               request.generation && request.generation != UINT64_MAX && request.occurrence &&
               request.occurrence != UINT64_MAX && request.leaf && request.noc < 2 && request.selector < 16 &&
               (request.issuer == 1 || request.issuer == 2) &&
               (request.completion == TT_NIU_ITEM_READ_DESTINATION_LOCAL ||
                request.completion == TT_NIU_ITEM_NONPOSTED_WRITE_ACKNOWLEDGED) && request.reserved == 0;
    }
    static bool same(const tt_niu_item_witness_request& lhs, const tt_niu_item_witness_request& rhs) {
        return lhs.abi_version == rhs.abi_version && lhs.request_generation == rhs.request_generation &&
               lhs.boot_cookie == rhs.boot_cookie && lhs.invocation_id == rhs.invocation_id && lhs.item == rhs.item &&
               lhs.generation == rhs.generation && lhs.occurrence == rhs.occurrence && lhs.leaf == rhs.leaf &&
               lhs.noc == rhs.noc && lhs.selector == rhs.selector && lhs.issuer == rhs.issuer &&
               lhs.completion == rhs.completion && lhs.reserved == rhs.reserved;
    }
    template <typename IO>
    bool unchanged(IO& io) const {
        io.invalidate();
        const auto generation = io.requestGeneration();
        const auto request = io.readRequest();
        io.invalidate();
        return generation == request_.request_generation && generation == io.requestGeneration() &&
               same(request, request_);
    }
    template <typename IO>
    bool publish(IO& io, uint32_t state) {
        if (receipt_.publish_sequence > UINT32_MAX - 2) {
            phase_ = Phase::Poisoned;
            return false;
        }
        receipt_.state = state;
        receipt_.publish_sequence += 2;
        io.publish(receipt_);
        return true;
    }
    template <typename IO>
    bool reject(IO& io) {
        phase_ = Phase::Poisoned;
        receipt_.closed_occurrence = 0;
        publish(io, TT_NIU_ITEM_WITNESS_REJECTED);
        return false;
    }
    Phase phase_ = Phase::Cold;
    tt_niu_item_witness_request request_{};
    tt_niu_item_witness_receipt receipt_{};
};

// The IO boundary is the actual firmware MMIO/L1 adapter. No caller can
// supply a clean flag: source counts, command readiness and hardware responses
// are read here, after the last issuing processor has returned.
class Receiver {
public:
    using ItemPublisher = ItemWitnessPublisher;
    using ItemWitnessRequest = tt_niu_item_witness_request;
    using ItemWitnessReceipt = tt_niu_item_witness_receipt;
    using ItemWitnessRegion = tt_niu_item_witness_region;
    template <typename Platform>
    using ItemMemory = ItemWitnessMemory<Platform>;
    template <typename IO>
    bool initializeBoot(IO& io) {
        if (phase_ != Phase::Cold || consumed_go_sequence_ != 0 || fault_) return false;
        closeProcessors(io);
        closeControl(io);
        closeDescriptors(io);
        receipt_.header.abi_version = TT_NIU_HANDOFF_VERSION;
        receipt_.header.publish_sequence = publish_sequence_ += 2;
        receipt_.header.status = TT_NIU_HANDOFF_BOOT_READY;
        receipt_.header.control_valid_noc_mask = 3;
        setControl(receipt_.control);
        io.publish(receipt_);
        clearReceipt(receipt_);
        phase_ = Phase::Ready;
        return true;
    }

    // Must precede any stage NoC initialization or issuing processor release.
    // Only boot or an actually closed managed tail can supply this baseline.
    template <typename IO>
    void prepareStage(IO& io) {
        if (!active_) return;
        if (phase_ != Phase::Running || stage_open_ || fault_) {
            reject(TT_NIU_HANDOFF_ERROR_SOURCE);
            return;
        }
        closeControl(io);
        closeDescriptors(io);
    }

    template <typename IO, typename Body>
    Admission runUnmanagedControl(IO& io, Body&& body) {
        io.invalidate();
        if (fault_ || phase_ == Phase::Cold || phase_ == Phase::Running || io.requestGeneration() != 0)
            return reject(TT_NIU_HANDOFF_ERROR_IDENTITY);
        // A host read-pointer-reset signal may remain asserted across polls.
        // Once invalidated, repeated control handling cannot revive a receipt.
        if (phase_ != Phase::Unmanaged && !publishPending(io, consumed_go_sequence_))
            return reject(TT_NIU_HANDOFF_ERROR_SEQUENCE);
        phase_ = Phase::Unmanaged;
        body();
        return Admission::Disabled;
    }

    template <typename IO, typename Body>
    Admission runGo(IO& io, Body&& body) {
        if (fault_) return Admission::Rejected;
        const auto consumed = advanceGoSequence(consumed_go_sequence_);
        if (!publishPending(io, consumed.sequence)) return reject(TT_NIU_HANDOFF_ERROR_SEQUENCE);
        if (consumed.status == GoSequenceStatus::Exhausted) return reject(TT_NIU_HANDOFF_ERROR_SEQUENCE);
        consumed_go_sequence_ = consumed.sequence;
        if (phase_ == Phase::Cold) return reject(TT_NIU_HANDOFF_ERROR_SOURCE);
        io.invalidate();
        const uint32_t generation = io.requestGeneration();
        if (generation == 0) {
            if (armed_ && completed_go_count_ != request_.final_go_count)
                return reject(TT_NIU_HANDOFF_ERROR_IDENTITY);
            active_ = false;
            phase_ = Phase::Unmanaged;
            body();
            io.notify();
            return Admission::Disabled;
        }
        if (phase_ != Phase::Ready) return reject(TT_NIU_HANDOFF_ERROR_SOURCE);
        const auto request = io.readRequest();
        io.invalidate();
        if (generation != io.requestGeneration() || generation != request.request_generation || !valid(request))
            return Admission::Rejected;
        if (armed_) {
            if (generation == request_.request_generation) {
                if (!same(request, request_) || completed_go_count_ == request_.final_go_count)
                    return Admission::Rejected;
            } else {
                if (generation < request_.request_generation || completed_go_count_ != request_.final_go_count ||
                    request.boot_cookie_lo != request_.boot_cookie_lo ||
                    request.boot_cookie_hi != request_.boot_cookie_hi ||
                    !laterChallenge(request, request_))
                    return Admission::Rejected;
                armed_ = false;
            }
        }
        if (!armed_) {
            copyRequest(request_, request);
            clearReceipt(receipt_);
            completed_go_count_ = 0;
            armed_ = true;
        }
        if (consumed_go_sequence_ != request_.first_go_sequence + completed_go_count_)
            return reject(TT_NIU_HANDOFF_ERROR_SEQUENCE);
        active_ = true;
        phase_ = Phase::Running;
        go_stage_count_ = 0;
        body();  // All body-local profiler scopes end before notification.
        if (fault_ || stage_open_ || go_stage_count_ == 0)
            return reject(TT_NIU_HANDOFF_ERROR_SOURCE);
        closeProcessors(io);
        if (!unchanged(io)) return reject(TT_NIU_HANDOFF_ERROR_IDENTITY);
        io.notify();
        closeDescriptors(io);
        closeControl(io);
        if (!unchanged(io)) return reject(TT_NIU_HANDOFF_ERROR_IDENTITY);
        if (publish_sequence_ > UINT32_MAX - 2) return reject(TT_NIU_HANDOFF_ERROR_SEQUENCE);
#pragma GCC unroll 0
        for (unsigned noc = 0; noc != 2; ++noc) {
#pragma GCC unroll 0
            for (unsigned selector = 0; selector != 16; ++selector)
                receipt_.selector_observed[noc][selector] = io.selectorObserved(noc, selector);
        }
        auto& header = receipt_.header;
        header.abi_version = TT_NIU_HANDOFF_VERSION;
        header.challenge_lo = request_.challenge_lo;
        header.challenge_hi = request_.challenge_hi;
        header.boot_cookie_lo = request_.boot_cookie_lo;
        header.boot_cookie_hi = request_.boot_cookie_hi;
        header.invocation_id_lo = request_.invocation_id_lo;
        header.invocation_id_hi = request_.invocation_id_hi;
        header.group_generation = request_.group_generation;
        header.completed_go_count = ++completed_go_count_;
        header.status = TT_NIU_HANDOFF_TRAFFIC_CLOSED;
        header.error = TT_NIU_HANDOFF_ERROR_NONE;
        header.control_valid_noc_mask = 3;
        header.publish_sequence = publish_sequence_ += 2;
        setControl(receipt_.control);
        receipt_.consumed_go_sequence = consumed_go_sequence_;
        io.publish(receipt_);
        active_ = false;
        phase_ = Phase::Ready;
        return Admission::Accepted;
    }

    template <typename IO>
    void beginStage(IO& io, Source source, uint32_t processor_mask) {
        if (!active_ || fault_) return;
        if (stage_open_ || (processor_mask & ~TT_NIU_HANDOFF_PROCESSOR_MASK) != 0 ||
            (source != Source::None && source != Source::BlackholeDynamic)) {
            reject(TT_NIU_HANDOFF_ERROR_SOURCE);
            return;
        }
        stage_open_ = true;
        source_ = source;
        if (source == Source::None) return;
        io.invalidate();
#pragma GCC unroll 0
        for (unsigned noc = 0; noc != 2; ++noc) {
#pragma GCC unroll 0
            for (unsigned counter = 0; counter != 5; ++counter) {
                baseline_observed_[noc][counter] = io.observedCounter(noc, counter);
#pragma GCC unroll 0
                for (unsigned risc = 0; risc != 2; ++risc)
                    baseline_expected_[noc][risc][counter] = io.sourceExpected(noc, risc, counter);
            }
        }
    }

    template <typename IO>
    void closeStage(IO& io) { closeStage(io, [] {}); }

    template <typename IO, typename ControlBody>
    void closeStage(IO& io, ControlBody&& control_body) {
        if (!active_) {
            control_body();
            return;
        }
        if (fault_ || !stage_open_) {
            reject(TT_NIU_HANDOFF_ERROR_SOURCE);
            return;
        }
        closeProcessors(io);
        if (source_ == Source::BlackholeDynamic) {
            io.invalidate();
#pragma GCC unroll 0
            for (unsigned noc = 0; noc != 2; ++noc) {
#pragma GCC unroll 0
                for (unsigned counter = 0; counter != 5; ++counter) {
                    const uint32_t first = io.sourceExpected(noc, 0, counter);
                    const uint32_t second = io.sourceExpected(noc, 1, counter);
                    const uint32_t expected = first + second;
                    while (io.observedCounter(noc, counter) != expected) io.poll();
                    receipt_.source_expected[noc][0][counter] += first - baseline_expected_[noc][0][counter];
                    receipt_.source_expected[noc][1][counter] += second - baseline_expected_[noc][1][counter];
                    receipt_.source_observed[noc][counter] += expected - baseline_observed_[noc][counter];
                }
            }
            receipt_.header.source_valid_noc_mask = 3;
        } else {
            // NoPayload has no dynamic source proof. Its actual inherited
            // dedicated control counts must close before any new baseline.
            closeControl(io);
        }
        // Rebase only after this stage's absolute counters close. Aggregated
        // deltas must never conceal an incomplete stage, including at wrap.
        io.rebaseControl();
        control_body();
        closeControl(io);
        if (receipt_.header.completed_stage_count == UINT32_MAX || go_stage_count_ == UINT32_MAX) {
            reject(TT_NIU_HANDOFF_ERROR_SEQUENCE);
            return;
        }
        ++receipt_.header.completed_stage_count;
        ++go_stage_count_;
        stage_open_ = false;
    }

    bool active() const { return active_; }

private:
    enum class Phase { Cold, Ready, Running, Unmanaged };

    // Keep wire housekeeping compact: aggregate assignments otherwise pull
    // general-purpose memcpy/memset into this fixed-size firmware image.
    __attribute__((noinline)) static void clearReceipt(tt_niu_handoff_receipt& receipt) {
        auto* bytes = reinterpret_cast<volatile unsigned char*>(&receipt);
#pragma GCC unroll 0
        for (unsigned byte = 0; byte != sizeof(receipt); ++byte) bytes[byte] = 0;
    }
    __attribute__((noinline)) static void copyRequest(
        tt_niu_handoff_request& destination, const tt_niu_handoff_request& source) {
        auto* output = reinterpret_cast<volatile unsigned char*>(&destination);
        const auto* input = reinterpret_cast<const unsigned char*>(&source);
#pragma GCC unroll 0
        for (unsigned byte = 0; byte != sizeof(source); ++byte) output[byte] = input[byte];
    }
    static void setControl(tt_niu_handoff_control& control) {
        control.valid_noc_mask = 3;
        control.ready_cmd_mask_noc0 = 15;
        control.ready_cmd_mask_noc1 = 15;
        control.control_contract = 1;
    }

    template <typename IO>
    bool publishPending(IO& io, uint32_t consumed) {
        if (publish_sequence_ > UINT32_MAX - 2) {
            // A permanently odd seqlock invalidates the previous clean record
            // even when no fresh even publication number remains.
            io.invalidateReceipt();
            return false;
        }
        tt_niu_handoff_receipt pending;
        clearReceipt(pending);
        pending.header.abi_version = TT_NIU_HANDOFF_VERSION;
        pending.header.publish_sequence = publish_sequence_ += 2;
        pending.header.status = TT_NIU_HANDOFF_PENDING;
        pending.consumed_go_sequence = consumed;
        io.publish(pending);
        return true;
    }

    template <typename IO>
    __attribute__((noinline)) void closeDescriptors(IO& io) {
#pragma GCC unroll 0
        for (unsigned noc = 0; noc != 2; ++noc) {
#pragma GCC unroll 0
            for (unsigned buffer = 0; buffer != 4; ++buffer) {
                while (!io.descriptorReady(noc, buffer)) io.poll();
                io.descriptorReadback(noc, buffer);
            }
        }
        io.invalidate();
    }

    template <typename IO>
    void closeProcessors(IO& io) {
        // BRISC has returned from its own kernel. The other four processors
        // must report actual DONE, including processors disabled in this GO.
        io.invalidate();
#pragma GCC unroll 0
        for (unsigned role = 0; role != 4; ++role) {
            while (io.subordinateSignal(role) != io.doneSignal()) io.poll();
        }
        receipt_.header.completed_processor_mask = TT_NIU_HANDOFF_PROCESSOR_MASK;
    }
    static bool valid(const tt_niu_handoff_request& request) {
        return request.abi_version == TT_NIU_HANDOFF_VERSION && request.request_generation != 0 &&
               (request.challenge_lo || request.challenge_hi) &&
               (request.boot_cookie_lo || request.boot_cookie_hi) &&
               (request.invocation_id_lo || request.invocation_id_hi) && request.group_generation != 0 &&
               request.final_go_count != 0 && request.first_go_sequence != 0 &&
               request.first_go_sequence <= UINT32_MAX - (request.final_go_count - 1) &&
               request.selector_mask_noc0 == 0xffff &&
               request.selector_mask_noc1 == 0xffff &&
               request.traffic_contract == TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1 &&
               request.required_processor_mask == TT_NIU_HANDOFF_PROCESSOR_MASK &&
               request.reserved1 == 0;
    }
    static bool laterChallenge(const tt_niu_handoff_request& a, const tt_niu_handoff_request& b) {
        return a.challenge_hi > b.challenge_hi ||
               (a.challenge_hi == b.challenge_hi && a.challenge_lo > b.challenge_lo);
    }
    __attribute__((noinline)) static bool same(const tt_niu_handoff_request& a, const tt_niu_handoff_request& b) {
        // The wire request consists of sixteen uint32 fields with no padding.
        // Compare its complete representation, including reserved fields,
        // without emitting a separate branch for every field.
        static_assert(sizeof(tt_niu_handoff_request) == 16 * sizeof(uint32_t));
        const auto* first = reinterpret_cast<const unsigned char*>(&a);
        const auto* second = reinterpret_cast<const unsigned char*>(&b);
#pragma GCC unroll 0
        for (unsigned byte = 0; byte != sizeof(a); ++byte)
            if (first[byte] != second[byte]) return false;
        return true;
    }
    template <typename IO>
    bool unchanged(IO& io) {
        io.invalidate();
        const uint32_t generation = io.requestGeneration();
        const auto request = io.readRequest();
        io.invalidate();
        return generation == io.requestGeneration() && generation == request_.request_generation &&
               same(request, request_);
    }
    template <typename IO>
    __attribute__((noinline)) void closeControl(IO& io) {
        io.invalidate();
#pragma GCC unroll 0
        for (unsigned noc = 0; noc != 2; ++noc) {
#pragma GCC unroll 0
            for (unsigned counter = 0; counter != 5; ++counter) {
                const uint32_t expected = io.controlExpected(noc, counter);
                while (io.observedCounter(noc, counter) != expected) io.poll();
                receipt_.control_expected[noc][counter] = expected;
                receipt_.control_observed[noc][counter] = expected;
            }
        }
    }
    __attribute__((noinline)) Admission reject(uint32_t error) {
        fault_ = true;
        receipt_.header.error = error;
        return Admission::Rejected;
    }

    tt_niu_handoff_request request_{};
    tt_niu_handoff_receipt receipt_{};
    uint32_t baseline_expected_[2][2][5]{};
    uint32_t baseline_observed_[2][5]{};
    uint32_t consumed_go_sequence_ = 0;
    Phase phase_ = Phase::Cold;
    uint32_t completed_go_count_ = 0;
    uint32_t go_stage_count_ = 0;
    uint32_t publish_sequence_ = 0;
    Source source_ = Source::None;
    bool armed_ = false;
    bool active_ = false;
    bool stage_open_ = false;
    bool fault_ = false;
};
}  // namespace tt::niu_handoff

#if defined(__riscv) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
