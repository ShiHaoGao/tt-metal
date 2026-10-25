// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

#include "hostdev/niu_handoff_contract.h"

namespace tt::tt_metal::detail {

// This internal transport/decoder observes bytes only. Neither an observation
// nor a successful request comparison is a boot, lease, clean-state receipt,
// selector normalization, or payload-visibility capability.
enum class NiuHandoffIoStatus { Complete, Failed };
enum class NiuHandoffReadStatus { Observed, Unpublished, InProgress, Changed, Invalid, IoFailure };
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

class NiuHandoffReceiver final {
public:
    // Offsets are relative to the actual HAL-validated mailbox. The future
    // owner adapter captures the original Cluster/chip/core/base, performs no
    // admission-locked IO, and propagates transport exceptions to that owner.
    using Read = std::function<NiuHandoffIoStatus(uint32_t, std::span<std::byte>)>;
    static NiuHandoffReadResult read(const Read& read);
    static NiuHandoffCheck check(
        const NiuHandoffObservation& observation, const NiuHandoffExpectedExecution& expected);
};

}  // namespace tt::tt_metal::detail
