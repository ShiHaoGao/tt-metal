// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace tt::worker_stream_state {
// Schema constants shared by host projection and firmware interpretation.
// Actual firmware-provider admission is separate; header presence is no proof.
inline constexpr uint8_t kVersion = 1;
#if defined(TT_WORKER_STREAM_STATE_ABI_VERSION)
static_assert(TT_WORKER_STREAM_STATE_ABI_VERSION == kVersion,
              "worker stream-state ABI differs between host build and firmware source");
#endif
enum class Owner : uint8_t { SdkCircularBuffers = 1, Program = 2 };
enum class FirmwareRole : uint8_t { Brisc = 1, Trisc0 = 2 };
// Exact bytes in the selected image's .tt_worker_stream_state ELF section.
// No magic/signature is needed: the section name and closed record own the ABI.
struct ImageRecord {
    uint8_t version = 0;
    uint8_t role = 0;
    uint8_t reserved0 = 0;
    uint8_t reserved1 = 0;
};
static_assert(sizeof(ImageRecord) == 4);
constexpr bool acceptsImage(ImageRecord image, FirmwareRole role) {
    return image.version == kVersion && image.role == static_cast<uint8_t>(role) &&
           image.reserved0 == 0 && image.reserved1 == 0;
}
enum class Boundary : uint8_t { LaunchEntry, LaunchExit };
enum class Counter : uint8_t { Received, Acknowledged };
enum class Decision : uint8_t {
    Preserve,
    ClearSdkCounters,
    RejectVersion,
    RejectOwner,
    RejectSdkDescriptor,
};

// sdkDescriptors means actual local or remote SDK CB configuration on this
// worker launch. It is a conflicting input check, never ownership inference.
constexpr Decision receive(uint8_t version, uint8_t owner, Boundary boundary, bool sdkDescriptors) {
    (void)boundary;  // Both boundaries obey the same ownership; boot is separate.
    if (version != kVersion)
        return Decision::RejectVersion;
    if (owner == static_cast<uint8_t>(Owner::Program))
        return sdkDescriptors ? Decision::RejectSdkDescriptor : Decision::Preserve;
    if (owner == static_cast<uint8_t>(Owner::SdkCircularBuffers))
        return Decision::ClearSdkCounters;
    return Decision::RejectOwner;
}

// Firmware uses the existing get_cb_tiles_received_ptr/get_cb_tiles_acked_ptr
// mapping in its writer. NUM_CIRCULAR_BUFFERS comes from that architecture.
// Host tests use nonzero mock registers. This helper does not signal completion
// or authorize launch: callers must reject invalid decisions before any setup.
template <typename CounterWriter>
inline void apply(Decision decision, uint32_t counterCount, CounterWriter &write) {
    if (decision != Decision::ClearSdkCounters)
        return;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 0
#endif
    for (uint32_t counter = 0; counter != counterCount; ++counter) {
        write(counter, Counter::Received, 0);
        write(counter, Counter::Acknowledged, 0);
    }
}
// BRISC's adapter owns the request and reads the actual TRISC0 done byte.
// Returning true is the only path which permits following setup/publication.
// No launch-slot pointer or mutable ownership data is reread while waiting.
template <typename ResetHandoff>
inline bool completeReset(Decision decision, ResetHandoff &handoff) {
    if (decision == Decision::Preserve)
        return true;
    if (decision != Decision::ClearSdkCounters)
        return false;
    handoff.request();
    while (!handoff.complete())
        handoff.poll();
    return true;
}
}  // namespace tt::worker_stream_state
