// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace tt::tt_metal {

enum class ProfilerRawProtocol { ClassicDram = 1 };
enum class ProfilerRawDecodeStatus { Complete, Truncated, Malformed, UnsupportedProtocol };
enum class ProfilerRawSlot { FirmwareStart, FirmwareEnd, KernelStart, KernelEnd, Custom };

struct ProfilerRawPacket {
    uint32_t word_offset;
    uint32_t timer_id;
    // Reassembled wire bits, without epoch repair or a claim of monotonicity.
    // For ZONE_TOTAL the low word holds a sum, not a timestamp.
    uint64_t timestamp;
    ProfilerRawSlot slot;
    std::array<uint64_t, 2> data{};
    uint8_t data_count = 0;
};

struct ProfilerRawRun {
    uint32_t word_offset;
    uint32_t risc_id;
    uint32_t flat_id;
    uint32_t runtime_id;
    uint32_t trace_counter;
    std::vector<ProfilerRawPacket> packets;
};

struct ProfilerRawDecodeResult {
    ProfilerRawDecodeStatus status;
    std::vector<ProfilerRawRun> runs;
};

// Decode one RISC's classic DRAM region, bounded by its saved control end index.
// Offsets are relative to words, and runs/packets retain storage order and all
// duplicates. Exact unused guaranteed-slot and custom-padding encodings are
// omitted. Guaranteed slots identify wrapper scopes independently of marker names.
//
// The caller must establish the actual protocol (no accumulate, trace-only,
// streaming, pre-sentinel perf-counter data or dispatch debug-dump feed). This
// parser cannot infer that deployment from bytes. Drops and device read/reset
// disposition are separate capture metadata. Complete proves structural decoding
// only, not complete kernel membership, temporal ordering or capture integrity.
// On failure, runs may retain a decoded prefix; callers must check status before
// accepting any result. No timestamp sorting, deduplication or ID remapping occurs.
ProfilerRawDecodeResult decodeProfilerRawStream(
    ProfilerRawProtocol protocol,
    std::span<const uint32_t> words,
    uint32_t end_index,
    uint32_t expected_risc,
    uint32_t expected_flat);

}  // namespace tt::tt_metal
