// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/profiler_raw_data.hpp>

#include "hostdev/profiler_common.h"

namespace tt::tt_metal {
namespace {

namespace kp = kernel_profiler;

uint32_t packetType(uint32_t timer_id) {
    return (timer_id >> kp::PROFILER_TIMER_PACKET_TYPE_SHIFT) & kp::PROFILER_TIMER_PACKET_TYPE_MASK;
}

ProfilerRawDecodeStatus decodePacket(
    std::span<const uint32_t> words,
    uint32_t end_index,
    uint32_t& offset,
    ProfilerRawSlot slot,
    std::vector<ProfilerRawPacket>& packets) {
    if (end_index - offset < kp::PROFILER_L1_MARKER_UINT32_SIZE) {
        return ProfilerRawDecodeStatus::Truncated;
    }
    const uint32_t high = words[offset];
    const uint32_t low = words[offset + 1];
    const bool guaranteed = slot != ProfilerRawSlot::Custom;
    // init_profiler fills both words of an unused guaranteed slot with VALID.
    // mark_padding writes VALID,0 only in the custom stream. Neither encoding
    // authorizes scanning the next guaranteed slot as a run header.
    if (high == kp::PROFILER_MARKER_VALID &&
        low == (guaranteed ? kp::PROFILER_MARKER_VALID : 0)) {
        offset += kp::PROFILER_L1_MARKER_UINT32_SIZE;
        return ProfilerRawDecodeStatus::Complete;
    }
    if (!(high & kp::PROFILER_MARKER_VALID)) {
        return ProfilerRawDecodeStatus::Malformed;
    }

    const uint32_t timer_id =
        (high >> kp::PROFILER_MARKER_TIMER_ID_SHIFT) & kp::PROFILER_MARKER_TIMER_ID_MASK;
    const uint32_t type = packetType(timer_id);
    if (type > kp::TS_DATA_16B) {
        return ProfilerRawDecodeStatus::Malformed;
    }
    if (guaranteed) {
        const bool start = slot == ProfilerRawSlot::FirmwareStart || slot == ProfilerRawSlot::KernelStart;
        if (type != (start ? kp::ZONE_START : kp::ZONE_END)) {
            return ProfilerRawDecodeStatus::Malformed;
        }
    }

    const uint8_t data_count =
        type == kp::TS_DATA       ? kp::TimestampedDataSize<kp::TS_DATA>::size
        : type == kp::TS_DATA_16B ? kp::TimestampedDataSize<kp::TS_DATA_16B>::size
                                 : 0;
    const uint32_t packet_words = kp::PROFILER_L1_MARKER_UINT32_SIZE * (1 + data_count);
    if (end_index - offset < packet_words) {
        return ProfilerRawDecodeStatus::Truncated;
    }
    ProfilerRawPacket packet{
        offset,
        timer_id,
        (uint64_t{high & kp::PROFILER_MARKER_TS_HIGH_MASK} << kp::PROFILER_MARKER_TS_LOW_BITS) | low,
        slot,
        {},
        data_count};
    for (uint8_t i = 0; i < data_count; ++i) {
        const uint32_t data_offset = offset + kp::PROFILER_L1_MARKER_UINT32_SIZE * (1 + i);
        packet.data[i] = (uint64_t{words[data_offset]} << 32) | words[data_offset + 1];
    }
    packets.push_back(packet);
    offset += packet_words;
    return ProfilerRawDecodeStatus::Complete;
}

}  // namespace

ProfilerRawDecodeResult decodeProfilerRawStream(
    ProfilerRawProtocol protocol,
    std::span<const uint32_t> words,
    uint32_t end_index,
    uint32_t expected_risc,
    uint32_t expected_flat) {
    if (protocol != ProfilerRawProtocol::ClassicDram) {
        return {ProfilerRawDecodeStatus::UnsupportedProtocol, {}};
    }
    if (end_index > words.size() || end_index % kp::PROFILER_L1_MARKER_UINT32_SIZE != 0) {
        return {ProfilerRawDecodeStatus::Truncated, {}};
    }
    if (expected_risc >= kp::PROFILER_MAX_RISC_COUNT || expected_flat > kp::PROFILER_ID_FLAT_MASK) {
        return {ProfilerRawDecodeStatus::Malformed, {}};
    }

    ProfilerRawDecodeResult result{ProfilerRawDecodeStatus::Complete, {}};
    uint32_t offset = 0;
    while (offset < end_index) {
        const uint32_t run_offset = offset;
        if (words[offset] != 0 || words[offset + 1] != 0) {
            result.status = ProfilerRawDecodeStatus::Malformed;
            return result;
        }
        if (end_index - offset < kp::CUSTOM_MARKERS) {
            result.status = ProfilerRawDecodeStatus::Truncated;
            return result;
        }

        const uint32_t identity = words[offset + kp::ID_LH];
        const uint32_t risc = (identity >> kp::PROFILER_ID_RISC_SHIFT) & kp::PROFILER_ID_RISC_MASK;
        const uint32_t flat = (identity >> kp::PROFILER_ID_FLAT_SHIFT) & kp::PROFILER_ID_FLAT_MASK;
        constexpr uint32_t identity_mask = kp::PROFILER_ID_RISC_FLAT_FIELD_MASK | kp::PROFILER_ID_TRACE_FIELD_MASK;
        if ((identity & ~identity_mask) != 0 || risc != expected_risc || flat != expected_flat) {
            result.status = ProfilerRawDecodeStatus::Malformed;
            return result;
        }
        result.runs.push_back({
            run_offset,
            risc,
            flat,
            words[offset + kp::ID_LL],
            (identity >> kp::PROFILER_ID_TRACE_SHIFT) & kp::PROFILER_ID_TRACE_MASK,
            {}});
        auto& run = result.runs.back();
        offset += kp::GUARANTEED_MARKER_1_H;
        constexpr std::array slots{
            ProfilerRawSlot::FirmwareStart,
            ProfilerRawSlot::FirmwareEnd,
            ProfilerRawSlot::KernelStart,
            ProfilerRawSlot::KernelEnd};
        static_assert(slots.size() == kp::PROFILER_L1_GUARANTEED_MARKER_COUNT);
        for (auto slot : slots) {
            result.status = decodePacket(words, end_index, offset, slot, run.packets);
            if (result.status != ProfilerRawDecodeStatus::Complete) {
                return result;
            }
        }

        while (offset < end_index) {
            // A sentinel is meaningful only between custom packets. Payload
            // words have already been consumed atomically by decodePacket.
            if (words[offset] == 0 && words[offset + 1] == 0) {
                break;
            }
            result.status = decodePacket(words, end_index, offset, ProfilerRawSlot::Custom, run.packets);
            if (result.status != ProfilerRawDecodeStatus::Complete) {
                return result;
            }
        }
    }
    return result;
}

}  // namespace tt::tt_metal
