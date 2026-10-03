// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/profiler_raw_data.hpp>

namespace tt::tt_metal {

enum class ProfilerRawReadDisposition { Unavailable, ReadStarted, ReadComplete, ResetStarted, ResetComplete };
enum class ProfilerRawCoreKind { Tensix, Ethernet };

struct ProfilerRawRiscCapture {
    uint32_t risc_id = 0;
    uint64_t dram_word_offset = 0;
    uint32_t end_index = 0;
    bool dropped_zones = false;
    // Bounded by both this RISC's allocated region and the read image. An
    // end_index larger than words.size() remains visible as truncation.
    std::vector<uint32_t> words;
};

struct ProfilerRawCoreCapture {
    CoreCoord virtual_core;
    CoreCoord physical_core;
    ProfilerRawCoreKind kind = ProfilerRawCoreKind::Tensix;
    uint32_t flat_id = 0;
    std::vector<uint32_t> control_words;
    std::vector<ProfilerRawRiscCapture> riscs;
};

struct ProfilerRawDeviceCapture {
    uint32_t device_id = 0;
    ProfilerRawProtocol protocol = ProfilerRawProtocol::ClassicDram;
    uint8_t timestamp_width = 0;
    ProfilerRawReadDisposition disposition = ProfilerRawReadDisposition::Unavailable;
    std::vector<ProfilerRawCoreCapture> cores;
};

// Caller-owned, single-drain data. Device read/reset disposition survives host
// decoding/Tracy exceptions. It is not a kernel completion receipt. A fresh
// object is required for each read; SDK marker-map cleanup cannot mutate it.
struct ProfilerRawCapture {
    int context_id = -1;
    std::vector<ProfilerRawDeviceCapture> devices;
};

}  // namespace tt::tt_metal
