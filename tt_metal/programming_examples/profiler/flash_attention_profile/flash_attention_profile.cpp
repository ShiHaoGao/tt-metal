// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

#include <flash_attention_profile_runner.hpp>

using namespace tt;
using namespace tt::tt_metal;
namespace distributed = tt::tt_metal::distributed;
namespace fap = flash_attention_profile;

namespace {

struct Options {
    uint32_t warmup_iters = 1;
    uint32_t measured_iters = 3;
    uint32_t device_id = 0;
    bool high_precision = false;
    bool read_device_profiler = true;
    bool check_correctness = false;
    fap::PipelineMode pipeline_mode = fap::PipelineMode::Auto;
    uint32_t pipeline_depth = 2;
    fap::CopiedKernelOptions copied_kernel_options;
    fap::GridPolicy grid_policy = fap::GridPolicy::Default;
    std::optional<CoreCoord> grid_override = std::nullopt;
    std::optional<std::pair<uint32_t, uint32_t>> chunk_override = std::nullopt;
    std::vector<fap::RunMode> modes = {fap::RunMode::Eager};
    std::vector<fap::Variant> variants = {
        fap::Variant::TtnnSdpaBaseline,
        fap::Variant::TtnnChunkedBaseline,
        fap::Variant::CopiedSdpa,
        fap::Variant::CopiedChunked};
    std::vector<fap::ShapeConfig> shapes = default_shapes();

    static std::vector<fap::ShapeConfig> default_shapes() {
        return {
            {.name = "smoke",
             .b = 1,
             .nh = 1,
             .nkv = 1,
             .s = 1024,
             .d = 128,
             .prefill = 256,
             .q_chunk = 128,
             .k_chunk = 128,
             .page = 128},
            {.name = "llama_prefill_2k",
             .b = 1,
             .nh = 8,
             .nkv = 1,
             .s = 2048,
             .d = 128,
             .prefill = 2048,
             .q_chunk = 256,
             .k_chunk = 128,
             .page = 128},
            {.name = "llama_prefill_2k_q128_k128",
             .b = 1,
             .nh = 8,
             .nkv = 1,
             .s = 2048,
             .d = 128,
             .prefill = 2048,
             .q_chunk = 128,
             .k_chunk = 128,
             .page = 128},
            {.name = "llama_prefill_2k_q256_k256",
             .b = 1,
             .nh = 8,
             .nkv = 1,
             .s = 2048,
             .d = 128,
             .prefill = 2048,
             .q_chunk = 256,
             .k_chunk = 256,
             .page = 128},
            {.name = "llama_prefill_16k_chunked",
             .b = 1,
             .nh = 8,
             .nkv = 1,
             .s = 16 * 1024,
             .d = 128,
             .prefill = 2048,
             .q_chunk = 256,
             .k_chunk = 128,
             .page = 128},
            {.name = "llama_prefill_16k_chunked_q128_k128",
             .b = 1,
             .nh = 8,
             .nkv = 1,
             .s = 16 * 1024,
             .d = 128,
             .prefill = 2048,
             .q_chunk = 128,
             .k_chunk = 128,
             .page = 128},
            {.name = "llama_prefill_16k_chunked_q256_k256",
             .b = 1,
             .nh = 8,
             .nkv = 1,
             .s = 16 * 1024,
             .d = 128,
             .prefill = 2048,
             .q_chunk = 256,
             .k_chunk = 256,
             .page = 128},
        };
    }
};

struct SummaryStats {
    uint64_t avg_us = 0;
    uint64_t best_us = 0;
    uint64_t worst_us = 0;
};

struct ZoneStats {
    uint64_t count = 0;
    uint64_t min_cycles = 0;
    uint64_t avg_cycles = 0;
    uint64_t max_cycles = 0;
    uint64_t first_start_cycles = 0;
    uint64_t last_end_cycles = 0;
    uint64_t span_cycles = 0;
    double first_start_us = 0.0;
    double last_end_us = 0.0;
    double span_us = 0.0;
    double critical_us = 0.0;
};

struct DeviceZoneSummary {
    std::filesystem::path csv_path;
    double chip_freq_mhz = 0.0;
    std::map<std::string, ZoneStats> zones;
};

std::string trim(std::string_view text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

std::vector<std::string> split_csv_line(std::string_view line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(trim(field));
    return fields;
}

int column_index(const std::vector<std::string>& header, std::string_view name) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::filesystem::path profiler_csv_path() {
    std::filesystem::path profiler_root;
    if (const char* profiler_dir = std::getenv("TT_METAL_PROFILER_DIR")) {
        profiler_root = profiler_dir;
    } else if (const char* metal_home = std::getenv("TT_METAL_HOME")) {
        profiler_root = std::filesystem::path(metal_home) / "generated" / "profiler";
    } else {
        profiler_root = std::filesystem::path("generated") / "profiler";
    }
    return profiler_root / ".logs" / "profile_log_device.csv";
}

uintmax_t safe_file_size(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return 0;
    }
    auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : size;
}

bool phase_is_start(std::string_view phase) { return phase == "ZONE_START" || phase == "START" || phase == "BEGIN"; }

bool phase_is_end(std::string_view phase) { return phase == "ZONE_END" || phase == "END" || phase == "STOP"; }

std::optional<uint64_t> parse_u64(std::string_view text) {
    try {
        size_t consumed = 0;
        uint64_t value = std::stoull(std::string(text), &consumed);
        if (consumed != text.size()) {
            return std::nullopt;
        }
        return value;
    } catch (...) {
        return std::nullopt;
    }
}

struct ProfilerCsvColumns {
    int time = -1;
    int zone = -1;
    int phase = -1;
    int core_x = -1;
    int core_y = -1;
    int risc = -1;
    int run_host_id = -1;
    double chip_freq_mhz = 0.0;

    bool valid() const {
        return time >= 0 && zone >= 0 && phase >= 0 && core_x >= 0 && core_y >= 0 && risc >= 0;
    }
};

ProfilerCsvColumns read_profiler_csv_columns(const std::filesystem::path& path) {
    ProfilerCsvColumns columns;
    std::ifstream file(path);
    if (!file) {
        return columns;
    }

    std::string line;
    while (std::getline(file, line)) {
        constexpr std::string_view freq_label = "CHIP_FREQ[MHz]:";
        auto freq_pos = line.find(freq_label);
        if (freq_pos != std::string::npos) {
            auto freq_text = line.substr(freq_pos + freq_label.size());
            if (auto comma_pos = freq_text.find(','); comma_pos != std::string::npos) {
                freq_text = freq_text.substr(0, comma_pos);
            }
            try {
                columns.chip_freq_mhz = std::stod(trim(freq_text));
            } catch (...) {
                columns.chip_freq_mhz = 0.0;
            }
        }

        auto header = split_csv_line(line);
        if (column_index(header, "zone name") < 0 || column_index(header, "time[cycles since reset]") < 0) {
            continue;
        }

        columns.time = column_index(header, "time[cycles since reset]");
        columns.zone = column_index(header, "zone name");
        columns.phase = column_index(header, "type");
        if (columns.phase < 0) {
            columns.phase = column_index(header, "zone phase");
        }
        columns.core_x = column_index(header, "core_x");
        columns.core_y = column_index(header, "core_y");
        columns.risc = column_index(header, "RISC processor type");
        columns.run_host_id = column_index(header, "run host ID");
        return columns;
    }

    return columns;
}

struct ZoneKey {
    std::string zone;
    std::string core_x;
    std::string core_y;
    std::string risc;
    std::string run_host_id;

    bool operator<(const ZoneKey& other) const {
        return std::tie(zone, core_x, core_y, risc, run_host_id) <
               std::tie(other.zone, other.core_x, other.core_y, other.risc, other.run_host_id);
    }
};

struct ZoneAggregate {
    std::vector<uint64_t> durations;
    uint64_t first_start_cycles = std::numeric_limits<uint64_t>::max();
    uint64_t last_end_cycles = 0;
};

ZoneStats make_zone_stats(ZoneAggregate aggregate, double chip_freq_mhz, uint64_t run_start_cycles) {
    ZoneStats stats;
    if (aggregate.durations.empty()) {
        return stats;
    }
    auto& durations = aggregate.durations;
    std::sort(durations.begin(), durations.end());
    stats.count = durations.size();
    stats.min_cycles = durations.front();
    stats.max_cycles = durations.back();
    stats.avg_cycles = std::accumulate(durations.begin(), durations.end(), uint64_t{0}) / durations.size();
    stats.first_start_cycles = aggregate.first_start_cycles;
    stats.last_end_cycles = aggregate.last_end_cycles;
    stats.span_cycles = stats.last_end_cycles - stats.first_start_cycles;
    if (chip_freq_mhz > 0.0) {
        stats.first_start_us = static_cast<double>(stats.first_start_cycles - run_start_cycles) / chip_freq_mhz;
        stats.last_end_us = static_cast<double>(stats.last_end_cycles - run_start_cycles) / chip_freq_mhz;
        stats.span_us = static_cast<double>(stats.span_cycles) / chip_freq_mhz;
        stats.critical_us = static_cast<double>(stats.max_cycles) / chip_freq_mhz;
    }
    return stats;
}

DeviceZoneSummary read_device_zone_summary(const std::filesystem::path& csv_path, uintmax_t csv_offset) {
    DeviceZoneSummary summary;
    summary.csv_path = csv_path;

    auto columns = read_profiler_csv_columns(csv_path);
    summary.chip_freq_mhz = columns.chip_freq_mhz;
    if (!columns.valid()) {
        return summary;
    }

    std::ifstream file(csv_path);
    if (!file) {
        return summary;
    }
    file.seekg(static_cast<std::streamoff>(csv_offset), std::ios::beg);

    std::map<ZoneKey, std::vector<uint64_t>> open_zones;
    std::map<std::string, ZoneAggregate> aggregates_by_zone;
    uint64_t run_start_cycles = std::numeric_limits<uint64_t>::max();
    const int max_column = std::max({columns.time, columns.zone, columns.phase, columns.core_x, columns.core_y, columns.risc});

    std::string line;
    while (std::getline(file, line)) {
        auto row = split_csv_line(line);
        if (static_cast<int>(row.size()) <= max_column || row[columns.zone] == "zone name") {
            continue;
        }
        auto timestamp = parse_u64(row[columns.time]);
        if (!timestamp.has_value()) {
            continue;
        }
        ZoneKey key{
            .zone = row[columns.zone],
            .core_x = row[columns.core_x],
            .core_y = row[columns.core_y],
            .risc = row[columns.risc],
            .run_host_id = columns.run_host_id >= 0 && columns.run_host_id < static_cast<int>(row.size())
                               ? row[columns.run_host_id]
                               : ""};
        const auto& phase = row[columns.phase];
        if (phase_is_start(phase)) {
            open_zones[key].push_back(*timestamp);
        } else if (phase_is_end(phase)) {
            auto& starts = open_zones[key];
            if (!starts.empty() && *timestamp >= starts.back()) {
                const uint64_t start = starts.back();
                const uint64_t end = *timestamp;
                auto& aggregate = aggregates_by_zone[key.zone];
                aggregate.durations.push_back(end - start);
                aggregate.first_start_cycles = std::min(aggregate.first_start_cycles, start);
                aggregate.last_end_cycles = std::max(aggregate.last_end_cycles, end);
                run_start_cycles = std::min(run_start_cycles, start);
                starts.pop_back();
            }
        }
    }

    if (run_start_cycles == std::numeric_limits<uint64_t>::max()) {
        return summary;
    }

    for (const auto& [zone, aggregate] : aggregates_by_zone) {
        summary.zones[zone] = make_zone_stats(aggregate, summary.chip_freq_mhz, run_start_cycles);
    }
    return summary;
}

SummaryStats summarize_us(std::vector<uint64_t> values) {
    SummaryStats stats;
    if (values.empty()) {
        return stats;
    }
    std::sort(values.begin(), values.end());
    stats.best_us = values.front();
    stats.worst_us = values.back();
    stats.avg_us = std::accumulate(values.begin(), values.end(), uint64_t{0}) / values.size();
    return stats;
}

std::size_t q_index(const fap::ShapeConfig& shape, uint32_t b, uint32_t h, uint32_t s, uint32_t d) {
    return (((static_cast<std::size_t>(b) * shape.nh + h) * shape.s + s) * shape.d + d);
}

std::size_t paged_kv_index(const fap::ShapeConfig& shape, uint32_t block, uint32_t kvh, uint32_t page_offset, uint32_t d) {
    return (((static_cast<std::size_t>(block) * shape.nkv + kvh) * shape.page + page_offset) * shape.d + d);
}

std::size_t kv_index(const fap::ShapeConfig& shape, uint32_t b, uint32_t kvh, uint32_t s, uint32_t d) {
    return (((static_cast<std::size_t>(b) * shape.nkv + kvh) * shape.s + s) * shape.d + d);
}

float deterministic_value(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t salt) {
    uint32_t x = 0x9e3779b9u;
    x ^= a + 0x85ebca6bu + (x << 6) + (x >> 2);
    x ^= b + 0xc2b2ae35u + (x << 6) + (x >> 2);
    x ^= c + salt + (x << 6) + (x >> 2);
    x ^= d + 0x27d4eb2du + (x << 6) + (x >> 2);
    return (static_cast<float>(x % 2048u) / 1024.0f) - 1.0f;
}

fap::HostInputs make_inputs(const fap::ShapeConfig& shape) {
    const uint32_t blocks_per_seq = shape.s / shape.page;
    const uint32_t total_blocks = shape.b * blocks_per_seq;
    fap::HostInputs inputs;
    inputs.q.resize(static_cast<std::size_t>(shape.b) * shape.nh * shape.s * shape.d);
    inputs.k.resize(static_cast<std::size_t>(shape.b) * shape.nkv * shape.s * shape.d);
    inputs.v.resize(inputs.k.size());
    inputs.paged_k.resize(static_cast<std::size_t>(total_blocks) * shape.nkv * shape.page * shape.d);
    inputs.paged_v.resize(inputs.paged_k.size());
    inputs.page_table.resize(static_cast<std::size_t>(shape.b) * blocks_per_seq);

    for (uint32_t b = 0; b < shape.b; ++b) {
        for (uint32_t block = 0; block < blocks_per_seq; ++block) {
            inputs.page_table[static_cast<std::size_t>(b) * blocks_per_seq + block] =
                static_cast<int32_t>(b * blocks_per_seq + block);
        }
    }
    for (uint32_t b = 0; b < shape.b; ++b) {
        for (uint32_t h = 0; h < shape.nh; ++h) {
            for (uint32_t s = 0; s < shape.s; ++s) {
                for (uint32_t d = 0; d < shape.d; ++d) {
                    inputs.q[q_index(shape, b, h, s, d)] = deterministic_value(b, h, s, d, 17);
                }
            }
        }
    }
    for (uint32_t b = 0; b < shape.b; ++b) {
        for (uint32_t kvh = 0; kvh < shape.nkv; ++kvh) {
            for (uint32_t s = 0; s < shape.s; ++s) {
                const uint32_t physical_block =
                    static_cast<uint32_t>(inputs.page_table[static_cast<std::size_t>(b) * blocks_per_seq + s / shape.page]);
                for (uint32_t d = 0; d < shape.d; ++d) {
                    const float k_value = deterministic_value(b, kvh, s, d, 101);
                    const float v_value = deterministic_value(b, kvh, s, d, 211);
                    inputs.k[kv_index(shape, b, kvh, s, d)] = k_value;
                    inputs.v[kv_index(shape, b, kvh, s, d)] = v_value;
                    inputs.paged_k[paged_kv_index(shape, physical_block, kvh, s % shape.page, d)] = k_value;
                    inputs.paged_v[paged_kv_index(shape, physical_block, kvh, s % shape.page, d)] = v_value;
                }
            }
        }
    }
    return inputs;
}

void validate_shape(const fap::ShapeConfig& shape) {
    if (shape.b == 0 || shape.nh == 0 || shape.nkv == 0 || shape.s == 0 || shape.d == 0 || shape.prefill == 0 ||
        shape.q_chunk == 0 || shape.k_chunk == 0 || shape.page == 0) {
        TT_THROW("shape dimensions must be non-zero");
    }
    if (shape.nh % shape.nkv != 0) {
        TT_THROW("nh must be divisible by nkv, got nh={} nkv={}", shape.nh, shape.nkv);
    }
    if (shape.s % shape.prefill != 0) {
        TT_THROW("s must be divisible by prefill, got s={} prefill={}", shape.s, shape.prefill);
    }
    if (shape.s % shape.page != 0) {
        TT_THROW("s must be divisible by page, got s={} page={}", shape.s, shape.page);
    }
    if (shape.prefill % shape.q_chunk != 0 || shape.prefill % shape.k_chunk != 0) {
        TT_THROW("prefill must be divisible by q_chunk and k_chunk");
    }
    if (shape.s % shape.q_chunk != 0 || shape.s % shape.k_chunk != 0) {
        TT_THROW("s must be divisible by q_chunk and k_chunk");
    }
    if (shape.q_chunk % tt::constants::TILE_HEIGHT != 0 || shape.k_chunk % tt::constants::TILE_HEIGHT != 0 ||
        shape.d % tt::constants::TILE_WIDTH != 0 || shape.page % tt::constants::TILE_HEIGHT != 0 ||
        shape.prefill % tt::constants::TILE_HEIGHT != 0 || shape.s % tt::constants::TILE_HEIGHT != 0) {
        TT_THROW("s, prefill, q_chunk, k_chunk, d, and page must be tile aligned");
    }
}

std::optional<fap::RunMode> parse_mode(std::string_view text) {
    if (text == "eager") {
        return fap::RunMode::Eager;
    }
    if (text == "prepared") {
        return fap::RunMode::Prepared;
    }
    if (text == "prepared_no_q_copy") {
        return fap::RunMode::PreparedNoQCopy;
    }
    if (text == "trace") {
        return fap::RunMode::Trace;
    }
    return std::nullopt;
}

std::optional<fap::Variant> parse_variant(std::string_view text) {
    if (text == "ttnn_sdpa_baseline") {
        return fap::Variant::TtnnSdpaBaseline;
    }
    if (text == "ttnn_chunked_baseline") {
        return fap::Variant::TtnnChunkedBaseline;
    }
    if (text == "copied_sdpa") {
        return fap::Variant::CopiedSdpa;
    }
    if (text == "copied_chunked") {
        return fap::Variant::CopiedChunked;
    }
    return std::nullopt;
}

std::optional<fap::PipelineMode> parse_pipeline_mode(std::string_view text) {
    if (text == "auto") {
        return fap::PipelineMode::Auto;
    }
    if (text == "stream_h1") {
        return fap::PipelineMode::StreamH1;
    }
    if (text == "qktv_h1") {
        return fap::PipelineMode::QktvH1;
    }
    if (text == "salad_first") {
        return fap::PipelineMode::SaladFirst;
    }
    if (text == "qktv_h1_salad_first") {
        return fap::PipelineMode::QktvH1SaladFirst;
    }
    if (text == "non_streaming") {
        return fap::PipelineMode::NonStreaming;
    }
    return std::nullopt;
}

std::optional<fap::GridPolicy> parse_grid_policy(std::string_view text) {
    if (text == "default") {
        return fap::GridPolicy::Default;
    }
    if (text == "copied_balanced_q") {
        return fap::GridPolicy::CopiedBalancedQ;
    }
    return std::nullopt;
}

std::optional<fap::ShapeConfig> preset_shape(std::string_view name) {
    for (const auto& shape : Options::default_shapes()) {
        if (shape.name == name) {
            return shape;
        }
    }
    return std::nullopt;
}

std::optional<fap::ShapeConfig> parse_shape(std::string_view text) {
    uint32_t values[9] = {};
    size_t value_index = 0;
    size_t start = 0;
    while (start <= text.size() && value_index < 9) {
        size_t end = text.find(',', start);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        auto token = text.substr(start, end - start);
        if (token.empty()) {
            return std::nullopt;
        }
        try {
            values[value_index++] = static_cast<uint32_t>(std::stoul(std::string(token)));
        } catch (...) {
            return std::nullopt;
        }
        if (end == text.size()) {
            break;
        }
        start = end + 1;
    }
    if (value_index != 9) {
        return std::nullopt;
    }
    return fap::ShapeConfig{
        .name = fmt::format("custom_B{}_H{}_KVH{}_S{}_D{}_P{}_Q{}_K{}_PAGE{}",
                            values[0],
                            values[1],
                            values[2],
                            values[3],
                            values[4],
                            values[5],
                            values[6],
                            values[7],
                            values[8]),
        .b = values[0],
        .nh = values[1],
        .nkv = values[2],
        .s = values[3],
        .d = values[4],
        .prefill = values[5],
        .q_chunk = values[6],
        .k_chunk = values[7],
        .page = values[8]};
}

std::optional<CoreCoord> parse_grid(std::string_view text) {
    const size_t comma = text.find(',');
    if (comma == std::string_view::npos || comma == 0 || comma + 1 >= text.size()) {
        return std::nullopt;
    }
    auto x = parse_u64(text.substr(0, comma));
    auto y = parse_u64(text.substr(comma + 1));
    if (!x.has_value() || !y.has_value() || *x == 0 || *y == 0 ||
        *x > std::numeric_limits<uint32_t>::max() || *y > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return CoreCoord{static_cast<uint32_t>(*x), static_cast<uint32_t>(*y)};
}

std::optional<std::pair<uint32_t, uint32_t>> parse_chunks(std::string_view text) {
    const size_t comma = text.find(',');
    if (comma == std::string_view::npos || comma == 0 || comma + 1 >= text.size()) {
        return std::nullopt;
    }
    auto q = parse_u64(text.substr(0, comma));
    auto k = parse_u64(text.substr(comma + 1));
    if (!q.has_value() || !k.has_value() || *q == 0 || *k == 0 ||
        *q > std::numeric_limits<uint32_t>::max() || *k > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(*q), static_cast<uint32_t>(*k)};
}

std::optional<uint32_t> parse_qk_softmax_profile_stage(std::string_view value) {
    if (value == "none" || value == "0") {
        return 0;
    }
    if (value == "wait_max" || value == "1") {
        return 1;
    }
    if (value == "sub_math" || value == "2") {
        return 2;
    }
    if (value == "wait_sub" || value == "3") {
        return 3;
    }
    if (value == "exp_sfpu" || value == "4") {
        return 4;
    }
    if (value == "pack" || value == "5") {
        return 5;
    }
    return std::nullopt;
}

std::string qk_softmax_profile_stage_label(uint32_t stage) {
    switch (stage) {
        case 0: return "none";
        case 1: return "wait_max";
        case 2: return "sub_math";
        case 3: return "wait_sub";
        case 4: return "exp_sfpu";
        case 5: return "pack";
        default: return fmt::format("unknown_{}", stage);
    }
}

std::optional<uint32_t> parse_qk_softmax_schedule(std::string_view value) {
    if (value == "before_matmul" || value == "0") {
        return 0;
    }
    if (value == "after_matmul" || value == "1") {
        return 1;
    }
    if (value == "after_matmul_except_final_kt" || value == "2") {
        return 2;
    }
    return std::nullopt;
}

std::string qk_softmax_schedule_label(uint32_t schedule) {
    switch (schedule) {
        case 0: return "before_matmul";
        case 1: return "after_matmul";
        case 2: return "after_matmul_except_final_kt";
        default: return fmt::format("unknown_{}", schedule);
    }
}

std::string copied_qk_subblock_label(const fap::CopiedKernelOptions& options) {
    if (!options.qk_subblock_override.has_value()) {
        return "auto";
    }
    return fmt::format("{}x{}", options.qk_subblock_override->first, options.qk_subblock_override->second);
}

std::string copied_q_buffer_factor_label(const fap::CopiedKernelOptions& options) {
    if (!options.q_buffer_factor_override.has_value()) {
        return "auto";
    }
    return fmt::format("{}", *options.q_buffer_factor_override);
}

std::string copied_dst_sync_label(const fap::CopiedKernelOptions& options) {
    if (!options.dst_full_sync_override.has_value()) {
        return "auto";
    }
    return *options.dst_full_sync_override ? "full" : "half";
}

void apply_chunk_override(std::vector<fap::ShapeConfig>& shapes, std::pair<uint32_t, uint32_t> chunks) {
    for (auto& shape : shapes) {
        if (shape.q_chunk == chunks.first && shape.k_chunk == chunks.second) {
            continue;
        }
        shape.q_chunk = chunks.first;
        shape.k_chunk = chunks.second;
        shape.name = fmt::format("{}_q{}_k{}", shape.name, chunks.first, chunks.second);
    }
}

std::string grid_label(
    const Options& options,
    fap::Variant variant,
    const fap::ShapeConfig& shape,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const auto resolved_grid = fap::resolve_flash_attention_grid(
        variant, options.grid_policy, options.grid_override, shape, mesh_device->compute_with_storage_grid_size());
    if (!resolved_grid.has_value()) {
        return "auto";
    }
    return fmt::format("{}x{}", resolved_grid->x, resolved_grid->y);
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool saw_shape_filter = false;
    bool saw_variant_filter = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        auto require_value = [&](std::string_view flag) -> const char* {
            if (i + 1 >= argc) {
                TT_THROW("{} requires a value", flag);
            }
            return argv[++i];
        };

        if (arg == "--iters") {
            options.measured_iters = static_cast<uint32_t>(std::stoul(require_value(arg)));
        } else if (arg == "--warmup") {
            options.warmup_iters = static_cast<uint32_t>(std::stoul(require_value(arg)));
        } else if (arg == "--device-id") {
            options.device_id = static_cast<uint32_t>(std::stoul(require_value(arg)));
        } else if (arg == "--high-precision") {
            options.high_precision = true;
        } else if (arg == "--no-device-profiler-read") {
            options.read_device_profiler = false;
        } else if (arg == "--check-correctness") {
            options.check_correctness = true;
        } else if (arg == "--pipeline") {
            auto pipeline_mode = parse_pipeline_mode(require_value(arg));
            if (!pipeline_mode.has_value()) {
                TT_THROW("--pipeline expects auto, stream_h1, qktv_h1, salad_first, qktv_h1_salad_first, or non_streaming");
            }
            options.pipeline_mode = *pipeline_mode;
        } else if (arg == "--pipeline-depth") {
            options.pipeline_depth = static_cast<uint32_t>(std::stoul(require_value(arg)));
        } else if (arg == "--grid-policy") {
            auto grid_policy = parse_grid_policy(require_value(arg));
            if (!grid_policy.has_value()) {
                TT_THROW("--grid-policy expects default or copied_balanced_q");
            }
            options.grid_policy = *grid_policy;
        } else if (arg == "--grid") {
            auto grid = parse_grid(require_value(arg));
            if (!grid.has_value()) {
                TT_THROW("--grid expects X,Y with positive integer dimensions");
            }
            options.grid_override = *grid;
        } else if (arg == "--chunks") {
            auto chunks = parse_chunks(require_value(arg));
            if (!chunks.has_value()) {
                TT_THROW("--chunks expects Q,K with positive integer chunk sizes");
            }
            options.chunk_override = *chunks;
        } else if (arg == "--qk-subblock") {
            auto subblock = parse_chunks(require_value(arg));
            if (!subblock.has_value()) {
                TT_THROW("--qk-subblock expects H,W with positive integer tile dimensions");
            }
            options.copied_kernel_options.qk_subblock_override = *subblock;
        } else if (arg == "--q-buffer-factor") {
            const uint32_t q_buffer_factor = static_cast<uint32_t>(std::stoul(require_value(arg)));
            if (q_buffer_factor == 0 || q_buffer_factor > 4) {
                TT_THROW("--q-buffer-factor expects a value in [1, 4]");
            }
            options.copied_kernel_options.q_buffer_factor_override = q_buffer_factor;
        } else if (arg == "--dst-full-sync") {
            options.copied_kernel_options.dst_full_sync_override = true;
        } else if (arg == "--dst-half-sync") {
            options.copied_kernel_options.dst_full_sync_override = false;
        } else if (arg == "--qk-softmax-profile") {
            auto stage = parse_qk_softmax_profile_stage(require_value(arg));
            if (!stage.has_value()) {
                TT_THROW("--qk-softmax-profile expects none, wait_max, sub_math, wait_sub, exp_sfpu, or pack");
            }
            options.copied_kernel_options.qk_softmax_profile_stage = *stage;
        } else if (arg == "--qk-softmax-schedule") {
            auto schedule = parse_qk_softmax_schedule(require_value(arg));
            if (!schedule.has_value()) {
                TT_THROW("--qk-softmax-schedule expects before_matmul, after_matmul, or after_matmul_except_final_kt");
            }
            options.copied_kernel_options.qk_softmax_schedule = *schedule;
        } else if (arg == "--preset") {
            std::string_view value(require_value(arg));
            if (!saw_shape_filter) {
                options.shapes.clear();
                saw_shape_filter = true;
            }
            if (value == "all") {
                auto defaults = Options::default_shapes();
                options.shapes.insert(options.shapes.end(), defaults.begin(), defaults.end());
            } else {
                auto shape = preset_shape(value);
                if (!shape.has_value()) {
                    TT_THROW("--preset expects a built-in preset name or all");
                }
                options.shapes.push_back(*shape);
            }
        } else if (arg == "--shape") {
            auto shape = parse_shape(require_value(arg));
            if (!shape.has_value()) {
                TT_THROW("--shape expects B,H,KVH,S,D,prefill,q,k,page");
            }
            if (!saw_shape_filter) {
                options.shapes.clear();
                saw_shape_filter = true;
            }
            options.shapes.push_back(*shape);
        } else if (arg == "--mode") {
            std::string_view value(require_value(arg));
            if (value == "all") {
                options.modes = {
                    fap::RunMode::Eager,
                    fap::RunMode::Prepared,
                    fap::RunMode::PreparedNoQCopy,
                    fap::RunMode::Trace};
            } else {
                auto mode = parse_mode(value);
                if (!mode.has_value()) {
                    TT_THROW("--mode expects eager, prepared, prepared_no_q_copy, trace, or all");
                }
                options.modes = {*mode};
            }
        } else if (arg == "--variant") {
            std::string_view value(require_value(arg));
            if (value == "all") {
                options.variants = {
                    fap::Variant::TtnnSdpaBaseline,
                    fap::Variant::TtnnChunkedBaseline,
                    fap::Variant::CopiedSdpa,
                    fap::Variant::CopiedChunked};
                saw_variant_filter = true;
                continue;
            }
            auto variant = parse_variant(value);
            if (!variant.has_value()) {
                TT_THROW("--variant expects ttnn_sdpa_baseline, ttnn_chunked_baseline, copied_sdpa, copied_chunked, or all");
            }
            if (!saw_variant_filter) {
                options.variants.clear();
                saw_variant_filter = true;
            }
            options.variants.push_back(*variant);
        } else if (arg == "--help") {
            fmt::print(
                "Usage: flash_attention_profile [--preset BUILTIN_PRESET|all] "
                "[--shape B,H,KVH,S,D,prefill,q,k,page] "
                "[--chunks Q,K] "
                "[--variant ttnn_sdpa_baseline|ttnn_chunked_baseline|copied_sdpa|copied_chunked|all] "
                "[--mode eager|prepared|prepared_no_q_copy|trace|all] "
                "[--pipeline auto|stream_h1|qktv_h1|salad_first|qktv_h1_salad_first|non_streaming] "
                "[--pipeline-depth N] [--grid-policy default|copied_balanced_q] "
                "[--qk-subblock H,W] [--q-buffer-factor N] [--dst-full-sync|--dst-half-sync] "
                "[--qk-softmax-profile none|wait_max|sub_math|wait_sub|exp_sfpu|pack] "
                "[--qk-softmax-schedule before_matmul|after_matmul|after_matmul_except_final_kt] "
                "[--grid X,Y] [--warmup N] [--iters N] [--device-id N] "
                "[--high-precision] [--check-correctness] [--no-device-profiler-read]\n");
            std::exit(0);
        } else {
            TT_THROW("Unknown argument: {}", arg);
        }
    }
    TT_FATAL(options.measured_iters > 0, "--iters must be > 0");
    TT_FATAL(options.pipeline_depth > 0, "--pipeline-depth must be > 0");
    TT_FATAL(!options.shapes.empty(), "at least one shape must be selected");
    if (options.chunk_override.has_value()) {
        apply_chunk_override(options.shapes, *options.chunk_override);
    }
    return options;
}

const char* unsupported_combination_reason(
    fap::Variant variant,
    fap::RunMode mode,
    fap::PipelineMode pipeline_mode,
    uint32_t pipeline_depth) {
    if (pipeline_mode != fap::PipelineMode::Auto && !fap::variant_is_copied(variant)) {
        return "pipeline modes other than auto are only implemented for copied variants";
    }
    if (pipeline_depth != 2 && !fap::variant_is_copied(variant)) {
        return "pipeline-depth overrides are only implemented for copied variants";
    }
    if (mode == fap::RunMode::Trace && !fap::variant_is_chunked(variant)) {
        return "trace requires a chunked runtime chunk-start tensor";
    }
    return nullptr;
}

std::optional<DeviceZoneSummary> read_device_profiler(
    const Options& options,
    std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const fap::ShapeConfig& shape,
    fap::Variant variant,
    fap::RunMode mode,
    uintmax_t csv_offset) {
    if (!options.read_device_profiler) {
        return std::nullopt;
    }
    ReadMeshDeviceProfilerResults(*mesh_device);
    auto csv_path = profiler_csv_path();
    auto summary = read_device_zone_summary(csv_path, csv_offset);
    fmt::print(
        "FLASH_ATTN_PROFILE_DEVICE shape={} variant={} mode={} pipeline={} pipeline_depth={} qk_subblock={} q_buffer_factor={} dst_sync={} qk_softmax_profile={} qk_softmax_schedule={} grid_policy={} grid={} csv={} chip_freq_mhz={:.3f} zones={}\n",
        shape.name,
        fap::variant_name(variant),
        fap::run_mode_name(mode),
        fap::pipeline_mode_name(options.pipeline_mode),
        options.pipeline_depth,
        copied_qk_subblock_label(options.copied_kernel_options),
        copied_q_buffer_factor_label(options.copied_kernel_options),
        copied_dst_sync_label(options.copied_kernel_options),
        qk_softmax_profile_stage_label(options.copied_kernel_options.qk_softmax_profile_stage),
        qk_softmax_schedule_label(options.copied_kernel_options.qk_softmax_schedule),
        fap::grid_policy_name(options.grid_policy),
        grid_label(options, variant, shape, mesh_device),
        csv_path.string(),
        summary.chip_freq_mhz,
        summary.zones.size());
    for (const auto& [zone, stats] : summary.zones) {
        fmt::print(
            "FLASH_ATTN_PROFILE_STAGE_RESULT shape={} variant={} mode={} pipeline={} pipeline_depth={} qk_subblock={} q_buffer_factor={} dst_sync={} qk_softmax_profile={} qk_softmax_schedule={} grid_policy={} grid={} zone=\"{}\" count={} min_cycles={} "
            "avg_cycles={} max_cycles={} critical_us={:.3f} first_start_us={:.3f} last_end_us={:.3f} span_us={:.3f}\n",
            shape.name,
            fap::variant_name(variant),
            fap::run_mode_name(mode),
            fap::pipeline_mode_name(options.pipeline_mode),
            options.pipeline_depth,
            copied_qk_subblock_label(options.copied_kernel_options),
            copied_q_buffer_factor_label(options.copied_kernel_options),
            copied_dst_sync_label(options.copied_kernel_options),
            qk_softmax_profile_stage_label(options.copied_kernel_options.qk_softmax_profile_stage),
            qk_softmax_schedule_label(options.copied_kernel_options.qk_softmax_schedule),
            fap::grid_policy_name(options.grid_policy),
            grid_label(options, variant, shape, mesh_device),
            zone,
            stats.count,
            stats.min_cycles,
            stats.avg_cycles,
            stats.max_cycles,
            stats.critical_us,
            stats.first_start_us,
            stats.last_end_us,
            stats.span_us);
    }
    return summary;
}

void run_one(
    const Options& options,
    const fap::ShapeConfig& shape,
    const fap::HostInputs& inputs,
    std::shared_ptr<distributed::MeshDevice>& mesh_device,
    fap::Variant variant,
    fap::RunMode mode) {
    if (const char* reason = unsupported_combination_reason(
            variant,
            mode,
            options.pipeline_mode,
            options.pipeline_depth)) {
        fmt::print(
            "FLASH_ATTN_PROFILE_SKIP shape={} variant={} mode={} pipeline={} pipeline_depth={} grid_policy={} grid={} reason=\"{}\"\n",
            shape.name,
            fap::variant_name(variant),
            fap::run_mode_name(mode),
            fap::pipeline_mode_name(options.pipeline_mode),
            options.pipeline_depth,
            fap::grid_policy_name(options.grid_policy),
            grid_label(options, variant, shape, mesh_device),
            reason);
        return;
    }

    mesh_device->disable_and_clear_program_cache();
    mesh_device->enable_program_cache();

    auto runner = fap::prepare_flash_attention_runner(
        variant,
        mode,
        options.pipeline_mode,
        options.pipeline_depth,
        options.copied_kernel_options,
        options.grid_policy,
        options.grid_override,
        shape,
        inputs,
        options.high_precision,
        mesh_device);
    for (uint32_t warmup = 0; warmup < options.warmup_iters; ++warmup) {
        (void)runner->run(warmup, false);
    }
    if (options.read_device_profiler) {
        ReadMeshDeviceProfilerResults(*mesh_device);
    }
    if (mode == fap::RunMode::Trace) {
        runner->prepare_trace(options.read_device_profiler);
    }

    const auto csv_offset = safe_file_size(profiler_csv_path());
    std::vector<uint64_t> total_us;
    std::vector<uint64_t> call_us;
    std::vector<uint64_t> sync_us;
    std::vector<uint64_t> copy_q_us;
    std::vector<uint64_t> copy_start_us;
    std::size_t cache_entries = 0;
    for (uint32_t iter = 0; iter < options.measured_iters; ++iter) {
        auto stats = runner->run(iter, true);
        total_us.push_back(stats.total_us);
        call_us.push_back(stats.call_us);
        sync_us.push_back(stats.sync_us);
        copy_q_us.push_back(stats.copy_q_us);
        copy_start_us.push_back(stats.copy_start_us);
        cache_entries = stats.cache_entries;
    }
    distributed::Synchronize(mesh_device.get(), std::nullopt);

    auto total = summarize_us(total_us);
    auto call = summarize_us(call_us);
    auto sync = summarize_us(sync_us);
    auto copy_q = summarize_us(copy_q_us);
    auto copy_start = summarize_us(copy_start_us);
    fmt::print(
        "FLASH_ATTN_PROFILE_RESULT shape={} variant={} mode={} pipeline={} pipeline_depth={} qk_subblock={} q_buffer_factor={} dst_sync={} qk_softmax_profile={} qk_softmax_schedule={} grid_policy={} B={} H={} KVH={} S={} D={} prefill={} q={} k={} page={} "
        "grid={} iters={} avg_ms={:.3f} best_ms={:.3f} worst_ms={:.3f} call_avg_ms={:.3f} sync_avg_ms={:.3f} "
        "copy_q_avg_ms={:.3f} copy_start_avg_ms={:.3f} cache_entries={}\n",
        shape.name,
        fap::variant_name(variant),
        fap::run_mode_name(mode),
        fap::pipeline_mode_name(options.pipeline_mode),
        options.pipeline_depth,
        copied_qk_subblock_label(options.copied_kernel_options),
        copied_q_buffer_factor_label(options.copied_kernel_options),
        copied_dst_sync_label(options.copied_kernel_options),
        qk_softmax_profile_stage_label(options.copied_kernel_options.qk_softmax_profile_stage),
        qk_softmax_schedule_label(options.copied_kernel_options.qk_softmax_schedule),
        fap::grid_policy_name(options.grid_policy),
        shape.b,
        shape.nh,
        shape.nkv,
        shape.s,
        shape.d,
        shape.prefill,
        shape.q_chunk,
        shape.k_chunk,
        shape.page,
        grid_label(options, variant, shape, mesh_device),
        options.measured_iters,
        static_cast<double>(total.avg_us) / 1000.0,
        static_cast<double>(total.best_us) / 1000.0,
        static_cast<double>(total.worst_us) / 1000.0,
        static_cast<double>(call.avg_us) / 1000.0,
        static_cast<double>(sync.avg_us) / 1000.0,
        static_cast<double>(copy_q.avg_us) / 1000.0,
        static_cast<double>(copy_start.avg_us) / 1000.0,
        cache_entries);

    (void)read_device_profiler(options, mesh_device, shape, variant, mode, csv_offset);
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    try {
        const Options options = parse_options(argc, argv);
        for (const auto& shape : options.shapes) {
            validate_shape(shape);
        }
        if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
            fmt::print(
                stderr,
                "WARNING: set TT_METAL_DEVICE_PROFILER=1 to collect device profiler zones in "
                "generated/profiler/.logs/profile_log_device.csv\n");
        }

        mesh_device = distributed::MeshDevice::create_unit_mesh(options.device_id);
        mesh_device->enable_program_cache();

        for (const auto& shape : options.shapes) {
            fmt::print(
                "# FlashAttention profile shape={} B={} H={} KVH={} S={} D={} prefill={} q={} k={} page={} "
                "warmup={} iters={} high_precision={} pipeline={} pipeline_depth={} qk_subblock={} "
                "q_buffer_factor={} dst_sync={} qk_softmax_profile={} qk_softmax_schedule={} grid_policy={} grid={}\n",
                shape.name,
                shape.b,
                shape.nh,
                shape.nkv,
                shape.s,
                shape.d,
                shape.prefill,
                shape.q_chunk,
                shape.k_chunk,
                shape.page,
                options.warmup_iters,
                options.measured_iters,
                options.high_precision ? "true" : "false",
                fap::pipeline_mode_name(options.pipeline_mode),
                options.pipeline_depth,
                copied_qk_subblock_label(options.copied_kernel_options),
                copied_q_buffer_factor_label(options.copied_kernel_options),
                copied_dst_sync_label(options.copied_kernel_options),
                qk_softmax_profile_stage_label(options.copied_kernel_options.qk_softmax_profile_stage),
                qk_softmax_schedule_label(options.copied_kernel_options.qk_softmax_schedule),
                fap::grid_policy_name(options.grid_policy),
                "per-variant");
            auto inputs = make_inputs(shape);
            if (options.check_correctness) {
                auto correctness_results = fap::check_flash_attention_correctness(
                    shape,
                    inputs,
                    options.high_precision,
                    mesh_device,
                    options.variants,
                    options.pipeline_mode,
                    options.pipeline_depth,
                    options.copied_kernel_options,
                    options.grid_policy,
                    options.grid_override);
                for (const auto& result : correctness_results) {
                    fmt::print(
                        "FLASH_ATTN_PROFILE_CORRECTNESS shape={} pipeline={} pipeline_depth={} grid_policy={} baseline={} candidate={} elements={} "
                        "max_abs_diff={:.6f} mean_abs_diff={:.6f} tolerance={:.6f} passed={}\n",
                        shape.name,
                        fap::pipeline_mode_name(options.pipeline_mode),
                        options.pipeline_depth,
                        fap::grid_policy_name(options.grid_policy),
                        fap::variant_name(result.baseline),
                        fap::variant_name(result.candidate),
                        result.elements,
                        result.max_abs_diff,
                        result.mean_abs_diff,
                        result.tolerance,
                        result.passed ? "true" : "false");
                    if (!result.passed) {
                        TT_THROW(
                            "correctness check failed for {} vs {}",
                            fap::variant_name(result.baseline),
                            fap::variant_name(result.candidate));
                    }
                }
                if (correctness_results.empty()) {
                    fmt::print(
                        "FLASH_ATTN_PROFILE_CORRECTNESS_SKIP shape={} pipeline={} pipeline_depth={} grid_policy={} reason=\"select both baseline and copied variants to compare\"\n",
                        shape.name,
                        fap::pipeline_mode_name(options.pipeline_mode),
                        options.pipeline_depth,
                        fap::grid_policy_name(options.grid_policy));
                }
            }
            for (auto variant : options.variants) {
                for (auto mode : options.modes) {
                    run_one(options, shape, inputs, mesh_device, variant, mode);
                }
            }
        }
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
        if (mesh_device) {
            mesh_device->close();
        }
    }

    if (!pass) {
        TT_THROW("FlashAttention profile failed");
    }
    return 0;
}
