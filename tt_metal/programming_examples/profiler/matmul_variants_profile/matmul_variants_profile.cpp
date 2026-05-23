// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <fmt/core.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/experimental/profiler.hpp>

#include <bmm_op.hpp>
#include <matmul_profile_runner.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

void matmul_single_core(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device);

void matmul_multi_core(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device);

void matmul_multicore_reuse(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t B,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device);

void matmul_multicore_reuse_mcast(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t B,
    std::shared_ptr<distributed::MeshDevice>& mesh_device);

namespace {

using Clock = std::chrono::steady_clock;

struct Shape {
    std::string name;
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

enum class RunMode {
    Eager,
    Prepared,
    Trace,
};

enum class Variant {
    SingleCore,
    MultiCore,
    MulticoreReuse,
    MulticoreReuseMcast,
};

struct Options {
    uint32_t warmup_iters = 1;
    uint32_t measured_iters = 3;
    bool include_single_core = false;
    bool read_device_profiler = true;
    std::vector<RunMode> modes = {RunMode::Eager};
    std::optional<std::vector<Variant>> variant_filter;
    std::vector<Shape> shapes = {
        {"tall_2x2_low_k", 2304, 128, 64},
        {"tall_3x2_low_k", 2400, 128, 64},
        {"tall_2x3_low_k", 2112, 192, 64},
        {"square_2x2_mid_k", 1024, 1024, 64},
        {"wide_2x2_low_k", 1024, 2048, 64},
    };
};

struct PreparedShape {
    std::vector<bfloat16> src0_tilized;
    std::vector<bfloat16> src1_tilized;
    std::vector<bfloat16> output;
};

struct Support {
    bool ok = true;
    std::string reason;
};

struct DeviceKernelSummary {
    uint64_t count = 0;
    uint64_t min_ns = 0;
    uint64_t avg_ns = 0;
    uint64_t max_ns = 0;
};

enum class Stage {
    Reader,
    Compute,
    Writer,
};

constexpr std::array<Stage, 3> kStages = {Stage::Reader, Stage::Compute, Stage::Writer};

struct StageStats {
    uint64_t count = 0;
    uint64_t min_cycles = 0;
    uint64_t avg_cycles = 0;
    uint64_t max_cycles = 0;
    double critical_us = 0.0;
};

struct DeviceStageSummary {
    std::filesystem::path csv_path;
    double chip_freq_mhz = 0.0;
    std::map<Stage, StageStats> stages;
};

const char* variant_name(Variant variant) {
    switch (variant) {
        case Variant::SingleCore: return "matmul_single_core";
        case Variant::MultiCore: return "matmul_multi_core";
        case Variant::MulticoreReuse: return "matmul_multicore_reuse";
        case Variant::MulticoreReuseMcast: return "matmul_multicore_reuse_mcast";
    }
    return "unknown";
}

const char* run_mode_name(RunMode mode) {
    switch (mode) {
        case RunMode::Eager: return "eager";
        case RunMode::Prepared: return "prepared";
        case RunMode::Trace: return "trace";
    }
    return "unknown";
}

const char* stage_name(Stage stage) {
    switch (stage) {
        case Stage::Reader: return "reader";
        case Stage::Compute: return "compute";
        case Stage::Writer: return "writer";
    }
    return "unknown";
}

const char* stage_zone_name(Variant variant, Stage stage) {
    switch (variant) {
        case Variant::SingleCore:
            switch (stage) {
                case Stage::Reader: return "MMVP_SINGLE_CORE_READER";
                case Stage::Compute: return "MMVP_SINGLE_CORE_COMPUTE";
                case Stage::Writer: return "MMVP_SINGLE_CORE_WRITER";
            }
            break;
        case Variant::MultiCore:
            switch (stage) {
                case Stage::Reader: return "MMVP_MULTI_CORE_READER";
                case Stage::Compute: return "MMVP_MULTI_CORE_COMPUTE";
                case Stage::Writer: return "MMVP_MULTI_CORE_WRITER";
            }
            break;
        case Variant::MulticoreReuse:
            switch (stage) {
                case Stage::Reader: return "MMVP_REUSE_READER";
                case Stage::Compute: return "MMVP_REUSE_COMPUTE";
                case Stage::Writer: return "MMVP_REUSE_WRITER";
            }
            break;
        case Variant::MulticoreReuseMcast:
            switch (stage) {
                case Stage::Reader: return "MMVP_REUSE_MCAST_READER";
                case Stage::Compute: return "MMVP_REUSE_MCAST_COMPUTE";
                case Stage::Writer: return "MMVP_REUSE_MCAST_WRITER";
            }
            break;
    }
    return "";
}

std::optional<Stage> stage_from_zone(Variant variant, std::string_view zone_name) {
    for (Stage stage : kStages) {
        if (zone_name == stage_zone_name(variant, stage)) {
            return stage;
        }
    }
    return std::nullopt;
}

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

bool phase_is_start(std::string_view phase) {
    return phase == "ZONE_START" || phase == "START" || phase == "BEGIN";
}

bool phase_is_end(std::string_view phase) {
    return phase == "ZONE_END" || phase == "END" || phase == "STOP";
}

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

int column_index(const std::vector<std::string>& header, std::string_view name) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

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
            auto comma_pos = freq_text.find(',');
            if (comma_pos != std::string::npos) {
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

StageStats make_stage_stats(std::vector<uint64_t> durations, double chip_freq_mhz) {
    StageStats stats;
    if (durations.empty()) {
        return stats;
    }
    std::sort(durations.begin(), durations.end());
    stats.count = durations.size();
    stats.min_cycles = durations.front();
    stats.max_cycles = durations.back();
    stats.avg_cycles = std::accumulate(durations.begin(), durations.end(), uint64_t{0}) / durations.size();
    if (chip_freq_mhz > 0.0) {
        stats.critical_us = static_cast<double>(stats.max_cycles) / chip_freq_mhz;
    }
    return stats;
}

DeviceStageSummary read_device_stage_summary(
    const std::filesystem::path& csv_path,
    uintmax_t csv_offset,
    Variant variant) {
    DeviceStageSummary summary;
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
    std::map<Stage, std::vector<uint64_t>> durations_by_stage;
    const int max_column =
        std::max({columns.time, columns.zone, columns.phase, columns.core_x, columns.core_y, columns.risc});

    std::string line;
    while (std::getline(file, line)) {
        auto row = split_csv_line(line);
        if (static_cast<int>(row.size()) <= max_column) {
            continue;
        }
        if (row[columns.zone] == "zone name") {
            continue;
        }

        auto stage = stage_from_zone(variant, row[columns.zone]);
        if (!stage.has_value()) {
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
                durations_by_stage[*stage].push_back(*timestamp - starts.back());
                starts.pop_back();
            }
        }
    }

    for (const auto& [stage, durations] : durations_by_stage) {
        summary.stages[stage] = make_stage_stats(durations, summary.chip_freq_mhz);
    }
    return summary;
}

std::optional<Shape> parse_shape(std::string_view text) {
    uint32_t values[3] = {};
    size_t value_index = 0;
    size_t start = 0;
    while (start <= text.size() && value_index < 3) {
        size_t end = text.find(',', start);
        if (end == std::string_view::npos) {
            end = text.size();
        }
        std::string token(text.substr(start, end - start));
        if (token.empty()) {
            return std::nullopt;
        }
        try {
            values[value_index++] = static_cast<uint32_t>(std::stoul(token));
        } catch (...) {
            return std::nullopt;
        }
        if (end == text.size()) {
            break;
        }
        start = end + 1;
    }
    if (value_index != 3) {
        return std::nullopt;
    }
    return Shape{"custom", values[0], values[1], values[2]};
}

std::optional<RunMode> parse_run_mode(std::string_view text) {
    if (text == "eager") {
        return RunMode::Eager;
    }
    if (text == "prepared") {
        return RunMode::Prepared;
    }
    if (text == "trace") {
        return RunMode::Trace;
    }
    return std::nullopt;
}

std::optional<Variant> parse_variant(std::string_view text) {
    if (text == "single_core" || text == "matmul_single_core") {
        return Variant::SingleCore;
    }
    if (text == "multi_core" || text == "matmul_multi_core") {
        return Variant::MultiCore;
    }
    if (text == "reuse" || text == "matmul_multicore_reuse") {
        return Variant::MulticoreReuse;
    }
    if (text == "reuse_mcast" || text == "matmul_multicore_reuse_mcast") {
        return Variant::MulticoreReuseMcast;
    }
    return std::nullopt;
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool saw_custom_shape = false;
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
        } else if (arg == "--shape") {
            auto shape = parse_shape(require_value(arg));
            if (!shape.has_value()) {
                TT_THROW("--shape expects M,N,K");
            }
            if (!saw_custom_shape) {
                options.shapes.clear();
                saw_custom_shape = true;
            }
            shape->name = fmt::format("custom_{}_{}_{}", shape->M, shape->N, shape->K);
            options.shapes.push_back(*shape);
        } else if (arg == "--skip-single-core") {
            options.include_single_core = false;
        } else if (arg == "--include-single-core") {
            options.include_single_core = true;
        } else if (arg == "--no-device-profiler-read") {
            options.read_device_profiler = false;
        } else if (arg == "--mode") {
            std::string_view mode_text(require_value(arg));
            if (mode_text == "all") {
                options.modes = {RunMode::Eager, RunMode::Prepared, RunMode::Trace};
            } else {
                auto mode = parse_run_mode(mode_text);
                if (!mode.has_value()) {
                    TT_THROW("--mode expects eager, prepared, trace, or all");
                }
                options.modes = {*mode};
            }
        } else if (arg == "--variant") {
            auto variant = parse_variant(require_value(arg));
            if (!variant.has_value()) {
                TT_THROW("--variant expects single_core, multi_core, reuse, or reuse_mcast");
            }
            if (!options.variant_filter.has_value()) {
                options.variant_filter = std::vector<Variant>{};
            }
            options.variant_filter->push_back(*variant);
        } else if (arg == "--help") {
            fmt::print(
                "Usage: matmul_variants_profile [--iters N] [--warmup N] [--shape M,N,K]... "
                "[--mode eager|prepared|trace|all] [--include-single-core] [--skip-single-core] "
                "[--variant single_core|multi_core|reuse|reuse_mcast]... [--no-device-profiler-read]\n");
            std::exit(0);
        } else {
            TT_THROW("Unknown argument: {}", arg);
        }
    }
    TT_FATAL(options.measured_iters > 0, "--iters must be > 0");
    return options;
}

Support common_tile_support(const Shape& shape) {
    if (shape.M % TILE_HEIGHT != 0 || shape.N % TILE_WIDTH != 0 || shape.K % TILE_WIDTH != 0) {
        return {false, "M/N/K must be divisible by 32"};
    }
    return {};
}

DeviceKernelSummary latest_kernel_summary() {
    DeviceKernelSummary summary;
    auto latest = experimental::GetLatestKernelDurationSummary();
    for (const auto& [_, chip_summary] : latest) {
        summary.count += chip_summary.count;
        if (chip_summary.count == 0) {
            continue;
        }
        summary.min_ns = summary.min_ns == 0 ? chip_summary.min_ns : std::min(summary.min_ns, chip_summary.min_ns);
        summary.max_ns = std::max(summary.max_ns, chip_summary.max_ns);
        summary.avg_ns = static_cast<uint64_t>(chip_summary.avg_ns);
    }
    return summary;
}

std::optional<DeviceStageSummary> read_device_profiler(
    const Options& options,
    std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Shape& shape,
    Variant variant,
    RunMode mode,
    std::string_view phase,
    uint32_t iter,
    uintmax_t csv_offset) {
    if (!options.read_device_profiler) {
        return std::nullopt;
    }
    ReadMeshDeviceProfilerResults(*mesh_device);
    auto summary = latest_kernel_summary();
    const auto csv_path = profiler_csv_path();
    auto stage_summary = read_device_stage_summary(csv_path, csv_offset, variant);
    fmt::print(
        "MATMUL_PROFILE_DEVICE shape={} variant={} mode={} phase={} iter={} kernel_count={} kernel_min_ns={} "
        "kernel_avg_ns={} kernel_max_ns={} csv={}\n",
        shape.name,
        variant_name(variant),
        run_mode_name(mode),
        phase,
        iter,
        summary.count,
        summary.min_ns,
        summary.avg_ns,
        summary.max_ns,
        csv_path.string());

    for (Stage stage : kStages) {
        auto it = stage_summary.stages.find(stage);
        if (it == stage_summary.stages.end() || it->second.count == 0) {
            fmt::print(
                "MATMUL_PROFILE_STAGE shape={} variant={} mode={} phase={} iter={} stage={} count=0 max_cycles=0 "
                "avg_cycles=0 critical_us=0.000\n",
                shape.name,
                variant_name(variant),
                run_mode_name(mode),
                phase,
                iter,
                stage_name(stage));
            continue;
        }
        const StageStats& stats = it->second;
        fmt::print(
            "MATMUL_PROFILE_STAGE shape={} variant={} mode={} phase={} iter={} stage={} count={} min_cycles={} "
            "avg_cycles={} max_cycles={} critical_us={:.3f}\n",
            shape.name,
            variant_name(variant),
            run_mode_name(mode),
            phase,
            iter,
            stage_name(stage),
            stats.count,
            stats.min_cycles,
            stats.avg_cycles,
            stats.max_cycles,
            stats.critical_us);
    }
    return stage_summary;
}

Support block_matmul_support(
    const Shape& shape,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    bool require_mcast_grid) {
    auto common = common_tile_support(shape);
    if (!common.ok) {
        return common;
    }

    constexpr uint32_t in0_block_w = 2;
    const uint32_t Mt = shape.M / TILE_HEIGHT;
    const uint32_t Kt = shape.K / TILE_WIDTH;
    const uint32_t Nt = shape.N / TILE_WIDTH;
    if (Kt % in0_block_w != 0) {
        return {false, "Kt must be divisible by in0_block_w=2"};
    }

    auto grid = mesh_device->compute_with_storage_grid_size();
    auto params = bmm_op_utils::get_large_matmul_params(Mt, Nt, grid.y, grid.x, in0_block_w);
    const uint32_t per_core_M = std::get<0>(params);
    const uint32_t per_core_N = std::get<1>(params);
    if (per_core_M == 0 || per_core_N == 0) {
        return {false, "get_large_matmul_params returned no valid per-core block"};
    }
    if (Mt % per_core_M != 0 || Nt % per_core_N != 0) {
        return {false, "tile shape is not divisible by chosen per-core block"};
    }

    const uint32_t num_blocks_y = Mt / per_core_M;
    const uint32_t num_blocks_x = Nt / per_core_N;
    if (num_blocks_y * num_blocks_x > grid.x * grid.y) {
        return {false, "block grid exceeds available compute grid"};
    }
    if (require_mcast_grid) {
        auto mcast_core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, grid.y, grid.x);
        if (mcast_core_range.x <= 1 || mcast_core_range.y <= 1) {
            return {false, "mcast requires a 2D block/core range with more than one row and column"};
        }
    }
    return {};
}

Support variant_support(
    Variant variant,
    const Shape& shape,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto common = common_tile_support(shape);
    if (!common.ok) {
        return common;
    }

    switch (variant) {
        case Variant::SingleCore: {
            const uint32_t output_tiles = (shape.M / TILE_HEIGHT) * (shape.N / TILE_WIDTH);
            if (output_tiles > 64) {
                return {false, "single-core example is disabled for shapes above 64 output tiles"};
            }
            return {};
        }
        case Variant::MultiCore:
            return {};
        case Variant::MulticoreReuse:
            return block_matmul_support(shape, mesh_device, false);
        case Variant::MulticoreReuseMcast:
            return block_matmul_support(shape, mesh_device, true);
    }
    return {false, "unknown variant"};
}

Support mode_support(RunMode mode, Variant variant) {
    if (mode == RunMode::Eager) {
        return {};
    }
    if (variant == Variant::SingleCore) {
        return {false, "single-core prepared/trace runner is not implemented"};
    }
    return {};
}

PreparedShape prepare_shape(const Shape& shape) {
    const uint32_t tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    const uint32_t Mt = shape.M / TILE_HEIGHT;
    const uint32_t Kt = shape.K / TILE_WIDTH;
    const uint32_t Nt = shape.N / TILE_WIDTH;

    auto src0 = create_random_vector_of_bfloat16_native(tile_size * Mt * Kt, 1, 123, -0.4);
    auto src1 = create_random_vector_of_bfloat16_native(tile_size * Kt * Nt, 1, 12522, -0.3);

    return {
        .src0_tilized = tilize_nfaces(src0, shape.M, shape.K),
        .src1_tilized = tilize_nfaces(src1, shape.K, shape.N),
        .output = std::vector<bfloat16>(tile_size * Mt * Nt / sizeof(bfloat16)),
    };
}

std::unique_ptr<MatmulProfileRunner> prepare_runner(
    Variant variant,
    const Shape& shape,
    std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    switch (variant) {
        case Variant::MultiCore:
            return prepare_matmul_multi_core_profile_runner(shape.M, shape.N, shape.K, mesh_device);
        case Variant::MulticoreReuse:
            return prepare_matmul_multicore_reuse_profile_runner(false, shape.M, shape.N, shape.K, 1, mesh_device);
        case Variant::MulticoreReuseMcast:
            return prepare_matmul_multicore_reuse_mcast_profile_runner(
                false, shape.M, shape.N, shape.K, 1, mesh_device);
        case Variant::SingleCore:
            break;
    }
    TT_THROW("No prepared runner for variant {}", variant_name(variant));
}

void run_variant(
    Variant variant,
    RunMode mode,
    const Shape& shape,
    PreparedShape& data,
    const Options& options,
    std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    fmt::print(
        "MATMUL_PROFILE_RUN_START shape={} variant={} mode={}\n",
        shape.name,
        variant_name(variant),
        run_mode_name(mode));
    std::cout.flush();

    std::unique_ptr<MatmulProfileRunner> runner;
    if (mode == RunMode::Prepared || mode == RunMode::Trace) {
        runner = prepare_runner(variant, shape, mesh_device);
        if (mode == RunMode::Trace) {
            runner->prepare_trace(data.src0_tilized, data.src1_tilized, data.output, options.read_device_profiler);
        }
    }

    auto invoke = [&]() {
        if (runner) {
            runner->run(data.src0_tilized, data.src1_tilized, data.output);
            return;
        }
        switch (variant) {
            case Variant::SingleCore:
                matmul_single_core(
                    data.src0_tilized, data.src1_tilized, data.output, false, shape.M, shape.N, shape.K, mesh_device);
                break;
            case Variant::MultiCore:
                matmul_multi_core(
                    data.src0_tilized, data.src1_tilized, data.output, shape.M, shape.N, shape.K, mesh_device);
                break;
            case Variant::MulticoreReuse:
                matmul_multicore_reuse(
                    data.src0_tilized,
                    data.src1_tilized,
                    data.output,
                    false,
                    shape.M,
                    shape.N,
                    shape.K,
                    1,
                    mesh_device);
                break;
            case Variant::MulticoreReuseMcast:
                matmul_multicore_reuse_mcast(
                    data.src0_tilized,
                    data.src1_tilized,
                    data.output,
                    false,
                    shape.M,
                    shape.N,
                    shape.K,
                    1,
                    mesh_device);
                break;
        }
    };

    for (uint32_t i = 0; i < options.warmup_iters; ++i) {
        const auto csv_offset = safe_file_size(profiler_csv_path());
        invoke();
        read_device_profiler(options, mesh_device, shape, variant, mode, "warmup", i, csv_offset);
    }

    double total_ms = 0.0;
    double best_ms = std::numeric_limits<double>::max();
    double worst_ms = 0.0;
    std::map<Stage, std::vector<double>> critical_us_by_stage;
    std::map<Stage, std::vector<uint64_t>> max_cycles_by_stage;
    for (uint32_t i = 0; i < options.measured_iters; ++i) {
        const auto csv_offset = safe_file_size(profiler_csv_path());
        const auto start = Clock::now();
        invoke();
        const auto end = Clock::now();
        auto stage_summary = read_device_profiler(
            options, mesh_device, shape, variant, mode, "measured", i, csv_offset);
        if (stage_summary.has_value()) {
            for (Stage stage : kStages) {
                auto it = stage_summary->stages.find(stage);
                if (it == stage_summary->stages.end() || it->second.count == 0) {
                    continue;
                }
                critical_us_by_stage[stage].push_back(it->second.critical_us);
                max_cycles_by_stage[stage].push_back(it->second.max_cycles);
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_ms += ms;
        best_ms = std::min(best_ms, ms);
        worst_ms = std::max(worst_ms, ms);
        fmt::print(
            "MATMUL_PROFILE_ITER shape={} variant={} mode={} iter={} ms={:.3f}\n",
            shape.name,
            variant_name(variant),
            run_mode_name(mode),
            i,
            ms);
    }

    fmt::print(
        "MATMUL_PROFILE_RESULT shape={} M={} N={} K={} variant={} mode={} avg_ms={:.3f} best_ms={:.3f} worst_ms={:.3f} "
        "warmup={} iters={}\n",
        shape.name,
        shape.M,
        shape.N,
        shape.K,
        variant_name(variant),
        run_mode_name(mode),
        total_ms / options.measured_iters,
        best_ms,
        worst_ms,
        options.warmup_iters,
        options.measured_iters);
    std::cout.flush();

    for (Stage stage : kStages) {
        const auto critical_it = critical_us_by_stage.find(stage);
        const auto cycles_it = max_cycles_by_stage.find(stage);
        if (critical_it == critical_us_by_stage.end() || critical_it->second.empty() ||
            cycles_it == max_cycles_by_stage.end() || cycles_it->second.empty()) {
            continue;
        }
        const auto& critical_us = critical_it->second;
        const auto& max_cycles = cycles_it->second;
        const double total_critical_us = std::accumulate(critical_us.begin(), critical_us.end(), 0.0);
        const uint64_t total_max_cycles = std::accumulate(max_cycles.begin(), max_cycles.end(), uint64_t{0});
        const auto [best_critical_it, worst_critical_it] =
            std::minmax_element(critical_us.begin(), critical_us.end());
        const auto [best_cycles_it, worst_cycles_it] = std::minmax_element(max_cycles.begin(), max_cycles.end());
        fmt::print(
            "MATMUL_PROFILE_STAGE_RESULT shape={} variant={} mode={} stage={} avg_critical_us={:.3f} "
            "best_critical_us={:.3f} worst_critical_us={:.3f} avg_max_cycles={} best_max_cycles={} "
            "worst_max_cycles={} iters={}\n",
            shape.name,
            variant_name(variant),
            run_mode_name(mode),
            stage_name(stage),
            total_critical_us / critical_us.size(),
            *best_critical_it,
            *worst_critical_it,
            total_max_cycles / max_cycles.size(),
            *best_cycles_it,
            *worst_cycles_it,
            critical_us.size());
    }
}

}  // namespace

int main(int argc, char** argv) {
    auto options = parse_options(argc, argv);
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    bool pass = true;

    std::vector<Variant> variants = {
        Variant::SingleCore,
        Variant::MultiCore,
        Variant::MulticoreReuse,
        Variant::MulticoreReuseMcast,
    };
    if (!options.include_single_core) {
        variants.erase(std::remove(variants.begin(), variants.end(), Variant::SingleCore), variants.end());
    }
    if (options.variant_filter.has_value()) {
        variants = *options.variant_filter;
        const bool explicit_single_core =
            std::find(variants.begin(), variants.end(), Variant::SingleCore) != variants.end();
        if (!options.include_single_core && !explicit_single_core) {
            variants.erase(std::remove(variants.begin(), variants.end(), Variant::SingleCore), variants.end());
        }
    }

    for (const auto& shape : options.shapes) {
        fmt::print("MATMUL_PROFILE_SHAPE name={} M={} N={} K={}\n", shape.name, shape.M, shape.N, shape.K);
        std::cout.flush();
        auto data = prepare_shape(shape);
        for (auto variant : variants) {
            auto support = variant_support(variant, shape, mesh_device);
            if (!support.ok) {
                fmt::print(
                    "MATMUL_PROFILE_SKIP shape={} variant={} reason=\"{}\"\n",
                    shape.name,
                    variant_name(variant),
                    support.reason);
                continue;
            }

            for (auto mode : options.modes) {
                auto mode_status = mode_support(mode, variant);
                if (!mode_status.ok) {
                    fmt::print(
                        "MATMUL_PROFILE_SKIP shape={} variant={} mode={} reason=\"{}\"\n",
                        shape.name,
                        variant_name(variant),
                        run_mode_name(mode),
                        mode_status.reason);
                    continue;
                }

                try {
                    run_variant(variant, mode, shape, data, options, mesh_device);
                } catch (const std::exception& e) {
                    pass = false;
                    fmt::print(
                        stderr,
                        "MATMUL_PROFILE_ERROR shape={} variant={} mode={} error=\"{}\"\n",
                        shape.name,
                        variant_name(variant),
                        run_mode_name(mode),
                        e.what());
                }
            }
        }
    }

    pass &= mesh_device->close();
    TT_FATAL(pass, "One or more matmul profile runs failed");
    return 0;
}
