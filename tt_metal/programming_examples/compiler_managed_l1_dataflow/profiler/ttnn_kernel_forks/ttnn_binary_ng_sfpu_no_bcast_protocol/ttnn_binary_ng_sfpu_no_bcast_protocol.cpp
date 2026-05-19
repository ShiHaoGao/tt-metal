// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace per_core_allocation = tt::tt_metal::experimental::per_core_allocation;

namespace {

constexpr CoreCoord kDefaultCore = {0, 0};
constexpr uint32_t kTileElements = tt::constants::TILE_HW;
constexpr uint32_t kTileSizeBytes = sizeof(bfloat16) * kTileElements;
constexpr uint32_t kProtocolStartBytes = 64;
constexpr uint32_t kStreamRegCounterMask = 0x00ffffffu;
constexpr uint32_t kStreamRegStartStreamId = 3;
constexpr uint32_t kStreamRegInputReady0StreamId = 4;
constexpr uint32_t kStreamRegInputReady1StreamId = 5;
constexpr uint32_t kStreamRegInputConsumed0StreamId = 6;
constexpr uint32_t kStreamRegInputConsumed1StreamId = 7;
constexpr uint32_t kStreamRegOutputReadyStreamId = 8;
constexpr uint32_t kStreamRegOutputConsumedStreamId = 9;

constexpr std::string_view kReaderKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_binary_ng_sfpu_no_bcast_protocol/kernels/dataflow/"
    "reader_interleaved_no_bcast_protocol.cpp";
constexpr std::string_view kWriterKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_binary_ng_sfpu_no_bcast_protocol/kernels/dataflow/"
    "writer_interleaved_no_bcast_protocol.cpp";
constexpr std::string_view kComputeKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_binary_ng_sfpu_no_bcast_protocol/kernels/compute/"
    "eltwise_binary_sfpu_no_bcast_protocol.cpp";

enum class Mode : uint32_t {
    Cb = 0,
    StaticRuntime = 1,
    StaticStreamReg = 2,
    StaticStreamRegCbRegs = 3,
    StaticStreamRegCbRegsCompileTime = 4,
};

struct Options {
    std::string mode = "all";
    uint32_t tiles = 1024;
    uint32_t num_pages = 2;
    uint32_t repeats = 1;
    uint32_t device_id = 0;
    CoreCoord core = kDefaultCore;
    uint32_t core_grid_x = 1;
    uint32_t core_grid_y = 1;
    uint32_t num_tiles_per_cycle = 1;
};

struct CoreWork {
    CoreCoord core;
    uint32_t start_tile = 0;
    uint32_t tile_count = 0;
    std::shared_ptr<Buffer> src0_ring_buffer;
    std::shared_ptr<Buffer> src1_ring_buffer;
    std::shared_ptr<Buffer> dst_ring_buffer;
    std::shared_ptr<Buffer> protocol_start_buffer;
};

struct RunResult {
    Mode mode;
    uint32_t repeat = 0;
    uint64_t enqueue_finish_us = 0;
    bool ok = false;
    float max_abs_error = 0.0f;
};

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.substr(0, prefix.size()) == prefix;
}

uint32_t parse_u32(std::string_view value, std::string_view name) {
    std::string owned(value);
    size_t parsed = 0;
    unsigned long result = 0;
    try {
        result = std::stoul(owned, &parsed, 0);
    } catch (const std::exception&) {
        throw std::invalid_argument(fmt::format("Invalid value for {}: {}", name, value));
    }
    if (parsed != owned.size() || result > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(fmt::format("Invalid value for {}: {}", name, value));
    }
    return static_cast<uint32_t>(result);
}

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::Cb: return "cb";
        case Mode::StaticRuntime: return "static-runtime";
        case Mode::StaticStreamReg: return "static-streamreg";
        case Mode::StaticStreamRegCbRegs: return "static-streamreg-cbregs";
        case Mode::StaticStreamRegCbRegsCompileTime: return "static-streamreg-cbregs-compiletime";
    }
    return "unknown";
}

std::optional<Mode> parse_mode(std::string_view mode) {
    if (mode == "cb") {
        return Mode::Cb;
    }
    if (mode == "static-runtime" || mode == "static") {
        return Mode::StaticRuntime;
    }
    if (mode == "static-streamreg") {
        return std::nullopt;
    }
    if (mode == "static-streamreg-cbregs") {
        return Mode::StaticStreamRegCbRegs;
    }
    if (mode == "static-streamreg-cbregs-compiletime") {
        return Mode::StaticStreamRegCbRegsCompileTime;
    }
    return std::nullopt;
}

std::vector<Mode> modes_to_run(const std::string& mode) {
    if (mode == "all") {
        return {Mode::Cb, Mode::StaticRuntime, Mode::StaticStreamRegCbRegs, Mode::StaticStreamRegCbRegsCompileTime};
    }
    auto parsed = parse_mode(mode);
    if (!parsed.has_value()) {
        throw std::invalid_argument(
            "Unknown --mode. Valid values are all, cb, static-runtime, static-streamreg-cbregs, "
            "static-streamreg-cbregs-compiletime, static. "
            "The old static-streamreg compute mode is disabled.");
    }
    return {*parsed};
}

bool is_static_mode(Mode mode) {
    return mode == Mode::StaticRuntime || mode == Mode::StaticStreamReg || mode == Mode::StaticStreamRegCbRegs ||
           mode == Mode::StaticStreamRegCbRegsCompileTime;
}

bool uses_stream_reg_sync(Mode mode) { return mode == Mode::StaticStreamReg; }

bool uses_stream_reg_cbregs(Mode mode) {
    return mode == Mode::StaticStreamRegCbRegs || mode == Mode::StaticStreamRegCbRegsCompileTime;
}

bool uses_compile_time_protocol_args(Mode mode) {
    return mode == Mode::StaticStreamRegCbRegsCompileTime;
}

bool uses_stream_reg_start_gate(Mode mode) { return uses_stream_reg_sync(mode) || uses_stream_reg_cbregs(mode); }

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--mode=all|cb|static-runtime|static-streamreg-cbregs|"
        "static-streamreg-cbregs-compiletime] "
        "[--tiles=N] [--num-pages=N] [--repeats=N] "
        "[--device-id=N] [--core-x=N] [--core-y=N] [--core-grid-x=N] [--core-grid-y=N]\n",
        argv0);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (starts_with(arg, "--mode=")) {
            options.mode = std::string(arg.substr(std::string_view("--mode=").size()));
        } else if (starts_with(arg, "--tiles=")) {
            options.tiles = parse_u32(arg.substr(std::string_view("--tiles=").size()), "tiles");
        } else if (starts_with(arg, "--num-pages=")) {
            options.num_pages = parse_u32(arg.substr(std::string_view("--num-pages=").size()), "num-pages");
        } else if (starts_with(arg, "--repeats=")) {
            options.repeats = parse_u32(arg.substr(std::string_view("--repeats=").size()), "repeats");
        } else if (starts_with(arg, "--device-id=")) {
            options.device_id = parse_u32(arg.substr(std::string_view("--device-id=").size()), "device-id");
        } else if (starts_with(arg, "--core-x=")) {
            options.core.x = parse_u32(arg.substr(std::string_view("--core-x=").size()), "core-x");
        } else if (starts_with(arg, "--core-y=")) {
            options.core.y = parse_u32(arg.substr(std::string_view("--core-y=").size()), "core-y");
        } else if (starts_with(arg, "--core-grid-x=")) {
            options.core_grid_x = parse_u32(arg.substr(std::string_view("--core-grid-x=").size()), "core-grid-x");
        } else if (starts_with(arg, "--core-grid-y=")) {
            options.core_grid_y = parse_u32(arg.substr(std::string_view("--core-grid-y=").size()), "core-grid-y");
        } else if (starts_with(arg, "--num-tiles-per-cycle=")) {
            options.num_tiles_per_cycle =
                parse_u32(arg.substr(std::string_view("--num-tiles-per-cycle=").size()), "num-tiles-per-cycle");
        } else {
            throw std::invalid_argument(fmt::format("Unknown argument '{}'. Use --help to list options.", arg));
        }
    }

    if (options.tiles == 0 || options.num_pages == 0 || options.repeats == 0) {
        throw std::invalid_argument("--tiles, --num-pages, and --repeats must be greater than zero");
    }
    if (options.core_grid_x == 0 || options.core_grid_y == 0) {
        throw std::invalid_argument("--core-grid-x and --core-grid-y must be greater than zero");
    }
    if (options.num_tiles_per_cycle != 1) {
        throw std::invalid_argument("This protocol fork currently supports --num-tiles-per-cycle=1 only");
    }
    return options;
}

uint32_t payload_size_bytes(const Options& options) {
    const uint64_t size = static_cast<uint64_t>(options.tiles) * kTileSizeBytes;
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Payload exceeds uint32_t addressable size");
    }
    return static_cast<uint32_t>(size);
}

uint32_t ring_size_bytes(const Options& options) {
    const uint64_t size = static_cast<uint64_t>(options.num_pages) * kTileSizeBytes;
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Static L1 ring exceeds uint32_t addressable size");
    }
    return static_cast<uint32_t>(size);
}

std::vector<CoreCoord> selected_cores(const Options& options) {
    std::vector<CoreCoord> cores;
    cores.reserve(static_cast<size_t>(options.core_grid_x) * static_cast<size_t>(options.core_grid_y));
    for (uint32_t y = 0; y < options.core_grid_y; ++y) {
        for (uint32_t x = 0; x < options.core_grid_x; ++x) {
            cores.emplace_back(options.core.x + x, options.core.y + y);
        }
    }
    return cores;
}

std::vector<CoreWork> partition_work(const Options& options, const std::vector<CoreCoord>& cores) {
    std::vector<CoreWork> work;
    const uint32_t base_tiles = options.tiles / static_cast<uint32_t>(cores.size());
    const uint32_t remainder = options.tiles % static_cast<uint32_t>(cores.size());
    uint32_t start_tile = 0;
    for (size_t i = 0; i < cores.size(); ++i) {
        const uint32_t tile_count = base_tiles + (static_cast<uint32_t>(i) < remainder ? 1u : 0u);
        if (tile_count == 0) {
            continue;
        }
        work.push_back(CoreWork{.core = cores[i], .start_tile = start_tile, .tile_count = tile_count});
        start_tile += tile_count;
    }
    return work;
}

CoreRangeSet core_range_set_from_work(const std::vector<CoreWork>& work) {
    std::vector<CoreRange> ranges;
    ranges.reserve(work.size());
    for (const auto& item : work) {
        ranges.emplace_back(item.core);
    }
    return CoreRangeSet(std::move(ranges)).merge_ranges();
}

std::shared_ptr<distributed::MeshBuffer> create_dram_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size_bytes) {
    distributed::DeviceLocalBufferConfig dram_config{.page_size = kTileSizeBytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = size_bytes};
    return distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
}

std::shared_ptr<Buffer> create_core_local_l1_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    CoreCoord core,
    uint32_t size_bytes,
    uint32_t page_size_bytes) {
    if (page_size_bytes == 0 || size_bytes % page_size_bytes != 0) {
        throw std::invalid_argument("Static L1 buffer size must be a whole number of pages");
    }

    const uint32_t num_pages = size_bytes / page_size_bytes;
    ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange(core)),
        std::array<uint32_t, 2>{num_pages, 1},
        ShardOrientation::ROW_MAJOR,
        std::array<uint32_t, 2>{1, 1},
        std::array<uint32_t, 2>{num_pages, 1});
    BufferShardingArgs sharding_args(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    per_core_allocation::set_per_core_allocation(sharding_args, true);

    const auto devices = mesh_device->get_devices();
    if (devices.empty()) {
        throw std::runtime_error("MeshDevice has no local device for per-core L1 allocation");
    }

    return Buffer::create(devices.front(), size_bytes, page_size_bytes, BufferType::L1, sharding_args, true);
}

uint32_t core_local_l1_address(const std::shared_ptr<Buffer>& buffer, CoreCoord core) {
    if (!buffer) {
        return 0;
    }
    return static_cast<uint32_t>(per_core_allocation::get_per_core_address(*buffer, core));
}

std::vector<bfloat16> make_input(uint32_t tiles, uint32_t salt) {
    std::vector<bfloat16> data(static_cast<size_t>(tiles) * kTileElements);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        for (uint32_t i = 0; i < kTileElements; ++i) {
            const uint32_t pattern = (tile * 131u + i * 17u + salt * 53u) % 97u;
            data[static_cast<size_t>(tile) * kTileElements + i] = bfloat16(static_cast<float>(pattern + 1u) * 0.125f);
        }
    }
    return data;
}

std::map<std::string, std::string> protocol_defines(
    Mode mode,
    uint32_t protocol_start_value,
    uint32_t src0_ring_addr,
    uint32_t src1_ring_addr,
    uint32_t dst_ring_addr,
    uint32_t protocol_start_addr,
    const Options& options) {
    return {
        {"BENCH_STATIC_PROTOCOL", is_static_mode(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_SYNC", uses_stream_reg_sync(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_CBREGS", uses_stream_reg_cbregs(mode) ? "1" : "0"},
        {"BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS", uses_compile_time_protocol_args(mode) ? "1" : "0"},
        {"BENCH_PROTOCOL_START_VALUE", std::to_string(protocol_start_value)},
        {"BENCH_SRC0_RING_ADDR", std::to_string(src0_ring_addr)},
        {"BENCH_SRC1_RING_ADDR", std::to_string(src1_ring_addr)},
        {"BENCH_DST_RING_ADDR", std::to_string(dst_ring_addr)},
        {"BENCH_PAGE_SIZE", std::to_string(kTileSizeBytes)},
        {"BENCH_NUM_PAGES", std::to_string(options.num_pages)},
        {"BENCH_PROTOCOL_START_SEM_ADDR", std::to_string(protocol_start_addr)},
        {"BENCH_STREAM_REG_START_STREAM_ID", std::to_string(kStreamRegStartStreamId)},
        {"BENCH_STREAM_REG_INPUT_READY0_STREAM_ID", std::to_string(kStreamRegInputReady0StreamId)},
        {"BENCH_STREAM_REG_INPUT_READY1_STREAM_ID", std::to_string(kStreamRegInputReady1StreamId)},
        {"BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID", std::to_string(kStreamRegInputConsumed0StreamId)},
        {"BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID", std::to_string(kStreamRegInputConsumed1StreamId)},
        {"BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID", std::to_string(kStreamRegOutputReadyStreamId)},
        {"BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID", std::to_string(kStreamRegOutputConsumedStreamId)},
        {"BENCH_STREAM_REG_VALUE_MASK", std::to_string(kStreamRegCounterMask)},
    };
}

std::map<std::string, std::string> reader_defines(
    Mode mode,
    uint32_t protocol_start_value,
    uint32_t src0_ring_addr,
    uint32_t src1_ring_addr,
    uint32_t dst_ring_addr,
    uint32_t protocol_start_addr,
    const Options& options) {
    auto defines =
        protocol_defines(mode, protocol_start_value, src0_ring_addr, src1_ring_addr, dst_ring_addr, protocol_start_addr, options);
    defines["SRC_SHARDED"] = "0";
    defines["SRC_SHARDED_B"] = "0";
    return defines;
}

std::map<std::string, std::string> writer_defines(
    Mode mode,
    uint32_t protocol_start_value,
    uint32_t src0_ring_addr,
    uint32_t src1_ring_addr,
    uint32_t dst_ring_addr,
    uint32_t protocol_start_addr,
    const Options& options) {
    auto defines =
        protocol_defines(mode, protocol_start_value, src0_ring_addr, src1_ring_addr, dst_ring_addr, protocol_start_addr, options);
    defines["DST_SHARDED"] = "0";
    return defines;
}

std::map<std::string, std::string> compute_defines(
    Mode mode,
    uint32_t protocol_start_value,
    uint32_t src0_ring_addr,
    uint32_t src1_ring_addr,
    uint32_t dst_ring_addr,
    uint32_t protocol_start_addr,
    const Options& options) {
    auto defines =
        protocol_defines(mode, protocol_start_value, src0_ring_addr, src1_ring_addr, dst_ring_addr, protocol_start_addr, options);
    defines.insert({
        {"SRC_SHARDED", "0"},
        {"SRC_SHARDED_B", "0"},
        {"DST_SHARDED", "0"},
        {"BINARY_SFPU_INIT", "div_binary_tile_init();"},
        {"BINARY_SFPU_OP", "div_binary_tile"},
        {"PROCESS_LHS_ACTIVATIONS(i)", ""},
        {"PROCESS_RHS_ACTIVATIONS(i)", ""},
        {"PROCESS_POST_ACTIVATIONS(i)", ""},
    });
    return defines;
}

void create_circular_buffers(
    Program& program,
    CoreCoord core,
    Mode mode,
    const Options& options,
    const std::shared_ptr<Buffer>& src0_ring_buffer,
    const std::shared_ptr<Buffer>& src1_ring_buffer,
    const std::shared_ptr<Buffer>& dst_ring_buffer) {
    const uint32_t cb_size = options.num_pages * kTileSizeBytes;
    auto make_cb_config = [&](CBIndex cb_index, const std::shared_ptr<Buffer>& static_ring_buffer) {
        auto config =
            CircularBufferConfig(cb_size, {{cb_index, DataFormat::Float16_b}}).set_page_size(cb_index, kTileSizeBytes);
        if (is_static_mode(mode)) {
            if (!static_ring_buffer) {
                throw std::runtime_error("Static mode requires explicit L1 ring buffers");
            }
            config.set_globally_allocated_address(*static_ring_buffer);
        }
        return config;
    };

    CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0, src0_ring_buffer));
    CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1, src1_ring_buffer));
    CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_2, dst_ring_buffer));
}

bool validate_result(
    const std::vector<bfloat16>& src0,
    const std::vector<bfloat16>& src1,
    const std::vector<bfloat16>& result,
    float* max_abs_error) {
    if (result.size() != src0.size() || src1.size() != src0.size()) {
        *max_abs_error = std::numeric_limits<float>::infinity();
        return false;
    }

    bool ok = true;
    float max_error = 0.0f;
    uint32_t printed = 0;
    for (size_t i = 0; i < src0.size(); ++i) {
        const float rhs = static_cast<float>(src1[i]);
        const float expected = rhs == 0.0f ? 0.0f : static_cast<float>(src0[i]) / rhs;
        const float actual = static_cast<float>(result[i]);
        const float error = std::abs(expected - actual);
        max_error = std::max(max_error, error);
        if (error > 3e-1f) {
            ok = false;
            if (printed < 5) {
                fmt::print(stderr, "Mismatch at element {}: expected {}, got {}\n", i, expected, actual);
                ++printed;
            }
        }
    }
    *max_abs_error = max_error;
    return ok;
}

std::vector<uint32_t> reader_runtime_args(
    const CoreWork& work,
    uint32_t total_tiles,
    uint32_t src0_dram_addr,
    uint32_t src1_dram_addr) {
    return {
        src0_dram_addr,
        work.start_tile,
        work.tile_count,
        work.tile_count,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        total_tiles,
        1,
        src1_dram_addr,
        0,
        0,
        0,
        0,
        work.tile_count};
}

std::vector<uint32_t> writer_runtime_args(const CoreWork& work, uint32_t total_tiles, uint32_t dst_dram_addr) {
    return {dst_dram_addr, work.start_tile, work.tile_count, 0, 1, 1, 1, 1, total_tiles, 1, 0};
}

RunResult run_one(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Options& options,
    Mode mode,
    uint32_t repeat,
    const std::vector<bfloat16>& src0,
    const std::vector<bfloat16>& src1) {
    std::vector<CoreWork> core_work = partition_work(options, selected_cores(options));
    if (core_work.empty()) {
        throw std::invalid_argument("No active cores were selected");
    }
    if (uses_compile_time_protocol_args(mode) && core_work.size() != 1) {
        throw std::invalid_argument("static-streamreg-cbregs-compiletime mode currently supports only one active core");
    }
    if (uses_stream_reg_sync(mode)) {
        throw std::invalid_argument(
            "static-streamreg is disabled for ttnn_binary_ng_sfpu_no_bcast_protocol: the old idle-stream scratch-register "
            "variant is not a valid compute-path comparison. Use static-streamreg-cbregs with per-CB "
            "tiles_received/tiles_acked registers instead");
    }
    if (uses_stream_reg_start_gate(mode) && options.tiles > kStreamRegCounterMask) {
        throw std::invalid_argument("stream-register protocol modes support up to 24-bit tile counters");
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    const CoreRangeSet active_core_ranges = core_range_set_from_work(core_work);

    const uint32_t payload_bytes = payload_size_bytes(options);
    auto src0_dram_buffer = create_dram_buffer(mesh_device, payload_bytes);
    auto src1_dram_buffer = create_dram_buffer(mesh_device, payload_bytes);
    auto dst_dram_buffer = create_dram_buffer(mesh_device, payload_bytes);

    if (is_static_mode(mode)) {
        const uint32_t ring_bytes = ring_size_bytes(options);
        std::vector<uint8_t> zero_ring_buffer(ring_bytes, 0);
        std::vector<uint8_t> zero_sem_buffer(kProtocolStartBytes, 0);
        for (auto& work : core_work) {
            work.src0_ring_buffer = create_core_local_l1_buffer(mesh_device, work.core, ring_bytes, kTileSizeBytes);
            work.src1_ring_buffer = create_core_local_l1_buffer(mesh_device, work.core, ring_bytes, kTileSizeBytes);
            work.dst_ring_buffer = create_core_local_l1_buffer(mesh_device, work.core, ring_bytes, kTileSizeBytes);
            detail::WriteToBuffer(work.src0_ring_buffer, zero_ring_buffer);
            detail::WriteToBuffer(work.src1_ring_buffer, zero_ring_buffer);
            detail::WriteToBuffer(work.dst_ring_buffer, zero_ring_buffer);
            if (!uses_stream_reg_start_gate(mode)) {
                work.protocol_start_buffer =
                    create_core_local_l1_buffer(mesh_device, work.core, kProtocolStartBytes, kProtocolStartBytes);
                detail::WriteToBuffer(work.protocol_start_buffer, zero_sem_buffer);
            }
        }
    }

    for (const auto& work : core_work) {
        create_circular_buffers(
            program, work.core, mode, options, work.src0_ring_buffer, work.src1_ring_buffer, work.dst_ring_buffer);
    }

    const uint32_t src0_dram_addr = static_cast<uint32_t>(src0_dram_buffer->address());
    const uint32_t src1_dram_addr = static_cast<uint32_t>(src1_dram_buffer->address());
    const uint32_t dst_dram_addr = static_cast<uint32_t>(dst_dram_buffer->address());

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);
    reader_compile_time_args.push_back(0);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
    writer_compile_time_args.push_back(0);

    std::vector<uint32_t> compute_compile_time_args = {options.num_tiles_per_cycle};
    const uint32_t protocol_start_value = 0x5a5a0000u ^ options.tiles ^ (options.num_pages << 8) ^
                                          (repeat * 0x00010001u) ^ dst_dram_addr;
    const auto first_static_addrs = [&]() {
        struct Addresses {
            uint32_t src0_ring_addr;
            uint32_t src1_ring_addr;
            uint32_t dst_ring_addr;
            uint32_t protocol_start_addr;
        };
        if (!is_static_mode(mode)) {
            return Addresses{};
        }
        const auto& work = core_work.front();
        return Addresses{
            .src0_ring_addr = core_local_l1_address(work.src0_ring_buffer, work.core),
            .src1_ring_addr = core_local_l1_address(work.src1_ring_buffer, work.core),
            .dst_ring_addr = core_local_l1_address(work.dst_ring_buffer, work.core),
            .protocol_start_addr = core_local_l1_address(work.protocol_start_buffer, work.core)};
    }();
    auto reader_kernel_defines = reader_defines(
        mode,
        protocol_start_value,
        first_static_addrs.src0_ring_addr,
        first_static_addrs.src1_ring_addr,
        first_static_addrs.dst_ring_addr,
        first_static_addrs.protocol_start_addr,
        options);
    auto writer_kernel_defines = writer_defines(
        mode,
        protocol_start_value,
        first_static_addrs.src0_ring_addr,
        first_static_addrs.src1_ring_addr,
        first_static_addrs.dst_ring_addr,
        first_static_addrs.protocol_start_addr,
        options);
    auto compute_kernel_defines = compute_defines(
        mode,
        protocol_start_value,
        first_static_addrs.src0_ring_addr,
        first_static_addrs.src1_ring_addr,
        first_static_addrs.dst_ring_addr,
        first_static_addrs.protocol_start_addr,
        options);

    KernelHandle reader_kernel = CreateKernel(
        program,
        std::string(kReaderKernel),
        active_core_ranges,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = reader_kernel_defines});
    KernelHandle writer_kernel = CreateKernel(
        program,
        std::string(kWriterKernel),
        active_core_ranges,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = writer_kernel_defines});
    KernelHandle compute_kernel = CreateKernel(
        program,
        std::string(kComputeKernel),
        active_core_ranges,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .compile_args = compute_compile_time_args,
            .defines = compute_kernel_defines});

    for (const auto& work : core_work) {
        std::vector<uint32_t> reader_args = reader_runtime_args(work, options.tiles, src0_dram_addr, src1_dram_addr);
        std::vector<uint32_t> writer_args = writer_runtime_args(work, options.tiles, dst_dram_addr);
        std::vector<uint32_t> compute_args = {work.tile_count};

        if (is_static_mode(mode)) {
            const uint32_t src0_ring_addr = core_local_l1_address(work.src0_ring_buffer, work.core);
            const uint32_t src1_ring_addr = core_local_l1_address(work.src1_ring_buffer, work.core);
            const uint32_t dst_ring_addr = core_local_l1_address(work.dst_ring_buffer, work.core);
            const uint32_t protocol_start_addr = core_local_l1_address(work.protocol_start_buffer, work.core);

            if (!uses_compile_time_protocol_args(mode)) {
                reader_args.insert(
                    reader_args.end(),
                    {src0_ring_addr, src1_ring_addr, kTileSizeBytes, options.num_pages, protocol_start_addr});
                writer_args.insert(writer_args.end(), {dst_ring_addr, kTileSizeBytes, options.num_pages, protocol_start_addr});
                compute_args.insert(
                    compute_args.end(),
                    {src0_ring_addr, src1_ring_addr, dst_ring_addr, kTileSizeBytes, options.num_pages, protocol_start_addr});
            }
        }

        SetRuntimeArgs(program, reader_kernel, work.core, reader_args);
        SetRuntimeArgs(program, writer_kernel, work.core, writer_args);
        SetRuntimeArgs(program, compute_kernel, work.core, compute_args);
    }

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1, false);

    workload.add_program(device_range, std::move(program));

    const auto start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();

    std::vector<bfloat16> result;
    distributed::EnqueueReadMeshBuffer(cq, result, dst_dram_buffer, true);

    float max_abs_error = 0.0f;
    const bool ok = validate_result(src0, src1, result, &max_abs_error);
    return {
        .mode = mode,
        .repeat = repeat,
        .enqueue_finish_us =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
        .ok = ok,
        .max_abs_error = max_abs_error};
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;

    try {
        const Options options = parse_options(argc, argv);
        const std::vector<Mode> modes = modes_to_run(options.mode);
        if (std::any_of(modes.begin(), modes.end(), is_static_mode)) {
            setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);
        }

        if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
            fmt::print(
                "WARNING: set TT_METAL_DEVICE_PROFILER=1 to collect device cycle zones in "
                "generated/profiler/.logs/profile_log_device.csv\n");
        }

        auto mesh_device = distributed::MeshDevice::create_unit_mesh(options.device_id);
        const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
        if (options.core.x >= grid.x || options.core.y >= grid.y ||
            static_cast<uint64_t>(options.core.x) + options.core_grid_x > grid.x ||
            static_cast<uint64_t>(options.core.y) + options.core_grid_y > grid.y) {
            throw std::invalid_argument(fmt::format(
                "Core grid origin=({}, {}) size=({}, {}) is outside compute_with_storage_grid_size ({}, {})",
                options.core.x,
                options.core.y,
                options.core_grid_x,
                options.core_grid_y,
                grid.x,
                grid.y));
        }

        const auto src0 = make_input(options.tiles, 1);
        const auto src1 = make_input(options.tiles, 2);

        fmt::print(
            "ttnn_binary_ng_sfpu_no_bcast_protocol: op=div, tiles={}, tile_size={}, num_pages={}, repeats={}, core=({}, {}), "
            "core_grid=({}, {}), num_tiles_per_cycle={}\n",
            options.tiles,
            kTileSizeBytes,
            options.num_pages,
            options.repeats,
            options.core.x,
            options.core.y,
            options.core_grid_x,
            options.core_grid_y,
            options.num_tiles_per_cycle);

        for (uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
            for (Mode mode : modes) {
                RunResult result = run_one(mesh_device, options, mode, repeat, src0, src1);
                pass &= result.ok;
                fmt::print(
                    "mode={:<20} repeat={} enqueue_finish_us={} max_abs_error={} {}\n",
                    mode_name(result.mode),
                    result.repeat,
                    result.enqueue_finish_us,
                    result.max_abs_error,
                    result.ok ? "ok" : "FAILED");
            }
        }

        ReadMeshDeviceProfilerResults(*mesh_device);
        fmt::print("Profiler CSV: generated/profiler/.logs/profile_log_device.csv\n");

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
