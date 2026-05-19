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
#include <memory>
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

constexpr std::string_view kReaderKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_slice_tile_protocol/"
    "kernels/dataflow/reader_slice_tile_protocol.cpp";
constexpr std::string_view kWriterKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_slice_tile_protocol/"
    "kernels/dataflow/writer_slice_tile_protocol.cpp";

enum class Mode : uint32_t {
    Cb = 0,
    StaticRuntime = 1,
    StaticStreamRegCbRegs = 2,
};

struct Options {
    std::string mode = "all";
    uint32_t input_height_tiles = 64;
    uint32_t input_width_tiles = 64;
    uint32_t output_height_tiles = 32;
    uint32_t output_width_tiles = 32;
    uint32_t start_height_tile = 0;
    uint32_t start_width_tile = 0;
    uint32_t batches = 1;
    uint32_t num_pages = 2;
    uint32_t repeats = 1;
    uint32_t device_id = 0;
    CoreCoord core = kDefaultCore;
    uint32_t core_grid_x = 1;
    uint32_t core_grid_y = 1;
};

struct CoreWork {
    CoreCoord core;
    uint32_t start_output_tile = 0;
    uint32_t tile_count = 0;
    std::shared_ptr<Buffer> ring_buffer;
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
        case Mode::StaticStreamRegCbRegs: return "static-streamreg-cbregs";
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
    if (mode == "static-streamreg-cbregs") {
        return Mode::StaticStreamRegCbRegs;
    }
    return std::nullopt;
}

std::vector<Mode> modes_to_run(const std::string& mode) {
    if (mode == "all") {
        return {Mode::Cb, Mode::StaticRuntime, Mode::StaticStreamRegCbRegs};
    }
    auto parsed = parse_mode(mode);
    if (!parsed.has_value()) {
        throw std::invalid_argument("Unknown --mode. Valid values are all, cb, static-runtime, static-streamreg-cbregs.");
    }
    return {*parsed};
}

bool is_static_mode(Mode mode) { return mode == Mode::StaticRuntime || mode == Mode::StaticStreamRegCbRegs; }

bool uses_stream_reg_cbregs(Mode mode) { return mode == Mode::StaticStreamRegCbRegs; }

uint32_t input_tiles(const Options& options) {
    return options.batches * options.input_height_tiles * options.input_width_tiles;
}

uint32_t output_tiles(const Options& options) {
    return options.batches * options.output_height_tiles * options.output_width_tiles;
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--mode=all|cb|static-runtime|static-streamreg-cbregs] "
        "[--input-height-tiles=N] [--input-width-tiles=N] [--output-height-tiles=N] "
        "[--output-width-tiles=N] [--start-height-tile=N] [--start-width-tile=N] "
        "[--batches=N] [--num-pages=N] [--repeats=N] [--device-id=N] "
        "[--core-x=N] [--core-y=N] [--core-grid-x=N] [--core-grid-y=N]\n",
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
        } else if (starts_with(arg, "--input-height-tiles=")) {
            options.input_height_tiles =
                parse_u32(arg.substr(std::string_view("--input-height-tiles=").size()), "input-height-tiles");
        } else if (starts_with(arg, "--input-width-tiles=")) {
            options.input_width_tiles =
                parse_u32(arg.substr(std::string_view("--input-width-tiles=").size()), "input-width-tiles");
        } else if (starts_with(arg, "--output-height-tiles=")) {
            options.output_height_tiles =
                parse_u32(arg.substr(std::string_view("--output-height-tiles=").size()), "output-height-tiles");
        } else if (starts_with(arg, "--output-width-tiles=")) {
            options.output_width_tiles =
                parse_u32(arg.substr(std::string_view("--output-width-tiles=").size()), "output-width-tiles");
        } else if (starts_with(arg, "--start-height-tile=")) {
            options.start_height_tile =
                parse_u32(arg.substr(std::string_view("--start-height-tile=").size()), "start-height-tile");
        } else if (starts_with(arg, "--start-width-tile=")) {
            options.start_width_tile =
                parse_u32(arg.substr(std::string_view("--start-width-tile=").size()), "start-width-tile");
        } else if (starts_with(arg, "--batches=")) {
            options.batches = parse_u32(arg.substr(std::string_view("--batches=").size()), "batches");
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
        } else {
            throw std::invalid_argument(fmt::format("Unknown argument '{}'. Use --help to list options.", arg));
        }
    }

    if (options.input_height_tiles == 0 || options.input_width_tiles == 0 || options.output_height_tiles == 0 ||
        options.output_width_tiles == 0 || options.batches == 0 || options.num_pages == 0 || options.repeats == 0) {
        throw std::invalid_argument("Tile dimensions, batches, num-pages, and repeats must be greater than zero");
    }
    if (options.start_height_tile + options.output_height_tiles > options.input_height_tiles ||
        options.start_width_tile + options.output_width_tiles > options.input_width_tiles) {
        throw std::invalid_argument("Requested slice is outside the input tile rectangle");
    }
    if (options.core_grid_x == 0 || options.core_grid_y == 0) {
        throw std::invalid_argument("--core-grid-x and --core-grid-y must be greater than zero");
    }
    if (output_tiles(options) > kStreamRegCounterMask) {
        throw std::invalid_argument("stream-register protocol modes support up to 24-bit tile counters");
    }
    return options;
}

uint32_t checked_payload_size_bytes(uint32_t tiles, std::string_view name) {
    const uint64_t size = static_cast<uint64_t>(tiles) * kTileSizeBytes;
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(fmt::format("{} exceeds uint32_t addressable size", name));
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
    const uint32_t tiles = output_tiles(options);
    const uint32_t base_tiles = tiles / static_cast<uint32_t>(cores.size());
    const uint32_t remainder = tiles % static_cast<uint32_t>(cores.size());
    uint32_t start_tile = 0;
    for (size_t i = 0; i < cores.size(); ++i) {
        const uint32_t tile_count = base_tiles + (static_cast<uint32_t>(i) < remainder ? 1u : 0u);
        if (tile_count == 0) {
            continue;
        }
        work.push_back(CoreWork{.core = cores[i], .start_output_tile = start_tile, .tile_count = tile_count});
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

std::vector<bfloat16> make_input(const Options& options) {
    std::vector<bfloat16> data(static_cast<size_t>(input_tiles(options)) * kTileElements);
    for (uint32_t tile = 0; tile < input_tiles(options); ++tile) {
        for (uint32_t i = 0; i < kTileElements; ++i) {
            const uint32_t pattern = (tile * 131u + i * 17u + 53u) % 251u;
            data[static_cast<size_t>(tile) * kTileElements + i] = bfloat16(static_cast<float>(pattern) * 0.03125f);
        }
    }
    return data;
}

uint32_t input_tile_for_output_tile(const Options& options, uint32_t output_tile) {
    const uint32_t output_hw = options.output_height_tiles * options.output_width_tiles;
    const uint32_t batch = output_tile / output_hw;
    const uint32_t within_batch = output_tile % output_hw;
    const uint32_t out_h = within_batch / options.output_width_tiles;
    const uint32_t out_w = within_batch % options.output_width_tiles;
    return batch * options.input_height_tiles * options.input_width_tiles +
           (options.start_height_tile + out_h) * options.input_width_tiles + (options.start_width_tile + out_w);
}

std::vector<bfloat16> expected_output(const Options& options, const std::vector<bfloat16>& src) {
    std::vector<bfloat16> expected(static_cast<size_t>(output_tiles(options)) * kTileElements);
    for (uint32_t out_tile = 0; out_tile < output_tiles(options); ++out_tile) {
        const uint32_t in_tile = input_tile_for_output_tile(options, out_tile);
        std::copy_n(
            src.begin() + static_cast<size_t>(in_tile) * kTileElements,
            kTileElements,
            expected.begin() + static_cast<size_t>(out_tile) * kTileElements);
    }
    return expected;
}

std::map<std::string, std::string> protocol_defines(
    Mode mode,
    uint32_t protocol_start_value,
    uint32_t ring_addr,
    uint32_t protocol_start_addr,
    const Options& options) {
    return {
        {"BENCH_STATIC_PROTOCOL", is_static_mode(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_CBREGS", uses_stream_reg_cbregs(mode) ? "1" : "0"},
        {"BENCH_PROTOCOL_START_VALUE", std::to_string(protocol_start_value)},
        {"BENCH_RING_ADDR", std::to_string(ring_addr)},
        {"BENCH_PAGE_SIZE", std::to_string(kTileSizeBytes)},
        {"BENCH_NUM_PAGES", std::to_string(options.num_pages)},
        {"BENCH_PROTOCOL_START_SEM_ADDR", std::to_string(protocol_start_addr)},
        {"BENCH_STREAM_REG_START_STREAM_ID", std::to_string(kStreamRegStartStreamId)},
        {"BENCH_STREAM_REG_VALUE_MASK", std::to_string(kStreamRegCounterMask)},
    };
}

void create_circular_buffer(
    Program& program,
    CoreCoord core,
    Mode mode,
    const Options& options,
    const std::shared_ptr<Buffer>& ring_buffer) {
    auto config = CircularBufferConfig(options.num_pages * kTileSizeBytes, {{CBIndex::c_0, DataFormat::Float16_b}})
                      .set_page_size(CBIndex::c_0, kTileSizeBytes);
    if (is_static_mode(mode)) {
        if (!ring_buffer) {
            throw std::runtime_error("Static mode requires explicit L1 ring buffer");
        }
        config.set_globally_allocated_address(*ring_buffer);
    }
    CreateCircularBuffer(program, core, config);
}

bool validate_result(
    const std::vector<bfloat16>& expected,
    const std::vector<bfloat16>& result,
    float* max_abs_error) {
    if (result.size() != expected.size()) {
        *max_abs_error = std::numeric_limits<float>::infinity();
        return false;
    }

    bool ok = true;
    float max_error = 0.0f;
    uint32_t printed = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float error = std::abs(static_cast<float>(expected[i]) - static_cast<float>(result[i]));
        max_error = std::max(max_error, error);
        if (error > 0.0f) {
            ok = false;
            if (printed < 5) {
                fmt::print(
                    stderr,
                    "Mismatch at element {}: expected {}, got {}\n",
                    i,
                    static_cast<float>(expected[i]),
                    static_cast<float>(result[i]));
                ++printed;
            }
        }
    }
    *max_abs_error = max_error;
    return ok;
}

std::vector<uint32_t> reader_runtime_args(const CoreWork& work, const Options& options, uint32_t src_dram_addr) {
    const uint32_t start_input_tile = input_tile_for_output_tile(options, work.start_output_tile);
    const uint32_t start_within_output = work.start_output_tile %
                                         (options.output_height_tiles * options.output_width_tiles);
    const uint32_t start_out_h = start_within_output / options.output_width_tiles;
    const uint32_t start_out_w = start_within_output % options.output_width_tiles;
    return {
        src_dram_addr,
        work.tile_count,
        start_input_tile,
        start_out_h,
        start_out_w,
        options.output_height_tiles,
        options.output_width_tiles,
        options.input_width_tiles};
}

std::vector<uint32_t> writer_runtime_args(const CoreWork& work, uint32_t dst_dram_addr) {
    return {dst_dram_addr, work.tile_count, work.start_output_tile};
}

RunResult run_one(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Options& options,
    Mode mode,
    uint32_t repeat,
    const std::vector<bfloat16>& src,
    const std::vector<bfloat16>& expected) {
    std::vector<CoreWork> core_work = partition_work(options, selected_cores(options));
    if (core_work.empty()) {
        throw std::invalid_argument("No active cores were selected");
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    const CoreRangeSet active_core_ranges = core_range_set_from_work(core_work);

    auto src_dram_buffer = create_dram_buffer(mesh_device, checked_payload_size_bytes(input_tiles(options), "input"));
    auto dst_dram_buffer = create_dram_buffer(mesh_device, checked_payload_size_bytes(output_tiles(options), "output"));

    if (is_static_mode(mode)) {
        const uint32_t ring_bytes = options.num_pages * kTileSizeBytes;
        std::vector<uint8_t> zero_ring_buffer(ring_bytes, 0);
        std::vector<uint8_t> zero_sem_buffer(kProtocolStartBytes, 0);
        for (auto& work : core_work) {
            work.ring_buffer = create_core_local_l1_buffer(mesh_device, work.core, ring_bytes, kTileSizeBytes);
            detail::WriteToBuffer(work.ring_buffer, zero_ring_buffer);
            if (!uses_stream_reg_cbregs(mode)) {
                work.protocol_start_buffer =
                    create_core_local_l1_buffer(mesh_device, work.core, kProtocolStartBytes, kProtocolStartBytes);
                detail::WriteToBuffer(work.protocol_start_buffer, zero_sem_buffer);
            }
        }
    }

    for (const auto& work : core_work) {
        create_circular_buffer(program, work.core, mode, options, work.ring_buffer);
    }

    const uint32_t src_dram_addr = static_cast<uint32_t>(src_dram_buffer->address());
    const uint32_t dst_dram_addr = static_cast<uint32_t>(dst_dram_buffer->address());

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src_dram_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);

    const uint32_t protocol_start_value =
        (0x534c0000u ^ output_tiles(options) ^ (options.num_pages << 8) ^ (repeat * 0x00010001u) ^ dst_dram_addr) &
        kStreamRegCounterMask;
    const uint32_t first_ring_addr =
        is_static_mode(mode) ? core_local_l1_address(core_work.front().ring_buffer, core_work.front().core) : 0;
    const uint32_t first_start_addr =
        is_static_mode(mode) ? core_local_l1_address(core_work.front().protocol_start_buffer, core_work.front().core) : 0;
    auto common_defines = protocol_defines(mode, protocol_start_value, first_ring_addr, first_start_addr, options);

    KernelHandle reader_kernel = CreateKernel(
        program,
        std::string(kReaderKernel),
        active_core_ranges,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = common_defines});
    KernelHandle writer_kernel = CreateKernel(
        program,
        std::string(kWriterKernel),
        active_core_ranges,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = common_defines});

    for (const auto& work : core_work) {
        std::vector<uint32_t> reader_args = reader_runtime_args(work, options, src_dram_addr);
        std::vector<uint32_t> writer_args = writer_runtime_args(work, dst_dram_addr);

        if (is_static_mode(mode)) {
            const uint32_t ring_addr = core_local_l1_address(work.ring_buffer, work.core);
            const uint32_t protocol_start_addr = core_local_l1_address(work.protocol_start_buffer, work.core);
            reader_args.insert(reader_args.end(), {ring_addr, kTileSizeBytes, options.num_pages, protocol_start_addr});
            writer_args.insert(writer_args.end(), {ring_addr, kTileSizeBytes, options.num_pages, protocol_start_addr});
        }

        SetRuntimeArgs(program, reader_kernel, work.core, reader_args);
        SetRuntimeArgs(program, writer_kernel, work.core, writer_args);
    }

    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src, false);

    workload.add_program(device_range, std::move(program));

    const auto start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();

    std::vector<bfloat16> result;
    distributed::EnqueueReadMeshBuffer(cq, result, dst_dram_buffer, true);

    float max_abs_error = 0.0f;
    const bool ok = validate_result(expected, result, &max_abs_error);
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

        const auto src = make_input(options);
        const auto expected = expected_output(options, src);

        fmt::print(
            "ttnn_slice_tile_protocol: input_height_tiles={}, input_width_tiles={}, output_height_tiles={}, "
            "output_width_tiles={}, start_height_tile={}, start_width_tile={}, batches={}, output_tiles={}, "
            "tile_size={}, num_pages={}, repeats={}, core=({}, {}), core_grid=({}, {})\n",
            options.input_height_tiles,
            options.input_width_tiles,
            options.output_height_tiles,
            options.output_width_tiles,
            options.start_height_tile,
            options.start_width_tile,
            options.batches,
            output_tiles(options),
            kTileSizeBytes,
            options.num_pages,
            options.repeats,
            options.core.x,
            options.core.y,
            options.core_grid_x,
            options.core_grid_y);

        for (uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
            for (Mode mode : modes) {
                RunResult result = run_one(mesh_device, options, mode, repeat, src, expected);
                pass &= result.ok;
                fmt::print(
                    "mode={:<24} repeat={} enqueue_finish_us={} max_abs_error={} {}\n",
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
