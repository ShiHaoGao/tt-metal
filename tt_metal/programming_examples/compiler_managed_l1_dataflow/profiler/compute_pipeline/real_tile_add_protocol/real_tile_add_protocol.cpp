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
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace per_core_allocation = tt::tt_metal::experimental::per_core_allocation;

namespace {

constexpr CoreCoord kDefaultCore = {0, 0};
constexpr uint32_t kTileElements = tt::constants::TILE_HW;
constexpr uint32_t kTileSizeBytes = sizeof(bfloat16) * kTileElements;
constexpr uint32_t kSemSlotBytes = 64;
constexpr uint32_t kStreamRegCounterMask = 0x00ffffffu;
constexpr uint32_t kStreamRegStartStreamId = 3;
constexpr uint32_t kStreamRegInputReady0StreamId = 3;
constexpr uint32_t kStreamRegInputReady1StreamId = 3;
constexpr uint32_t kStreamRegInputConsumed0StreamId = 3;
constexpr uint32_t kStreamRegInputConsumed1StreamId = 3;
constexpr uint32_t kStreamRegOutputReadyStreamId = 3;
constexpr uint32_t kStreamRegOutputConsumedStreamId = 3;

constexpr std::string_view kReaderKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/kernels/dataflow/reader_binary_tiles.cpp";
constexpr std::string_view kWriterKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/kernels/dataflow/writer_tiles.cpp";
constexpr std::string_view kComputeKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/real_tile_add_protocol/kernels/compute/add_tiles.cpp";

enum class Mode : uint32_t {
    Cb = 0,
    StaticRuntime = 1,
    StaticCompileTime = 2,
    StaticStreamReg = 3,
    StaticStreamRegCbRegs = 4,
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
    bool serialized_static = false;
};

struct CoreWork {
    CoreCoord core;
    uint32_t start_tile = 0;
    uint32_t tile_count = 0;
    std::shared_ptr<Buffer> src0_ring_buffer;
    std::shared_ptr<Buffer> src1_ring_buffer;
    std::shared_ptr<Buffer> dst_ring_buffer;
    std::shared_ptr<Buffer> protocol_sem_buffer;
};

struct RunResult {
    Mode mode;
    uint32_t repeat;
    uint64_t enqueue_finish_us;
    bool ok;
    float max_abs_error;
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
        case Mode::StaticCompileTime: return "static-compiletime";
        case Mode::StaticStreamReg: return "static-streamreg";
        case Mode::StaticStreamRegCbRegs: return "static-streamreg-cbregs";
    }
    return "unknown";
}

std::optional<Mode> parse_mode(std::string_view mode) {
    if (mode == "cb") {
        return Mode::Cb;
    }
    if (mode == "static-runtime") {
        return Mode::StaticRuntime;
    }
    if (mode == "static-compiletime") {
        return Mode::StaticCompileTime;
    }
    if (mode == "static-streamreg") {
        return std::nullopt;
    }
    if (mode == "static-streamreg-cbregs") {
        return Mode::StaticStreamRegCbRegs;
    }
    return std::nullopt;
}

std::vector<Mode> modes_to_run(const std::string& mode) {
    if (mode == "all") {
        return {Mode::Cb, Mode::StaticRuntime, Mode::StaticCompileTime, Mode::StaticStreamRegCbRegs};
    }
    auto parsed = parse_mode(mode);
    if (!parsed.has_value()) {
        throw std::invalid_argument(
            "Unknown --mode. Valid values are all, cb, static-runtime, static-compiletime, "
            "static-streamreg-cbregs. The old static-streamreg compute mode is disabled.");
    }
    return {*parsed};
}

bool is_static_mode(Mode mode) {
    return mode == Mode::StaticRuntime || mode == Mode::StaticCompileTime || mode == Mode::StaticStreamReg ||
           mode == Mode::StaticStreamRegCbRegs;
}

bool uses_compile_time_args(Mode mode) {
    return mode == Mode::StaticCompileTime;
}

bool uses_stream_reg_sync(Mode mode) {
    return mode == Mode::StaticStreamReg;
}

bool uses_stream_reg_cbregs(Mode mode) {
    return mode == Mode::StaticStreamRegCbRegs;
}

bool uses_stream_reg_start_gate(Mode mode) {
    return uses_stream_reg_sync(mode) || uses_stream_reg_cbregs(mode);
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--mode=all|cb|static-runtime|static-compiletime|static-streamreg-cbregs] [--tiles=N] "
        "[--num-pages=N] "
        "[--repeats=N] [--device-id=N] [--core-x=N] [--core-y=N] "
        "[--core-grid-x=N] [--core-grid-y=N] "
        "[--serialized-static]\n",
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
        } else if (starts_with(arg, "--iterations=")) {
            options.tiles = parse_u32(arg.substr(std::string_view("--iterations=").size()), "iterations");
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
        } else if (arg == "--serialized-static") {
            options.serialized_static = true;
        } else {
            throw std::invalid_argument(fmt::format("Unknown argument '{}'. Use --help to list options.", arg));
        }
    }

    if (options.tiles == 0) {
        throw std::invalid_argument("--tiles must be greater than zero");
    }
    if (options.num_pages == 0) {
        throw std::invalid_argument("--num-pages must be greater than zero");
    }
    if (options.repeats == 0) {
        throw std::invalid_argument("--repeats must be greater than zero");
    }
    if (options.core_grid_x == 0 || options.core_grid_y == 0) {
        throw std::invalid_argument("--core-grid-x and --core-grid-y must be greater than zero");
    }
    return options;
}

uint32_t payload_size_bytes(const Options& options) {
    const uint64_t size = static_cast<uint64_t>(options.tiles) * kTileSizeBytes;
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Input/output payload exceeds uint32_t addressable size");
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

uint32_t protocol_sem_slots(const Options& options) {
    return options.num_pages * 4 + 1;
}

uint32_t protocol_sem_size_bytes(const Options& options) {
    const uint64_t size = static_cast<uint64_t>(protocol_sem_slots(options)) * kSemSlotBytes;
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Static protocol semaphore buffer exceeds uint32_t addressable size");
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
    if (cores.empty()) {
        return work;
    }

    const uint32_t base_tiles = options.tiles / static_cast<uint32_t>(cores.size());
    const uint32_t remainder = options.tiles % static_cast<uint32_t>(cores.size());
    uint32_t start_tile = 0;
    for (size_t i = 0; i < cores.size(); ++i) {
        const uint32_t tile_count = base_tiles + (static_cast<uint32_t>(i) < remainder ? 1u : 0u);
        if (tile_count == 0) {
            continue;
        }
        work.push_back(CoreWork{
            .core = cores[i],
            .start_tile = start_tile,
            .tile_count = tile_count,
        });
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
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = kTileSizeBytes, .buffer_type = tt_metal::BufferType::DRAM};
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

    return Buffer::create(
        devices.front(),
        size_bytes,
        page_size_bytes,
        BufferType::L1,
        sharding_args,
        true /* bottom_up */);
}

std::shared_ptr<Buffer> create_core_local_l1_ring_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, CoreCoord core, uint32_t size_bytes) {
    return create_core_local_l1_buffer(mesh_device, core, size_bytes, kTileSizeBytes);
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
            data[static_cast<size_t>(tile) * kTileElements + i] = bfloat16(static_cast<float>(pattern) * 0.125f);
        }
    }
    return data;
}

std::map<std::string, std::string> kernel_defines(
    const Options& options,
    Mode mode,
    uint32_t src0_dram_addr,
    uint32_t src1_dram_addr,
    uint32_t dst_dram_addr,
    uint32_t src0_ring_addr,
    uint32_t src1_ring_addr,
    uint32_t dst_ring_addr,
    uint32_t input_ready_sem_addr,
    uint32_t input_consumed_sem_addr,
    uint32_t output_ready_sem_addr,
    uint32_t output_consumed_sem_addr,
    uint32_t protocol_start_sem_addr,
    uint32_t repeat) {
    const uint32_t protocol_start_value = 0x5a5a0000u ^ options.tiles ^ (options.num_pages << 8) ^
                                          protocol_start_sem_addr ^ dst_ring_addr ^ (repeat * 0x00010001u);
    return {
        {"BENCH_STATIC_PROTOCOL", is_static_mode(mode) ? "1" : "0"},
        {"BENCH_SERIAL_STATIC_PROTOCOL", (is_static_mode(mode) && options.serialized_static) ? "1" : "0"},
        {"BENCH_TRACE_STATIC_PROTOCOL",
         (is_static_mode(mode) && std::getenv("RTADD_TRACE_STATIC") != nullptr) ? "1" : "0"},
        {"BENCH_USE_COMPILE_TIME_ARGS", uses_compile_time_args(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_SYNC", uses_stream_reg_sync(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_CBREGS", uses_stream_reg_cbregs(mode) ? "1" : "0"},
        {"BENCH_ITERATIONS", std::to_string(options.tiles)},
        {"BENCH_START_TILE", "0"},
        {"BENCH_PAGE_SIZE", std::to_string(kTileSizeBytes)},
        {"BENCH_NUM_PAGES", std::to_string(options.num_pages)},
        {"BENCH_SEM_SLOT_BYTES", std::to_string(kSemSlotBytes)},
        {"BENCH_SRC0_DRAM_ADDR", std::to_string(src0_dram_addr)},
        {"BENCH_SRC1_DRAM_ADDR", std::to_string(src1_dram_addr)},
        {"BENCH_DST_DRAM_ADDR", std::to_string(dst_dram_addr)},
        {"BENCH_SRC0_RING_ADDR", std::to_string(src0_ring_addr)},
        {"BENCH_SRC1_RING_ADDR", std::to_string(src1_ring_addr)},
        {"BENCH_DST_RING_ADDR", std::to_string(dst_ring_addr)},
        {"BENCH_INPUT_READY_SEM_ADDR", std::to_string(input_ready_sem_addr)},
        {"BENCH_INPUT_CONSUMED_SEM_ADDR", std::to_string(input_consumed_sem_addr)},
        {"BENCH_OUTPUT_READY_SEM_ADDR", std::to_string(output_ready_sem_addr)},
        {"BENCH_OUTPUT_CONSUMED_SEM_ADDR", std::to_string(output_consumed_sem_addr)},
        {"BENCH_PROTOCOL_START_SEM_ADDR", std::to_string(protocol_start_sem_addr)},
        {"BENCH_PROTOCOL_START_VALUE", std::to_string(protocol_start_value)},
        {"BENCH_STREAM_REG_START_STREAM_ID", std::to_string(kStreamRegStartStreamId)},
        {"BENCH_STREAM_REG_INPUT_READY0_STREAM_ID", std::to_string(kStreamRegInputReady0StreamId)},
        {"BENCH_STREAM_REG_INPUT_READY1_STREAM_ID", std::to_string(kStreamRegInputReady1StreamId)},
        {"BENCH_STREAM_REG_INPUT_CONSUMED0_STREAM_ID", std::to_string(kStreamRegInputConsumed0StreamId)},
        {"BENCH_STREAM_REG_INPUT_CONSUMED1_STREAM_ID", std::to_string(kStreamRegInputConsumed1StreamId)},
        {"BENCH_STREAM_REG_OUTPUT_READY_STREAM_ID", std::to_string(kStreamRegOutputReadyStreamId)},
        {"BENCH_STREAM_REG_OUTPUT_CONSUMED_STREAM_ID", std::to_string(kStreamRegOutputConsumedStreamId)},
        {"BENCH_STREAM_REG_VALUE_MASK", std::to_string(kStreamRegCounterMask)}};
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
                throw std::runtime_error("Static mode requires an explicit L1 ring buffer for each CB descriptor");
            }
            config.set_globally_allocated_address(*static_ring_buffer);
        }
        return config;
    };

    CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_0, src0_ring_buffer));
    CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_1, src1_ring_buffer));
    CreateCircularBuffer(program, core, make_cb_config(CBIndex::c_16, dst_ring_buffer));
}

bool validate_result(
    const std::vector<bfloat16>& src0,
    const std::vector<bfloat16>& src1,
    const std::vector<bfloat16>& result,
    float* max_abs_error) {
    bool ok = true;
    float max_error = 0.0f;
    const size_t n = src0.size();
    if (result.size() != n) {
        fmt::print(stderr, "Result size mismatch: expected {}, got {}\n", n, result.size());
        *max_abs_error = std::numeric_limits<float>::infinity();
        return false;
    }

    if (std::getenv("RTADD_DUMP_RESULT") != nullptr) {
        const size_t tiles_to_print = std::min<size_t>(n / kTileElements, 8);
        for (size_t tile = 0; tile < tiles_to_print; ++tile) {
            const size_t index = tile * kTileElements;
            const float expected = static_cast<float>(src0[index]) + static_cast<float>(src1[index]);
            const float actual = static_cast<float>(result[index]);
            fmt::print("tile {} first_element expected={} actual={}\n", tile, expected, actual);
        }
    }

    uint32_t printed = 0;
    for (size_t i = 0; i < n; ++i) {
        const float expected = static_cast<float>(src0[i]) + static_cast<float>(src1[i]);
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

RunResult run_one(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Options& options,
    Mode mode,
    uint32_t repeat,
    const std::vector<bfloat16>& src0,
    const std::vector<bfloat16>& src1) {
    const std::vector<CoreCoord> requested_cores = selected_cores(options);
    std::vector<CoreWork> core_work = partition_work(options, requested_cores);
    if (core_work.empty()) {
        throw std::invalid_argument("No active cores were selected");
    }
    if (uses_compile_time_args(mode) && core_work.size() != 1) {
        throw std::invalid_argument("static-compiletime mode currently supports only one active core");
    }
    if (uses_stream_reg_sync(mode)) {
        throw std::invalid_argument(
            "static-streamreg is disabled for real_tile_add_protocol: the old idle-stream scratch-register "
            "variant is not a valid compute-path comparison. Implement static-streamreg-cbregs with per-CB "
            "tiles_received/tiles_acked registers before enabling this mode");
    }
    if (uses_stream_reg_sync(mode) && options.tiles > kStreamRegCounterMask) {
        throw std::invalid_argument("static-streamreg mode supports up to 24-bit tile counters");
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
        const uint32_t sem_bytes = protocol_sem_size_bytes(options);
        std::vector<uint8_t> zero_ring_buffer(ring_bytes, 0);
        std::vector<uint8_t> zero_sem_buffer(sem_bytes, 0);
        for (auto& work : core_work) {
            work.src0_ring_buffer = create_core_local_l1_ring_buffer(mesh_device, work.core, ring_bytes);
            work.src1_ring_buffer = create_core_local_l1_ring_buffer(mesh_device, work.core, ring_bytes);
            work.dst_ring_buffer = create_core_local_l1_ring_buffer(mesh_device, work.core, ring_bytes);
            detail::WriteToBuffer(work.src0_ring_buffer, zero_ring_buffer);
            detail::WriteToBuffer(work.src1_ring_buffer, zero_ring_buffer);
            detail::WriteToBuffer(work.dst_ring_buffer, zero_ring_buffer);
            if (!uses_stream_reg_start_gate(mode)) {
                work.protocol_sem_buffer =
                    create_core_local_l1_buffer(mesh_device, work.core, sem_bytes, kSemSlotBytes);
                detail::WriteToBuffer(work.protocol_sem_buffer, zero_sem_buffer);
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

    auto core_addresses = [&](const CoreWork& work) {
        struct Addresses {
            uint32_t src0_ring_addr;
            uint32_t src1_ring_addr;
            uint32_t dst_ring_addr;
            uint32_t input_ready_sem_addr;
            uint32_t input_consumed_sem_addr;
            uint32_t output_ready_sem_addr;
            uint32_t output_consumed_sem_addr;
            uint32_t protocol_start_sem_addr;
        };

        const bool has_protocol_sem_buffer = work.protocol_sem_buffer != nullptr;
        const uint32_t sem_base_addr =
            has_protocol_sem_buffer ? core_local_l1_address(work.protocol_sem_buffer, work.core) : 0;
        const uint32_t input_ready_sem_addr = has_protocol_sem_buffer ? sem_base_addr : 0;
        const uint32_t input_consumed_sem_addr =
            has_protocol_sem_buffer ? input_ready_sem_addr + options.num_pages * kSemSlotBytes : 0;
        const uint32_t output_ready_sem_addr =
            has_protocol_sem_buffer ? input_consumed_sem_addr + options.num_pages * kSemSlotBytes : 0;
        const uint32_t output_consumed_sem_addr =
            has_protocol_sem_buffer ? output_ready_sem_addr + options.num_pages * kSemSlotBytes : 0;
        const uint32_t protocol_start_sem_addr =
            has_protocol_sem_buffer ? output_consumed_sem_addr + options.num_pages * kSemSlotBytes : 0;
        return Addresses{
            .src0_ring_addr = core_local_l1_address(work.src0_ring_buffer, work.core),
            .src1_ring_addr = core_local_l1_address(work.src1_ring_buffer, work.core),
            .dst_ring_addr = core_local_l1_address(work.dst_ring_buffer, work.core),
            .input_ready_sem_addr = input_ready_sem_addr,
            .input_consumed_sem_addr = input_consumed_sem_addr,
            .output_ready_sem_addr = output_ready_sem_addr,
            .output_consumed_sem_addr = output_consumed_sem_addr,
            .protocol_start_sem_addr = protocol_start_sem_addr};
    };

    const auto first_addresses = core_addresses(core_work.front());
    if (std::getenv("RTADD_PRINT_ADDRS") != nullptr) {
        for (const auto& work : core_work) {
            const auto addrs = core_addresses(work);
            fmt::print(
                "mode={} core=({}, {}) start={} tiles={} src0_ring=0x{:x}/0x{:x} src1_ring=0x{:x}/0x{:x} dst_ring=0x{:x}/0x{:x} sems=0x{:x},0x{:x},0x{:x},0x{:x},0x{:x}\n",
                mode_name(mode),
                work.core.x,
                work.core.y,
                work.start_tile,
                work.tile_count,
                static_cast<uint32_t>(work.src0_ring_buffer ? work.src0_ring_buffer->address() : 0),
                addrs.src0_ring_addr,
                static_cast<uint32_t>(work.src1_ring_buffer ? work.src1_ring_buffer->address() : 0),
                addrs.src1_ring_addr,
                static_cast<uint32_t>(work.dst_ring_buffer ? work.dst_ring_buffer->address() : 0),
                addrs.dst_ring_addr,
                addrs.input_ready_sem_addr,
                addrs.input_consumed_sem_addr,
                addrs.output_ready_sem_addr,
                addrs.output_consumed_sem_addr,
                addrs.protocol_start_sem_addr);
        }
    }

    auto defines = kernel_defines(
        options,
        mode,
        src0_dram_addr,
        src1_dram_addr,
        dst_dram_addr,
        first_addresses.src0_ring_addr,
        first_addresses.src1_ring_addr,
        first_addresses.dst_ring_addr,
        first_addresses.input_ready_sem_addr,
        first_addresses.input_consumed_sem_addr,
        first_addresses.output_ready_sem_addr,
        first_addresses.output_consumed_sem_addr,
        first_addresses.protocol_start_sem_addr,
        repeat);

    KernelHandle reader_kernel = CreateKernel(
        program,
        std::string(kReaderKernel),
        active_core_ranges,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .defines = defines});
    KernelHandle writer_kernel = CreateKernel(
        program,
        std::string(kWriterKernel),
        active_core_ranges,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .defines = defines});
    KernelHandle compute_kernel = CreateKernel(
        program,
        std::string(kComputeKernel),
        active_core_ranges,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .defines = defines});

    for (const auto& work : core_work) {
        const auto addrs = core_addresses(work);
        if (uses_compile_time_args(mode)) {
            SetRuntimeArgs(program, reader_kernel, work.core, {});
            SetRuntimeArgs(program, writer_kernel, work.core, {});
            SetRuntimeArgs(program, compute_kernel, work.core, {});
        } else if (is_static_mode(mode)) {
            SetRuntimeArgs(
                program,
                reader_kernel,
                work.core,
                {work.tile_count,
                 src0_dram_addr,
                 src1_dram_addr,
                 work.start_tile,
                 addrs.src0_ring_addr,
                 addrs.src1_ring_addr,
                 kTileSizeBytes,
                 options.num_pages,
                 addrs.input_ready_sem_addr,
                 addrs.input_consumed_sem_addr,
                 addrs.output_ready_sem_addr,
                 addrs.output_consumed_sem_addr,
                 addrs.protocol_start_sem_addr});
            SetRuntimeArgs(
                program,
                writer_kernel,
                work.core,
                {work.tile_count,
                 dst_dram_addr,
                 work.start_tile,
                 addrs.dst_ring_addr,
                 kTileSizeBytes,
                 options.num_pages,
                 addrs.output_ready_sem_addr,
                 addrs.output_consumed_sem_addr,
                 addrs.protocol_start_sem_addr});
            SetRuntimeArgs(
                program,
                compute_kernel,
                work.core,
                {work.tile_count,
                 addrs.src0_ring_addr,
                 addrs.src1_ring_addr,
                 addrs.dst_ring_addr,
                 kTileSizeBytes,
                 options.num_pages,
                 addrs.input_ready_sem_addr,
                 addrs.input_consumed_sem_addr,
                 addrs.output_ready_sem_addr,
                 addrs.output_consumed_sem_addr,
                 addrs.protocol_start_sem_addr});
        } else {
            SetRuntimeArgs(
                program, reader_kernel, work.core, {work.tile_count, src0_dram_addr, src1_dram_addr, work.start_tile});
            SetRuntimeArgs(program, writer_kernel, work.core, {work.tile_count, dst_dram_addr, work.start_tile});
            SetRuntimeArgs(program, compute_kernel, work.core, {work.tile_count});
        }
    }

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1, false);

    workload.add_program(device_range, std::move(program));

    const auto start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();

    if (is_static_mode(mode) && std::getenv("RTADD_DUMP_RING") != nullptr) {
        std::vector<bfloat16> src0_ring_snapshot;
        std::vector<bfloat16> src1_ring_snapshot;
        std::vector<bfloat16> dst_ring_snapshot;
        detail::ReadFromBuffer(core_work.front().src0_ring_buffer, src0_ring_snapshot);
        detail::ReadFromBuffer(core_work.front().src1_ring_buffer, src1_ring_snapshot);
        detail::ReadFromBuffer(core_work.front().dst_ring_buffer, dst_ring_snapshot);
        const size_t tiles_to_print = std::min<size_t>(dst_ring_snapshot.size() / kTileElements, 8);
        for (size_t tile = 0; tile < tiles_to_print; ++tile) {
            const size_t index = tile * kTileElements;
            fmt::print(
                "ring tile {} src0={} src1={} dst={}\n",
                tile,
                static_cast<float>(src0_ring_snapshot[index]),
                static_cast<float>(src1_ring_snapshot[index]),
                static_cast<float>(dst_ring_snapshot[index]));
        }
    }

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

        std::shared_ptr<distributed::MeshDevice> mesh_device =
            distributed::MeshDevice::create_unit_mesh(options.device_id);

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
            "real_tile_add_protocol: tiles={}, tile_size={}, num_pages={}, repeats={}, core=({}, {}), "
            "core_grid=({}, {}), serialized_static={}\n",
            options.tiles,
            kTileSizeBytes,
            options.num_pages,
            options.repeats,
            options.core.x,
            options.core.y,
            options.core_grid_x,
            options.core_grid_y,
            options.serialized_static ? "true" : "false");

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
