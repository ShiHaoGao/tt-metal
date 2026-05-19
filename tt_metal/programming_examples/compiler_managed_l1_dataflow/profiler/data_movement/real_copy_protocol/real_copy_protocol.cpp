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
constexpr uint32_t kStreamRegStartStreamId = 0;
constexpr uint32_t kStreamRegReadyStreamId = 1;
constexpr uint32_t kStreamRegConsumedStreamId = 2;

constexpr std::string_view kReaderKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/data_movement/real_copy_protocol/kernels/dataflow/reader_copy.cpp";
constexpr std::string_view kWriterKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/data_movement/real_copy_protocol/kernels/dataflow/writer_copy.cpp";

enum class Mode : uint32_t {
    Cb = 0,
    StaticRuntime = 1,
    StaticCompileTime = 2,
    StaticStreamRegScratch = 3,
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
};

struct CoreWork {
    CoreCoord core;
    uint32_t start_tile = 0;
    uint32_t tile_count = 0;
    std::shared_ptr<Buffer> ring_buffer;
    std::shared_ptr<Buffer> sem_buffer;
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
        case Mode::StaticStreamRegScratch: return "static-streamreg-scratch";
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
    if (mode == "static-streamreg-scratch") {
        return Mode::StaticStreamRegScratch;
    }
    return std::nullopt;
}

std::vector<Mode> modes_to_run(const std::string& mode) {
    if (mode == "all") {
        return {Mode::Cb, Mode::StaticRuntime, Mode::StaticCompileTime, Mode::StaticStreamRegScratch};
    }
    auto parsed = parse_mode(mode);
    if (!parsed.has_value()) {
        throw std::invalid_argument(
            "Unknown --mode. Valid values are all, cb, static-runtime, static-compiletime, static-streamreg-scratch");
    }
    return {*parsed};
}

bool is_static_mode(Mode mode) {
    return mode == Mode::StaticRuntime || mode == Mode::StaticCompileTime || mode == Mode::StaticStreamRegScratch;
}

bool uses_compile_time_args(Mode mode) {
    return mode == Mode::StaticCompileTime;
}

bool uses_stream_reg_sync(Mode mode) {
    return mode == Mode::StaticStreamRegScratch;
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--mode=all|cb|static-runtime|static-compiletime|static-streamreg-scratch] [--tiles=N] "
        "[--num-pages=N] [--repeats=N] [--device-id=N] [--core-x=N] [--core-y=N] "
        "[--core-grid-x=N] [--core-grid-y=N]\n",
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

uint32_t sem_size_bytes() {
    return 3 * kSemSlotBytes;
}

uint32_t make_start_value(
    const Options& options,
    uint32_t repeat,
    uint32_t src_dram_addr,
    uint32_t dst_dram_addr,
    uint32_t ring_addr,
    uint32_t start_sem_addr) {
    uint32_t value = 0x5a5a0000u ^ options.tiles ^ (options.num_pages << 8) ^ repeat ^ src_dram_addr ^
                     dst_dram_addr ^ ring_addr ^ start_sem_addr;
    value &= kStreamRegCounterMask;
    return value == 0 ? 1 : value;
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

uint32_t core_local_l1_address(const std::shared_ptr<Buffer>& buffer, CoreCoord core) {
    if (!buffer) {
        return 0;
    }
    return static_cast<uint32_t>(per_core_allocation::get_per_core_address(*buffer, core));
}

std::vector<bfloat16> make_input(uint32_t tiles) {
    std::vector<bfloat16> data(static_cast<size_t>(tiles) * kTileElements);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        for (uint32_t i = 0; i < kTileElements; ++i) {
            const uint32_t pattern = (tile * 131u + i * 17u + 53u) % 97u;
            data[static_cast<size_t>(tile) * kTileElements + i] = bfloat16(static_cast<float>(pattern) * 0.125f);
        }
    }
    return data;
}

std::map<std::string, std::string> kernel_defines(
    const Options& options,
    Mode mode,
    uint32_t src_dram_addr,
    uint32_t dst_dram_addr,
    uint32_t ring_addr,
    uint32_t ready_sem_addr,
    uint32_t consumed_sem_addr,
    uint32_t start_sem_addr,
    uint32_t start_value) {
    return {
        {"BENCH_STATIC_PROTOCOL", is_static_mode(mode) ? "1" : "0"},
        {"BENCH_USE_COMPILE_TIME_ARGS", uses_compile_time_args(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_SYNC", uses_stream_reg_sync(mode) ? "1" : "0"},
        {"BENCH_ITERATIONS", std::to_string(options.tiles)},
        {"BENCH_START_TILE", "0"},
        {"BENCH_PAGE_SIZE", std::to_string(kTileSizeBytes)},
        {"BENCH_NUM_PAGES", std::to_string(options.num_pages)},
        {"BENCH_SEM_SLOT_BYTES", std::to_string(kSemSlotBytes)},
        {"BENCH_SRC_DRAM_ADDR", std::to_string(src_dram_addr)},
        {"BENCH_DST_DRAM_ADDR", std::to_string(dst_dram_addr)},
        {"BENCH_RING_ADDR", std::to_string(ring_addr)},
        {"BENCH_READY_SEM_ADDR", std::to_string(ready_sem_addr)},
        {"BENCH_CONSUMED_SEM_ADDR", std::to_string(consumed_sem_addr)},
        {"BENCH_START_SEM_ADDR", std::to_string(start_sem_addr)},
        {"BENCH_STREAM_REG_START_STREAM_ID", std::to_string(kStreamRegStartStreamId)},
        {"BENCH_STREAM_REG_READY_STREAM_ID", std::to_string(kStreamRegReadyStreamId)},
        {"BENCH_STREAM_REG_CONSUMED_STREAM_ID", std::to_string(kStreamRegConsumedStreamId)},
        {"BENCH_STREAM_REG_VALUE_MASK", std::to_string(kStreamRegCounterMask)},
        {"BENCH_START_VALUE", std::to_string(start_value)}};
}

void create_circular_buffer(Program& program, CoreCoord core, const Options& options) {
    const uint32_t cb_size = options.num_pages * kTileSizeBytes;
    CircularBufferConfig config(cb_size, {{CBIndex::c_0, DataFormat::Float16_b}});
    config.set_page_size(CBIndex::c_0, kTileSizeBytes);
    CreateCircularBuffer(program, core, config);
}

bool validate_result(const std::vector<bfloat16>& src, const std::vector<bfloat16>& result, float* max_abs_error) {
    bool ok = true;
    float max_error = 0.0f;
    if (src.size() != result.size()) {
        *max_abs_error = std::numeric_limits<float>::infinity();
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i) {
        const float expected = static_cast<float>(src[i]);
        const float actual = static_cast<float>(result[i]);
        const float error = std::abs(expected - actual);
        max_error = std::max(max_error, error);
        if (error > 0.0f) {
            ok = false;
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
    const std::vector<bfloat16>& src) {
    const std::vector<CoreCoord> requested_cores = selected_cores(options);
    std::vector<CoreWork> core_work = partition_work(options, requested_cores);
    if (core_work.empty()) {
        throw std::invalid_argument("No active cores were selected");
    }
    if (uses_compile_time_args(mode) && core_work.size() != 1) {
        throw std::invalid_argument("static-compiletime mode currently supports only one active core");
    }
    if (uses_stream_reg_sync(mode) && options.tiles > kStreamRegCounterMask) {
        throw std::invalid_argument("static-streamreg-scratch mode supports up to 24-bit tile counters");
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    const CoreRangeSet active_core_ranges = core_range_set_from_work(core_work);

    const uint32_t payload_bytes = payload_size_bytes(options);
    auto src_dram_buffer = create_dram_buffer(mesh_device, payload_bytes);
    auto dst_dram_buffer = create_dram_buffer(mesh_device, payload_bytes);

    if (is_static_mode(mode)) {
        const uint32_t ring_bytes = ring_size_bytes(options);
        std::vector<uint8_t> zero_ring(ring_bytes, 0);
        for (auto& work : core_work) {
            work.ring_buffer = create_core_local_l1_buffer(mesh_device, work.core, ring_bytes, kTileSizeBytes);
            detail::WriteToBuffer(work.ring_buffer, zero_ring);
            if (!uses_stream_reg_sync(mode)) {
                const uint32_t sem_bytes = sem_size_bytes();
                std::vector<uint8_t> zero_sem(sem_bytes, 0);
                work.sem_buffer = create_core_local_l1_buffer(mesh_device, work.core, sem_bytes, kSemSlotBytes);
                detail::WriteToBuffer(work.sem_buffer, zero_sem);
            }
        }
    } else {
        for (const auto& work : core_work) {
            create_circular_buffer(program, work.core, options);
        }
    }

    const uint32_t src_dram_addr = static_cast<uint32_t>(src_dram_buffer->address());
    const uint32_t dst_dram_addr = static_cast<uint32_t>(dst_dram_buffer->address());

    auto core_addresses = [&](const CoreWork& work) {
        struct Addresses {
            uint32_t ring_addr;
            uint32_t ready_sem_addr;
            uint32_t consumed_sem_addr;
            uint32_t start_sem_addr;
        };
        const uint32_t sem_base_addr = core_local_l1_address(work.sem_buffer, work.core);
        return Addresses{
            .ring_addr = core_local_l1_address(work.ring_buffer, work.core),
            .ready_sem_addr = sem_base_addr,
            .consumed_sem_addr = sem_base_addr + kSemSlotBytes,
            .start_sem_addr = sem_base_addr + 2 * kSemSlotBytes};
    };

    const auto first_addresses = core_addresses(core_work.front());
    const uint32_t start_value = make_start_value(
        options,
        repeat,
        src_dram_addr,
        dst_dram_addr,
        first_addresses.ring_addr,
        first_addresses.start_sem_addr);
    auto defines = kernel_defines(
        options,
        mode,
        src_dram_addr,
        dst_dram_addr,
        first_addresses.ring_addr,
        first_addresses.ready_sem_addr,
        first_addresses.consumed_sem_addr,
        first_addresses.start_sem_addr,
        start_value);

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

    for (const auto& work : core_work) {
        const auto addrs = core_addresses(work);
        if (uses_compile_time_args(mode)) {
            SetRuntimeArgs(program, reader_kernel, work.core, {});
            SetRuntimeArgs(program, writer_kernel, work.core, {});
        } else if (is_static_mode(mode)) {
            SetRuntimeArgs(
                program,
                reader_kernel,
                work.core,
                {work.tile_count,
                 src_dram_addr,
                 work.start_tile,
                 addrs.ring_addr,
                 kTileSizeBytes,
                 options.num_pages,
                 addrs.ready_sem_addr,
                 addrs.consumed_sem_addr,
                 addrs.start_sem_addr});
            SetRuntimeArgs(
                program,
                writer_kernel,
                work.core,
                {work.tile_count,
                 dst_dram_addr,
                 work.start_tile,
                 addrs.ring_addr,
                 kTileSizeBytes,
                 options.num_pages,
                 addrs.ready_sem_addr,
                 addrs.consumed_sem_addr,
                 addrs.start_sem_addr});
        } else {
            SetRuntimeArgs(program, reader_kernel, work.core, {work.tile_count, src_dram_addr, work.start_tile});
            SetRuntimeArgs(program, writer_kernel, work.core, {work.tile_count, dst_dram_addr, work.start_tile});
        }
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
    const bool ok = validate_result(src, result, &max_abs_error);
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

        const auto src = make_input(options.tiles);
        fmt::print(
            "real_copy_protocol: tiles={}, tile_size={}, num_pages={}, repeats={}, core=({}, {}), "
            "core_grid=({}, {})\n",
            options.tiles,
            kTileSizeBytes,
            options.num_pages,
            options.repeats,
            options.core.x,
            options.core.y,
            options.core_grid_x,
            options.core_grid_y);

        for (uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
            for (Mode mode : modes) {
                RunResult result = run_one(mesh_device, options, mode, repeat, src);
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
