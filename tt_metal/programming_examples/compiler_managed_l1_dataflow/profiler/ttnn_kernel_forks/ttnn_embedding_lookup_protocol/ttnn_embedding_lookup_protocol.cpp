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
constexpr uint32_t kIndexPageBytes = 32;
constexpr uint32_t kProtocolStartBytes = 64;
constexpr uint32_t kStreamRegCounterMask = 0x00ffffffu;
constexpr uint32_t kStreamRegStartStreamId = 3;

constexpr std::string_view kReaderKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_embedding_lookup_protocol/"
    "kernels/dataflow/reader_embedding_lookup_protocol.cpp";
constexpr std::string_view kWriterKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_embedding_lookup_protocol/"
    "kernels/dataflow/writer_embedding_lookup_protocol.cpp";

enum class Mode : uint32_t {
    Cb = 0,
    StaticRuntime = 1,
    StaticStreamRegCbRegs = 2,
};

struct Options {
    std::string mode = "all";
    uint32_t rows = 1024;
    uint32_t vocab_size = 32000;
    uint32_t dim = 128;
    uint32_t num_pages = 2;
    uint32_t repeats = 1;
    uint32_t device_id = 0;
    CoreCoord core = kDefaultCore;
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
        throw std::invalid_argument("Unknown --mode. Valid values are all, cb, static-runtime, static-streamreg-cbregs, static.");
    }
    return {*parsed};
}

bool is_static_mode(Mode mode) {
    return mode == Mode::StaticRuntime || mode == Mode::StaticStreamRegCbRegs;
}

bool uses_stream_reg_cbregs(Mode mode) {
    return mode == Mode::StaticStreamRegCbRegs;
}

uint32_t row_bytes(const Options& options) {
    const uint64_t bytes = static_cast<uint64_t>(options.dim) * sizeof(bfloat16);
    if (bytes == 0 || bytes % 32 != 0 || bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("--dim must produce a non-zero row size aligned to 32 bytes");
    }
    return static_cast<uint32_t>(bytes);
}

uint32_t checked_size_bytes(uint64_t pages, uint32_t page_bytes, std::string_view name) {
    const uint64_t bytes = pages * page_bytes;
    if (bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(fmt::format("{} exceeds uint32_t addressable size", name));
    }
    return static_cast<uint32_t>(bytes);
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--mode=all|cb|static-runtime|static-streamreg-cbregs] "
        "[--rows=N] [--vocab-size=N] [--dim=N] [--num-pages=N] [--repeats=N] "
        "[--device-id=N] [--core-x=N] [--core-y=N]\n",
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
        } else if (starts_with(arg, "--rows=")) {
            options.rows = parse_u32(arg.substr(std::string_view("--rows=").size()), "rows");
        } else if (starts_with(arg, "--vocab-size=")) {
            options.vocab_size = parse_u32(arg.substr(std::string_view("--vocab-size=").size()), "vocab-size");
        } else if (starts_with(arg, "--dim=")) {
            options.dim = parse_u32(arg.substr(std::string_view("--dim=").size()), "dim");
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
        } else {
            throw std::invalid_argument(fmt::format("Unknown argument '{}'. Use --help to list options.", arg));
        }
    }

    if (options.rows == 0 || options.vocab_size == 0 || options.dim == 0 || options.num_pages == 0 ||
        options.repeats == 0) {
        throw std::invalid_argument("--rows, --vocab-size, --dim, --num-pages, and --repeats must be greater than zero");
    }
    if (options.rows > kStreamRegCounterMask) {
        throw std::invalid_argument("stream-register protocol modes support up to 24-bit row counters");
    }
    (void)row_bytes(options);
    return options;
}

std::shared_ptr<distributed::MeshBuffer> create_dram_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size_bytes, uint32_t page_size) {
    distributed::DeviceLocalBufferConfig dram_config{.page_size = page_size, .buffer_type = BufferType::DRAM};
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

std::vector<uint32_t> make_indices(const Options& options) {
    const uint32_t words_per_page = kIndexPageBytes / sizeof(uint32_t);
    std::vector<uint32_t> indices(static_cast<size_t>(options.rows) * words_per_page, 0);
    for (uint32_t row = 0; row < options.rows; ++row) {
        indices[static_cast<size_t>(row) * words_per_page] = (row * 131u + 17u) % options.vocab_size;
    }
    return indices;
}

std::vector<bfloat16> make_weights(const Options& options) {
    std::vector<bfloat16> weights(static_cast<size_t>(options.vocab_size) * options.dim);
    for (uint32_t token = 0; token < options.vocab_size; ++token) {
        for (uint32_t d = 0; d < options.dim; ++d) {
            const uint32_t pattern = (token * 29u + d * 7u + 11u) % 997u;
            weights[static_cast<size_t>(token) * options.dim + d] = bfloat16(static_cast<float>(pattern) * 0.00390625f);
        }
    }
    return weights;
}

std::vector<bfloat16> expected_output(
    const Options& options, const std::vector<uint32_t>& indices, const std::vector<bfloat16>& weights) {
    const uint32_t words_per_page = kIndexPageBytes / sizeof(uint32_t);
    std::vector<bfloat16> expected(static_cast<size_t>(options.rows) * options.dim);
    for (uint32_t row = 0; row < options.rows; ++row) {
        const uint32_t token = indices[static_cast<size_t>(row) * words_per_page] % options.vocab_size;
        for (uint32_t d = 0; d < options.dim; ++d) {
            expected[static_cast<size_t>(row) * options.dim + d] =
                weights[static_cast<size_t>(token) * options.dim + d];
        }
    }
    return expected;
}

std::map<std::string, std::string> protocol_defines(
    Mode mode,
    uint32_t protocol_start_value,
    uint32_t row_ring_addr,
    uint32_t protocol_start_addr,
    const Options& options) {
    return {
        {"BENCH_STATIC_PROTOCOL", is_static_mode(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_CBREGS", uses_stream_reg_cbregs(mode) ? "1" : "0"},
        {"BENCH_PROTOCOL_START_VALUE", std::to_string(protocol_start_value)},
        {"BENCH_ROW_RING_ADDR", std::to_string(row_ring_addr)},
        {"BENCH_ROW_BYTES", std::to_string(row_bytes(options))},
        {"BENCH_NUM_PAGES", std::to_string(options.num_pages)},
        {"BENCH_PROTOCOL_START_SEM_ADDR", std::to_string(protocol_start_addr)},
        {"BENCH_STREAM_REG_START_STREAM_ID", std::to_string(kStreamRegStartStreamId)},
        {"BENCH_STREAM_REG_VALUE_MASK", std::to_string(kStreamRegCounterMask)},
    };
}

void create_circular_buffers(
    Program& program,
    CoreCoord core,
    Mode mode,
    const Options& options,
    const std::shared_ptr<Buffer>& row_ring_buffer,
    const std::shared_ptr<Buffer>& index_scratch_buffer) {
    const uint32_t row_page_bytes = row_bytes(options);
    auto row_config =
        CircularBufferConfig(options.num_pages * row_page_bytes, {{CBIndex::c_0, DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, row_page_bytes);
    if (is_static_mode(mode)) {
        if (!row_ring_buffer) {
            throw std::runtime_error("Static mode requires explicit row ring buffer");
        }
        row_config.set_globally_allocated_address(*row_ring_buffer);
    }
    CreateCircularBuffer(program, core, row_config);

    auto index_config = CircularBufferConfig(kIndexPageBytes, {{CBIndex::c_1, DataFormat::UInt32}})
                            .set_page_size(CBIndex::c_1, kIndexPageBytes);
    if (is_static_mode(mode)) {
        if (!index_scratch_buffer) {
            throw std::runtime_error("Static mode requires explicit index scratch buffer");
        }
        index_config.set_globally_allocated_address(*index_scratch_buffer);
    }
    CreateCircularBuffer(program, core, index_config);
}

bool validate_result(
    const std::vector<bfloat16>& expected, const std::vector<bfloat16>& result, float* max_abs_error) {
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

RunResult run_one(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Options& options,
    Mode mode,
    uint32_t repeat,
    const std::vector<uint32_t>& indices,
    const std::vector<bfloat16>& weights,
    const std::vector<bfloat16>& expected) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    const uint32_t row_page_bytes = row_bytes(options);
    auto index_buffer = create_dram_buffer(
        mesh_device, checked_size_bytes(options.rows, kIndexPageBytes, "index buffer"), kIndexPageBytes);
    auto weights_buffer = create_dram_buffer(
        mesh_device, checked_size_bytes(options.vocab_size, row_page_bytes, "weights buffer"), row_page_bytes);
    auto output_buffer =
        create_dram_buffer(mesh_device, checked_size_bytes(options.rows, row_page_bytes, "output buffer"), row_page_bytes);

    std::shared_ptr<Buffer> row_ring_buffer;
    std::shared_ptr<Buffer> index_scratch_buffer;
    std::shared_ptr<Buffer> protocol_start_buffer;
    if (is_static_mode(mode)) {
        const uint32_t ring_bytes = options.num_pages * row_page_bytes;
        std::vector<uint8_t> zero_ring_buffer(ring_bytes, 0);
        std::vector<uint8_t> zero_index_buffer(kIndexPageBytes, 0);
        std::vector<uint8_t> zero_sem_buffer(kProtocolStartBytes, 0);
        row_ring_buffer = create_core_local_l1_buffer(mesh_device, options.core, ring_bytes, row_page_bytes);
        index_scratch_buffer =
            create_core_local_l1_buffer(mesh_device, options.core, kIndexPageBytes, kIndexPageBytes);
        detail::WriteToBuffer(row_ring_buffer, zero_ring_buffer);
        detail::WriteToBuffer(index_scratch_buffer, zero_index_buffer);
        if (!uses_stream_reg_cbregs(mode)) {
            protocol_start_buffer =
                create_core_local_l1_buffer(mesh_device, options.core, kProtocolStartBytes, kProtocolStartBytes);
            detail::WriteToBuffer(protocol_start_buffer, zero_sem_buffer);
        }
    }

    create_circular_buffers(program, options.core, mode, options, row_ring_buffer, index_scratch_buffer);

    const uint32_t index_addr = static_cast<uint32_t>(index_buffer->address());
    const uint32_t weights_addr = static_cast<uint32_t>(weights_buffer->address());
    const uint32_t output_addr = static_cast<uint32_t>(output_buffer->address());

    const uint32_t row_ring_addr = core_local_l1_address(row_ring_buffer, options.core);
    const uint32_t protocol_start_addr = core_local_l1_address(protocol_start_buffer, options.core);
    const uint32_t protocol_start_value =
        (0x4d420000u ^ options.rows ^ options.vocab_size ^ (repeat * 0x00010001u) ^ output_addr) &
        kStreamRegCounterMask;

    auto defines = protocol_defines(mode, protocol_start_value, row_ring_addr, protocol_start_addr, options);

    std::vector<uint32_t> reader_compile_time_args = {kIndexPageBytes};
    TensorAccessorArgs(index_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(weights_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);

    KernelHandle reader_kernel = CreateKernel(
        program,
        std::string(kReaderKernel),
        options.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = defines});
    KernelHandle writer_kernel = CreateKernel(
        program,
        std::string(kWriterKernel),
        options.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = defines});

    std::vector<uint32_t> reader_args = {
        index_addr, weights_addr, options.rows, 0, row_page_bytes, options.vocab_size};
    std::vector<uint32_t> writer_args = {output_addr, options.rows, 0, row_page_bytes};
    if (is_static_mode(mode)) {
        reader_args.insert(reader_args.end(), {row_ring_addr, options.num_pages, protocol_start_addr});
        writer_args.insert(writer_args.end(), {row_ring_addr, options.num_pages, protocol_start_addr});
    }
    SetRuntimeArgs(program, reader_kernel, options.core, reader_args);
    SetRuntimeArgs(program, writer_kernel, options.core, writer_args);

    distributed::EnqueueWriteMeshBuffer(cq, index_buffer, indices, false);
    distributed::EnqueueWriteMeshBuffer(cq, weights_buffer, weights, false);

    workload.add_program(device_range, std::move(program));

    const auto start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();

    std::vector<bfloat16> result;
    distributed::EnqueueReadMeshBuffer(cq, result, output_buffer, true);

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
        if (options.core.x >= grid.x || options.core.y >= grid.y) {
            throw std::invalid_argument(fmt::format(
                "Core ({}, {}) is outside compute_with_storage_grid_size ({}, {})",
                options.core.x,
                options.core.y,
                grid.x,
                grid.y));
        }

        const auto indices = make_indices(options);
        const auto weights = make_weights(options);
        const auto expected = expected_output(options, indices, weights);

        fmt::print(
            "ttnn_embedding_lookup_protocol: rows={}, vocab_size={}, dim={}, row_bytes={}, index_page_bytes={}, "
            "num_pages={}, repeats={}, core=({}, {})\n",
            options.rows,
            options.vocab_size,
            options.dim,
            row_bytes(options),
            kIndexPageBytes,
            options.num_pages,
            options.repeats,
            options.core.x,
            options.core.y);

        for (uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
            for (Mode mode : modes) {
                RunResult result = run_one(mesh_device, options, mode, repeat, indices, weights, expected);
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
