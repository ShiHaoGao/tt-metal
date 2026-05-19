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
#include <tt-metalium/tilize_utils.hpp>
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
constexpr uint32_t kStreamRegInputReady0StreamId = 4;
constexpr uint32_t kStreamRegInputReady1StreamId = 5;
constexpr uint32_t kStreamRegInputConsumed0StreamId = 6;
constexpr uint32_t kStreamRegInputConsumed1StreamId = 7;
constexpr uint32_t kStreamRegOutputReadyStreamId = 8;
constexpr uint32_t kStreamRegOutputConsumedStreamId = 9;

constexpr std::string_view kBinaryReaderKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/static_protocol_modeling/kernels/dataflow/reader_binary_tiles.cpp";
constexpr std::string_view kMatmulReaderKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/static_protocol_modeling/kernels/dataflow/reader_matmul_tiles.cpp";
constexpr std::string_view kWriterKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/static_protocol_modeling/kernels/dataflow/writer_tiles.cpp";
constexpr std::string_view kEltwiseComputeKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/static_protocol_modeling/kernels/compute/eltwise_tiles.cpp";
constexpr std::string_view kMatmulComputeKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/compute_pipeline/static_protocol_modeling/kernels/compute/matmul_tiles.cpp";

enum class Op : uint32_t {
    TileAdd = 0,
    EltwiseChain = 1,
    MatmulSingle = 2,
    MatmulBlock = 3,
};

enum class Mode : uint32_t {
    Cb = 0,
    StaticRuntime = 1,
    StaticCompileTime = 2,
    StaticSerialized = 3,
    StaticStreamReg = 4,
    StaticStreamRegCbRegs = 5,
    StaticStreamRegCbRegsCompileTime = 6,
};

struct Options {
    Op op = Op::TileAdd;
    std::string mode = "all";
    std::string sweep;
    uint32_t tiles = 1024;
    uint32_t num_slots = 2;
    uint32_t slot_bytes = kTileSizeBytes;
    uint32_t repeats = 1;
    uint32_t chain_depth = 4;
    uint32_t matmul_m_tiles = 1;
    uint32_t matmul_n_tiles = 1;
    uint32_t matmul_k_tiles = 4;
    uint32_t core_grid_x = 1;
    uint32_t core_grid_y = 1;
    uint32_t device_id = 0;
    CoreCoord core = kDefaultCore;
    bool serialized_static = false;
    bool matmul_m_tiles_explicit = false;
};

struct RunResult {
    Op op;
    Mode mode;
    uint32_t repeat;
    uint64_t enqueue_finish_us;
    bool ok;
    float max_abs_error;
};

struct TestData {
    std::vector<bfloat16> src0;
    std::vector<bfloat16> src1;
    std::vector<bfloat16> expected;
    bool result_is_tilized = false;
    uint32_t matrix_m = 0;
    uint32_t matrix_n = 0;
};

struct CoreWork {
    CoreCoord core;
    uint32_t base_m_tile = 0;
    uint32_t base_n_tile = 0;
    uint32_t local_m_tiles = 1;
    uint32_t local_n_tiles = 1;
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

const char* op_name(Op op) {
    switch (op) {
        case Op::TileAdd: return "tile-add";
        case Op::EltwiseChain: return "eltwise-chain";
        case Op::MatmulSingle: return "matmul-single";
        case Op::MatmulBlock: return "matmul-block";
    }
    return "unknown";
}

std::optional<Op> parse_op(std::string_view op) {
    if (op == "tile-add") {
        return Op::TileAdd;
    }
    if (op == "eltwise-chain") {
        return Op::EltwiseChain;
    }
    if (op == "matmul-single") {
        return Op::MatmulSingle;
    }
    if (op == "matmul-block") {
        return Op::MatmulBlock;
    }
    return std::nullopt;
}

bool is_matmul_op(Op op) {
    return op == Op::MatmulSingle || op == Op::MatmulBlock;
}

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::Cb: return "cb";
        case Mode::StaticRuntime: return "static-runtime";
        case Mode::StaticCompileTime: return "static-compiletime";
        case Mode::StaticSerialized: return "static-serialized";
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
    if (mode == "static-runtime") {
        return Mode::StaticRuntime;
    }
    if (mode == "static-compiletime") {
        return Mode::StaticCompileTime;
    }
    if (mode == "static-serialized") {
        return Mode::StaticSerialized;
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
        return {
            Mode::Cb,
            Mode::StaticRuntime,
            Mode::StaticCompileTime,
            Mode::StaticSerialized,
            Mode::StaticStreamRegCbRegs,
            Mode::StaticStreamRegCbRegsCompileTime};
    }
    auto parsed = parse_mode(mode);
    if (!parsed.has_value()) {
        throw std::invalid_argument(
            "Unknown --mode. Valid values are all, cb, static-runtime, static-compiletime, "
            "static-serialized, static-streamreg-cbregs, static-streamreg-cbregs-compiletime. "
            "The old static-streamreg compute mode is disabled.");
    }
    return {*parsed};
}

bool is_static_mode(Mode mode) {
    return mode == Mode::StaticRuntime || mode == Mode::StaticCompileTime || mode == Mode::StaticSerialized ||
           mode == Mode::StaticStreamReg || mode == Mode::StaticStreamRegCbRegs ||
           mode == Mode::StaticStreamRegCbRegsCompileTime;
}

bool uses_compile_time_args(Mode mode) {
    return mode == Mode::StaticCompileTime || mode == Mode::StaticStreamRegCbRegsCompileTime;
}

bool uses_stream_reg_sync(Mode mode) {
    return mode == Mode::StaticStreamReg;
}

bool uses_stream_reg_cbregs(Mode mode) {
    return mode == Mode::StaticStreamRegCbRegs || mode == Mode::StaticStreamRegCbRegsCompileTime;
}

bool uses_stream_reg_start_gate(Mode mode) {
    return uses_stream_reg_sync(mode) || uses_stream_reg_cbregs(mode);
}

bool uses_serialized_static(const Options& options, Mode mode) {
    return mode == Mode::StaticSerialized || (is_static_mode(mode) && options.serialized_static && !uses_stream_reg_sync(mode));
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--op=tile-add|eltwise-chain|matmul-single|matmul-block] "
        "[--mode=all|cb|static-runtime|static-compiletime|static-serialized|static-streamreg-cbregs|"
        "static-streamreg-cbregs-compiletime] "
        "[--tiles=N] [--num-slots=N] [--slot-bytes=N] [--chain-depth=N] "
        "[--matmul-m-tiles=N] [--matmul-n-tiles=N] [--matmul-k-tiles=N] "
        "[--core-grid-x=N] [--core-grid-y=N] [--repeats=N] [--sweep=preset|matmul|matmul-targeted] [--device-id=N] "
        "[--core-x=N] [--core-y=N] [--serialized-static]\n",
        argv0);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (starts_with(arg, "--op=")) {
            auto op = parse_op(arg.substr(std::string_view("--op=").size()));
            if (!op.has_value()) {
                throw std::invalid_argument(
                    "Unknown --op. Valid values are tile-add, eltwise-chain, matmul-single, matmul-block");
            }
            options.op = *op;
        } else if (starts_with(arg, "--mode=")) {
            options.mode = std::string(arg.substr(std::string_view("--mode=").size()));
        } else if (starts_with(arg, "--tiles=")) {
            options.tiles = parse_u32(arg.substr(std::string_view("--tiles=").size()), "tiles");
            options.matmul_n_tiles = options.tiles;
        } else if (starts_with(arg, "--iterations=")) {
            options.tiles = parse_u32(arg.substr(std::string_view("--iterations=").size()), "iterations");
        } else if (starts_with(arg, "--num-slots=")) {
            options.num_slots = parse_u32(arg.substr(std::string_view("--num-slots=").size()), "num-slots");
        } else if (starts_with(arg, "--num-pages=")) {
            options.num_slots = parse_u32(arg.substr(std::string_view("--num-pages=").size()), "num-pages");
        } else if (starts_with(arg, "--slot-bytes=")) {
            options.slot_bytes = parse_u32(arg.substr(std::string_view("--slot-bytes=").size()), "slot-bytes");
        } else if (starts_with(arg, "--page-size=")) {
            options.slot_bytes = parse_u32(arg.substr(std::string_view("--page-size=").size()), "page-size");
        } else if (starts_with(arg, "--chain-depth=")) {
            options.chain_depth = parse_u32(arg.substr(std::string_view("--chain-depth=").size()), "chain-depth");
        } else if (starts_with(arg, "--matmul-m-tiles=")) {
            options.matmul_m_tiles =
                parse_u32(arg.substr(std::string_view("--matmul-m-tiles=").size()), "matmul-m-tiles");
            options.matmul_m_tiles_explicit = true;
        } else if (starts_with(arg, "--matmul-n-tiles=")) {
            options.matmul_n_tiles =
                parse_u32(arg.substr(std::string_view("--matmul-n-tiles=").size()), "matmul-n-tiles");
            options.tiles = options.matmul_n_tiles;
        } else if (starts_with(arg, "--matmul-k-tiles=")) {
            options.matmul_k_tiles =
                parse_u32(arg.substr(std::string_view("--matmul-k-tiles=").size()), "matmul-k-tiles");
        } else if (starts_with(arg, "--core-grid-x=")) {
            options.core_grid_x = parse_u32(arg.substr(std::string_view("--core-grid-x=").size()), "core-grid-x");
        } else if (starts_with(arg, "--core-grid-y=")) {
            options.core_grid_y = parse_u32(arg.substr(std::string_view("--core-grid-y=").size()), "core-grid-y");
        } else if (starts_with(arg, "--repeats=")) {
            options.repeats = parse_u32(arg.substr(std::string_view("--repeats=").size()), "repeats");
        } else if (starts_with(arg, "--sweep=")) {
            options.sweep = std::string(arg.substr(std::string_view("--sweep=").size()));
        } else if (starts_with(arg, "--device-id=")) {
            options.device_id = parse_u32(arg.substr(std::string_view("--device-id=").size()), "device-id");
        } else if (starts_with(arg, "--core-x=")) {
            options.core.x = parse_u32(arg.substr(std::string_view("--core-x=").size()), "core-x");
        } else if (starts_with(arg, "--core-y=")) {
            options.core.y = parse_u32(arg.substr(std::string_view("--core-y=").size()), "core-y");
        } else if (arg == "--serialized-static") {
            options.serialized_static = true;
        } else {
            throw std::invalid_argument(fmt::format("Unknown argument '{}'. Use --help to list options.", arg));
        }
    }

    if (options.tiles == 0) {
        throw std::invalid_argument("--tiles must be greater than zero");
    }
    if (options.num_slots == 0) {
        throw std::invalid_argument("--num-slots must be greater than zero");
    }
    if (options.slot_bytes != kTileSizeBytes) {
        throw std::invalid_argument(fmt::format(
            "--slot-bytes must be {} for this BF16 tile benchmark version", kTileSizeBytes));
    }
    if (options.repeats == 0) {
        throw std::invalid_argument("--repeats must be greater than zero");
    }
    if (options.chain_depth == 0) {
        throw std::invalid_argument("--chain-depth must be greater than zero");
    }
    if (options.matmul_k_tiles == 0) {
        throw std::invalid_argument("--matmul-k-tiles must be greater than zero");
    }
    if (options.matmul_m_tiles == 0) {
        throw std::invalid_argument("--matmul-m-tiles must be greater than zero");
    }
    if (options.matmul_n_tiles == 0) {
        throw std::invalid_argument("--matmul-n-tiles must be greater than zero");
    }
    if (options.core_grid_x == 0 || options.core_grid_y == 0) {
        throw std::invalid_argument("--core-grid-x and --core-grid-y must be greater than zero");
    }
    if (options.op == Op::MatmulSingle && (options.core_grid_x != 1 || options.core_grid_y != 1)) {
        throw std::invalid_argument("matmul-single requires --core-grid-x=1 and --core-grid-y=1");
    }
    if (options.op == Op::MatmulBlock && !options.matmul_m_tiles_explicit) {
        options.matmul_m_tiles = options.core_grid_y;
    }
    if (options.op == Op::MatmulBlock) {
        if (options.matmul_m_tiles % options.core_grid_y != 0) {
            throw std::invalid_argument("matmul-block requires matmul-m-tiles to be divisible by core-grid-y");
        }
        if (options.matmul_n_tiles % options.core_grid_x != 0) {
            throw std::invalid_argument("matmul-block requires matmul-n-tiles to be divisible by core-grid-x");
        }
    }
    if (!options.sweep.empty() && options.sweep != "preset" && options.sweep != "matmul" &&
        options.sweep != "matmul-targeted") {
        throw std::invalid_argument("Unknown --sweep. Valid values are preset, matmul, and matmul-targeted");
    }
    return options;
}

uint32_t checked_bytes(size_t elements, std::string_view label) {
    const uint64_t size = static_cast<uint64_t>(elements) * sizeof(bfloat16);
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(fmt::format("{} payload exceeds uint32_t addressable size", label));
    }
    return static_cast<uint32_t>(size);
}

uint32_t output_tiles(const Options& options) {
    if (is_matmul_op(options.op)) {
        return options.matmul_m_tiles * options.matmul_n_tiles;
    }
    return options.tiles;
}

uint32_t input_pairs(const Options& options) {
    if (is_matmul_op(options.op)) {
        return options.matmul_m_tiles * options.matmul_n_tiles * options.matmul_k_tiles;
    }
    return options.tiles;
}

uint32_t ring_size_bytes(const Options& options) {
    const uint64_t size = static_cast<uint64_t>(options.num_slots) * options.slot_bytes;
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Static L1 ring exceeds uint32_t addressable size");
    }
    return static_cast<uint32_t>(size);
}

uint32_t protocol_sem_slots(const Options& options) {
    return options.num_slots * 4 + 1;
}

uint32_t protocol_sem_size_bytes(const Options& options) {
    const uint64_t size = static_cast<uint64_t>(protocol_sem_slots(options)) * kSemSlotBytes;
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Static protocol semaphore buffer exceeds uint32_t addressable size");
    }
    return static_cast<uint32_t>(size);
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

std::vector<bfloat16> make_binary_input(uint32_t tiles, uint32_t salt) {
    std::vector<bfloat16> data(static_cast<size_t>(tiles) * kTileElements);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        for (uint32_t i = 0; i < kTileElements; ++i) {
            const uint32_t pattern = (tile * 131u + i * 17u + salt * 53u) % 97u;
            data[static_cast<size_t>(tile) * kTileElements + i] = bfloat16(static_cast<float>(pattern) * 0.0625f);
        }
    }
    return data;
}

std::vector<bfloat16> make_matmul_input(uint32_t rows, uint32_t cols, uint32_t salt) {
    std::vector<bfloat16> data(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t pattern = (r * 19u + c * 7u + salt * 11u) % 7u;
            data[static_cast<size_t>(r) * cols + c] = bfloat16(static_cast<float>(pattern + 1) * 0.03125f);
        }
    }
    return data;
}

std::vector<bfloat16> golden_matmul(
    const std::vector<bfloat16>& a, const std::vector<bfloat16>& b, uint32_t m, uint32_t n, uint32_t k) {
    std::vector<bfloat16> output(static_cast<size_t>(m) * n, bfloat16(0.0f));
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (uint32_t kk = 0; kk < k; ++kk) {
                sum += static_cast<float>(a[static_cast<size_t>(row) * k + kk]) *
                       static_cast<float>(b[static_cast<size_t>(kk) * n + col]);
            }
            output[static_cast<size_t>(row) * n + col] = bfloat16(sum);
        }
    }
    return output;
}

TestData make_test_data(const Options& options) {
    if (is_matmul_op(options.op)) {
        const uint32_t m = options.matmul_m_tiles * tt::constants::TILE_HEIGHT;
        const uint32_t n = options.matmul_n_tiles * tt::constants::TILE_WIDTH;
        const uint32_t k = options.matmul_k_tiles * tt::constants::TILE_WIDTH;
        auto src0_row_major = make_matmul_input(m, k, 1);
        auto src1_row_major = make_matmul_input(k, n, 2);
        auto expected = golden_matmul(src0_row_major, src1_row_major, m, n, k);
        return {
            .src0 = tilize_nfaces(src0_row_major, m, k),
            .src1 = tilize_nfaces(src1_row_major, k, n),
            .expected = std::move(expected),
            .result_is_tilized = true,
            .matrix_m = m,
            .matrix_n = n};
    }

    auto src0 = make_binary_input(options.tiles, 1);
    auto src1 = make_binary_input(options.tiles, 2);
    std::vector<bfloat16> expected(src0.size(), bfloat16(0.0f));
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = bfloat16(static_cast<float>(src0[i]) + static_cast<float>(src1[i]));
    }
    return {.src0 = std::move(src0), .src1 = std::move(src1), .expected = std::move(expected)};
}

std::map<std::string, std::string> kernel_defines(
    const Options& options,
    Mode mode,
    const CoreWork& work,
    uint32_t iterations,
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
    return {
        {"BENCH_STATIC_PROTOCOL", is_static_mode(mode) ? "1" : "0"},
        {"BENCH_SERIAL_STATIC_PROTOCOL", uses_serialized_static(options, mode) ? "1" : "0"},
        {"BENCH_TRACE_STATIC_PROTOCOL",
         (is_static_mode(mode) && std::getenv("SPM_TRACE_STATIC") != nullptr) ? "1" : "0"},
        {"BENCH_PROFILE_CASE_LABELS", std::getenv("SPM_PROFILE_CASE_LABELS") != nullptr ? "1" : "0"},
        {"BENCH_USE_COMPILE_TIME_ARGS", uses_compile_time_args(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_SYNC", uses_stream_reg_sync(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_CBREGS", uses_stream_reg_cbregs(mode) ? "1" : "0"},
        {"BENCH_OP_TILE_ADD", options.op == Op::TileAdd ? "1" : "0"},
        {"BENCH_OP_ELTWISE_CHAIN", options.op == Op::EltwiseChain ? "1" : "0"},
        {"BENCH_OP_MATMUL", is_matmul_op(options.op) ? "1" : "0"},
        {"BENCH_OP_MATMUL_BLOCK", options.op == Op::MatmulBlock ? "1" : "0"},
        {"BENCH_ITERATIONS", std::to_string(iterations)},
        {"BENCH_CHAIN_DEPTH", std::to_string(options.op == Op::EltwiseChain ? options.chain_depth : 1)},
        {"BENCH_MATMUL_MT", std::to_string(work.local_m_tiles)},
        {"BENCH_MATMUL_GLOBAL_MT", std::to_string(options.matmul_m_tiles)},
        {"BENCH_MATMUL_KT", std::to_string(options.matmul_k_tiles)},
        {"BENCH_MATMUL_NT", std::to_string(work.local_n_tiles)},
        {"BENCH_MATMUL_GLOBAL_NT", std::to_string(options.matmul_n_tiles)},
        {"BENCH_MATMUL_BASE_MT", std::to_string(work.base_m_tile)},
        {"BENCH_MATMUL_BASE_NT", std::to_string(work.base_n_tile)},
        {"BENCH_CORE_GRID_X", std::to_string(options.core_grid_x)},
        {"BENCH_CORE_GRID_Y", std::to_string(options.core_grid_y)},
        {"BENCH_PAGE_SIZE", std::to_string(options.slot_bytes)},
        {"BENCH_NUM_PAGES", std::to_string(options.num_slots)},
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
        {"BENCH_PROTOCOL_START_VALUE",
         std::to_string(
             0x5a5a0000u ^ static_cast<uint32_t>(options.op) ^ output_tiles(options) ^
             (options.num_slots << 8) ^ protocol_start_sem_addr ^ (repeat * 0x00010001u))},
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
    const uint32_t cb_size = options.num_slots * options.slot_bytes;
    auto make_cb_config = [&](CBIndex cb_index, const std::shared_ptr<Buffer>& static_ring_buffer) {
        auto config =
            CircularBufferConfig(cb_size, {{cb_index, DataFormat::Float16_b}}).set_page_size(cb_index, options.slot_bytes);
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

std::vector<CoreWork> make_core_works(const Options& options) {
    if (options.op != Op::MatmulBlock) {
        return {CoreWork{
            .core = options.core,
            .base_m_tile = 0,
            .base_n_tile = 0,
            .local_m_tiles = options.matmul_m_tiles,
            .local_n_tiles = options.matmul_n_tiles}};
    }

    std::vector<CoreWork> works;
    const uint32_t local_m_tiles = options.matmul_m_tiles / options.core_grid_y;
    const uint32_t local_n_tiles = options.matmul_n_tiles / options.core_grid_x;
    for (uint32_t gy = 0; gy < options.core_grid_y; ++gy) {
        for (uint32_t gx = 0; gx < options.core_grid_x; ++gx) {
            works.push_back(CoreWork{
                .core = CoreCoord{options.core.x + gx, options.core.y + gy},
                .base_m_tile = gy * local_m_tiles,
                .base_n_tile = gx * local_n_tiles,
                .local_m_tiles = local_m_tiles,
                .local_n_tiles = local_n_tiles});
        }
    }
    return works;
}

void validate_core_works(const Options& options, const CoreCoord& grid) {
    for (const auto& work : make_core_works(options)) {
        if (work.core.x >= grid.x || work.core.y >= grid.y) {
            throw std::invalid_argument(fmt::format(
                "Core ({}, {}) for op {} is outside compute_with_storage_grid_size ({}, {})",
                work.core.x,
                work.core.y,
                op_name(options.op),
                grid.x,
                grid.y));
        }
    }
}

uint32_t local_output_tiles(const Options& options, const CoreWork& work) {
    if (is_matmul_op(options.op)) {
        return work.local_m_tiles * work.local_n_tiles;
    }
    return options.tiles;
}

bool validate_result(
    const Options& options, const TestData& data, const std::vector<bfloat16>& raw_result, float* max_abs_error) {
    std::vector<bfloat16> result = raw_result;
    if (data.result_is_tilized) {
        result = untilize_nfaces(result, data.matrix_m, data.matrix_n);
    }

    if (result.size() != data.expected.size()) {
        fmt::print(stderr, "Result size mismatch: expected {}, got {}\n", data.expected.size(), result.size());
        *max_abs_error = std::numeric_limits<float>::infinity();
        return false;
    }

    bool ok = true;
    float max_error = 0.0f;
    const float tolerance = is_matmul_op(options.op) ? 0.75f : std::max(0.5f, 0.25f * options.chain_depth);
    uint32_t printed = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        const float expected = static_cast<float>(data.expected[i]);
        const float actual = static_cast<float>(result[i]);
        const float error = std::abs(expected - actual);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            ok = false;
            if (printed < 5) {
                fmt::print(stderr, "Mismatch at element {}: expected {}, got {}, error {}\n", i, expected, actual, error);
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
    const TestData& data) {
    if (uses_stream_reg_start_gate(mode) && std::max(input_pairs(options), output_tiles(options)) > kStreamRegCounterMask) {
        throw std::invalid_argument("stream-register protocol modes support up to 24-bit protocol counters");
    }
    if (uses_stream_reg_sync(mode)) {
        throw std::invalid_argument(
            "static-streamreg is disabled for static_protocol_modeling: the old idle-stream scratch-register "
            "variant is not a valid compute-path comparison. Use static-streamreg-cbregs with per-CB "
            "tiles_received/tiles_acked registers instead");
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    auto src0_dram_buffer = create_dram_buffer(mesh_device, checked_bytes(data.src0.size(), "src0"));
    auto src1_dram_buffer = create_dram_buffer(mesh_device, checked_bytes(data.src1.size(), "src1"));
    auto dst_dram_buffer = create_dram_buffer(mesh_device, checked_bytes(data.expected.size(), "dst"));

    const uint32_t src0_dram_addr = static_cast<uint32_t>(src0_dram_buffer->address());
    const uint32_t src1_dram_addr = static_cast<uint32_t>(src1_dram_buffer->address());
    const uint32_t dst_dram_addr = static_cast<uint32_t>(dst_dram_buffer->address());

    const std::string reader_kernel_path =
        is_matmul_op(options.op) ? std::string(kMatmulReaderKernel) : std::string(kBinaryReaderKernel);
    const std::string compute_kernel_path =
        is_matmul_op(options.op) ? std::string(kMatmulComputeKernel) : std::string(kEltwiseComputeKernel);

    std::vector<std::shared_ptr<Buffer>> l1_buffer_keepalive;
    const auto works = make_core_works(options);
    for (const auto& work : works) {
        std::shared_ptr<Buffer> src0_ring_buffer;
        std::shared_ptr<Buffer> src1_ring_buffer;
        std::shared_ptr<Buffer> dst_ring_buffer;
        std::shared_ptr<Buffer> protocol_sem_buffer;

        if (is_static_mode(mode)) {
            const uint32_t ring_bytes = ring_size_bytes(options);
            src0_ring_buffer = create_core_local_l1_ring_buffer(mesh_device, work.core, ring_bytes);
            src1_ring_buffer = create_core_local_l1_ring_buffer(mesh_device, work.core, ring_bytes);
            dst_ring_buffer = create_core_local_l1_ring_buffer(mesh_device, work.core, ring_bytes);
            std::vector<uint8_t> zero_ring_buffer(ring_bytes, 0);
            detail::WriteToBuffer(src0_ring_buffer, zero_ring_buffer);
            detail::WriteToBuffer(src1_ring_buffer, zero_ring_buffer);
            detail::WriteToBuffer(dst_ring_buffer, zero_ring_buffer);
            l1_buffer_keepalive.push_back(src0_ring_buffer);
            l1_buffer_keepalive.push_back(src1_ring_buffer);
            l1_buffer_keepalive.push_back(dst_ring_buffer);

            if (!uses_stream_reg_start_gate(mode)) {
                protocol_sem_buffer = create_core_local_l1_buffer(
                    mesh_device, work.core, protocol_sem_size_bytes(options), kSemSlotBytes);
                std::vector<uint8_t> zero_sem_buffer(protocol_sem_buffer->size(), 0);
                detail::WriteToBuffer(protocol_sem_buffer, zero_sem_buffer);
                l1_buffer_keepalive.push_back(protocol_sem_buffer);
            }
        }

        create_circular_buffers(
            program, work.core, mode, options, src0_ring_buffer, src1_ring_buffer, dst_ring_buffer);

        const uint32_t src0_ring_addr = core_local_l1_address(src0_ring_buffer, work.core);
        const uint32_t src1_ring_addr = core_local_l1_address(src1_ring_buffer, work.core);
        const uint32_t dst_ring_addr = core_local_l1_address(dst_ring_buffer, work.core);
        const uint32_t protocol_sem_base_addr = core_local_l1_address(protocol_sem_buffer, work.core);
        const uint32_t input_ready_sem_addr = protocol_sem_base_addr;
        const uint32_t input_consumed_sem_addr = input_ready_sem_addr + options.num_slots * kSemSlotBytes;
        const uint32_t output_ready_sem_addr = input_consumed_sem_addr + options.num_slots * kSemSlotBytes;
        const uint32_t output_consumed_sem_addr = output_ready_sem_addr + options.num_slots * kSemSlotBytes;
        const uint32_t protocol_start_sem_addr = output_consumed_sem_addr + options.num_slots * kSemSlotBytes;
        const uint32_t iterations = local_output_tiles(options, work);

        auto defines = kernel_defines(
            options,
            mode,
            work,
            iterations,
            src0_dram_addr,
            src1_dram_addr,
            dst_dram_addr,
            src0_ring_addr,
            src1_ring_addr,
            dst_ring_addr,
            input_ready_sem_addr,
            input_consumed_sem_addr,
            output_ready_sem_addr,
            output_consumed_sem_addr,
            protocol_start_sem_addr,
            repeat);

        KernelHandle reader_kernel = CreateKernel(
            program,
            reader_kernel_path,
            work.core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .defines = defines});
        KernelHandle writer_kernel = CreateKernel(
            program,
            std::string(kWriterKernel),
            work.core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = defines});
        KernelHandle compute_kernel = CreateKernel(
            program,
            compute_kernel_path,
            work.core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .defines = defines});

        const uint32_t kt = options.matmul_k_tiles;
        if (uses_compile_time_args(mode)) {
            SetRuntimeArgs(program, reader_kernel, work.core, {});
            SetRuntimeArgs(program, writer_kernel, work.core, {});
            SetRuntimeArgs(program, compute_kernel, work.core, {});
        } else if (is_static_mode(mode)) {
            if (is_matmul_op(options.op)) {
                SetRuntimeArgs(
                    program,
                    reader_kernel,
                    work.core,
                    {src0_dram_addr,
                     src1_dram_addr,
                     work.local_m_tiles,
                     kt,
                     work.local_n_tiles,
                     src0_ring_addr,
                     src1_ring_addr,
                     options.slot_bytes,
                     options.num_slots,
                     input_ready_sem_addr,
                     input_consumed_sem_addr,
                     output_ready_sem_addr,
                     output_consumed_sem_addr,
                     protocol_start_sem_addr,
                     options.matmul_n_tiles,
                     work.base_m_tile,
                     work.base_n_tile});
                SetRuntimeArgs(
                    program,
                    compute_kernel,
                    work.core,
                    {work.local_m_tiles,
                     kt,
                     work.local_n_tiles,
                     src0_ring_addr,
                     src1_ring_addr,
                     dst_ring_addr,
                     options.slot_bytes,
                     options.num_slots,
                     input_ready_sem_addr,
                     input_consumed_sem_addr,
                     output_ready_sem_addr,
                     output_consumed_sem_addr,
                     protocol_start_sem_addr});
            } else {
                SetRuntimeArgs(
                    program,
                    reader_kernel,
                    work.core,
                    {options.tiles,
                     src0_dram_addr,
                     src1_dram_addr,
                     src0_ring_addr,
                     src1_ring_addr,
                     options.slot_bytes,
                     options.num_slots,
                     input_ready_sem_addr,
                     input_consumed_sem_addr,
                     output_ready_sem_addr,
                     output_consumed_sem_addr,
                     protocol_start_sem_addr});
                SetRuntimeArgs(
                    program,
                    compute_kernel,
                    work.core,
                    {options.tiles,
                     src0_ring_addr,
                     src1_ring_addr,
                     dst_ring_addr,
                     options.slot_bytes,
                     options.num_slots,
                     input_ready_sem_addr,
                     input_consumed_sem_addr,
                     output_ready_sem_addr,
                     output_consumed_sem_addr,
                     protocol_start_sem_addr});
            }

            SetRuntimeArgs(
                program,
                writer_kernel,
                work.core,
                {iterations,
                 dst_dram_addr,
                 dst_ring_addr,
                 options.slot_bytes,
                 options.num_slots,
                 output_ready_sem_addr,
                 output_consumed_sem_addr,
                 protocol_start_sem_addr,
                 work.local_n_tiles,
                 options.matmul_n_tiles,
                 work.base_m_tile,
                 work.base_n_tile});
        } else {
            if (is_matmul_op(options.op)) {
                SetRuntimeArgs(
                    program,
                    reader_kernel,
                    work.core,
                    {src0_dram_addr,
                     src1_dram_addr,
                     work.local_m_tiles,
                     kt,
                     work.local_n_tiles,
                     options.matmul_n_tiles,
                     work.base_m_tile,
                     work.base_n_tile});
                SetRuntimeArgs(program, compute_kernel, work.core, {work.local_m_tiles, kt, work.local_n_tiles});
                SetRuntimeArgs(
                    program,
                    writer_kernel,
                    work.core,
                    {iterations,
                     dst_dram_addr,
                     work.local_n_tiles,
                     options.matmul_n_tiles,
                     work.base_m_tile,
                     work.base_n_tile});
            } else {
                SetRuntimeArgs(program, reader_kernel, work.core, {options.tiles, src0_dram_addr, src1_dram_addr});
                SetRuntimeArgs(program, compute_kernel, work.core, {options.tiles});
                SetRuntimeArgs(program, writer_kernel, work.core, {iterations, dst_dram_addr});
            }
        }
    }

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, data.src0, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, data.src1, false);

    workload.add_program(device_range, std::move(program));

    const auto start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();

    std::vector<bfloat16> raw_result;
    distributed::EnqueueReadMeshBuffer(cq, raw_result, dst_dram_buffer, true);

    float max_abs_error = 0.0f;
    const bool ok = validate_result(options, data, raw_result, &max_abs_error);
    return {
        .op = options.op,
        .mode = mode,
        .repeat = repeat,
        .enqueue_finish_us =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
        .ok = ok,
        .max_abs_error = max_abs_error};
}

std::vector<Options> expand_sweep(const Options& base) {
    if (base.sweep.empty()) {
        return {base};
    }

    std::vector<Options> cases;
    auto add_unique_case = [&](const Options& next) {
        const auto same_case = [&](const Options& existing) {
            return existing.op == next.op && existing.tiles == next.tiles && existing.num_slots == next.num_slots &&
                   existing.chain_depth == next.chain_depth && existing.matmul_m_tiles == next.matmul_m_tiles &&
                   existing.matmul_n_tiles == next.matmul_n_tiles &&
                   existing.matmul_k_tiles == next.matmul_k_tiles && existing.core_grid_x == next.core_grid_x &&
                   existing.core_grid_y == next.core_grid_y;
        };
        if (std::none_of(cases.begin(), cases.end(), same_case)) {
            cases.push_back(next);
        }
    };

    if (base.sweep == "matmul-targeted") {
        auto add_single = [&](uint32_t mt, uint32_t nt, uint32_t kt, uint32_t slots) {
            Options next = base;
            next.sweep.clear();
            next.op = Op::MatmulSingle;
            next.core_grid_x = 1;
            next.core_grid_y = 1;
            next.matmul_m_tiles = mt;
            next.matmul_n_tiles = nt;
            next.matmul_k_tiles = kt;
            next.tiles = nt;
            next.num_slots = slots;
            add_unique_case(next);
        };
        auto add_block = [&](uint32_t grid_x, uint32_t grid_y, uint32_t nt_per_core, uint32_t kt, uint32_t slots) {
            Options next = base;
            next.sweep.clear();
            next.op = Op::MatmulBlock;
            next.core_grid_x = grid_x;
            next.core_grid_y = grid_y;
            next.matmul_m_tiles = grid_y;
            next.matmul_n_tiles = grid_x * nt_per_core;
            next.matmul_k_tiles = kt;
            next.tiles = next.matmul_n_tiles;
            next.num_slots = slots;
            add_unique_case(next);
        };

        for (uint32_t kt : {1u, 2u, 4u, 8u, 16u}) {
            add_single(1, 4, kt, 2);
        }
        for (uint32_t nt : {1u, 4u, 16u}) {
            add_single(1, nt, 4, 2);
        }
        for (uint32_t slots : {1u, 2u, 4u}) {
            add_single(1, 4, 4, slots);
            add_single(1, 16, 8, slots);
        }
        for (uint32_t mt : {2u, 4u}) {
            add_single(mt, 4, 4, 2);
        }
        for (auto [grid_x, grid_y] : {std::array<uint32_t, 2>{1, 2}, std::array<uint32_t, 2>{2, 2}}) {
            for (uint32_t nt_per_core : {1u, 4u}) {
                for (uint32_t kt : {2u, 8u}) {
                    add_block(grid_x, grid_y, nt_per_core, kt, 2);
                }
            }
        }
        for (uint32_t slots : {1u, 4u}) {
            add_block(2, 2, 4, 8, slots);
        }
        return cases;
    }

    if (base.sweep == "matmul") {
        for (uint32_t mt : {1u, 2u, 4u}) {
            for (uint32_t nt : {1u, 4u, 16u}) {
                for (uint32_t kt : {1u, 2u, 4u, 8u, 16u}) {
                    for (uint32_t slots : {1u, 2u, 4u}) {
                        Options next = base;
                        next.sweep.clear();
                        next.op = Op::MatmulSingle;
                        next.core_grid_x = 1;
                        next.core_grid_y = 1;
                        next.matmul_m_tiles = mt;
                        next.matmul_n_tiles = nt;
                        next.matmul_k_tiles = kt;
                        next.tiles = nt;
                        next.num_slots = slots;
                        cases.push_back(next);
                    }
                }
            }
        }

        for (auto [grid_x, grid_y] : {std::array<uint32_t, 2>{1, 2}, std::array<uint32_t, 2>{2, 2}}) {
            for (uint32_t nt_per_core : {1u, 4u}) {
                for (uint32_t kt : {2u, 8u}) {
                    for (uint32_t slots : {1u, 2u, 4u}) {
                        Options next = base;
                        next.sweep.clear();
                        next.op = Op::MatmulBlock;
                        next.core_grid_x = grid_x;
                        next.core_grid_y = grid_y;
                        next.matmul_m_tiles = grid_y;
                        next.matmul_n_tiles = grid_x * nt_per_core;
                        next.matmul_k_tiles = kt;
                        next.tiles = next.matmul_n_tiles;
                        next.num_slots = slots;
                        cases.push_back(next);
                    }
                }
            }
        }
        return cases;
    }

    for (Op op : {Op::TileAdd, Op::EltwiseChain, Op::MatmulSingle, Op::MatmulBlock}) {
        const std::vector<uint32_t> tile_counts = op == Op::MatmulSingle ? std::vector<uint32_t>{1, 4, 16}
                                                                         : op == Op::MatmulBlock
                                                                               ? std::vector<uint32_t>{4, 8, 16}
                                                                               : std::vector<uint32_t>{16, 256, 1024};
        for (uint32_t tiles : tile_counts) {
            for (uint32_t slots : {1u, 2u, 4u}) {
                Options next = base;
                next.sweep.clear();
                next.op = op;
                next.tiles = tiles;
                if (op == Op::MatmulSingle) {
                    next.matmul_m_tiles = 1;
                    next.matmul_n_tiles = tiles;
                } else if (op == Op::MatmulBlock) {
                    next.core_grid_x = 1;
                    next.core_grid_y = 2;
                    next.matmul_m_tiles = 2;
                    next.matmul_n_tiles = tiles;
                    next.matmul_k_tiles = 2;
                }
                next.num_slots = slots;
                cases.push_back(next);
            }
        }
    }
    return cases;
}

void print_result_header_once() {
    static bool printed = false;
    if (!printed) {
        fmt::print(
            "RESULT_HEADER,op,mode,repeat,tiles,num_slots,slot_bytes,chain_depth,"
            "matmul_m_tiles,matmul_n_tiles,matmul_k_tiles,input_tile_pairs,output_tiles,core_grid_x,core_grid_y,"
            "enqueue_finish_us,max_abs_error,ok\n");
        printed = true;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;

    try {
        const Options base_options = parse_options(argc, argv);
        const auto cases = expand_sweep(base_options);
        const std::vector<Mode> modes = modes_to_run(base_options.mode);
        if (std::any_of(modes.begin(), modes.end(), is_static_mode)) {
            setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);
        }

        if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
            fmt::print(
                "WARNING: set TT_METAL_DEVICE_PROFILER=1 to collect device cycle zones in "
                "generated/profiler/.logs/profile_log_device.csv\n");
        }

        std::shared_ptr<distributed::MeshDevice> mesh_device =
            distributed::MeshDevice::create_unit_mesh(base_options.device_id);

        const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
        for (const Options& options : cases) {
            validate_core_works(options, grid);
        }

        fmt::print(
            "static_protocol_modeling: cases={}, modes={}, repeats={}, core=({}, {})\n",
            cases.size(),
            modes.size(),
            base_options.repeats,
            base_options.core.x,
            base_options.core.y);
        print_result_header_once();

        for (const Options& options : cases) {
            const TestData data = make_test_data(options);
            fmt::print(
                "CASE op={} tiles={} num_slots={} slot_bytes={} chain_depth={} matmul_m_tiles={} "
                "matmul_n_tiles={} matmul_k_tiles={} core_grid=({}, {})\n",
                op_name(options.op),
                output_tiles(options),
                options.num_slots,
                options.slot_bytes,
                options.chain_depth,
                options.matmul_m_tiles,
                options.matmul_n_tiles,
                options.matmul_k_tiles,
                options.core_grid_x,
                options.core_grid_y);

            for (uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
                for (Mode mode : modes) {
                    RunResult result = run_one(mesh_device, options, mode, repeat, data);
                    pass &= result.ok;
                    fmt::print(
                        "RESULT,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                        op_name(result.op),
                        mode_name(result.mode),
                        result.repeat,
                        output_tiles(options),
                        options.num_slots,
                        options.slot_bytes,
                        options.chain_depth,
                        options.matmul_m_tiles,
                        options.matmul_n_tiles,
                        options.matmul_k_tiles,
                        input_pairs(options),
                        output_tiles(options),
                        options.core_grid_x,
                        options.core_grid_y,
                        result.enqueue_finish_us,
                        result.max_abs_error,
                        result.ok ? 1 : 0);
                }
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
