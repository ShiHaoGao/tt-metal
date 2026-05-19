// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "../../matmul/matmul_common/bmm_op.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace per_core_allocation = tt::tt_metal::experimental::per_core_allocation;

namespace {

constexpr uint32_t kTileHeight = tt::constants::TILE_HEIGHT;
constexpr uint32_t kTileWidth = tt::constants::TILE_WIDTH;
constexpr uint32_t kSemSlotBytes = 64;
constexpr uint32_t kIn0BlockW = 2;

constexpr std::string_view kReaderKernel =
    OVERRIDE_KERNEL_PREFIX "pipeline_warmup_experiments/matmul/kernels/dataflow/reader_bmm_tile_layout_protocol.cpp";
constexpr std::string_view kWriterKernel =
    OVERRIDE_KERNEL_PREFIX "pipeline_warmup_experiments/matmul/kernels/dataflow/writer_bmm_tile_layout_protocol.cpp";
constexpr std::string_view kComputeKernel =
    OVERRIDE_KERNEL_PREFIX "pipeline_warmup_experiments/matmul/kernels/compute/bmm_large_block_zm_protocol.cpp";

enum class Mode : uint32_t {
    ProfiledCb = 0,
    ProfiledCbDynamic = 1,
    StaticInputOnly = 2,
    StaticOutputOnly = 3,
    StaticInputOutput = 4,
};

struct Options {
    std::string mode = "all";
    std::string sweep;
    uint32_t M = 512;
    uint32_t N = 512;
    uint32_t K = 64;
    uint32_t B = 1;
    uint32_t num_pages = 2;
    uint32_t prologue_blocks = 1;
    uint32_t steady_blocks = 2;
    uint32_t repeats = 1;
    uint32_t device_id = 0;
    bool skip_check = false;
};

struct CoreWork {
    CoreCoord core;
    uint32_t output_idx_y = 0;
    uint32_t output_idx_x = 0;
};

struct MatmulShape {
    uint32_t Mt = 0;
    uint32_t Nt = 0;
    uint32_t Kt = 0;
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;
    uint32_t out_subblock_h = 0;
    uint32_t out_subblock_w = 0;
    uint32_t in0_block_num_tiles = 0;
    uint32_t in1_block_num_tiles = 0;
    uint32_t out_subblock_num_tiles = 0;
    uint32_t out_block_tiles = 0;
    uint32_t num_blocks = 0;
    uint32_t num_blocks_y = 0;
    uint32_t num_blocks_x = 0;
    uint32_t num_active_cores = 0;
};

struct TestData {
    std::vector<bfloat16> src0_tilized;
    std::vector<bfloat16> src1_tilized;
    std::vector<bfloat16> expected_row_major;
};

struct StaticCoreResources {
    CoreCoord core;
    std::shared_ptr<Buffer> src0_ring;
    std::shared_ptr<Buffer> src1_ring;
    std::shared_ptr<Buffer> out_ring;
    std::shared_ptr<Buffer> protocol_start_sem;
};

struct RunResult {
    Mode mode;
    uint32_t repeat = 0;
    uint64_t enqueue_finish_us = 0;
    bool ok = false;
    float pcc = 0.0f;
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
        case Mode::ProfiledCb: return "profiled-cb";
        case Mode::ProfiledCbDynamic: return "profiled-cb-dynamic";
        case Mode::StaticInputOnly: return "static-input-only";
        case Mode::StaticOutputOnly: return "static-output-only";
        case Mode::StaticInputOutput: return "static-input-output";
    }
    return "unknown";
}

std::optional<Mode> parse_mode(std::string_view mode) {
    if (mode == "profiled-cb" || mode == "cb") {
        return Mode::ProfiledCb;
    }
    if (mode == "profiled-cb-dynamic" || mode == "cb-dynamic" || mode == "dynamic") {
        return Mode::ProfiledCbDynamic;
    }
    if (mode == "static-input-only") {
        return Mode::StaticInputOnly;
    }
    if (mode == "static-output-only") {
        return Mode::StaticOutputOnly;
    }
    if (mode == "static-input-output" || mode == "static") {
        return Mode::StaticInputOutput;
    }
    return std::nullopt;
}

bool is_static_mode(Mode mode) {
    return mode != Mode::ProfiledCb && mode != Mode::ProfiledCbDynamic;
}

bool uses_static_input(Mode mode) {
    return mode == Mode::StaticInputOnly || mode == Mode::StaticInputOutput;
}

bool uses_static_output(Mode mode) {
    return mode == Mode::StaticOutputOnly || mode == Mode::StaticInputOutput;
}

std::vector<Mode> modes_to_run(const std::string& mode) {
    if (mode == "all") {
        return {
            Mode::ProfiledCb,
            Mode::ProfiledCbDynamic,
            Mode::StaticInputOnly,
            Mode::StaticOutputOnly,
            Mode::StaticInputOutput};
    }
    auto parsed = parse_mode(mode);
    if (!parsed.has_value()) {
        throw std::invalid_argument(
            "Unknown --mode. Valid values are all, profiled-cb, profiled-cb-dynamic, cb, cb-dynamic, "
            "static-input-only, static-output-only, "
            "static-input-output, static");
    }
    return {*parsed};
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--mode=all|profiled-cb|static-input-only|static-output-only|static-input-output] "
        "[--M=N] [--N=N] [--K=N] "
        "[--B=N] [--num-pages=N] [--prologue-blocks=N] [--steady-blocks=N] "
        "[--repeats=N] [--device-id=N] [--sweep=targeted] [--skip-check]\n",
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
        } else if (starts_with(arg, "--M=")) {
            options.M = parse_u32(arg.substr(std::string_view("--M=").size()), "M");
        } else if (starts_with(arg, "--N=")) {
            options.N = parse_u32(arg.substr(std::string_view("--N=").size()), "N");
        } else if (starts_with(arg, "--K=")) {
            options.K = parse_u32(arg.substr(std::string_view("--K=").size()), "K");
        } else if (starts_with(arg, "--B=")) {
            options.B = parse_u32(arg.substr(std::string_view("--B=").size()), "B");
        } else if (starts_with(arg, "--num-pages=")) {
            options.num_pages = parse_u32(arg.substr(std::string_view("--num-pages=").size()), "num-pages");
        } else if (starts_with(arg, "--prologue-blocks=")) {
            options.prologue_blocks =
                parse_u32(arg.substr(std::string_view("--prologue-blocks=").size()), "prologue-blocks");
        } else if (starts_with(arg, "--steady-blocks=")) {
            options.steady_blocks =
                parse_u32(arg.substr(std::string_view("--steady-blocks=").size()), "steady-blocks");
        } else if (starts_with(arg, "--repeats=")) {
            options.repeats = parse_u32(arg.substr(std::string_view("--repeats=").size()), "repeats");
        } else if (starts_with(arg, "--device-id=")) {
            options.device_id = parse_u32(arg.substr(std::string_view("--device-id=").size()), "device-id");
        } else if (starts_with(arg, "--sweep=")) {
            options.sweep = std::string(arg.substr(std::string_view("--sweep=").size()));
        } else if (arg == "--skip-check") {
            options.skip_check = true;
        } else {
            throw std::invalid_argument(fmt::format("Unknown argument '{}'. Use --help to list options.", arg));
        }
    }

    if (options.M == 0 || options.N == 0 || options.K == 0) {
        throw std::invalid_argument("M, N, and K must be greater than zero");
    }
    if (options.B != 1) {
        throw std::invalid_argument("This profiler currently supports B=1 only");
    }
    if (options.num_pages == 0) {
        throw std::invalid_argument("--num-pages must be greater than zero");
    }
    if (options.prologue_blocks == 0 || options.steady_blocks == 0) {
        throw std::invalid_argument("--prologue-blocks and --steady-blocks must be greater than zero");
    }
    if (options.repeats == 0) {
        throw std::invalid_argument("--repeats must be greater than zero");
    }
    return options;
}

std::vector<Options> expand_cases(const Options& options) {
    if (options.sweep.empty()) {
        return {options};
    }
    if (options.sweep != "targeted") {
        throw std::invalid_argument("Unknown --sweep. Valid value is targeted");
    }

    std::vector<Options> cases;
    for (uint32_t dim : {512u, 1024u}) {
        for (uint32_t k : {64u, 128u, 256u}) {
            Options next = options;
            next.sweep.clear();
            next.M = dim;
            next.N = dim;
            next.K = k;
            next.num_pages = 2;
            cases.push_back(next);
        }
    }
    return cases;
}

uint32_t checked_size_bytes(size_t elements, std::string_view label) {
    const uint64_t bytes = static_cast<uint64_t>(elements) * sizeof(bfloat16);
    if (bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(fmt::format("{} exceeds uint32_t addressable size", label));
    }
    return static_cast<uint32_t>(bytes);
}

uint32_t checked_mul_u32(uint32_t a, uint32_t b, std::string_view label) {
    const uint64_t value = static_cast<uint64_t>(a) * b;
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(fmt::format("{} exceeds uint32_t range", label));
    }
    return static_cast<uint32_t>(value);
}

MatmulShape compute_shape(const Options& options, CoreCoord compute_grid) {
    if (options.M % kTileHeight != 0 || options.N % kTileWidth != 0 || options.K % kTileWidth != 0) {
        throw std::invalid_argument("M, N, and K must be multiples of 32");
    }

    MatmulShape shape;
    shape.Mt = options.M / kTileHeight;
    shape.Nt = options.N / kTileWidth;
    shape.Kt = options.K / kTileWidth;
    if (shape.Kt % kIn0BlockW != 0) {
        throw std::invalid_argument("Kt must be divisible by in0_block_w=2, so K must be a multiple of 64");
    }

    auto params = bmm_op_utils::get_large_matmul_params(
        shape.Mt,
        shape.Nt,
        static_cast<uint32_t>(compute_grid.y),
        static_cast<uint32_t>(compute_grid.x),
        kIn0BlockW);
    shape.per_core_M = std::get<0>(params);
    shape.per_core_N = std::get<1>(params);
    shape.out_subblock_h = std::get<2>(params);
    shape.out_subblock_w = std::get<3>(params);
    if (shape.per_core_M == 0 || shape.per_core_N == 0 || shape.out_subblock_h == 0 || shape.out_subblock_w == 0) {
        throw std::invalid_argument("get_large_matmul_params could not find a valid core blocking for this shape");
    }
    if (shape.Mt % shape.per_core_M != 0 || shape.Nt % shape.per_core_N != 0) {
        throw std::invalid_argument("Invalid per-core block returned by get_large_matmul_params");
    }

    shape.num_blocks_y = shape.Mt / shape.per_core_M;
    shape.num_blocks_x = shape.Nt / shape.per_core_N;
    shape.num_active_cores = shape.num_blocks_y * shape.num_blocks_x;
    if (shape.num_active_cores > compute_grid.x * compute_grid.y) {
        throw std::invalid_argument("Shape requires more cores than compute_with_storage_grid_size");
    }

    shape.in0_block_num_tiles = shape.per_core_M * kIn0BlockW;
    shape.in1_block_num_tiles = shape.per_core_N * kIn0BlockW;
    shape.out_subblock_num_tiles = shape.out_subblock_h * shape.out_subblock_w;
    shape.out_block_tiles = shape.per_core_M * shape.per_core_N;
    shape.num_blocks = shape.Kt / kIn0BlockW;
    return shape;
}

uint32_t output_subblocks(const MatmulShape& shape) {
    if (shape.out_subblock_num_tiles == 0 || shape.out_block_tiles % shape.out_subblock_num_tiles != 0) {
        throw std::invalid_argument("Invalid output subblock geometry");
    }
    return shape.out_block_tiles / shape.out_subblock_num_tiles;
}

std::vector<CoreWork> make_core_works(const MatmulShape& shape, CoreCoord compute_grid) {
    std::vector<CoreWork> works;
    works.reserve(shape.num_active_cores);
    uint32_t core_index = 0;
    for (uint32_t output_idx_y = 0; output_idx_y < shape.num_blocks_y; ++output_idx_y) {
        for (uint32_t output_idx_x = 0; output_idx_x < shape.num_blocks_x; ++output_idx_x) {
            const uint32_t core_idx_x = core_index % static_cast<uint32_t>(compute_grid.x);
            const uint32_t core_idx_y = core_index / static_cast<uint32_t>(compute_grid.x);
            works.push_back(CoreWork{
                .core = CoreCoord{core_idx_x, core_idx_y},
                .output_idx_y = output_idx_y,
                .output_idx_x = output_idx_x});
            ++core_index;
        }
    }
    return works;
}

std::vector<bfloat16> make_matrix(uint32_t rows, uint32_t cols, uint32_t salt) {
    std::vector<bfloat16> data(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t pattern = (r * 17u + c * 29u + salt * 31u) % 13u;
            data[static_cast<size_t>(r) * cols + c] = bfloat16(static_cast<float>(pattern + 1) * 0.03125f);
        }
    }
    return data;
}

std::vector<bfloat16> golden_matmul(
    const std::vector<bfloat16>& a, const std::vector<bfloat16>& b, uint32_t M, uint32_t N, uint32_t K) {
    std::vector<bfloat16> output(static_cast<size_t>(M) * N, bfloat16(0.0f));
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                acc += static_cast<float>(a[static_cast<size_t>(i) * K + k]) *
                       static_cast<float>(b[static_cast<size_t>(k) * N + j]);
            }
            output[static_cast<size_t>(i) * N + j] = bfloat16(acc);
        }
    }
    return output;
}

TestData make_test_data(const Options& options) {
    auto src0_row_major = make_matrix(options.M, options.K, 1);
    auto src1_row_major = make_matrix(options.K, options.N, 2);
    auto expected = golden_matmul(src0_row_major, src1_row_major, options.M, options.N, options.K);
    return {
        .src0_tilized = tilize_nfaces(src0_row_major, options.M, options.K),
        .src1_tilized = tilize_nfaces(src1_row_major, options.K, options.N),
        .expected_row_major = std::move(expected)};
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
        throw std::invalid_argument("L1 buffer size must be a whole number of pages");
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
        false /* bottom_up */);
}

uint32_t core_local_l1_address(const std::shared_ptr<Buffer>& buffer, CoreCoord core) {
    if (!buffer) {
        return 0;
    }
    return static_cast<uint32_t>(per_core_allocation::get_per_core_address(*buffer, core));
}

uint32_t protocol_start_value() {
    return 1;
}

std::map<std::string, std::string> kernel_defines(const Options& options, Mode mode) {
    return {
        {"BENCH_STATIC_PROTOCOL", is_static_mode(mode) ? "1" : "0"},
        {"BENCH_STATIC_INPUT_PROTOCOL", uses_static_input(mode) ? "1" : "0"},
        {"BENCH_STATIC_OUTPUT_PROTOCOL", uses_static_output(mode) ? "1" : "0"},
        {"BENCH_DYNAMIC_BLOCKS", (mode == Mode::ProfiledCbDynamic) ? "1" : "0"},
        {"BENCH_PROLOGUE_BLOCKS", std::to_string(options.prologue_blocks)},
        {"BENCH_STEADY_BLOCKS", std::to_string(options.steady_blocks)},
        {"BENCH_PROTOCOL_START_VALUE", std::to_string(protocol_start_value())},
    };
}

void zero_buffer(const std::shared_ptr<Buffer>& buffer) {
    std::vector<uint8_t> zeros(buffer->size(), 0);
    detail::WriteToBuffer(buffer, zeros);
}

std::vector<StaticCoreResources> create_static_resources(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const MatmulShape& shape,
    const std::vector<CoreWork>& works,
    Mode mode,
    uint32_t single_tile_size,
    uint32_t num_pages) {
    const uint32_t src0_slot_bytes = checked_mul_u32(shape.in0_block_num_tiles, single_tile_size, "src0 slot");
    const uint32_t src1_slot_bytes = checked_mul_u32(shape.in1_block_num_tiles, single_tile_size, "src1 slot");
    const uint32_t out_ring_bytes = checked_mul_u32(shape.out_block_tiles, single_tile_size, "out ring");

    std::vector<StaticCoreResources> resources;
    resources.reserve(works.size());
    for (const auto& work : works) {
        StaticCoreResources r;
        r.core = work.core;
        if (uses_static_input(mode)) {
            r.src0_ring = create_core_local_l1_buffer(
                mesh_device, work.core, checked_mul_u32(num_pages, src0_slot_bytes, "src0 ring"), single_tile_size);
            r.src1_ring = create_core_local_l1_buffer(
                mesh_device, work.core, checked_mul_u32(num_pages, src1_slot_bytes, "src1 ring"), single_tile_size);
            zero_buffer(r.src0_ring);
            zero_buffer(r.src1_ring);
            r.protocol_start_sem = create_core_local_l1_buffer(mesh_device, work.core, kSemSlotBytes, kSemSlotBytes);
            zero_buffer(r.protocol_start_sem);
        }
        if (uses_static_output(mode)) {
            r.out_ring = create_core_local_l1_buffer(mesh_device, work.core, out_ring_bytes, single_tile_size);
            zero_buffer(r.out_ring);
        }
        resources.push_back(std::move(r));
    }
    return resources;
}

const StaticCoreResources* find_static_resources(
    const std::vector<StaticCoreResources>& resources, const CoreCoord& core) {
    for (const auto& resource : resources) {
        if (resource.core.x == core.x && resource.core.y == core.y) {
            return &resource;
        }
    }
    return nullptr;
}

void create_circular_buffers(
    Program& program,
    const MatmulShape& shape,
    const std::vector<CoreWork>& works,
    const std::vector<StaticCoreResources>& static_resources,
    Mode mode,
    uint32_t single_tile_size,
    uint32_t num_pages) {
    const bool static_input = uses_static_input(mode);
    const bool static_output = uses_static_output(mode);

    for (const auto& work : works) {
        const StaticCoreResources* r = is_static_mode(mode) ? find_static_resources(static_resources, work.core) : nullptr;
        if (is_static_mode(mode) && r == nullptr) {
            throw std::runtime_error("Missing static L1 resources for core");
        }

        const uint32_t dynamic_depth = std::max(2u, num_pages);
        const uint32_t in0_cb_tiles = static_input ? shape.in0_block_num_tiles * num_pages
                                                   : shape.in0_block_num_tiles * dynamic_depth;
        const uint32_t in1_cb_tiles = static_input ? shape.in1_block_num_tiles * num_pages
                                                   : shape.in1_block_num_tiles * dynamic_depth;
        const uint32_t out_cb_tiles = shape.out_block_tiles;

        auto in0_config = CircularBufferConfig(
                              checked_mul_u32(in0_cb_tiles, single_tile_size, "in0 CB"),
                              {{CBIndex::c_0, DataFormat::Float16_b}})
                              .set_page_size(CBIndex::c_0, single_tile_size);
        auto in1_config = CircularBufferConfig(
                              checked_mul_u32(in1_cb_tiles, single_tile_size, "in1 CB"),
                              {{CBIndex::c_1, DataFormat::Float16_b}})
                              .set_page_size(CBIndex::c_1, single_tile_size);

        if (static_input) {
            in0_config.set_globally_allocated_address(*r->src0_ring);
            in1_config.set_globally_allocated_address(*r->src1_ring);
        }

        CreateCircularBuffer(program, work.core, in0_config);
        CreateCircularBuffer(program, work.core, in1_config);

        if (static_output) {
            std::map<uint8_t, DataFormat> output_cb_data_format_spec{
                {CBIndex::c_16, DataFormat::Float16_b}, {CBIndex::c_24, DataFormat::Float16_b}};
            auto out_config = CircularBufferConfig(
                                  checked_mul_u32(out_cb_tiles, single_tile_size, "out CB"),
                                  output_cb_data_format_spec)
                                  .set_page_size(CBIndex::c_16, single_tile_size)
                                  .set_page_size(CBIndex::c_24, single_tile_size);
            out_config.set_globally_allocated_address(*r->out_ring);
            CreateCircularBuffer(program, work.core, out_config);
        } else {
            std::map<uint8_t, DataFormat> output_cb_data_format_spec{
                {CBIndex::c_16, DataFormat::Float16_b}, {CBIndex::c_24, DataFormat::Float16_b}};
            auto out_config = CircularBufferConfig(
                                  checked_mul_u32(out_cb_tiles, single_tile_size, "out CB"),
                                  output_cb_data_format_spec)
                                  .set_page_size(CBIndex::c_16, single_tile_size)
                                  .set_page_size(CBIndex::c_24, single_tile_size);
            CreateCircularBuffer(program, work.core, out_config);
        }
    }
}

bool validate_result(
    const Options& options,
    const TestData& data,
    const std::vector<bfloat16>& raw_result,
    float* pcc,
    float* max_abs_error) {
    if (options.skip_check) {
        *pcc = 1.0f;
        *max_abs_error = 0.0f;
        return true;
    }

    auto result = untilize_nfaces(raw_result, options.M, options.N);
    if (result.size() != data.expected_row_major.size()) {
        fmt::print(stderr, "Result size mismatch: expected {}, got {}\n", data.expected_row_major.size(), result.size());
        *pcc = 0.0f;
        *max_abs_error = std::numeric_limits<float>::infinity();
        return false;
    }

    float max_error = 0.0f;
    for (size_t i = 0; i < result.size(); ++i) {
        max_error = std::max(
            max_error,
            std::abs(static_cast<float>(data.expected_row_major[i]) - static_cast<float>(result[i])));
    }
    *max_abs_error = max_error;
    *pcc = check_bfloat16_vector_pcc(data.expected_row_major, result);
    return *pcc > 0.99f;
}

RunResult run_one(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Options& options,
    Mode mode,
    uint32_t repeat,
    const TestData& data) {
    const tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const auto compute_grid = mesh_device->compute_with_storage_grid_size();
    const MatmulShape shape = compute_shape(options, compute_grid);
    const auto works = make_core_works(shape, compute_grid);
    const CoreRangeSet all_cores(tt::tt_metal::num_cores_to_corerangeset(
        shape.num_active_cores, compute_grid, true));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    auto src0_dram_buffer =
        create_dram_buffer(mesh_device, checked_size_bytes(data.src0_tilized.size(), "src0"), single_tile_size);
    auto src1_dram_buffer =
        create_dram_buffer(mesh_device, checked_size_bytes(data.src1_tilized.size(), "src1"), single_tile_size);
    auto dst_dram_buffer =
        create_dram_buffer(mesh_device, checked_size_bytes(data.expected_row_major.size(), "dst"), single_tile_size);

    const uint32_t src0_dram_addr = static_cast<uint32_t>(src0_dram_buffer->address());
    const uint32_t src1_dram_addr = static_cast<uint32_t>(src1_dram_buffer->address());
    const uint32_t dst_dram_addr = static_cast<uint32_t>(dst_dram_buffer->address());

    std::vector<StaticCoreResources> static_resources;
    if (is_static_mode(mode)) {
        static_resources = create_static_resources(mesh_device, shape, works, mode, single_tile_size, options.num_pages);
    }

    const uint32_t cb_depth_blocks =
        mode == Mode::ProfiledCbDynamic ? std::max(options.num_pages, options.steady_blocks) : options.num_pages;
    create_circular_buffers(program, shape, works, static_resources, mode, single_tile_size, cb_depth_blocks);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    const uint32_t in0_num_subblocks = shape.per_core_M / shape.out_subblock_h;
    const uint32_t in0_subblock_num_tiles = shape.out_subblock_h * kIn0BlockW;
    const uint32_t in1_num_subblocks = shape.per_core_N / shape.out_subblock_w;
    const uint32_t in1_per_core_w = shape.out_subblock_w * in1_num_subblocks;

    std::vector<uint32_t> compute_kernel_args = {
        kIn0BlockW,
        in0_num_subblocks,
        shape.in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        shape.in1_block_num_tiles,
        in1_per_core_w,
        shape.num_blocks,
        shape.out_subblock_h,
        shape.out_subblock_w,
        shape.out_subblock_num_tiles,
        options.B};

    auto defines = kernel_defines(options, mode);
    auto reader_id = CreateKernel(
        program,
        std::string(kReaderKernel),
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = defines});
    auto writer_id = CreateKernel(
        program,
        std::string(kWriterKernel),
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = defines});
    auto compute_id = CreateKernel(
        program,
        std::string(kComputeKernel),
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_kernel_args, .defines = defines});

    const uint32_t src0_slot_bytes = checked_mul_u32(shape.in0_block_num_tiles, single_tile_size, "src0 slot");
    const uint32_t src1_slot_bytes = checked_mul_u32(shape.in1_block_num_tiles, single_tile_size, "src1 slot");
    const uint32_t out_slot_bytes = checked_mul_u32(shape.out_subblock_num_tiles, single_tile_size, "out slot");
    const uint32_t out_num_pages = output_subblocks(shape);

    for (const auto& work : works) {
        std::vector<uint32_t> reader_args = {
            src0_dram_addr,
            shape.Kt * shape.per_core_M * work.output_idx_y,
            1,
            shape.Kt,
            kIn0BlockW,
            kIn0BlockW,
            shape.per_core_M,
            shape.in0_block_num_tiles,
            src1_dram_addr,
            shape.per_core_N * work.output_idx_x,
            1,
            shape.Nt,
            kIn0BlockW * shape.Nt,
            shape.per_core_N,
            kIn0BlockW,
            shape.in1_block_num_tiles,
            shape.num_blocks,
            shape.Mt * shape.Kt,
            shape.Kt * shape.Nt,
            options.B,
            0};

        std::vector<uint32_t> writer_args = {
            dst_dram_addr,
            work.output_idx_x * shape.per_core_N + work.output_idx_y * shape.per_core_M * shape.Nt,
            1,
            shape.Nt,
            shape.out_subblock_w,
            shape.out_subblock_h * shape.Nt,
            shape.out_subblock_w,
            shape.out_subblock_h,
            shape.out_subblock_num_tiles,
            shape.per_core_N / shape.out_subblock_w,
            shape.per_core_M / shape.out_subblock_h,
            shape.Mt * shape.Nt,
            options.B};

        std::vector<uint32_t> compute_args;
        if (is_static_mode(mode)) {
            const StaticCoreResources* r = find_static_resources(static_resources, work.core);
            if (r == nullptr) {
                throw std::runtime_error("Missing static L1 resources for runtime args");
            }
            const uint32_t src0_ring_addr = core_local_l1_address(r->src0_ring, work.core);
            const uint32_t src1_ring_addr = core_local_l1_address(r->src1_ring, work.core);
            const uint32_t out_ring_addr = core_local_l1_address(r->out_ring, work.core);
            const uint32_t protocol_start_sem_addr = core_local_l1_address(r->protocol_start_sem, work.core);

            if (uses_static_input(mode)) {
                reader_args.push_back(src0_ring_addr);
                reader_args.push_back(src1_ring_addr);
                reader_args.push_back(src0_slot_bytes);
                reader_args.push_back(src1_slot_bytes);
                reader_args.push_back(options.num_pages);
                reader_args.push_back(protocol_start_sem_addr);
            }

            if (uses_static_output(mode)) {
                writer_args.push_back(out_ring_addr);
                writer_args.push_back(out_slot_bytes);
                writer_args.push_back(out_num_pages);
                writer_args.push_back(protocol_start_sem_addr);
            }

            compute_args = {
                src0_ring_addr,
                src1_ring_addr,
                out_ring_addr,
                src0_slot_bytes,
                src1_slot_bytes,
                out_slot_bytes,
                options.num_pages,
                out_num_pages,
                protocol_start_sem_addr};
        }

        SetRuntimeArgs(program, reader_id, work.core, reader_args);
        SetRuntimeArgs(program, writer_id, work.core, writer_args);
        SetRuntimeArgs(program, compute_id, work.core, compute_args);
    }

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, data.src0_tilized, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, data.src1_tilized, false);
    workload.add_program(device_range, std::move(program));

    const auto start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();

    std::vector<bfloat16> raw_result;
    distributed::EnqueueReadMeshBuffer(cq, raw_result, dst_dram_buffer, true);

    float pcc = 0.0f;
    float max_abs_error = 0.0f;
    const bool ok = validate_result(options, data, raw_result, &pcc, &max_abs_error);
    return {
        .mode = mode,
        .repeat = repeat,
        .enqueue_finish_us =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
        .ok = ok,
        .pcc = pcc,
        .max_abs_error = max_abs_error};
}

void print_result_header() {
    fmt::print(
        "RESULT_HEADER,mode,repeat,M,N,K,B,num_pages,out_num_pages,Mt,Nt,Kt,per_core_M,per_core_N,out_subblock_h,"
        "out_subblock_w,num_blocks,num_active_cores,enqueue_finish_us,ok,pcc,max_abs_error\n");
}

void print_result(const Options& options, const MatmulShape& shape, const RunResult& result) {
    fmt::print(
        "RESULT,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6f},{:.6f}\n",
        mode_name(result.mode),
        result.repeat,
        options.M,
        options.N,
        options.K,
        options.B,
        options.num_pages,
        output_subblocks(shape),
        shape.Mt,
        shape.Nt,
        shape.Kt,
        shape.per_core_M,
        shape.per_core_N,
        shape.out_subblock_h,
        shape.out_subblock_w,
        shape.num_blocks,
        shape.num_active_cores,
        result.enqueue_finish_us,
        result.ok ? 1 : 0,
        result.pcc,
        result.max_abs_error);
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        Options base_options = parse_options(argc, argv);
        const std::vector<Mode> modes = modes_to_run(base_options.mode);
        if (std::any_of(modes.begin(), modes.end(), is_static_mode)) {
            setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);
        }
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(base_options.device_id);
        const auto compute_grid = mesh_device->compute_with_storage_grid_size();

        print_result_header();
        for (const auto& options : expand_cases(base_options)) {
            const MatmulShape shape = compute_shape(options, compute_grid);
            fmt::print(
                "real_matmul_protocol: M={} N={} K={} B={} mode={} repeats={} pages={} grid={}x{} "
                "per_core={}x{} subblock={}x{} num_blocks={} active_cores={}\n",
                options.M,
                options.N,
                options.K,
                options.B,
                options.mode,
                options.repeats,
                options.num_pages,
                compute_grid.x,
                compute_grid.y,
                shape.per_core_M,
                shape.per_core_N,
                shape.out_subblock_h,
                shape.out_subblock_w,
                shape.num_blocks,
                shape.num_active_cores);

            TestData data = make_test_data(options);
            for (Mode mode : modes_to_run(options.mode)) {
                for (uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
                    RunResult result = run_one(mesh_device, options, mode, repeat, data);
                    print_result(options, shape, result);
                    pass &= result.ok;
                }
            }
        }

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "real_matmul_protocol failed: {}\n", e.what());
        throw;
    }

    if (!pass) {
        TT_THROW("real_matmul_protocol validation failed");
    }
    return 0;
}
