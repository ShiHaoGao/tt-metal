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
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace per_core_allocation = tt::tt_metal::experimental::per_core_allocation;

namespace {

constexpr CoreCoord kDefaultCore = {0, 0};
constexpr uint32_t kTileHeight = tt::constants::TILE_HEIGHT;
constexpr uint32_t kTileWidth = tt::constants::TILE_WIDTH;
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
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_paged_update_cache_protocol/kernels/dataflow/"
    "reader_update_cache_interleaved_start_id_protocol.cpp";
constexpr std::string_view kWriterKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_paged_update_cache_protocol/kernels/dataflow/"
    "writer_update_cache_interleaved_start_id_protocol.cpp";
constexpr std::string_view kComputeKernel =
    "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/ttnn_kernel_forks/ttnn_paged_update_cache_protocol/kernels/compute/"
    "update_cache_protocol.cpp";

enum class Mode : uint32_t {
    Cb = 0,
    StaticRuntime = 1,
    StaticStreamRegCbRegs = 2,
    StaticStreamRegCbRegsCompileTime = 3,
};

struct Options {
    std::string mode = "all";
    uint32_t users = 32;
    uint32_t kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t block_size = 64;
    uint32_t max_seq_len = 2048;
    uint32_t cache_idx = 127;
    uint32_t per_user_stride = 17;
    uint32_t num_pages = 2;
    uint32_t repeats = 1;
    uint32_t device_id = 0;
    CoreCoord core = kDefaultCore;
    uint32_t core_grid_x = 0;
    uint32_t core_grid_y = 0;
    bool skip_check = false;
};

struct ShapeInfo {
    uint32_t Wt = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t num_blocks = 0;
    uint32_t cache_tiles = 0;
    uint32_t input_tiles_per_user = 0;
    uint32_t ring_tiles = 0;
    uint32_t index_stick_size = 0;
    uint32_t index_cb_page_size = 0;
    uint32_t page_table_stick_size = 0;
};

struct CoreResources {
    CoreCoord core;
    std::shared_ptr<Buffer> input_tiles_l1;
    std::shared_ptr<Buffer> cache_ring_l1;
    std::shared_ptr<Buffer> intermed_shared_l1;
    std::shared_ptr<Buffer> input_untilized_l1;
    std::shared_ptr<Buffer> output_ring_l1;
    std::shared_ptr<Buffer> protocol_start_l1;
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
            "static-streamreg-cbregs-compiletime, static");
    }
    return {*parsed};
}

bool is_static_mode(Mode mode) {
    return mode == Mode::StaticRuntime || mode == Mode::StaticStreamRegCbRegs ||
           mode == Mode::StaticStreamRegCbRegsCompileTime;
}

bool uses_stream_reg_cbregs(Mode mode) {
    return mode == Mode::StaticStreamRegCbRegs || mode == Mode::StaticStreamRegCbRegsCompileTime;
}

bool uses_compile_time_protocol_args(Mode mode) {
    return mode == Mode::StaticStreamRegCbRegsCompileTime;
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--mode=all|cb|static-runtime|static-streamreg-cbregs|"
        "static-streamreg-cbregs-compiletime] "
        "[--users=N] [--kv-heads=N] [--head-dim=N] [--block-size=N] [--max-seq-len=N] "
        "[--cache-idx=N] [--per-user-stride=N] [--num-pages=N] [--repeats=N] "
        "[--device-id=N] [--core-x=N] [--core-y=N] [--core-grid-x=N] [--core-grid-y=N] [--skip-check]\n",
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
        } else if (starts_with(arg, "--users=")) {
            options.users = parse_u32(arg.substr(std::string_view("--users=").size()), "users");
        } else if (starts_with(arg, "--kv-heads=")) {
            options.kv_heads = parse_u32(arg.substr(std::string_view("--kv-heads=").size()), "kv-heads");
        } else if (starts_with(arg, "--head-dim=")) {
            options.head_dim = parse_u32(arg.substr(std::string_view("--head-dim=").size()), "head-dim");
        } else if (starts_with(arg, "--block-size=")) {
            options.block_size = parse_u32(arg.substr(std::string_view("--block-size=").size()), "block-size");
        } else if (starts_with(arg, "--max-seq-len=")) {
            options.max_seq_len = parse_u32(arg.substr(std::string_view("--max-seq-len=").size()), "max-seq-len");
        } else if (starts_with(arg, "--cache-idx=")) {
            options.cache_idx = parse_u32(arg.substr(std::string_view("--cache-idx=").size()), "cache-idx");
        } else if (starts_with(arg, "--per-user-stride=")) {
            options.per_user_stride =
                parse_u32(arg.substr(std::string_view("--per-user-stride=").size()), "per-user-stride");
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
        } else if (arg == "--skip-check") {
            options.skip_check = true;
        } else {
            throw std::invalid_argument(fmt::format("Unknown argument '{}'. Use --help to list options.", arg));
        }
    }

    if (options.users == 0 || options.kv_heads == 0 || options.head_dim == 0 || options.block_size == 0 ||
        options.max_seq_len == 0 || options.num_pages == 0 || options.repeats == 0) {
        throw std::invalid_argument("Shape, --num-pages, and --repeats values must be greater than zero");
    }
    if (options.kv_heads > 32) {
        throw std::invalid_argument("This BF16 decode fork expects kv_heads <= 32, matching the TTNN padded input row");
    }
    if (options.head_dim % kTileWidth != 0 || options.block_size % kTileHeight != 0 ||
        options.max_seq_len % options.block_size != 0) {
        throw std::invalid_argument("--head-dim and --block-size must be tile multiples, and seq length must be block aligned");
    }
    if (options.cache_idx >= options.max_seq_len) {
        throw std::invalid_argument("--cache-idx must be smaller than --max-seq-len");
    }
    if (options.head_dim > 256) {
        throw std::invalid_argument(
            "This first real-kernel protocol fork supports head_dim <= 256. Wider Wt needs the block-splitting "
            "static untilize/tilize path before it can be used as attribution evidence.");
    }
    const uint64_t last_update_idx =
        static_cast<uint64_t>(options.cache_idx) + static_cast<uint64_t>(options.users - 1) * options.per_user_stride;
    if (last_update_idx >= options.max_seq_len) {
        throw std::invalid_argument("cache_idx + (users - 1) * per_user_stride must stay inside max_seq_len");
    }
    return options;
}

uint32_t checked_u32(uint64_t value, std::string_view label) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(fmt::format("{} exceeds uint32_t range", label));
    }
    return static_cast<uint32_t>(value);
}

uint32_t checked_size_bytes(uint64_t pages, uint32_t page_size, std::string_view label) {
    return checked_u32(pages * page_size, label);
}

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    if (divisor == 0) {
        throw std::invalid_argument("ceil_div_u32 divisor must be non-zero");
    }
    return (value + divisor - 1) / divisor;
}

ShapeInfo derive_shape(const Options& options) {
    ShapeInfo shape;
    shape.Wt = options.head_dim / kTileWidth;
    shape.block_size_t = options.block_size / kTileHeight;
    shape.max_blocks_per_seq = options.max_seq_len / options.block_size;
    shape.num_blocks = checked_u32(static_cast<uint64_t>(options.users) * shape.max_blocks_per_seq, "num_blocks");
    shape.cache_tiles = checked_u32(
        static_cast<uint64_t>(shape.num_blocks) * options.kv_heads * shape.block_size_t * shape.Wt, "cache_tiles");
    shape.input_tiles_per_user = shape.Wt;
    shape.ring_tiles = options.num_pages * shape.Wt;
    shape.index_stick_size = options.users * sizeof(uint32_t);
    shape.index_cb_page_size = tt::tile_size(DataFormat::Int32);
    shape.page_table_stick_size = shape.max_blocks_per_seq * sizeof(uint32_t);
    return shape;
}

std::vector<CoreCoord> select_cores(const Options& options, CoreCoord compute_grid) {
    const uint32_t available_x = compute_grid.x - options.core.x;
    const uint32_t available_y = compute_grid.y - options.core.y;
    uint32_t grid_x = options.core_grid_x;
    uint32_t grid_y = options.core_grid_y;

    if (grid_x == 0 && grid_y == 0) {
        grid_x = std::min({available_x, options.users, 4u});
        grid_y = ceil_div_u32(options.users, grid_x);
        if (grid_y > available_y) {
            grid_y = available_y;
            grid_x = ceil_div_u32(options.users, grid_y);
        }
    } else if (grid_x == 0) {
        grid_x = ceil_div_u32(options.users, grid_y);
    } else if (grid_y == 0) {
        grid_y = ceil_div_u32(options.users, grid_x);
    }

    if (grid_x == 0 || grid_y == 0 || options.core.x + grid_x > compute_grid.x || options.core.y + grid_y > compute_grid.y) {
        throw std::invalid_argument("Requested core grid is outside compute_with_storage_grid_size");
    }
    if (static_cast<uint64_t>(grid_x) * grid_y < options.users) {
        throw std::invalid_argument("Requested core grid does not contain enough cores for --users");
    }

    std::vector<CoreCoord> cores;
    cores.reserve(options.users);
    for (uint32_t y = 0; y < grid_y && cores.size() < options.users; ++y) {
        for (uint32_t x = 0; x < grid_x && cores.size() < options.users; ++x) {
            cores.emplace_back(options.core.x + x, options.core.y + y);
        }
    }
    return cores;
}

CoreRangeSet core_range_set_from_cores(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& core : cores) {
        ranges.emplace_back(core);
    }
    return CoreRangeSet(std::move(ranges)).merge_ranges();
}

std::shared_ptr<distributed::MeshBuffer> create_dram_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t size_bytes,
    uint32_t page_size_bytes) {
    distributed::DeviceLocalBufferConfig dram_config{.page_size = page_size_bytes, .buffer_type = BufferType::DRAM};
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
    // These buffers are bound to globally allocated CBs or used as explicit static-protocol rings.
    // Allocate them from the top of L1 so the remaining non-global CBs (index/page-table scratch)
    // can still occupy the normal static CB region without overlap.
    return Buffer::create(devices.front(), size_bytes, page_size_bytes, BufferType::L1, sharding_args, false);
}

uint32_t core_local_l1_address(const std::shared_ptr<Buffer>& buffer, CoreCoord core) {
    return buffer ? static_cast<uint32_t>(per_core_allocation::get_per_core_address(*buffer, core)) : 0;
}

bfloat16 cache_value(uint32_t user, uint32_t head, uint32_t seq, uint32_t d) {
    const uint32_t pattern = (user * 193u + head * 37u + seq * 17u + d * 11u + 5u) % 251u;
    return bfloat16(static_cast<float>(pattern) * 0.03125f);
}

bfloat16 update_value(uint32_t user, uint32_t head, uint32_t d) {
    const uint32_t pattern = (user * 157u + head * 53u + d * 19u + 101u) % 251u;
    return bfloat16(static_cast<float>(pattern) * 0.03125f);
}

std::vector<uint32_t> make_page_table(const Options& options, const ShapeInfo& shape) {
    std::vector<uint32_t> page_table(static_cast<size_t>(options.users) * shape.max_blocks_per_seq);
    for (uint32_t user = 0; user < options.users; ++user) {
        for (uint32_t block = 0; block < shape.max_blocks_per_seq; ++block) {
            const uint32_t logical_block = user * shape.max_blocks_per_seq + block;
            page_table[static_cast<size_t>(user) * shape.max_blocks_per_seq + block] =
                (logical_block * 17u + 3u) % shape.num_blocks;
        }
    }
    return page_table;
}

std::vector<uint32_t> make_update_indices(const Options& options) {
    std::vector<uint32_t> update_indices(options.users);
    for (uint32_t user = 0; user < options.users; ++user) {
        update_indices[user] = options.cache_idx + user * options.per_user_stride;
    }
    return update_indices;
}

void fill_block_head_tiles(
    std::vector<bfloat16>& dst,
    uint32_t physical_block,
    uint32_t head,
    const std::vector<bfloat16>& block_row_major,
    const Options& options,
    const ShapeInfo& shape) {
    const auto tiled = tilize_nfaces(block_row_major, options.block_size, options.head_dim);
    const size_t tile_offset =
        (static_cast<size_t>(physical_block) * options.kv_heads * shape.block_size_t * shape.Wt +
         static_cast<size_t>(head) * shape.block_size_t * shape.Wt) *
        kTileElements;
    std::copy(tiled.begin(), tiled.end(), dst.begin() + tile_offset);
}

std::vector<bfloat16> make_cache_tiles(
    const Options& options,
    const ShapeInfo& shape,
    const std::vector<uint32_t>& page_table,
    const std::optional<std::vector<uint32_t>>& update_indices = std::nullopt) {
    std::vector<std::pair<uint32_t, uint32_t>> physical_to_logical(shape.num_blocks);
    for (uint32_t user = 0; user < options.users; ++user) {
        for (uint32_t block = 0; block < shape.max_blocks_per_seq; ++block) {
            physical_to_logical[page_table[static_cast<size_t>(user) * shape.max_blocks_per_seq + block]] =
                {user, block};
        }
    }

    std::vector<bfloat16> cache(static_cast<size_t>(shape.cache_tiles) * kTileElements);
    std::vector<bfloat16> block_row_major(static_cast<size_t>(options.block_size) * options.head_dim);
    for (uint32_t physical_block = 0; physical_block < shape.num_blocks; ++physical_block) {
        const auto [user, virtual_block] = physical_to_logical[physical_block];
        for (uint32_t head = 0; head < options.kv_heads; ++head) {
            for (uint32_t row = 0; row < options.block_size; ++row) {
                const uint32_t seq = virtual_block * options.block_size + row;
                for (uint32_t d = 0; d < options.head_dim; ++d) {
                    block_row_major[static_cast<size_t>(row) * options.head_dim + d] =
                        cache_value(user, head, seq, d);
                }
            }
            if (update_indices.has_value()) {
                const uint32_t update_idx = update_indices->at(user);
                if (update_idx / options.block_size == virtual_block) {
                    const uint32_t row = update_idx % options.block_size;
                    for (uint32_t d = 0; d < options.head_dim; ++d) {
                        block_row_major[static_cast<size_t>(row) * options.head_dim + d] =
                            update_value(user, head, d);
                    }
                }
            }
            fill_block_head_tiles(cache, physical_block, head, block_row_major, options, shape);
        }
    }
    return cache;
}

std::vector<bfloat16> make_input_tiles_for_user(uint32_t user, const Options& options) {
    std::vector<bfloat16> row_major(static_cast<size_t>(kTileHeight) * options.head_dim, bfloat16(0.0f));
    for (uint32_t head = 0; head < options.kv_heads; ++head) {
        for (uint32_t d = 0; d < options.head_dim; ++d) {
            row_major[static_cast<size_t>(head) * options.head_dim + d] = update_value(user, head, d);
        }
    }
    return tilize_nfaces(row_major, kTileHeight, options.head_dim);
}

void create_data_cb(
    Program& program,
    CoreCoord core,
    CBIndex cb,
    uint32_t page_size,
    uint32_t num_pages,
    DataFormat data_format,
    const std::shared_ptr<Buffer>& global_buffer) {
    auto config = CircularBufferConfig(num_pages * page_size, {{cb, data_format}}).set_page_size(cb, page_size);
    if (global_buffer) {
        config.set_globally_allocated_address(*global_buffer);
    }
    CreateCircularBuffer(program, core, config);
}

void create_shared_intermed_cb(
    Program& program,
    CoreCoord core,
    uint32_t page_size,
    uint32_t num_pages,
    DataFormat data_format,
    const std::shared_ptr<Buffer>& global_buffer) {
    auto config =
        CircularBufferConfig(num_pages * page_size, {{CBIndex::c_24, data_format}, {CBIndex::c_25, data_format}})
            .set_page_size(CBIndex::c_24, page_size)
            .set_page_size(CBIndex::c_25, page_size);
    if (global_buffer) {
        config.set_globally_allocated_address(*global_buffer);
    }
    CreateCircularBuffer(program, core, config);
}

void create_circular_buffers(
    Program& program,
    CoreResources& resources,
    Mode mode,
    const Options& options,
    const ShapeInfo& shape) {
    const uint32_t ring_tiles = shape.ring_tiles;
    const DataFormat data_format = DataFormat::Float16_b;
    const DataFormat intermed_format = DataFormat::Float16_b;

    create_data_cb(
        program,
        resources.core,
        CBIndex::c_0,
        kTileSizeBytes,
        ring_tiles,
        data_format,
        is_static_mode(mode) ? resources.cache_ring_l1 : nullptr);
    create_data_cb(
        program,
        resources.core,
        CBIndex::c_1,
        kTileSizeBytes,
        shape.input_tiles_per_user,
        data_format,
        resources.input_tiles_l1);
    create_shared_intermed_cb(
        program,
        resources.core,
        kTileSizeBytes,
        ring_tiles,
        intermed_format,
        is_static_mode(mode) ? resources.intermed_shared_l1 : nullptr);
    create_data_cb(
        program,
        resources.core,
        CBIndex::c_26,
        kTileSizeBytes,
        shape.input_tiles_per_user,
        intermed_format,
        is_static_mode(mode) ? resources.input_untilized_l1 : nullptr);
    create_data_cb(
        program,
        resources.core,
        CBIndex::c_16,
        kTileSizeBytes,
        is_static_mode(mode) ? ring_tiles : std::max(ring_tiles, options.users * shape.Wt),
        data_format,
        is_static_mode(mode) ? resources.output_ring_l1 : nullptr);
    create_data_cb(
        program, resources.core, CBIndex::c_2, shape.index_cb_page_size, 1, DataFormat::Int32, nullptr);
    create_data_cb(
        program, resources.core, CBIndex::c_3, shape.page_table_stick_size, 1, DataFormat::Int32, nullptr);
}

std::map<std::string, std::string> protocol_defines(
    Mode mode,
    uint32_t protocol_start_value,
    uint32_t cache_ring_addr,
    uint32_t intermed_ring_addr,
    uint32_t input_untilized_addr,
    uint32_t output_ring_addr,
    uint32_t input_tiles_addr,
    uint32_t protocol_start_addr,
    const Options& options) {
    return {
        {"BENCH_STATIC_PROTOCOL", is_static_mode(mode) ? "1" : "0"},
        {"BENCH_USE_STREAM_REG_CBREGS", uses_stream_reg_cbregs(mode) ? "1" : "0"},
        {"BENCH_USE_COMPILE_TIME_PROTOCOL_ARGS", uses_compile_time_protocol_args(mode) ? "1" : "0"},
        {"BENCH_PROTOCOL_START_VALUE", std::to_string(protocol_start_value)},
        {"BENCH_CACHE_RING_ADDR", std::to_string(cache_ring_addr)},
        {"BENCH_INTERMED_RING_ADDR", std::to_string(intermed_ring_addr)},
        {"BENCH_INPUT_UNTILIZED_ADDR", std::to_string(input_untilized_addr)},
        {"BENCH_OUTPUT_RING_ADDR", std::to_string(output_ring_addr)},
        {"BENCH_INPUT_TILES_ADDR", std::to_string(input_tiles_addr)},
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

bool validate_result(
    const std::vector<bfloat16>& expected,
    const std::vector<bfloat16>& result,
    float* max_abs_error) {
    if (result.size() != expected.size()) {
        *max_abs_error = std::numeric_limits<float>::infinity();
        fmt::print(stderr, "Result size mismatch: expected {}, got {}\n", expected.size(), result.size());
        return false;
    }
    bool ok = true;
    float max_error = 0.0f;
    uint32_t printed = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float error = std::abs(static_cast<float>(expected[i]) - static_cast<float>(result[i]));
        max_error = std::max(max_error, error);
        if (error > 3e-1f) {
            ok = false;
            if (printed < 8) {
                fmt::print(
                    stderr,
                    "Mismatch at raw tiled element {}: expected {}, got {}\n",
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
    const ShapeInfo& shape,
    const std::vector<CoreCoord>& cores,
    Mode mode,
    uint32_t repeat,
    const std::vector<bfloat16>& initial_cache,
    const std::vector<bfloat16>& expected_cache,
    const std::vector<uint32_t>& update_indices,
    const std::vector<uint32_t>& page_table) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    const CoreRangeSet active_cores = core_range_set_from_cores(cores);
    const uint32_t sequential_mode_semaphore_id = CreateSemaphore(program, active_cores, 0);

    auto cache_buffer = create_dram_buffer(
        mesh_device, checked_size_bytes(shape.cache_tiles, kTileSizeBytes, "cache buffer"), kTileSizeBytes);
    auto index_buffer = create_dram_buffer(mesh_device, shape.index_stick_size, shape.index_stick_size);
    auto page_table_buffer = create_dram_buffer(
        mesh_device,
        checked_size_bytes(options.users, shape.page_table_stick_size, "page table buffer"),
        shape.page_table_stick_size);

    std::vector<CoreResources> resources;
    resources.reserve(cores.size());
    const uint32_t input_bytes = checked_size_bytes(shape.input_tiles_per_user, kTileSizeBytes, "input L1");
    const uint32_t ring_bytes = checked_size_bytes(shape.ring_tiles, kTileSizeBytes, "static ring");
    const std::vector<uint8_t> zero_ring(ring_bytes, 0);
    const std::vector<uint8_t> zero_input(input_bytes, 0);
    const std::vector<uint8_t> zero_sem(kProtocolStartBytes, 0);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        CoreResources item{.core = cores[i]};
        item.input_tiles_l1 = create_core_local_l1_buffer(mesh_device, item.core, input_bytes, kTileSizeBytes);
        detail::WriteToBuffer(item.input_tiles_l1, zero_input);
        const auto input_tiles = make_input_tiles_for_user(i, options);
        detail::WriteToBuffer(item.input_tiles_l1, input_tiles);

        if (is_static_mode(mode)) {
            item.cache_ring_l1 = create_core_local_l1_buffer(mesh_device, item.core, ring_bytes, kTileSizeBytes);
            item.intermed_shared_l1 = create_core_local_l1_buffer(mesh_device, item.core, ring_bytes, kTileSizeBytes);
            item.input_untilized_l1 = create_core_local_l1_buffer(mesh_device, item.core, input_bytes, kTileSizeBytes);
            item.output_ring_l1 = create_core_local_l1_buffer(mesh_device, item.core, ring_bytes, kTileSizeBytes);
            detail::WriteToBuffer(item.cache_ring_l1, zero_ring);
            detail::WriteToBuffer(item.intermed_shared_l1, zero_ring);
            detail::WriteToBuffer(item.input_untilized_l1, zero_input);
            detail::WriteToBuffer(item.output_ring_l1, zero_ring);
            if (!uses_stream_reg_cbregs(mode)) {
                item.protocol_start_l1 =
                    create_core_local_l1_buffer(mesh_device, item.core, kProtocolStartBytes, kProtocolStartBytes);
                detail::WriteToBuffer(item.protocol_start_l1, zero_sem);
            }
        }
        create_circular_buffers(program, item, mode, options, shape);
        resources.push_back(std::move(item));
    }

    const uint32_t cache_addr = static_cast<uint32_t>(cache_buffer->address());
    const uint32_t index_addr = static_cast<uint32_t>(index_buffer->address());
    const uint32_t page_table_addr = static_cast<uint32_t>(page_table_buffer->address());
    const uint32_t protocol_start_value =
        (0x5a5a0000u ^ options.users ^ (options.kv_heads << 4) ^ (options.head_dim << 8) ^
         (repeat * 0x00010001u) ^ cache_addr) &
        kStreamRegCounterMask;
    if (uses_compile_time_protocol_args(mode) && resources.size() != 1) {
        throw std::invalid_argument("static-streamreg-cbregs-compiletime mode currently supports only one active core");
    }
    const auto first_static_addrs = [&]() {
        struct Addresses {
            uint32_t cache_ring_addr;
            uint32_t intermed_ring_addr;
            uint32_t input_untilized_addr;
            uint32_t output_ring_addr;
            uint32_t input_tiles_addr;
            uint32_t protocol_start_addr;
        };
        if (!is_static_mode(mode)) {
            return Addresses{};
        }
        const auto& item = resources.front();
        return Addresses{
            .cache_ring_addr = core_local_l1_address(item.cache_ring_l1, item.core),
            .intermed_ring_addr = core_local_l1_address(item.intermed_shared_l1, item.core),
            .input_untilized_addr = core_local_l1_address(item.input_untilized_l1, item.core),
            .output_ring_addr = core_local_l1_address(item.output_ring_l1, item.core),
            .input_tiles_addr = core_local_l1_address(item.input_tiles_l1, item.core),
            .protocol_start_addr = core_local_l1_address(item.protocol_start_l1, item.core)};
    }();
    auto defines = protocol_defines(
        mode,
        protocol_start_value == 0 ? 1 : protocol_start_value,
        first_static_addrs.cache_ring_addr,
        first_static_addrs.intermed_ring_addr,
        first_static_addrs.input_untilized_addr,
        first_static_addrs.output_ring_addr,
        first_static_addrs.input_tiles_addr,
        first_static_addrs.protocol_start_addr,
        options);

    std::vector<uint32_t> reader_compile_args = {
        static_cast<uint32_t>(CBIndex::c_0),
        static_cast<uint32_t>(CBIndex::c_1),
        1,
        static_cast<uint32_t>(CBIndex::c_2),
        0,
        shape.Wt,
        0,
        shape.index_stick_size,
        1,
        options.kv_heads,
        options.block_size,
        shape.block_size_t,
        shape.max_blocks_per_seq,
        0,
        shape.page_table_stick_size,
        static_cast<uint32_t>(CBIndex::c_3),
        shape.block_size_t,
        sequential_mode_semaphore_id,
    };
    TensorAccessorArgs(cache_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(index_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(page_table_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {
        static_cast<uint32_t>(CBIndex::c_16),
        static_cast<uint32_t>(CBIndex::c_24),
        static_cast<uint32_t>(CBIndex::c_25),
        static_cast<uint32_t>(CBIndex::c_26),
        1,
        static_cast<uint32_t>(CBIndex::c_2),
        0,
        shape.Wt,
        options.head_dim * sizeof(bfloat16),
        1,
        options.kv_heads,
        options.block_size,
        shape.block_size_t,
        shape.max_blocks_per_seq,
        static_cast<uint32_t>(CBIndex::c_3),
        shape.block_size_t,
        sequential_mode_semaphore_id,
    };
    TensorAccessorArgs(cache_buffer).append_to(writer_compile_args);

    std::vector<uint32_t> compute_compile_args = {
        static_cast<uint32_t>(CBIndex::c_0),
        static_cast<uint32_t>(CBIndex::c_1),
        static_cast<uint32_t>(CBIndex::c_24),
        static_cast<uint32_t>(CBIndex::c_25),
        static_cast<uint32_t>(CBIndex::c_26),
        static_cast<uint32_t>(CBIndex::c_16),
        shape.Wt,
        options.kv_heads,
    };

    auto reader_kernel = CreateKernel(
        program,
        std::string(kReaderKernel),
        active_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args,
            .defines = defines});
    auto writer_kernel = CreateKernel(
        program,
        std::string(kWriterKernel),
        active_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args,
            .defines = defines});
    auto compute_kernel = CreateKernel(
        program,
        std::string(kComputeKernel),
        active_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .compile_args = compute_compile_args,
            .defines = defines});

    for (uint32_t i = 0; i < resources.size(); ++i) {
        const auto& item = resources[i];
        std::vector<uint32_t> reader_args = {cache_addr, 0, index_addr, i, page_table_addr, 0};
        std::vector<uint32_t> writer_args = {cache_addr, 0, 0, i, 0, 0, 0};
        std::vector<uint32_t> compute_args;
        if (is_static_mode(mode)) {
            const uint32_t protocol_start_addr = core_local_l1_address(item.protocol_start_l1, item.core);
            if (!uses_compile_time_protocol_args(mode)) {
                reader_args.insert(
                    reader_args.end(),
                    {core_local_l1_address(item.cache_ring_l1, item.core),
                     kTileSizeBytes,
                     options.num_pages,
                     protocol_start_addr});
                writer_args.insert(
                    writer_args.end(),
                    {core_local_l1_address(item.intermed_shared_l1, item.core),
                     core_local_l1_address(item.input_untilized_l1, item.core),
                     core_local_l1_address(item.output_ring_l1, item.core),
                     kTileSizeBytes,
                     kTileSizeBytes,
                     options.num_pages,
                     protocol_start_addr});
                compute_args.insert(
                    compute_args.end(),
                    {core_local_l1_address(item.cache_ring_l1, item.core),
                     core_local_l1_address(item.intermed_shared_l1, item.core),
                     core_local_l1_address(item.input_untilized_l1, item.core),
                     core_local_l1_address(item.output_ring_l1, item.core),
                     kTileSizeBytes,
                     options.num_pages,
                     protocol_start_addr,
                     core_local_l1_address(item.input_tiles_l1, item.core)});
            }
        }
        SetRuntimeArgs(program, reader_kernel, item.core, reader_args);
        SetRuntimeArgs(program, writer_kernel, item.core, writer_args);
        if (is_static_mode(mode)) {
            SetRuntimeArgs(program, compute_kernel, item.core, compute_args);
        }
    }

    distributed::EnqueueWriteMeshBuffer(cq, cache_buffer, initial_cache, false);
    distributed::EnqueueWriteMeshBuffer(cq, index_buffer, update_indices, false);
    distributed::EnqueueWriteMeshBuffer(cq, page_table_buffer, page_table, false);

    workload.add_program(device_range, std::move(program));
    const auto start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();

    std::vector<bfloat16> result;
    distributed::EnqueueReadMeshBuffer(cq, result, cache_buffer, true);

    float max_abs_error = 0.0f;
    const bool ok = options.skip_check || validate_result(expected_cache, result, &max_abs_error);
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
        const ShapeInfo shape = derive_shape(options);
        const auto modes = modes_to_run(options.mode);
        (void)modes;
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", 1);
        if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
            fmt::print(
                "WARNING: set TT_METAL_DEVICE_PROFILER=1 to collect device cycle zones in "
                "generated/profiler/.logs/profile_log_device.csv\n");
        }

        auto mesh_device = distributed::MeshDevice::create_unit_mesh(options.device_id);
        const auto cores = select_cores(options, mesh_device->compute_with_storage_grid_size());
        const auto page_table = make_page_table(options, shape);
        const auto update_indices = make_update_indices(options);
        const auto initial_cache = make_cache_tiles(options, shape, page_table);
        const auto expected_cache = make_cache_tiles(options, shape, page_table, update_indices);

        fmt::print(
            "ttnn_paged_update_cache_protocol: users={}, kv_heads={}, head_dim={}, Wt={}, block_size={}, "
            "block_size_t={}, max_seq_len={}, max_blocks_per_seq={}, cache_idx={}, per_user_stride={}, "
            "cache_tiles={}, ring_rows={}, repeats={}, core=({}, {}), active_cores={}\n",
            options.users,
            options.kv_heads,
            options.head_dim,
            shape.Wt,
            options.block_size,
            shape.block_size_t,
            options.max_seq_len,
            shape.max_blocks_per_seq,
            options.cache_idx,
            options.per_user_stride,
            shape.cache_tiles,
            options.num_pages,
            options.repeats,
            options.core.x,
            options.core.y,
            cores.size());

        for (uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
            for (Mode mode : modes) {
                RunResult result = run_one(
                    mesh_device,
                    options,
                    shape,
                    cores,
                    mode,
                    repeat,
                    initial_cache,
                    expected_cache,
                    update_indices,
                    page_table);
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
