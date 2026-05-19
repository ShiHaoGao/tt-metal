// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cerrno>
#include <chrono>
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

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr CoreCoord kDefaultCore = {0, 0};
constexpr uint32_t kL1Alignment = 64;
constexpr uint32_t kSinkBufferBytes = 2 * kL1Alignment;
constexpr uint32_t kConsumerSinkOffsetBytes = sizeof(uint32_t);
constexpr uint32_t kMaxCbApiIterations = 60000;

enum class BenchMode : uint32_t {
    Empty = 0,
    CbGetWritePtr = 1,
    CbGetReadWritePtr = 2,
    CbGetTileSize = 3,
    CbApiRoundtrip = 4,
    StaticRing = 5,
    StaticCounter = 6,
    CrossEmpty = 7,
    CrossCb = 8,
    CrossStatic = 9,
    CbSystem = 10,
    StaticRuntimeAddr = 11,
    StaticCompileTimeAddr = 12,
    StaticNoSync = 13,
    DramCb = 14,
    DramStaticRuntimeAddr = 15,
    DramStaticCompileTimeAddr = 16,
    DramSingleNoSync = 17,
    CrossStreamReg = 18,
    StaticStreamRegAddr = 19,
    DramStaticStreamRegAddr = 20,
};

enum class BenchRole : uint32_t {
    Producer = 0,
    Consumer = 1,
};

struct Options {
    std::string mode = "all";
    uint32_t iterations = 10000;
    uint32_t page_size = 64;
    uint32_t num_pages = 8;
    uint32_t num_cbs = 1;
    uint32_t num_rv = 2;
    uint32_t repeats = 1;
    uint32_t device_id = 0;
    CoreCoord core = kDefaultCore;
};

struct RunResult {
    BenchMode mode;
    uint32_t repeat;
    uint64_t enqueue_finish_us;
    uint32_t producer_sink;
    uint32_t consumer_sink;
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

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

const char* mode_name(BenchMode mode) {
    switch (mode) {
        case BenchMode::Empty: return "empty";
        case BenchMode::CbGetWritePtr: return "cb-get-write-ptr";
        case BenchMode::CbGetReadWritePtr: return "cb-get-read-write-ptr";
        case BenchMode::CbGetTileSize: return "cb-get-tile-size";
        case BenchMode::CbApiRoundtrip: return "cb-api-roundtrip";
        case BenchMode::StaticRing: return "static-ring";
        case BenchMode::StaticCounter: return "static-counter";
        case BenchMode::CrossEmpty: return "cross-empty";
        case BenchMode::CrossCb: return "cross-cb";
        case BenchMode::CrossStatic: return "cross-static";
        case BenchMode::CbSystem: return "cb-system";
        case BenchMode::StaticRuntimeAddr: return "static-runtime-addr";
        case BenchMode::StaticCompileTimeAddr: return "static-compiletime-addr";
        case BenchMode::StaticNoSync: return "static-no-sync";
        case BenchMode::DramCb: return "dram-cb";
        case BenchMode::DramStaticRuntimeAddr: return "dram-static-runtime-addr";
        case BenchMode::DramStaticCompileTimeAddr: return "dram-static-compiletime-addr";
        case BenchMode::DramSingleNoSync: return "dram-single-nosync";
        case BenchMode::CrossStreamReg: return "cross-streamreg";
        case BenchMode::StaticStreamRegAddr: return "static-streamreg-addr";
        case BenchMode::DramStaticStreamRegAddr: return "dram-static-streamreg-addr";
    }
    return "unknown";
}

std::optional<BenchMode> parse_mode(std::string_view mode) {
    if (mode == "empty") {
        return BenchMode::Empty;
    }
    if (mode == "cb-get-write-ptr") {
        return BenchMode::CbGetWritePtr;
    }
    if (mode == "cb-get-read-write-ptr") {
        return BenchMode::CbGetReadWritePtr;
    }
    if (mode == "cb-get-tile-size") {
        return BenchMode::CbGetTileSize;
    }
    if (mode == "cb-api-roundtrip") {
        return BenchMode::CbApiRoundtrip;
    }
    if (mode == "static-ring") {
        return BenchMode::StaticRing;
    }
    if (mode == "static-counter") {
        return BenchMode::StaticCounter;
    }
    if (mode == "cross-empty") {
        return BenchMode::CrossEmpty;
    }
    if (mode == "cross-cb") {
        return BenchMode::CrossCb;
    }
    if (mode == "cross-static") {
        return BenchMode::CrossStatic;
    }
    if (mode == "cb-system") {
        return BenchMode::CbSystem;
    }
    if (mode == "static-runtime-addr") {
        return BenchMode::StaticRuntimeAddr;
    }
    if (mode == "static-compiletime-addr") {
        return BenchMode::StaticCompileTimeAddr;
    }
    if (mode == "static-no-sync") {
        return BenchMode::StaticNoSync;
    }
    if (mode == "dram-cb") {
        return BenchMode::DramCb;
    }
    if (mode == "dram-static-runtime-addr") {
        return BenchMode::DramStaticRuntimeAddr;
    }
    if (mode == "dram-static-compiletime-addr") {
        return BenchMode::DramStaticCompileTimeAddr;
    }
    if (mode == "dram-single-nosync") {
        return BenchMode::DramSingleNoSync;
    }
    if (mode == "cross-streamreg") {
        return BenchMode::CrossStreamReg;
    }
    if (mode == "static-streamreg-addr") {
        return BenchMode::StaticStreamRegAddr;
    }
    if (mode == "dram-static-streamreg-addr") {
        return BenchMode::DramStaticStreamRegAddr;
    }
    return std::nullopt;
}

bool uses_cb(BenchMode mode) {
    return mode == BenchMode::CbGetWritePtr || mode == BenchMode::CbGetReadWritePtr ||
           mode == BenchMode::CbGetTileSize || mode == BenchMode::CbApiRoundtrip || mode == BenchMode::CrossCb ||
           mode == BenchMode::CbSystem || mode == BenchMode::DramCb;
}

bool is_cross_mode(BenchMode mode) {
    return mode == BenchMode::CrossEmpty || mode == BenchMode::CrossCb || mode == BenchMode::CrossStatic ||
           mode == BenchMode::CbSystem || mode == BenchMode::StaticRuntimeAddr ||
           mode == BenchMode::StaticCompileTimeAddr || mode == BenchMode::DramCb ||
           mode == BenchMode::DramStaticRuntimeAddr || mode == BenchMode::DramStaticCompileTimeAddr ||
           mode == BenchMode::CrossStreamReg || mode == BenchMode::StaticStreamRegAddr ||
           mode == BenchMode::DramStaticStreamRegAddr;
}

bool uses_static_work_l1(BenchMode mode) {
    return mode == BenchMode::StaticRing || mode == BenchMode::StaticCounter || mode == BenchMode::CrossStatic ||
           mode == BenchMode::StaticRuntimeAddr || mode == BenchMode::StaticCompileTimeAddr ||
           mode == BenchMode::StaticNoSync || mode == BenchMode::DramStaticRuntimeAddr ||
           mode == BenchMode::DramStaticCompileTimeAddr || mode == BenchMode::DramSingleNoSync ||
           mode == BenchMode::CrossStreamReg || mode == BenchMode::StaticStreamRegAddr ||
           mode == BenchMode::DramStaticStreamRegAddr;
}

bool uses_static_sync(BenchMode mode) {
    return mode == BenchMode::CrossStatic || mode == BenchMode::StaticRuntimeAddr ||
           mode == BenchMode::StaticCompileTimeAddr || mode == BenchMode::DramStaticRuntimeAddr ||
           mode == BenchMode::DramStaticCompileTimeAddr;
}

bool uses_compile_time_addresses(BenchMode mode) {
    return mode == BenchMode::StaticCompileTimeAddr || mode == BenchMode::StaticNoSync ||
           mode == BenchMode::DramStaticCompileTimeAddr || mode == BenchMode::DramSingleNoSync;
}

bool is_system_mode(BenchMode mode) {
    return mode == BenchMode::CbSystem || mode == BenchMode::StaticRuntimeAddr ||
           mode == BenchMode::StaticCompileTimeAddr || mode == BenchMode::StaticNoSync ||
           mode == BenchMode::StaticStreamRegAddr;
}

bool is_dram_mode(BenchMode mode) {
    return mode == BenchMode::DramCb || mode == BenchMode::DramStaticRuntimeAddr ||
           mode == BenchMode::DramStaticCompileTimeAddr || mode == BenchMode::DramSingleNoSync ||
           mode == BenchMode::DramStaticStreamRegAddr;
}

std::vector<BenchMode> modes_to_run(const std::string& mode) {
    if (mode == "all") {
        return {
            BenchMode::Empty,
            BenchMode::CbGetWritePtr,
            BenchMode::CbGetReadWritePtr,
            BenchMode::CbGetTileSize,
            BenchMode::CbApiRoundtrip,
            BenchMode::StaticRing,
            BenchMode::StaticCounter,
            BenchMode::CrossEmpty,
            BenchMode::CrossCb,
            BenchMode::CrossStatic,
            BenchMode::CrossStreamReg,
            BenchMode::CbSystem,
            BenchMode::StaticRuntimeAddr,
            BenchMode::StaticCompileTimeAddr,
            BenchMode::StaticStreamRegAddr,
            BenchMode::StaticNoSync,
            BenchMode::DramCb,
            BenchMode::DramStaticRuntimeAddr,
            BenchMode::DramStaticCompileTimeAddr,
            BenchMode::DramStaticStreamRegAddr,
            BenchMode::DramSingleNoSync};
    }

    if (mode == "system") {
        return {
            BenchMode::CrossEmpty,
            BenchMode::CbSystem,
            BenchMode::StaticRuntimeAddr,
            BenchMode::StaticCompileTimeAddr,
            BenchMode::StaticStreamRegAddr,
            BenchMode::StaticNoSync};
    }

    if (mode == "dram") {
        return {
            BenchMode::CrossEmpty,
            BenchMode::DramCb,
            BenchMode::DramStaticRuntimeAddr,
            BenchMode::DramStaticCompileTimeAddr,
            BenchMode::DramStaticStreamRegAddr,
            BenchMode::DramSingleNoSync};
    }

    auto parsed_mode = parse_mode(mode);
    if (!parsed_mode.has_value()) {
        throw std::invalid_argument(fmt::format("Unknown mode '{}'. Use --help to list modes.", mode));
    }
    return {*parsed_mode};
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--mode=all|system|dram|empty|cb-get-write-ptr|cb-get-read-write-ptr|cb-get-tile-size|"
        "cb-api-roundtrip|static-ring|static-counter|cross-empty|cross-cb|cross-static|"
        "cross-streamreg|cb-system|static-runtime-addr|static-compiletime-addr|static-streamreg-addr|static-no-sync|"
        "dram-cb|dram-static-runtime-addr|dram-static-compiletime-addr|dram-static-streamreg-addr|dram-single-nosync] "
        "[--iterations=N] [--page-size=N] [--num-pages=N] [--num-cbs=N] [--num-rv=1|2|5] "
        "[--repeats=N] [--device-id=N] [--core-x=N] [--core-y=N]\n",
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
        } else if (starts_with(arg, "--iterations=")) {
            options.iterations = parse_u32(arg.substr(std::string_view("--iterations=").size()), "iterations");
        } else if (starts_with(arg, "--page-size=")) {
            options.page_size = parse_u32(arg.substr(std::string_view("--page-size=").size()), "page-size");
        } else if (starts_with(arg, "--num-pages=")) {
            options.num_pages = parse_u32(arg.substr(std::string_view("--num-pages=").size()), "num-pages");
        } else if (starts_with(arg, "--num-cbs=")) {
            options.num_cbs = parse_u32(arg.substr(std::string_view("--num-cbs=").size()), "num-cbs");
        } else if (starts_with(arg, "--num-rv=")) {
            options.num_rv = parse_u32(arg.substr(std::string_view("--num-rv=").size()), "num-rv");
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

    if (options.iterations == 0) {
        throw std::invalid_argument("--iterations must be greater than zero");
    }
    if (options.repeats == 0) {
        throw std::invalid_argument("--repeats must be greater than zero");
    }
    if (options.page_size < kL1Alignment || options.page_size % kL1Alignment != 0) {
        throw std::invalid_argument(fmt::format("--page-size must be a multiple of {}", kL1Alignment));
    }
    if (options.num_pages == 0) {
        throw std::invalid_argument("--num-pages must be greater than zero");
    }
    if (options.num_cbs == 0 || options.num_cbs > 64) {
        throw std::invalid_argument("--num-cbs must be in the range [1, 64]");
    }
    if (options.num_rv != 1 && options.num_rv != 2 && options.num_rv != 5) {
        throw std::invalid_argument("--num-rv must be one of 1, 2, or 5");
    }
    return options;
}

uint32_t scratch_size_bytes(const Options& options) {
    const uint32_t payload_bytes = options.page_size * options.num_pages;
    return align_up(payload_bytes + 2 * kL1Alignment, kL1Alignment);
}

uint32_t payload_dram_size_bytes(const Options& options) {
    const uint64_t payload_bytes = static_cast<uint64_t>(options.page_size) * options.iterations;
    if (payload_bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("payload DRAM size exceeds uint32_t addressable range");
    }
    return static_cast<uint32_t>(payload_bytes);
}

std::map<std::string, std::string> kernel_defines(
    const Options& options,
    BenchMode mode,
    BenchRole role,
    uint32_t repeat,
    uint32_t work_base_addr,
    uint32_t sink_l1_addr,
    uint32_t sink_dram_addr,
    uint32_t sink_dram_offset,
    uint32_t input_dram_addr,
    uint32_t output_dram_addr,
    uint32_t produced_sem_id,
    uint32_t consumed_sem_id) {
    std::map<std::string, std::string> defines = {
        {"BENCH_MODE", std::to_string(static_cast<uint32_t>(mode))},
        {"BENCH_ITERATIONS", std::to_string(options.iterations)},
        {"BENCH_ROLE", std::to_string(static_cast<uint32_t>(role))},
        {"BENCH_WORK_BASE_ADDR", std::to_string(work_base_addr)},
        {"BENCH_SINK_L1_ADDR", std::to_string(sink_l1_addr)},
        {"BENCH_SINK_DRAM_ADDR", std::to_string(sink_dram_addr)},
        {"BENCH_SINK_DRAM_OFFSET", std::to_string(sink_dram_offset)},
        {"BENCH_INPUT_DRAM_ADDR", std::to_string(input_dram_addr)},
        {"BENCH_OUTPUT_DRAM_ADDR", std::to_string(output_dram_addr)},
        {"BENCH_PAGE_SIZE", std::to_string(options.page_size)},
        {"BENCH_NUM_PAGES", std::to_string(options.num_pages)},
        {"BENCH_PRODUCED_SEM_ID", std::to_string(produced_sem_id)},
        {"BENCH_CONSUMED_SEM_ID", std::to_string(consumed_sem_id)},
        {"BENCH_STREAM_REG_PRODUCED_STREAM_ID", "1"},
        {"BENCH_STREAM_REG_CONSUMED_STREAM_ID", "2"},
        {"BENCH_STREAM_REG_START_STREAM_ID", "0"},
        {"BENCH_STREAM_REG_VALUE_MASK", "16777215"},
        {"BENCH_STREAM_REG_START_VALUE",
         std::to_string(0x6b6b0000u ^ static_cast<uint32_t>(mode) ^ repeat ^ options.iterations ^
                        (options.num_pages << 8) ^ options.page_size)}};
    if (uses_compile_time_addresses(mode)) {
        defines["BENCH_USE_COMPILE_TIME_ARGS"] = "1";
    }
    return defines;
}

std::map<std::string, std::string> aux_compute_defines(const Options& options, BenchMode mode) {
    return {
        {"BENCH_MODE", std::to_string(static_cast<uint32_t>(mode))},
        {"BENCH_ITERATIONS", std::to_string(options.iterations)}};
}

std::shared_ptr<distributed::MeshBuffer> create_l1_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size_bytes) {
    distributed::DeviceLocalBufferConfig config{
        .page_size = size_bytes, .buffer_type = tt_metal::BufferType::L1, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = size_bytes};
    return distributed::MeshBuffer::create(buffer_config, config, mesh_device.get());
}

std::shared_ptr<distributed::MeshBuffer> create_dram_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size_bytes, uint32_t page_size) {
    distributed::DeviceLocalBufferConfig config{.page_size = page_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = size_bytes};
    return distributed::MeshBuffer::create(buffer_config, config, mesh_device.get());
}

std::shared_ptr<distributed::MeshBuffer> create_dram_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size_bytes) {
    return create_dram_buffer(mesh_device, size_bytes, kL1Alignment);
}

void zero_l1_buffer(
    distributed::MeshCommandQueue& cq,
    std::shared_ptr<distributed::MeshBuffer>& buffer,
    uint32_t size_bytes) {
    std::vector<uint32_t> zeros(size_bytes / sizeof(uint32_t), 0);
    distributed::EnqueueWriteMeshBuffer(cq, buffer, zeros, true);
}

void initialize_dram_payload(
    distributed::MeshCommandQueue& cq,
    std::shared_ptr<distributed::MeshBuffer>& input_buffer,
    std::shared_ptr<distributed::MeshBuffer>& output_buffer,
    const Options& options) {
    const uint32_t size_bytes = payload_dram_size_bytes(options);
    std::vector<uint32_t> input(size_bytes / sizeof(uint32_t), 0);
    const uint32_t words_per_page = options.page_size / sizeof(uint32_t);
    for (uint32_t page = 0; page < options.iterations; ++page) {
        for (uint32_t word = 0; word < words_per_page; ++word) {
            input[page * words_per_page + word] = (page * 1315423911u) ^ (word * 2654435761u);
        }
    }
    std::vector<uint32_t> zeros(input.size(), 0);
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input, true);
    distributed::EnqueueWriteMeshBuffer(cq, output_buffer, zeros, true);
}

RunResult run_one(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Options& options,
    BenchMode mode,
    uint32_t repeat) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    const uint32_t cb_total_size = options.page_size * options.num_pages;
    const uint32_t cb_count =
        mode == BenchMode::CbSystem || mode == BenchMode::DramCb ? options.num_cbs : 1;
    if (uses_cb(mode)) {
        for (uint32_t cb_index = 0; cb_index < cb_count; ++cb_index) {
            CircularBufferConfig cb_config =
                CircularBufferConfig(cb_total_size, {{static_cast<uint8_t>(cb_index), tt::DataFormat::UInt32}})
                    .set_page_size(static_cast<uint8_t>(cb_index), options.page_size);
            CreateCircularBuffer(program, options.core, cb_config);
        }
    }

    uint32_t produced_sem_id = 0;
    uint32_t consumed_sem_id = 0;
    if (uses_static_sync(mode)) {
        const CoreRange core_range(options.core);
        produced_sem_id = CreateSemaphore(program, core_range, 0);
        consumed_sem_id = CreateSemaphore(program, core_range, 0);
    }

    const uint32_t work_size = scratch_size_bytes(options);
    auto work_buffer = create_l1_buffer(mesh_device, work_size);
    auto sink_l1_buffer = create_l1_buffer(mesh_device, kSinkBufferBytes);
    auto sink_dram_buffer = create_dram_buffer(mesh_device, kSinkBufferBytes);
    std::shared_ptr<distributed::MeshBuffer> input_dram_buffer;
    std::shared_ptr<distributed::MeshBuffer> output_dram_buffer;
    if (is_dram_mode(mode)) {
        const uint32_t payload_bytes = payload_dram_size_bytes(options);
        input_dram_buffer = create_dram_buffer(mesh_device, payload_bytes, options.page_size);
        output_dram_buffer = create_dram_buffer(mesh_device, payload_bytes, options.page_size);
        initialize_dram_payload(cq, input_dram_buffer, output_dram_buffer, options);
    }

    if (uses_static_work_l1(mode)) {
        zero_l1_buffer(cq, work_buffer, work_size);
    }

    const uint32_t input_dram_addr = input_dram_buffer ? input_dram_buffer->address() : 0;
    const uint32_t output_dram_addr = output_dram_buffer ? output_dram_buffer->address() : 0;

    auto make_runtime_args = [&](uint32_t sink_offset) {
        return std::vector<uint32_t>{
            work_buffer->address(),
            sink_l1_buffer->address() + sink_offset,
            sink_dram_buffer->address(),
            sink_offset,
            options.page_size,
            options.num_pages,
            produced_sem_id,
            consumed_sem_id,
            input_dram_addr,
            output_dram_addr};
    };

    auto make_defines = [&](BenchRole role, uint32_t sink_offset) {
        return kernel_defines(
            options,
            mode,
            role,
            repeat,
            work_buffer->address(),
            sink_l1_buffer->address() + sink_offset,
            sink_dram_buffer->address(),
            sink_offset,
            input_dram_addr,
            output_dram_addr,
            produced_sem_id,
            consumed_sem_id);
    };

    auto set_args_if_needed = [&](KernelHandle kernel, uint32_t sink_offset) {
        if (uses_compile_time_addresses(mode)) {
            SetRuntimeArgs(program, kernel, options.core, {});
        } else {
            SetRuntimeArgs(program, kernel, options.core, make_runtime_args(sink_offset));
        }
    };

    if (is_cross_mode(mode)) {
        KernelHandle producer = CreateKernel(
            program,
            "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/microbench/cb_protocol_overhead/kernels/cb_protocol_bench.cpp",
            options.core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = make_defines(BenchRole::Producer, 0)});

        KernelHandle consumer = CreateKernel(
            program,
            "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/microbench/cb_protocol_overhead/kernels/cb_protocol_bench.cpp",
            options.core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .defines = make_defines(BenchRole::Consumer, kConsumerSinkOffsetBytes)});

        set_args_if_needed(producer, 0);
        set_args_if_needed(consumer, kConsumerSinkOffsetBytes);
    } else {
        KernelHandle kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/microbench/cb_protocol_overhead/kernels/cb_protocol_bench.cpp",
            options.core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = make_defines(BenchRole::Producer, 0)});

        set_args_if_needed(kernel, 0);
    }

    if (options.num_rv == 5 && (is_system_mode(mode) || is_dram_mode(mode))) {
        KernelHandle aux_compute = CreateKernel(
            program,
            "tt_metal/programming_examples/compiler_managed_l1_dataflow/profiler/microbench/cb_protocol_overhead/kernels/aux_compute_empty.cpp",
            options.core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .defines = aux_compute_defines(options, mode)});
        SetRuntimeArgs(program, aux_compute, options.core, {});
    }

    workload.add_program(device_range, std::move(program));

    const auto start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    const auto end = std::chrono::steady_clock::now();

    std::vector<uint32_t> readback;
    distributed::EnqueueReadMeshBuffer(cq, readback, sink_dram_buffer, true);
    const uint32_t producer_sink = readback.empty() ? 0 : readback[0];
    const size_t consumer_sink_index = kConsumerSinkOffsetBytes / sizeof(uint32_t);
    const uint32_t consumer_sink =
        is_cross_mode(mode) && readback.size() > consumer_sink_index ? readback[consumer_sink_index] : 0;

    return {
        .mode = mode,
        .repeat = repeat,
        .enqueue_finish_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
        .producer_sink = producer_sink,
        .consumer_sink = consumer_sink};
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;

    try {
        const Options options = parse_options(argc, argv);
        const std::vector<BenchMode> modes = modes_to_run(options.mode);
        if (options.num_rv == 1 &&
            std::any_of(modes.begin(), modes.end(), [](BenchMode mode) { return is_cross_mode(mode); })) {
            throw std::invalid_argument("--num-rv=1 is only valid for single-RISC modes such as static-no-sync");
        }
        if (options.iterations > kMaxCbApiIterations &&
            std::find(modes.begin(), modes.end(), BenchMode::CbApiRoundtrip) != modes.end()) {
            throw std::invalid_argument(fmt::format(
                "--iterations is capped at {} when cb-api-roundtrip is enabled to avoid 16-bit CB counter wrap",
                kMaxCbApiIterations));
        }

        if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
            fmt::print(
                "WARNING: set TT_METAL_DEVICE_PROFILER=1 to collect device cycle zones in "
                "generated/profiler/.logs/profile_log_device.csv\n");
        }
        if (std::getenv("TT_METAL_DPRINT_CORES") != nullptr || std::getenv("TT_METAL_WATCHER") != nullptr) {
            fmt::print(
                "WARNING: device profiler, DPRINT, and watcher share L1 space; disable DPRINT/WATCHER for clean "
                "profiling.\n");
        }

        std::shared_ptr<distributed::MeshDevice> mesh_device =
            distributed::MeshDevice::create_unit_mesh(options.device_id);

        const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
        if (options.core.x >= grid.x || options.core.y >= grid.y) {
            throw std::invalid_argument(fmt::format(
                "Core ({}, {}) is outside compute_with_storage_grid_size ({}, {})",
                options.core.x,
                options.core.y,
                grid.x,
                grid.y));
        }

        fmt::print(
            "cb_protocol_overhead: iterations={}, page_size={}, num_pages={}, num_cbs={}, num_rv={}, repeats={}, "
            "core=({}, {})\n",
            options.iterations,
            options.page_size,
            options.num_pages,
            options.num_cbs,
            options.num_rv,
            options.repeats,
            options.core.x,
            options.core.y);

        for (uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
            for (BenchMode mode : modes) {
                RunResult result = run_one(mesh_device, options, mode, repeat);
                fmt::print(
                    "mode={:<30} repeat={} enqueue_finish_us={} cb_config_bytes_est={} producer_sink=0x{:08x} "
                    "consumer_sink=0x{:08x}\n",
                    mode_name(result.mode),
                    result.repeat,
                    result.enqueue_finish_us,
                    uses_cb(result.mode)
                        ? (((result.mode == BenchMode::CbSystem || result.mode == BenchMode::DramCb)
                                ? options.num_cbs
                                : 1) *
                           4 * sizeof(uint32_t))
                        : 0,
                    result.producer_sink,
                    result.consumer_sink);
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
