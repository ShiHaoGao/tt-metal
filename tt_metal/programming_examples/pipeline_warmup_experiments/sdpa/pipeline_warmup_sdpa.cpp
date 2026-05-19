// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/types.hpp"

#include "tt_metal/programming_examples/pipeline_warmup_experiments/sdpa/instrumented/device/instrumented_sdpa_device_operation.hpp"
#include "tt_metal/programming_examples/pipeline_warmup_experiments/sdpa/instrumented/instrumentation.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace distributed = tt::tt_metal::distributed;

namespace {

enum class Mode {
    ScalarStart,
    TensorStart,
};

struct ShapeConfig {
    uint32_t b = 1;
    uint32_t nh = 1;
    uint32_t nkv = 1;
    uint32_t s = 1024;
    uint32_t d = 128;
    uint32_t q_chunk = 128;
    uint32_t k_chunk = 128;
    uint32_t prefill = 256;
    uint32_t page = 128;
};

struct Options {
    std::string preset = "smoke";
    std::string mode = "all";
    uint32_t warmup = 1;
    uint32_t device_id = 0;
    bool high_precision = false;
    bool clear_cache_between_modes = true;
};

struct HostInputs {
    std::vector<float> q;
    std::vector<float> paged_k;
    std::vector<float> paged_v;
    std::vector<int32_t> page_table;
};

struct ChunkStats {
    uint32_t chunk_idx = 0;
    uint32_t chunk_start = 0;
    uint64_t copy_q_us = 0;
    uint64_t copy_start_us = 0;
    uint64_t op_call_us = 0;
    uint64_t sync_us = 0;
    uint64_t total_us = 0;
    std::size_t cache_entries = 0;
    ttnn::prim::sdpa_instrumentation::Snapshot instrumentation;
};

uint64_t elapsed_us(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

template <typename Func>
uint64_t time_us(Func&& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    return elapsed_us(start, std::chrono::steady_clock::now());
}

ShapeConfig preset_shape(std::string_view name) {
    if (name == "smoke") {
        return ShapeConfig{.b = 1,
                           .nh = 1,
                           .nkv = 1,
                           .s = 1024,
                           .d = 128,
                           .q_chunk = 128,
                           .k_chunk = 128,
                           .prefill = 256,
                           .page = 128};
    }
    if (name == "llama2-70b") {
        return ShapeConfig{.b = 1,
                           .nh = 8,
                           .nkv = 1,
                           .s = 16 * 1024,
                           .d = 128,
                           .q_chunk = 256,
                           .k_chunk = 128,
                           .prefill = 2048,
                           .page = 128};
    }
    throw std::invalid_argument(fmt::format("unknown preset '{}'", name));
}

void validate_shape(const ShapeConfig& shape) {
    if (shape.b == 0 || shape.nh == 0 || shape.nkv == 0 || shape.s == 0 || shape.d == 0) {
        throw std::invalid_argument("shape dimensions must be non-zero");
    }
    if (shape.nh % shape.nkv != 0) {
        throw std::invalid_argument(fmt::format("nh must be divisible by nkv, got nh={} nkv={}", shape.nh, shape.nkv));
    }
    if (shape.s % shape.prefill != 0) {
        throw std::invalid_argument(fmt::format("s must be divisible by prefill, got s={} prefill={}", shape.s, shape.prefill));
    }
    if (shape.s % shape.page != 0) {
        throw std::invalid_argument(fmt::format("s must be divisible by page, got s={} page={}", shape.s, shape.page));
    }
    if (shape.prefill % shape.q_chunk != 0) {
        throw std::invalid_argument(
            fmt::format("prefill must be divisible by q_chunk, got prefill={} q={}", shape.prefill, shape.q_chunk));
    }
    if (shape.prefill % shape.k_chunk != 0) {
        throw std::invalid_argument(
            fmt::format("prefill must be divisible by k_chunk, got prefill={} k={}", shape.prefill, shape.k_chunk));
    }
    if (shape.q_chunk % tt::constants::TILE_HEIGHT != 0 || shape.k_chunk % tt::constants::TILE_HEIGHT != 0 ||
        shape.d % tt::constants::TILE_WIDTH != 0 || shape.page % tt::constants::TILE_HEIGHT != 0) {
        throw std::invalid_argument("q_chunk, k_chunk, d, and page must be tile aligned");
    }
}

std::string_view mode_name(Mode mode) {
    switch (mode) {
        case Mode::ScalarStart: return "scalar-start";
        case Mode::TensorStart: return "tensor-start";
    }
    return "unknown";
}

std::vector<Mode> modes_to_run(std::string_view mode) {
    if (mode == "all") {
        return {Mode::ScalarStart, Mode::TensorStart};
    }
    if (mode == "scalar-start" || mode == "scalar") {
        return {Mode::ScalarStart};
    }
    if (mode == "tensor-start" || mode == "tensor") {
        return {Mode::TensorStart};
    }
    throw std::invalid_argument(fmt::format("unknown mode '{}'", mode));
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto read_value = [&](std::string_view name) {
            const std::string prefix = fmt::format("--{}=", name);
            if (arg.rfind(prefix, 0) == 0) {
                return arg.substr(prefix.size());
            }
            if (arg == fmt::format("--{}", name)) {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(fmt::format("missing value for --{}", name));
                }
                return std::string(argv[++i]);
            }
            return std::string();
        };

        if (auto value = read_value("preset"); !value.empty()) {
            options.preset = value;
        } else if (auto value = read_value("mode"); !value.empty()) {
            options.mode = value;
        } else if (auto value = read_value("warmup"); !value.empty()) {
            options.warmup = static_cast<uint32_t>(std::stoul(value));
        } else if (auto value = read_value("device-id"); !value.empty()) {
            options.device_id = static_cast<uint32_t>(std::stoul(value));
        } else if (arg == "--high-precision") {
            options.high_precision = true;
        } else if (arg == "--keep-cache-between-modes") {
            options.clear_cache_between_modes = false;
        } else if (arg == "--help" || arg == "-h") {
            fmt::print(
                "Usage: pipeline_warmup_sdpa [--preset smoke|llama2-70b] "
                "[--mode all|scalar-start|tensor-start] [--warmup N] [--device-id N]\n");
            std::exit(0);
        } else {
            throw std::invalid_argument(fmt::format("unknown argument '{}'", arg));
        }
    }
    return options;
}

std::size_t q_index(const ShapeConfig& shape, uint32_t b, uint32_t h, uint32_t s, uint32_t d) {
    return (((static_cast<std::size_t>(b) * shape.nh + h) * shape.s + s) * shape.d + d);
}

std::size_t paged_kv_index(const ShapeConfig& shape, uint32_t block, uint32_t kvh, uint32_t page_offset, uint32_t d) {
    return (((static_cast<std::size_t>(block) * shape.nkv + kvh) * shape.page + page_offset) * shape.d + d);
}

float deterministic_value(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t salt) {
    uint32_t x = 0x9e3779b9u;
    x ^= a + 0x85ebca6bu + (x << 6) + (x >> 2);
    x ^= b + 0xc2b2ae35u + (x << 6) + (x >> 2);
    x ^= c + salt + (x << 6) + (x >> 2);
    x ^= d + 0x27d4eb2du + (x << 6) + (x >> 2);
    return (static_cast<float>(x % 2048u) / 1024.0f) - 1.0f;
}

HostInputs make_inputs(const ShapeConfig& shape) {
    const uint32_t blocks_per_seq = shape.s / shape.page;
    const uint32_t total_blocks = shape.b * blocks_per_seq;
    HostInputs inputs;
    inputs.q.resize(static_cast<std::size_t>(shape.b) * shape.nh * shape.s * shape.d);
    inputs.paged_k.resize(static_cast<std::size_t>(total_blocks) * shape.nkv * shape.page * shape.d);
    inputs.paged_v.resize(inputs.paged_k.size());
    inputs.page_table.resize(static_cast<std::size_t>(shape.b) * blocks_per_seq);

    for (uint32_t b = 0; b < shape.b; ++b) {
        for (uint32_t block = 0; block < blocks_per_seq; ++block) {
            const uint32_t physical_block = b * blocks_per_seq + block;
            inputs.page_table[static_cast<std::size_t>(b) * blocks_per_seq + block] =
                static_cast<int32_t>(physical_block);
        }
    }

    for (uint32_t b = 0; b < shape.b; ++b) {
        for (uint32_t h = 0; h < shape.nh; ++h) {
            for (uint32_t s = 0; s < shape.s; ++s) {
                for (uint32_t d = 0; d < shape.d; ++d) {
                    inputs.q[q_index(shape, b, h, s, d)] = deterministic_value(b, h, s, d, 17);
                }
            }
        }
    }

    for (uint32_t b = 0; b < shape.b; ++b) {
        for (uint32_t kvh = 0; kvh < shape.nkv; ++kvh) {
            for (uint32_t s = 0; s < shape.s; ++s) {
                const uint32_t virtual_block = s / shape.page;
                const uint32_t page_offset = s % shape.page;
                const uint32_t physical_block =
                    static_cast<uint32_t>(inputs.page_table[static_cast<std::size_t>(b) * blocks_per_seq + virtual_block]);
                for (uint32_t d = 0; d < shape.d; ++d) {
                    inputs.paged_k[paged_kv_index(shape, physical_block, kvh, page_offset, d)] =
                        deterministic_value(b, kvh, s, d, 101);
                    inputs.paged_v[paged_kv_index(shape, physical_block, kvh, page_offset, d)] =
                        deterministic_value(b, kvh, s, d, 211);
                }
            }
        }
    }
    return inputs;
}

ttnn::TensorSpec tensor_spec(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return ttnn::TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config));
}

ttnn::Tensor make_device_tensor_float(
    const std::vector<float>& values,
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    distributed::MeshDevice* device) {
    return ttnn::Tensor::from_vector(values, tensor_spec(shape, dtype, layout), device);
}

ttnn::Tensor make_device_tensor_i32(
    const std::vector<int32_t>& values, const ttnn::Shape& shape, Layout layout, distributed::MeshDevice* device) {
    return ttnn::Tensor::from_vector(values, tensor_spec(shape, DataType::INT32, layout), device);
}

std::vector<float> q_chunk_values(const HostInputs& inputs, const ShapeConfig& shape, uint32_t chunk_start) {
    std::vector<float> chunk(static_cast<std::size_t>(shape.b) * shape.nh * shape.prefill * shape.d);
    for (uint32_t b = 0; b < shape.b; ++b) {
        for (uint32_t h = 0; h < shape.nh; ++h) {
            for (uint32_t s = 0; s < shape.prefill; ++s) {
                for (uint32_t d = 0; d < shape.d; ++d) {
                    const std::size_t out_idx =
                        (((static_cast<std::size_t>(b) * shape.nh + h) * shape.prefill + s) * shape.d + d);
                    chunk[out_idx] = inputs.q[q_index(shape, b, h, chunk_start + s, d)];
                }
            }
        }
    }
    return chunk;
}

ttnn::DeviceComputeKernelConfig make_compute_kernel_config(
    distributed::MeshDevice* device, bool high_precision) {
    return ttnn::init_device_compute_kernel_config(
        device->arch(),
        /*device_kernel_config=*/std::nullopt,
        high_precision ? MathFidelity::HiFi4 : MathFidelity::HiFi2,
        /*default_approx_mode=*/!high_precision,
        /*default_fp32_acc=*/high_precision,
        /*default_l1_acc=*/false);
}

ttnn::operations::transformer::SDPAProgramConfig make_program_config(
    distributed::MeshDevice* device, const ShapeConfig& shape) {
    return ttnn::operations::transformer::SDPAProgramConfig{
        .compute_with_storage_grid_size = device->compute_with_storage_grid_size(),
        .sub_core_grids = std::nullopt,
        .q_chunk_size = shape.q_chunk,
        .k_chunk_size = shape.k_chunk,
        .exp_approx_mode = true,
        .max_cores_per_head_batch = 16};
}

ttnn::Tensor run_instrumented_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& page_table,
    std::optional<int64_t> scalar_start,
    const std::optional<ttnn::Tensor>& tensor_start,
    const MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    return ttnn::prim::instrumented_sdpa(
        q,
        k,
        v,
        std::nullopt,
        page_table,
        std::nullopt,
        /*is_causal=*/true,
        std::nullopt,
        std::nullopt,
        scalar_start,
        tensor_start,
        /*use_mla=*/false,
        std::nullopt,
        output_mem_config,
        std::move(program_config),
        compute_kernel_config);
}

ChunkStats run_one_chunk(
    Mode mode,
    uint32_t chunk_idx,
    const ShapeConfig& shape,
    const HostInputs& inputs,
    distributed::MeshDevice* device,
    ttnn::Tensor& q_device,
    ttnn::Tensor& chunk_start_device,
    const ttnn::Tensor& k_device,
    const ttnn::Tensor& v_device,
    const ttnn::Tensor& page_table_device,
    const MemoryConfig& output_mem_config,
    const ttnn::operations::transformer::SDPAProgramConfig& program_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    bool measure_copy) {
    const uint32_t chunk_start = chunk_idx * shape.prefill;
    ChunkStats stats;
    stats.chunk_idx = chunk_idx;
    stats.chunk_start = chunk_start;

    const auto total_start = std::chrono::steady_clock::now();

    if (measure_copy) {
        const auto host_q = ttnn::Tensor::from_vector(
            q_chunk_values(inputs, shape, chunk_start),
            tensor_spec(ttnn::Shape{shape.b, shape.nh, shape.prefill, shape.d}, DataType::BFLOAT16, Layout::TILE));
        stats.copy_q_us = time_us([&]() { tt::tt_metal::copy_to_device(host_q, q_device); });
    }

    std::optional<ttnn::Tensor> tensor_start = std::nullopt;
    std::optional<int64_t> scalar_start = std::nullopt;
    if (mode == Mode::TensorStart) {
        if (measure_copy) {
            const std::vector<int32_t> start_data = {static_cast<int32_t>(chunk_start)};
            const auto host_start =
                ttnn::Tensor::from_vector(start_data, tensor_spec(ttnn::Shape{1}, DataType::INT32, Layout::ROW_MAJOR));
            stats.copy_start_us = time_us([&]() { tt::tt_metal::copy_to_device(host_start, chunk_start_device); });
        }
        tensor_start = chunk_start_device;
    } else {
        scalar_start = static_cast<int64_t>(chunk_start);
    }

    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    ttnn::prim::sdpa_instrumentation::reset();
    ttnn::Tensor output;
    stats.op_call_us = time_us([&]() {
        output = run_instrumented_sdpa(
            q_device,
            k_device,
            v_device,
            page_table_device,
            scalar_start,
            tensor_start,
            output_mem_config,
            program_config,
            compute_kernel_config);
    });
    stats.sync_us = time_us([&]() { tt::tt_metal::distributed::Synchronize(device, std::nullopt); });
    stats.instrumentation = ttnn::prim::sdpa_instrumentation::snapshot();
    stats.cache_entries = device->num_program_cache_entries();
    stats.total_us = elapsed_us(total_start, std::chrono::steady_clock::now());

    output.deallocate();
    return stats;
}

void print_stage(const ttnn::prim::sdpa_instrumentation::StageStats& stage) {
    fmt::print("{},{:.3f},{:.3f}", stage.calls, stage.total_us, stage.max_us);
}

void print_csv_header() {
    fmt::print(
        "mode,chunk_idx,chunk_start,copy_q_us,copy_start_us,op_call_us,sync_us,total_us,cache_entries,"
        "hash_calls,hash_total_us,hash_max_us,"
        "create_output_calls,create_output_total_us,create_output_max_us,"
        "factory_create_calls,factory_create_total_us,factory_create_max_us,"
        "override_calls,override_total_us,override_max_us\n");
}

void print_csv_row(Mode mode, const ChunkStats& stats) {
    fmt::print(
        "{},{},{},{},{},{},{},{},{},",
        mode_name(mode),
        stats.chunk_idx,
        stats.chunk_start,
        stats.copy_q_us,
        stats.copy_start_us,
        stats.op_call_us,
        stats.sync_us,
        stats.total_us,
        stats.cache_entries);
    print_stage(stats.instrumentation.compute_program_hash);
    fmt::print(",");
    print_stage(stats.instrumentation.create_output_tensors);
    fmt::print(",");
    print_stage(stats.instrumentation.program_factory_create);
    fmt::print(",");
    print_stage(stats.instrumentation.override_runtime_arguments);
    fmt::print("\n");
}

void run_mode(
    Mode mode,
    const Options& options,
    const ShapeConfig& shape,
    const HostInputs& inputs,
    const std::shared_ptr<distributed::MeshDevice>& device) {
    if (options.clear_cache_between_modes) {
        device->disable_and_clear_program_cache();
        device->enable_program_cache();
    }

    auto* dev_ptr = device.get();
    const auto q_shape = ttnn::Shape{shape.b, shape.nh, shape.prefill, shape.d};
    const auto kv_shape = ttnn::Shape{shape.b * (shape.s / shape.page), shape.nkv, shape.page, shape.d};
    const auto page_table_shape = ttnn::Shape{shape.b, shape.s / shape.page};

    const auto first_q = q_chunk_values(inputs, shape, 0);
    auto q_device = make_device_tensor_float(first_q, q_shape, DataType::BFLOAT16, Layout::TILE, dev_ptr);
    auto k_device = make_device_tensor_float(inputs.paged_k, kv_shape, DataType::BFLOAT8_B, Layout::TILE, dev_ptr);
    auto v_device = make_device_tensor_float(inputs.paged_v, kv_shape, DataType::BFLOAT8_B, Layout::TILE, dev_ptr);
    auto page_table_device = make_device_tensor_i32(inputs.page_table, page_table_shape, Layout::ROW_MAJOR, dev_ptr);
    auto chunk_start_device = make_device_tensor_i32(std::vector<int32_t>{0}, ttnn::Shape{1}, Layout::ROW_MAJOR, dev_ptr);

    auto program_config = make_program_config(dev_ptr, shape);
    auto compute_kernel_config = make_compute_kernel_config(dev_ptr, options.high_precision);
    const MemoryConfig output_mem_config = ttnn::DRAM_MEMORY_CONFIG;
    const uint32_t num_chunks = shape.s / shape.prefill;

    for (uint32_t warmup_idx = 0; warmup_idx < options.warmup; ++warmup_idx) {
        (void)run_one_chunk(
            mode,
            0,
            shape,
            inputs,
            dev_ptr,
            q_device,
            chunk_start_device,
            k_device,
            v_device,
            page_table_device,
            output_mem_config,
            program_config,
            compute_kernel_config,
            /*measure_copy=*/false);
    }

    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const auto stats = run_one_chunk(
            mode,
            chunk_idx,
            shape,
            inputs,
            dev_ptr,
            q_device,
            chunk_start_device,
            k_device,
            v_device,
            page_table_device,
            output_mem_config,
            program_config,
            compute_kernel_config,
            /*measure_copy=*/true);
        print_csv_row(mode, stats);
    }

    q_device.deallocate();
    k_device.deallocate();
    v_device.deallocate();
    page_table_device.deallocate();
    chunk_start_device.deallocate();
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const Options options = parse_options(argc, argv);
        const ShapeConfig shape = preset_shape(options.preset);
        validate_shape(shape);
        const HostInputs inputs = make_inputs(shape);

        if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
            fmt::print(
                stderr,
                "WARNING: set TT_METAL_DEVICE_PROFILER=1 to collect device critical-path zones in "
                "generated/profiler/.logs/profile_log_device.csv\n");
        }

        auto device = distributed::MeshDevice::create_unit_mesh(options.device_id);
        device->enable_program_cache();

        fmt::print(
            "# SDPA C++ warmup profiler: preset={} B={} H={} KVH={} S={} D={} q={} k={} prefill={} page={} "
            "warmup={} high_precision={}\n",
            options.preset,
            shape.b,
            shape.nh,
            shape.nkv,
            shape.s,
            shape.d,
            shape.q_chunk,
            shape.k_chunk,
            shape.prefill,
            shape.page,
            options.warmup,
            options.high_precision ? "true" : "false");
        print_csv_header();
        for (Mode mode : modes_to_run(options.mode)) {
            run_mode(mode, options, shape, inputs, device);
        }

        ReadMeshDeviceProfilerResults(*device);
        fmt::print(stderr, "Profiler CSV: generated/profiler/.logs/profile_log_device.csv\n");
        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }

    if (!pass) {
        TT_THROW("Test Failed");
    }
    return 0;
}
