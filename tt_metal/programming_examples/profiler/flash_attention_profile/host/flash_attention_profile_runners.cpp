// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/trace.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/types.hpp"

#include <flash_attention_profile_runner.hpp>

#include "copied_sdpa/device/sdpa_device_operation.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace distributed = tt::tt_metal::distributed;

namespace flash_attention_profile {
namespace {

using Clock = std::chrono::steady_clock;
using SDPAPipelineMode = ttnn::prim::flash_attention_profile_sdpa::SDPAPipelineMode;

template <typename Func>
uint64_t time_us(Func&& func) {
    const auto start = Clock::now();
    func();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
}

std::size_t q_index(const ShapeConfig& shape, uint32_t b, uint32_t h, uint32_t s, uint32_t d) {
    return (((static_cast<std::size_t>(b) * shape.nh + h) * shape.s + s) * shape.d + d);
}

std::vector<float> q_window_values(const HostInputs& inputs, const ShapeConfig& shape, uint32_t start, uint32_t length) {
    std::vector<float> window(static_cast<std::size_t>(shape.b) * shape.nh * length * shape.d);
    for (uint32_t b = 0; b < shape.b; ++b) {
        for (uint32_t h = 0; h < shape.nh; ++h) {
            for (uint32_t s = 0; s < length; ++s) {
                for (uint32_t d = 0; d < shape.d; ++d) {
                    const auto out_idx =
                        (((static_cast<std::size_t>(b) * shape.nh + h) * length + s) * shape.d + d);
                    window[out_idx] = inputs.q[q_index(shape, b, h, start + s, d)];
                }
            }
        }
    }
    return window;
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

ttnn::Tensor make_host_tensor_float(const std::vector<float>& values, const ttnn::Shape& shape, DataType dtype, Layout layout) {
    return ttnn::Tensor::from_vector(values, tensor_spec(shape, dtype, layout));
}

ttnn::Tensor make_device_tensor_i32(
    const std::vector<int32_t>& values,
    const ttnn::Shape& shape,
    Layout layout,
    distributed::MeshDevice* device) {
    return ttnn::Tensor::from_vector(values, tensor_spec(shape, DataType::INT32, layout), device);
}

ttnn::Tensor make_host_tensor_i32(const std::vector<int32_t>& values, const ttnn::Shape& shape, Layout layout) {
    return ttnn::Tensor::from_vector(values, tensor_spec(shape, DataType::INT32, layout));
}

ttnn::operations::transformer::SDPAProgramConfig make_program_config(
    distributed::MeshDevice* device,
    const ShapeConfig& shape,
    Variant variant,
    GridPolicy grid_policy,
    std::optional<CoreCoord> grid_override) {
    const auto resolved_grid =
        resolve_flash_attention_grid(variant, grid_policy, grid_override, shape, device->compute_with_storage_grid_size());
    return ttnn::operations::transformer::SDPAProgramConfig{
        .compute_with_storage_grid_size = resolved_grid.value_or(device->compute_with_storage_grid_size()),
        .sub_core_grids = std::nullopt,
        .q_chunk_size = shape.q_chunk,
        .k_chunk_size = shape.k_chunk,
        .exp_approx_mode = true,
        .max_cores_per_head_batch = 16};
}

ttnn::DeviceComputeKernelConfig make_compute_kernel_config(distributed::MeshDevice* device, bool high_precision) {
    return ttnn::init_device_compute_kernel_config(
        device->arch(),
        /*device_kernel_config=*/std::nullopt,
        high_precision ? MathFidelity::HiFi4 : MathFidelity::HiFi2,
        /*default_approx_mode=*/!high_precision,
        /*default_fp32_acc=*/high_precision,
        /*default_l1_acc=*/false);
}

bool variant_selected(const std::vector<Variant>& selected_variants, Variant variant) {
    return std::find(selected_variants.begin(), selected_variants.end(), variant) != selected_variants.end();
}

SDPAPipelineMode to_copied_pipeline_mode(PipelineMode mode) {
    switch (mode) {
        case PipelineMode::Auto: return SDPAPipelineMode::Auto;
        case PipelineMode::StreamH1: return SDPAPipelineMode::StreamH1;
        case PipelineMode::QktvH1: return SDPAPipelineMode::QktvH1;
        case PipelineMode::SaladFirst: return SDPAPipelineMode::SaladFirst;
        case PipelineMode::QktvH1SaladFirst: return SDPAPipelineMode::QktvH1SaladFirst;
        case PipelineMode::NonStreaming: return SDPAPipelineMode::NonStreaming;
    }
    return SDPAPipelineMode::Auto;
}

ttnn::Tensor copied_scaled_dot_product_attention(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options);

ttnn::Tensor copied_chunked_scaled_dot_product_attention(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& page_table,
    const ttnn::Tensor& chunk_start,
    const MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options);

std::vector<float> run_variant_for_correctness(
    Variant variant,
    const ShapeConfig& shape,
    const HostInputs& inputs,
    bool high_precision,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options,
    GridPolicy grid_policy,
    std::optional<CoreCoord> grid_override) {
    distributed::MeshDevice* device = mesh_device.get();
    auto program_config = make_program_config(device, shape, variant, grid_policy, grid_override);
    auto compute_kernel_config = make_compute_kernel_config(device, high_precision);
    const bool chunked = variant_is_chunked(variant);
    const uint32_t q_length = chunked ? shape.prefill : shape.s;
    const auto q_shape = ttnn::Shape{shape.b, shape.nh, q_length, shape.d};
    const auto kv_shape = chunked ? ttnn::Shape{shape.b * (shape.s / shape.page), shape.nkv, shape.page, shape.d}
                                  : ttnn::Shape{shape.b, shape.nkv, shape.s, shape.d};
    const auto page_table_shape = ttnn::Shape{shape.b, shape.s / shape.page};

    auto q = make_device_tensor_float(q_window_values(inputs, shape, 0, q_length), q_shape, DataType::BFLOAT16, Layout::TILE, device);
    auto k = make_device_tensor_float(
        chunked ? inputs.paged_k : inputs.k, kv_shape, DataType::BFLOAT8_B, Layout::TILE, device);
    auto v = make_device_tensor_float(
        chunked ? inputs.paged_v : inputs.v, kv_shape, DataType::BFLOAT8_B, Layout::TILE, device);
    auto page_table = make_device_tensor_i32(inputs.page_table, page_table_shape, Layout::ROW_MAJOR, device);
    auto chunk_start = make_device_tensor_i32({0}, ttnn::Shape{1}, Layout::ROW_MAJOR, device);

    ttnn::Tensor output;
    if (variant == Variant::TtnnSdpaBaseline) {
        output = ttnn::transformer::scaled_dot_product_attention(
            q,
            k,
            v,
            std::nullopt,
            /*is_causal=*/true,
            std::nullopt,
            std::nullopt,
            ttnn::DRAM_MEMORY_CONFIG,
            program_config,
            compute_kernel_config);
    } else if (variant == Variant::TtnnChunkedBaseline) {
        output = ttnn::transformer::chunked_scaled_dot_product_attention(
            q,
            k,
            v,
            page_table,
            chunk_start,
            std::nullopt,
            ttnn::DRAM_MEMORY_CONFIG,
            program_config,
            compute_kernel_config);
    } else if (variant == Variant::CopiedSdpa) {
        output = copied_scaled_dot_product_attention(
            q,
            k,
            v,
            ttnn::DRAM_MEMORY_CONFIG,
            program_config,
            compute_kernel_config,
            pipeline_mode,
            pipeline_depth,
            copied_kernel_options);
    } else {
        output = copied_chunked_scaled_dot_product_attention(
            q,
            k,
            v,
            page_table,
            chunk_start,
            ttnn::DRAM_MEMORY_CONFIG,
            program_config,
            compute_kernel_config,
            pipeline_mode,
            pipeline_depth,
            copied_kernel_options);
    }

    distributed::Synchronize(device, std::nullopt);
    std::vector<float> values = output.to_vector<float>();
    output.deallocate();
    q.deallocate();
    k.deallocate();
    v.deallocate();
    page_table.deallocate();
    chunk_start.deallocate();
    return values;
}

ttnn::Tensor copied_scaled_dot_product_attention(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options) {
    return ttnn::prim::flash_attention_profile_sdpa::sdpa(
        q,
        k,
        v,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        /*is_causal=*/true,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        /*use_mla=*/false,
        std::nullopt,
        output_mem_config,
        std::move(program_config),
        compute_kernel_config,
        to_copied_pipeline_mode(pipeline_mode),
        pipeline_depth,
        copied_kernel_options.qk_subblock_override,
        copied_kernel_options.q_buffer_factor_override,
        copied_kernel_options.dst_full_sync_override,
        copied_kernel_options.qk_softmax_profile_stage,
        copied_kernel_options.qk_softmax_schedule);
}

ttnn::Tensor copied_chunked_scaled_dot_product_attention(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& page_table,
    const ttnn::Tensor& chunk_start,
    const MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options) {
    return ttnn::prim::flash_attention_profile_sdpa::sdpa(
        q,
        k,
        v,
        std::nullopt,
        page_table,
        std::nullopt,
        /*is_causal=*/true,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        chunk_start,
        /*use_mla=*/false,
        std::nullopt,
        output_mem_config,
        std::move(program_config),
        compute_kernel_config,
        to_copied_pipeline_mode(pipeline_mode),
        pipeline_depth,
        copied_kernel_options.qk_subblock_override,
        copied_kernel_options.q_buffer_factor_override,
        copied_kernel_options.dst_full_sync_override,
        copied_kernel_options.qk_softmax_profile_stage,
        copied_kernel_options.qk_softmax_schedule);
}

class FlashAttentionRunner : public FlashAttentionProfileRunner {
public:
    FlashAttentionRunner(
        Variant variant,
        RunMode mode,
        PipelineMode pipeline_mode,
        uint32_t pipeline_depth,
        CopiedKernelOptions copied_kernel_options,
        GridPolicy grid_policy,
        std::optional<CoreCoord> grid_override,
        ShapeConfig shape,
        const HostInputs& inputs,
        bool high_precision,
        std::shared_ptr<distributed::MeshDevice> mesh_device) :
        variant_(variant),
        mode_(mode),
        pipeline_mode_(pipeline_mode),
        pipeline_depth_(pipeline_depth),
        copied_kernel_options_(copied_kernel_options),
        shape_(std::move(shape)),
        inputs_(inputs),
        mesh_device_(std::move(mesh_device)),
        dev_ptr_(mesh_device_.get()),
        program_config_(make_program_config(dev_ptr_, shape_, variant_, grid_policy, grid_override)),
        compute_kernel_config_(make_compute_kernel_config(dev_ptr_, high_precision)) {
        create_host_runtime_tensors();
        create_static_tensors();
        if (mode_ != RunMode::Eager) {
            create_q_and_start_tensors(0);
        }
    }

    ~FlashAttentionRunner() override {
        try {
            release_trace();
        } catch (...) {
        }
    }

    IterationStats run(uint32_t iter_index, bool include_input_copy) override {
        if (trace_ready()) {
            return run_trace(iter_index, include_input_copy);
        }
        if (mode_ == RunMode::Eager) {
            return run_eager(iter_index);
        }
        return run_prepared(iter_index, include_input_copy);
    }

    void prepare_trace(bool drain_profiler) override {
        if (!variant_is_chunked(variant_)) {
            TT_THROW("trace mode is only supported for chunked variants because full SDPA has no runtime chunk-start tensor");
        }
        if (trace_ready()) {
            return;
        }
        if (mode_ == RunMode::Eager) {
            TT_THROW("trace capture requires prepared device tensors");
        }

        (void)run_prepared(0, false);
        if (drain_profiler) {
            ReadMeshDeviceProfilerResults(*mesh_device_);
        }

        auto trace_id = ttnn::operations::trace::begin_trace_capture(dev_ptr_, std::nullopt);
        ttnn::Tensor output = call_chunked_with_tensor_start();
        output.deallocate();
        ttnn::operations::trace::end_trace_capture(dev_ptr_, trace_id, std::nullopt);
        distributed::Synchronize(dev_ptr_, std::nullopt);
        trace_id_ = trace_id;

        if (drain_profiler) {
            ReadMeshDeviceProfilerResults(*mesh_device_);
        }
    }

    bool trace_ready() const override { return trace_id_.has_value(); }

    void release_trace() override {
        if (!trace_id_.has_value()) {
            return;
        }
        ttnn::operations::trace::release_trace(dev_ptr_, *trace_id_);
        trace_id_.reset();
    }

private:
    uint32_t q_start_for_iter(uint32_t iter_index) const {
        if (mode_ == RunMode::PreparedNoQCopy) {
            return 0;
        }
        if (!variant_is_chunked(variant_)) {
            return 0;
        }
        const uint32_t chunks = shape_.s / shape_.prefill;
        return (iter_index % chunks) * shape_.prefill;
    }

    std::size_t q_window_slot(uint32_t q_start) const {
        if (!variant_is_chunked(variant_)) {
            return 0;
        }
        return q_start / shape_.prefill;
    }

    void create_host_runtime_tensors() {
        const uint32_t chunks = variant_is_chunked(variant_) ? shape_.s / shape_.prefill : 1;
        q_window_values_.reserve(chunks);
        q_host_tensors_.reserve(chunks);
        chunk_start_host_tensors_.reserve(chunks);
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            const uint32_t q_start = variant_is_chunked(variant_) ? chunk * shape_.prefill : 0;
            q_window_values_.push_back(q_window_values(inputs_, shape_, q_start, q_length()));
            q_host_tensors_.push_back(
                make_host_tensor_float(q_window_values_.back(), q_shape(), DataType::BFLOAT16, Layout::TILE));
            chunk_start_host_tensors_.push_back(
                make_host_tensor_i32({static_cast<int32_t>(q_start)}, ttnn::Shape{1}, Layout::ROW_MAJOR));
        }
    }

    void create_static_tensors() {
        const auto kv_shape = !variant_is_chunked(variant_)
                                  ? ttnn::Shape{shape_.b, shape_.nkv, shape_.s, shape_.d}
                                  : ttnn::Shape{shape_.b * (shape_.s / shape_.page), shape_.nkv, shape_.page, shape_.d};
        const auto page_table_shape = ttnn::Shape{shape_.b, shape_.s / shape_.page};
        k_device_ = make_device_tensor_float(
            !variant_is_chunked(variant_) ? inputs_.k : inputs_.paged_k, kv_shape, DataType::BFLOAT8_B, Layout::TILE, dev_ptr_);
        v_device_ = make_device_tensor_float(
            !variant_is_chunked(variant_) ? inputs_.v : inputs_.paged_v, kv_shape, DataType::BFLOAT8_B, Layout::TILE, dev_ptr_);
        page_table_device_ = make_device_tensor_i32(inputs_.page_table, page_table_shape, Layout::ROW_MAJOR, dev_ptr_);
    }

    uint32_t q_length() const { return variant_is_chunked(variant_) ? shape_.prefill : shape_.s; }

    ttnn::Shape q_shape() const { return ttnn::Shape{shape_.b, shape_.nh, q_length(), shape_.d}; }

    void create_q_and_start_tensors(uint32_t q_start) {
        const auto slot = q_window_slot(q_start);
        q_device_ =
            make_device_tensor_float(q_window_values_[slot], q_shape(), DataType::BFLOAT16, Layout::TILE, dev_ptr_);
        chunk_start_device_ =
            make_device_tensor_i32({static_cast<int32_t>(q_start)}, ttnn::Shape{1}, Layout::ROW_MAJOR, dev_ptr_);
    }

    IterationStats copy_runtime_inputs(uint32_t q_start, bool include_input_copy) {
        IterationStats stats;
        if (!include_input_copy || mode_ == RunMode::PreparedNoQCopy) {
            return stats;
        }
        const auto slot = q_window_slot(q_start);
        stats.copy_q_us = time_us([&]() { tt::tt_metal::copy_to_device(q_host_tensors_[slot], q_device_); });
        if (variant_is_chunked(variant_)) {
            stats.copy_start_us =
                time_us([&]() { tt::tt_metal::copy_to_device(chunk_start_host_tensors_[slot], chunk_start_device_); });
        }
        return stats;
    }

    ttnn::Tensor call_full_sdpa() {
        if (variant_ == Variant::CopiedSdpa) {
            return copied_scaled_dot_product_attention(
                q_device_,
                k_device_,
                v_device_,
                ttnn::DRAM_MEMORY_CONFIG,
                program_config_,
                compute_kernel_config_,
                pipeline_mode_,
                pipeline_depth_,
                copied_kernel_options_);
        }
        return ttnn::transformer::scaled_dot_product_attention(
            q_device_,
            k_device_,
            v_device_,
            std::nullopt,
            /*is_causal=*/true,
            std::nullopt,
            std::nullopt,
            ttnn::DRAM_MEMORY_CONFIG,
            program_config_,
            compute_kernel_config_);
    }

    ttnn::Tensor call_chunked_with_tensor_start() {
        if (variant_ == Variant::CopiedChunked) {
            return copied_chunked_scaled_dot_product_attention(
                q_device_,
                k_device_,
                v_device_,
                page_table_device_,
                chunk_start_device_,
                ttnn::DRAM_MEMORY_CONFIG,
                program_config_,
                compute_kernel_config_,
                pipeline_mode_,
                pipeline_depth_,
                copied_kernel_options_);
        }
        return ttnn::transformer::chunked_scaled_dot_product_attention(
            q_device_,
            k_device_,
            v_device_,
            page_table_device_,
            chunk_start_device_,
            std::nullopt,
            ttnn::DRAM_MEMORY_CONFIG,
            program_config_,
            compute_kernel_config_);
    }

    ttnn::Tensor call_variant(uint32_t q_start) {
        if (!variant_is_chunked(variant_)) {
            return call_full_sdpa();
        }
        (void)q_start;
        return call_chunked_with_tensor_start();
    }

    IterationStats run_eager(uint32_t iter_index) {
        const auto start = Clock::now();
        const uint32_t q_start = q_start_for_iter(iter_index);
        create_q_and_start_tensors(q_start);
        IterationStats stats;
        ttnn::Tensor output;
        stats.call_us = time_us([&]() { output = call_variant(q_start); });
        stats.sync_us = time_us([&]() { distributed::Synchronize(dev_ptr_, std::nullopt); });
        stats.cache_entries = dev_ptr_->num_program_cache_entries();
        output.deallocate();
        q_device_.deallocate();
        chunk_start_device_.deallocate();
        stats.total_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
        return stats;
    }

    IterationStats run_prepared(uint32_t iter_index, bool include_input_copy) {
        const auto start = Clock::now();
        const uint32_t q_start = q_start_for_iter(iter_index);
        IterationStats stats = copy_runtime_inputs(q_start, include_input_copy);
        ttnn::Tensor output;
        stats.call_us = time_us([&]() { output = call_variant(q_start); });
        stats.sync_us = time_us([&]() { distributed::Synchronize(dev_ptr_, std::nullopt); });
        stats.cache_entries = dev_ptr_->num_program_cache_entries();
        output.deallocate();
        stats.total_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
        return stats;
    }

    IterationStats run_trace(uint32_t iter_index, bool include_input_copy) {
        const auto start = Clock::now();
        const uint32_t q_start = q_start_for_iter(iter_index);
        IterationStats stats = copy_runtime_inputs(q_start, include_input_copy);
        stats.call_us = time_us([&]() { ttnn::operations::trace::execute_trace(dev_ptr_, *trace_id_, std::nullopt, false); });
        stats.sync_us = time_us([&]() { distributed::Synchronize(dev_ptr_, std::nullopt); });
        stats.cache_entries = dev_ptr_->num_program_cache_entries();
        stats.total_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
        return stats;
    }

    Variant variant_;
    RunMode mode_;
    PipelineMode pipeline_mode_;
    uint32_t pipeline_depth_ = 2;
    CopiedKernelOptions copied_kernel_options_;
    ShapeConfig shape_;
    const HostInputs& inputs_;
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    distributed::MeshDevice* dev_ptr_ = nullptr;
    ttnn::operations::transformer::SDPAProgramConfig program_config_;
    ttnn::DeviceComputeKernelConfig compute_kernel_config_;
    ttnn::Tensor q_device_;
    ttnn::Tensor k_device_;
    ttnn::Tensor v_device_;
    ttnn::Tensor page_table_device_;
    ttnn::Tensor chunk_start_device_;
    std::vector<std::vector<float>> q_window_values_;
    std::vector<ttnn::Tensor> q_host_tensors_;
    std::vector<ttnn::Tensor> chunk_start_host_tensors_;
    std::optional<ttnn::MeshTraceId> trace_id_;
};

}  // namespace

const char* variant_name(Variant variant) {
    switch (variant) {
        case Variant::TtnnSdpaBaseline: return "ttnn_sdpa_baseline";
        case Variant::TtnnChunkedBaseline: return "ttnn_chunked_baseline";
        case Variant::CopiedSdpa: return "copied_sdpa";
        case Variant::CopiedChunked: return "copied_chunked";
    }
    return "unknown";
}

const char* run_mode_name(RunMode mode) {
    switch (mode) {
        case RunMode::Eager: return "eager";
        case RunMode::Prepared: return "prepared";
        case RunMode::PreparedNoQCopy: return "prepared_no_q_copy";
        case RunMode::Trace: return "trace";
    }
    return "unknown";
}

const char* pipeline_mode_name(PipelineMode mode) {
    switch (mode) {
        case PipelineMode::Auto: return "auto";
        case PipelineMode::StreamH1: return "stream_h1";
        case PipelineMode::QktvH1: return "qktv_h1";
        case PipelineMode::SaladFirst: return "salad_first";
        case PipelineMode::QktvH1SaladFirst: return "qktv_h1_salad_first";
        case PipelineMode::NonStreaming: return "non_streaming";
    }
    return "unknown";
}

const char* grid_policy_name(GridPolicy policy) {
    switch (policy) {
        case GridPolicy::Default: return "default";
        case GridPolicy::CopiedBalancedQ: return "copied_balanced_q";
    }
    return "unknown";
}

bool variant_is_chunked(Variant variant) {
    return variant == Variant::TtnnChunkedBaseline || variant == Variant::CopiedChunked;
}

bool variant_is_copied(Variant variant) {
    return variant == Variant::CopiedSdpa || variant == Variant::CopiedChunked;
}

namespace {

uint32_t largest_divisor_at_most(uint32_t value, uint32_t limit) {
    uint32_t candidate = std::min(value, limit);
    while (candidate > 1 && value % candidate != 0) {
        --candidate;
    }
    return candidate;
}

std::optional<CoreCoord> rectangular_grid_with_area(uint32_t area, CoreCoord device_grid) {
    CoreCoord best{0, 0};
    uint32_t best_perimeter = std::numeric_limits<uint32_t>::max();
    for (uint32_t x = 1; x <= device_grid.x; ++x) {
        if (area % x != 0) {
            continue;
        }
        const uint32_t y = area / x;
        if (y == 0 || y > device_grid.y) {
            continue;
        }
        const uint32_t perimeter = x + y;
        if (perimeter < best_perimeter || (perimeter == best_perimeter && x > best.x)) {
            best = CoreCoord{x, y};
            best_perimeter = perimeter;
        }
    }
    if (best.x == 0 || best.y == 0) {
        return std::nullopt;
    }
    return best;
}

}  // namespace

std::optional<CoreCoord> resolve_flash_attention_grid(
    Variant variant,
    GridPolicy grid_policy,
    std::optional<CoreCoord> grid_override,
    const ShapeConfig& shape,
    CoreCoord device_grid) {
    if (grid_override.has_value()) {
        return grid_override;
    }
    if (grid_policy != GridPolicy::CopiedBalancedQ || !variant_is_copied(variant)) {
        return std::nullopt;
    }

    const uint32_t total_cores = device_grid.x * device_grid.y;
    const uint32_t batch_parallel_factor = std::min(shape.b, total_cores);
    if (batch_parallel_factor == 0) {
        return std::nullopt;
    }
    const uint32_t heads_per_batch_capacity = total_cores / batch_parallel_factor;
    const uint32_t nh_parallel_factor = std::min(heads_per_batch_capacity, shape.nh);
    if (nh_parallel_factor == 0) {
        return std::nullopt;
    }

    const uint32_t q_length = variant_is_chunked(variant) ? shape.prefill : shape.s;
    const uint32_t q_num_chunks = (q_length + shape.q_chunk - 1) / shape.q_chunk;
    const uint32_t max_q_parallel = std::min(total_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks);
    if (max_q_parallel == 0) {
        return std::nullopt;
    }
    if (q_num_chunks % max_q_parallel == 0) {
        return std::nullopt;
    }

    const uint32_t q_parallel_factor = largest_divisor_at_most(q_num_chunks, max_q_parallel);
    const uint32_t default_q_per_core = (q_num_chunks + max_q_parallel - 1) / max_q_parallel;
    const uint32_t balanced_q_per_core = q_num_chunks / q_parallel_factor;
    if (balanced_q_per_core > default_q_per_core) {
        return std::nullopt;
    }

    const uint32_t active_cores = batch_parallel_factor * nh_parallel_factor * q_parallel_factor;
    auto balanced_grid = rectangular_grid_with_area(active_cores, device_grid);
    if (!balanced_grid.has_value()) {
        return std::nullopt;
    }
    if (balanced_grid->x == device_grid.x && balanced_grid->y == device_grid.y) {
        return std::nullopt;
    }
    return balanced_grid;
}

std::unique_ptr<FlashAttentionProfileRunner> prepare_flash_attention_runner(
    Variant variant,
    RunMode mode,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options,
    GridPolicy grid_policy,
    std::optional<CoreCoord> grid_override,
    const ShapeConfig& shape,
    const HostInputs& inputs,
    bool high_precision,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    return std::make_unique<FlashAttentionRunner>(
        variant,
        mode,
        pipeline_mode,
        pipeline_depth,
        copied_kernel_options,
        grid_policy,
        grid_override,
        shape,
        inputs,
        high_precision,
        mesh_device);
}

std::vector<CorrectnessResult> check_flash_attention_correctness(
    const ShapeConfig& shape,
    const HostInputs& inputs,
    bool high_precision,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<Variant>& selected_variants,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options,
    GridPolicy grid_policy,
    std::optional<CoreCoord> grid_override) {
    std::vector<CorrectnessResult> results;
    constexpr double tolerance = 0.125;

    auto compare_pair = [&](Variant baseline_variant, Variant candidate_variant) {
        if (!variant_selected(selected_variants, baseline_variant) || !variant_selected(selected_variants, candidate_variant)) {
            return;
        }
        auto baseline =
            run_variant_for_correctness(
                baseline_variant,
                shape,
                inputs,
                high_precision,
                mesh_device,
                PipelineMode::Auto,
                2,
                CopiedKernelOptions{},
                grid_policy,
                grid_override);
        auto candidate =
            run_variant_for_correctness(
                candidate_variant,
                shape,
                inputs,
                high_precision,
                mesh_device,
                pipeline_mode,
                pipeline_depth,
                copied_kernel_options,
                grid_policy,
                grid_override);
        TT_FATAL(baseline.size() == candidate.size(), "baseline and copied outputs have different element counts");

        double max_abs_diff = 0.0;
        double sum_abs_diff = 0.0;
        for (std::size_t i = 0; i < baseline.size(); ++i) {
            const double diff = std::abs(static_cast<double>(baseline[i]) - static_cast<double>(candidate[i]));
            max_abs_diff = std::max(max_abs_diff, diff);
            sum_abs_diff += diff;
        }
        const double mean_abs_diff = baseline.empty() ? 0.0 : sum_abs_diff / static_cast<double>(baseline.size());
        results.push_back(CorrectnessResult{
            .baseline = baseline_variant,
            .candidate = candidate_variant,
            .elements = baseline.size(),
            .max_abs_diff = max_abs_diff,
            .mean_abs_diff = mean_abs_diff,
            .tolerance = tolerance,
            .passed = max_abs_diff <= tolerance});
    };

    compare_pair(Variant::TtnnSdpaBaseline, Variant::CopiedSdpa);
    compare_pair(Variant::TtnnChunkedBaseline, Variant::CopiedChunked);
    return results;
}

}  // namespace flash_attention_profile
