// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace flash_attention_profile {

enum class Variant {
    TtnnSdpaBaseline,
    TtnnChunkedBaseline,
    CopiedSdpa,
    CopiedChunked,
};

enum class RunMode {
    Eager,
    Prepared,
    PreparedNoQCopy,
    Trace,
};

enum class PipelineMode {
    Auto,
    StreamH1,
    QktvH1,
    SaladFirst,
    QktvH1SaladFirst,
    NonStreaming,
};

enum class GridPolicy {
    Default,
    CopiedBalancedQ,
};

struct CopiedKernelOptions {
    std::optional<std::pair<uint32_t, uint32_t>> qk_subblock_override = std::nullopt;
    std::optional<uint32_t> q_buffer_factor_override = std::nullopt;
    std::optional<bool> dst_full_sync_override = std::nullopt;
    uint32_t qk_softmax_profile_stage = 0;
    uint32_t qk_softmax_schedule = 0;
};

struct ShapeConfig {
    std::string name;
    uint32_t b = 1;
    uint32_t nh = 1;
    uint32_t nkv = 1;
    uint32_t s = 1024;
    uint32_t d = 128;
    uint32_t prefill = 256;
    uint32_t q_chunk = 128;
    uint32_t k_chunk = 128;
    uint32_t page = 128;
};

struct HostInputs {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> paged_k;
    std::vector<float> paged_v;
    std::vector<int32_t> page_table;
};

struct IterationStats {
    uint64_t copy_q_us = 0;
    uint64_t copy_start_us = 0;
    uint64_t call_us = 0;
    uint64_t sync_us = 0;
    uint64_t total_us = 0;
    std::size_t cache_entries = 0;
};

struct CorrectnessResult {
    Variant baseline;
    Variant candidate;
    std::size_t elements = 0;
    double max_abs_diff = 0.0;
    double mean_abs_diff = 0.0;
    double tolerance = 0.0;
    bool passed = false;
};

class FlashAttentionProfileRunner {
public:
    virtual ~FlashAttentionProfileRunner() = default;

    virtual IterationStats run(uint32_t iter_index, bool include_input_copy) = 0;
    virtual void prepare_trace(bool drain_profiler) = 0;
    virtual bool trace_ready() const = 0;
    virtual void release_trace() = 0;
};

const char* variant_name(Variant variant);
const char* run_mode_name(RunMode mode);
const char* pipeline_mode_name(PipelineMode mode);
const char* grid_policy_name(GridPolicy policy);
bool variant_is_chunked(Variant variant);
bool variant_is_copied(Variant variant);

std::optional<tt::tt_metal::CoreCoord> resolve_flash_attention_grid(
    Variant variant,
    GridPolicy grid_policy,
    std::optional<tt::tt_metal::CoreCoord> grid_override,
    const ShapeConfig& shape,
    tt::tt_metal::CoreCoord device_grid);

std::unique_ptr<FlashAttentionProfileRunner> prepare_flash_attention_runner(
    Variant variant,
    RunMode mode,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options,
    GridPolicy grid_policy,
    std::optional<tt::tt_metal::CoreCoord> grid_override,
    const ShapeConfig& shape,
    const HostInputs& inputs,
    bool high_precision,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device);

std::vector<CorrectnessResult> check_flash_attention_correctness(
    const ShapeConfig& shape,
    const HostInputs& inputs,
    bool high_precision,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<Variant>& selected_variants,
    PipelineMode pipeline_mode,
    uint32_t pipeline_depth,
    CopiedKernelOptions copied_kernel_options,
    GridPolicy grid_policy,
    std::optional<tt::tt_metal::CoreCoord> grid_override);

}  // namespace flash_attention_profile
