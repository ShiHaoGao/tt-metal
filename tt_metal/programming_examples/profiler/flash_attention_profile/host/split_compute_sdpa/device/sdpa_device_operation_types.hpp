// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <cstdint>
#include <optional>
#include <utility>

namespace ttnn::prim::flash_attention_profile_split_compute_sdpa {

enum class SDPAPipelineMode : uint32_t {
    Auto = 0,
    StreamH1 = 1,
    NonStreaming = 2,
    QktvH1 = 3,
    SaladFirst = 4,
    QktvH1SaladFirst = 5,
};

struct SDPAParams {
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    bool is_causal = false;
    std::optional<int64_t> chunk_start_idx;        // Chunked legacy: scalar offset, part of program cache key
    std::optional<Tensor> chunk_start_idx_tensor;  // Chunked flexible: device tensor [1] int32, read at runtime
    DeviceComputeKernelConfig compute_kernel_config;
    bool use_mla = false;
    std::optional<uint32_t> head_dim_v;
    std::optional<uint32_t> sliding_window_size;
    SDPAPipelineMode pipeline_mode = SDPAPipelineMode::Auto;
    uint32_t streaming_output_depth = 2;
    std::optional<std::pair<uint32_t, uint32_t>> qk_subblock_override;
    std::optional<uint32_t> q_buffer_factor_override;
    std::optional<bool> dst_full_sync_override;
    uint32_t qk_softmax_profile_stage = 0;
    uint32_t qk_softmax_schedule = 0;
    uint32_t qk_detail_profile_stage = 0;
    uint32_t q_reader_schedule = 0;
    uint32_t qk_first_body_warmup = 0;
    uint32_t compute_pipeline_schedule = 0;
};

struct SDPAInputs {
    Tensor q;
    Tensor k;
    std::optional<Tensor> v;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> page_table;
    std::optional<Tensor> attention_sink;
};

}  // namespace ttnn::prim::flash_attention_profile_split_compute_sdpa
