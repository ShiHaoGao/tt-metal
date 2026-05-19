// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/programming_examples/pipeline_warmup_experiments/sdpa/instrumented/device/instrumented_sdpa_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct InstrumentedSDPASharedVariables {
    tt::tt_metal::KernelHandle reader_kernels_id{};
    tt::tt_metal::KernelHandle writer_kernels_id{};
    tt::tt_metal::KernelHandle compute_kernels_id{};
    tt::tt_metal::CoreCoord grid_size;
    uint32_t num_cores = 0;
    bool is_chunked = false;
    uint32_t q_chunk_size = 0;
    bool use_mla = false;
};

struct InstrumentedSDPAProgramFactory {
    using shared_variables_t = InstrumentedSDPASharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const InstrumentedSDPAParams& operation_attributes, const InstrumentedSDPAInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const InstrumentedSDPAParams& operation_attributes,
        const InstrumentedSDPAInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
