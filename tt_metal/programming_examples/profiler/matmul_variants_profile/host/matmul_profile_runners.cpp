// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/work_split.hpp>

#include <bmm_op.hpp>
#include <matmul_profile_runner.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

namespace {

class PreparedMatmulProfileRunner : public MatmulProfileRunner {
public:
    PreparedMatmulProfileRunner(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        distributed::MeshWorkload&& workload,
        std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer,
        std::shared_ptr<distributed::MeshBuffer> src1_dram_buffer,
        std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer) :
        mesh_device_(mesh_device),
        workload_(std::move(workload)),
        src0_dram_buffer_(std::move(src0_dram_buffer)),
        src1_dram_buffer_(std::move(src1_dram_buffer)),
        dst_dram_buffer_(std::move(dst_dram_buffer)) {}

    ~PreparedMatmulProfileRunner() override {
        try {
            release_trace();
        } catch (...) {
        }
    }

    void run(
        const std::vector<bfloat16>& a,
        const std::vector<bfloat16>& b,
        std::vector<bfloat16>& output) override {
        if (trace_ready()) {
            run_trace(a, b, output);
            return;
        }
        run_workload(a, b, output);
    }

    void prepare_trace(
        const std::vector<bfloat16>& a,
        const std::vector<bfloat16>& b,
        std::vector<bfloat16>& output,
        bool drain_profiler) override {
        if (trace_ready()) {
            return;
        }

        // Compile/load once outside the measured loop. This execution is intentionally not timed.
        run_workload(a, b, output);
        if (drain_profiler) {
            ReadMeshDeviceProfilerResults(*mesh_device_);
        }

        auto& cq = mesh_device_->mesh_command_queue();
        auto trace_id = distributed::BeginTraceCapture(mesh_device_.get(), cq.id());
        distributed::EnqueueMeshWorkload(cq, workload_, false);
        mesh_device_->end_mesh_trace(cq.id(), trace_id);
        distributed::Finish(cq);
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
        mesh_device_->release_mesh_trace(*trace_id_);
        trace_id_.reset();
    }

private:
    void run_workload(
        const std::vector<bfloat16>& a,
        const std::vector<bfloat16>& b,
        std::vector<bfloat16>& output) {
        auto& cq = mesh_device_->mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer_, a, false);
        distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer_, b, false);
        distributed::EnqueueMeshWorkload(cq, workload_, false);
        distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer_, true);
    }

    void run_trace(
        const std::vector<bfloat16>& a,
        const std::vector<bfloat16>& b,
        std::vector<bfloat16>& output) {
        auto& cq = mesh_device_->mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer_, a, false);
        distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer_, b, false);
        mesh_device_->replay_mesh_trace(cq.id(), *trace_id_, false);
        distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer_, true);
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    distributed::MeshWorkload workload_;
    std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer_;
    std::shared_ptr<distributed::MeshBuffer> src1_dram_buffer_;
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer_;
    std::optional<distributed::MeshTraceId> trace_id_;
};

std::unique_ptr<MatmulProfileRunner> make_runner(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshWorkload&& workload,
    std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer,
    std::shared_ptr<distributed::MeshBuffer> src1_dram_buffer,
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer) {
    return std::make_unique<PreparedMatmulProfileRunner>(
        mesh_device,
        std::move(workload),
        std::move(src0_dram_buffer),
        std::move(src1_dram_buffer),
        std::move(dst_dram_buffer));
}

}  // namespace

std::unique_ptr<MatmulProfileRunner> prepare_matmul_multi_core_profile_runner(
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto num_output_tiles_total = (M * N) / TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, num_output_tiles_total);
    (void)num_cores;

    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config_A{.size = single_tile_size * Mt * Kt};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = single_tile_size * Nt * Kt};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = single_tile_size * Mt * Nt};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());

    const auto cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size));
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size));
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size));

    MathFidelity math_fidelity = MathFidelity::HiFi4;
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/multi_core/dataflow/reader_mm_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/multi_core/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/multi_core/compute/mm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                tt_metal::SetRuntimeArgs(
                    program,
                    reader_id,
                    core,
                    {src0_dram_buffer->address(),
                     src1_dram_buffer->address(),
                     Mt,
                     Kt,
                     Nt,
                     work_offset,
                     work_per_core});
                tt_metal::SetRuntimeArgs(
                    program, writer_id, core, {dst_dram_buffer->address(), work_per_core, work_offset});
                tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {work_per_core, Kt});
                work_offset += work_per_core;
            }
        }
    }

    workload.add_program(device_range, std::move(program));
    return make_runner(
        mesh_device, std::move(workload), std::move(src0_dram_buffer), std::move(src1_dram_buffer), std::move(dst_dram_buffer));
}

std::unique_ptr<MatmulProfileRunner> prepare_matmul_multicore_reuse_profile_runner(
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t B,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    auto compute_with_storage_grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t in0_block_w = 2;

    auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2;
    uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    uint32_t num_blocks = Kt / in0_block_w;
    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        B};

    uint32_t num_blocks_y = Mt / per_core_M;
    uint32_t num_blocks_x = Nt / per_core_N;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
    CoreRangeSet all_cores(
        tt::tt_metal::num_cores_to_corerangeset(num_blocks_x * num_blocks_y, compute_with_storage_grid_size, true));

    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt;
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt;
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config_A{.size = dram_buffer_A_size};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = dram_buffer_B_size};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = dram_buffer_C_size};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());

    uint32_t src0_cb_index = CBIndex::c_0;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size));

    uint32_t src1_cb_index = CBIndex::c_1;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size));

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, single_tile_size)
            .set_page_size(interm0_cb_index, single_tile_size));

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse/dataflow/reader_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});
    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse/dataflow/writer_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});
    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse/compute/bmm_large_block_zm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args});

    uint32_t num_blocks_read = 0;
    for (uint32_t output_idx_y = 0; output_idx_y < num_blocks_y; output_idx_y++) {
        for (uint32_t output_idx_x = 0; output_idx_x < num_blocks_x; output_idx_x++) {
            uint32_t core_idx_x = num_blocks_read % num_cores_x;
            uint32_t core_idx_y = num_blocks_read / num_cores_x;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            std::vector<uint32_t> mm_reader_args = {
                src0_dram_buffer->address(),
                Kt * per_core_M * output_idx_y,
                1,
                Kt,
                in0_block_w,
                in0_block_w,
                per_core_M,
                in0_block_w * per_core_M,
                src1_dram_buffer->address(),
                per_core_N * output_idx_x,
                1,
                Nt,
                in0_block_w * Nt,
                per_core_N,
                in0_block_w,
                per_core_N * in0_block_w,
                Kt / in0_block_w,
                Mt * Kt,
                Kt * Nt,
                B,
                static_cast<uint32_t>(bcast_batch)};
            std::vector<uint32_t> writer_args = {
                dst_dram_buffer->address(),
                output_idx_x * per_core_N + output_idx_y * per_core_M * Nt,
                1,
                Nt,
                out_subblock_w,
                out_subblock_h * Nt,
                out_subblock_w,
                out_subblock_h,
                out_subblock_w * out_subblock_h,
                per_core_N / out_subblock_w,
                per_core_M / out_subblock_h,
                Mt * Nt,
                B};

            tt_metal::SetRuntimeArgs(program, reader_id, core, mm_reader_args);
            tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
            num_blocks_read++;
        }
    }

    workload.add_program(device_range, std::move(program));
    return make_runner(
        mesh_device, std::move(workload), std::move(src0_dram_buffer), std::move(src1_dram_buffer), std::move(dst_dram_buffer));
}

std::unique_ptr<MatmulProfileRunner> prepare_matmul_multicore_reuse_mcast_profile_runner(
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t B,
    std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    auto compute_with_storage_grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t in0_block_w = 2;

    auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2;
    uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    uint32_t num_blocks = Kt / in0_block_w;
    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        B};

    uint32_t num_blocks_y = Mt / per_core_M;
    uint32_t num_blocks_x = Nt / per_core_N;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
    CoreCoord start_core = {0, 0};
    CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);

    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    CoreRange left_column(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});
    CoreRange all_except_left_column(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    CoreRange in0_sender_in1_sender(
        {(std::size_t)start_core_x, (std::size_t)start_core_y}, {(std::size_t)start_core_x, (std::size_t)start_core_y});
    CoreRange in0_sender_in1_receiver(
        {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});
    CoreRange in0_receiver_in1_sender(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});
    CoreRange in0_receiver_in1_receiver(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});

    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt;
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt;
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;
    distributed::ReplicatedBufferConfig buffer_config_A{.size = dram_buffer_A_size};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = dram_buffer_B_size};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = dram_buffer_C_size};
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, dram_config, mesh_device.get());

    uint32_t src0_cb_index = CBIndex::c_0;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size));
    uint32_t src1_cb_index = CBIndex::c_1;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size));

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
        {output_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
    tt_metal::CreateCircularBuffer(
        program,
        CoreRangeSet({all_cores}),
        CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, single_tile_size)
            .set_page_size(interm0_cb_index, single_tile_size));

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    auto mm_reader_kernel_in0_sender_in1_sender_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse_mcast/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
        in0_sender_in1_sender,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});
    auto mm_reader_kernel_in0_sender_in1_receiver_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse_mcast/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
        in0_sender_in1_receiver,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});
    auto mm_reader_kernel_in0_receiver_in1_sender_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse_mcast/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
        in0_receiver_in1_sender,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});
    auto mm_reader_kernel_in0_receiver_in1_receiver_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse_mcast/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
        in0_receiver_in1_receiver,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});
    auto unary_writer_kernel_noc0_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse_mcast/dataflow/writer_bmm_tile_layout.cpp",
        all_except_left_column,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});
    auto unary_writer_kernel_noc1_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse_mcast/dataflow/writer_bmm_tile_layout.cpp",
        left_column,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/matmul_variants_profile/kernels/reuse_mcast/compute/bmm_large_block_zm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args});

    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    for (uint32_t core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (uint32_t core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

            CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};
            CoreCoord left_core_plus_one = {(std::size_t)start_core_x + 1, (std::size_t)core.y};
            CoreCoord right_core = {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)core.y};
            CoreCoord top_core = {(std::size_t)core.x, (std::size_t)start_core_y};
            CoreCoord top_core_plus_one = {(std::size_t)core.x, (std::size_t)start_core_y + 1};
            CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)start_core_y + num_cores_r - 1};

            auto left_core_physical = mesh_device->worker_core_from_logical_core(left_core);
            auto left_core_plus_one_physical = mesh_device->worker_core_from_logical_core(left_core_plus_one);
            auto right_core_physical = mesh_device->worker_core_from_logical_core(right_core);
            auto top_core_physical = mesh_device->worker_core_from_logical_core(top_core);
            auto top_core_plus_one_physical = mesh_device->worker_core_from_logical_core(top_core_plus_one);
            auto bottom_core_physical = mesh_device->worker_core_from_logical_core(bottom_core);

            std::vector<uint32_t> mm_reader_args = {
                src0_dram_buffer->address(),
                Kt * per_core_M * core_idx_y,
                1,
                Kt,
                in0_block_w,
                in0_block_w,
                per_core_M,
                in0_block_w * per_core_M,
                src1_dram_buffer->address(),
                per_core_N * core_idx_x,
                1,
                Nt,
                in0_block_w * Nt,
                per_core_N,
                in0_block_w,
                per_core_N * in0_block_w,
                Kt / in0_block_w,
                static_cast<uint32_t>(right_core_physical.x),
                static_cast<uint32_t>(right_core_physical.y),
                static_cast<uint32_t>(left_core_plus_one_physical.x),
                static_cast<uint32_t>(left_core_plus_one_physical.y),
                num_cores_c - 1,
                static_cast<uint32_t>(left_core_physical.x),
                static_cast<uint32_t>(left_core_physical.y),
                in0_mcast_sender_semaphore_id,
                in0_mcast_receiver_semaphore_id,
                static_cast<uint32_t>(bottom_core_physical.x),
                static_cast<uint32_t>(bottom_core_physical.y),
                static_cast<uint32_t>(top_core_plus_one_physical.x),
                static_cast<uint32_t>(top_core_plus_one_physical.y),
                num_cores_r - 1,
                static_cast<uint32_t>(top_core_physical.x),
                static_cast<uint32_t>(top_core_physical.y),
                in1_mcast_sender_semaphore_id,
                in1_mcast_receiver_semaphore_id,
                Mt * Kt,
                Kt * Nt,
                B,
                static_cast<uint32_t>(bcast_batch)};

            std::vector<uint32_t> writer_args = {
                dst_dram_buffer->address(),
                core_idx_x * per_core_N + core_idx_y * per_core_M * Nt,
                1,
                Nt,
                out_subblock_w,
                out_subblock_h * Nt,
                out_subblock_w,
                out_subblock_h,
                out_subblock_w * out_subblock_h,
                per_core_N / out_subblock_w,
                per_core_M / out_subblock_h,
                Mt * Nt,
                B};

            if (core_idx_x == 0 && core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_sender_id, core, mm_reader_args);
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args);
            } else if (core_idx_x == 0 && core_idx_y != 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_receiver_id, core, mm_reader_args);
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args);
            } else if (core_idx_x != 0 && core_idx_y == 0) {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_sender_id, core, mm_reader_args);
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args);
            } else {
                tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_receiver_id, core, mm_reader_args);
                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args);
            }
        }
    }

    workload.add_program(device_range, std::move(program));
    return make_runner(
        mesh_device, std::move(workload), std::move(src0_dram_buffer), std::move(src1_dram_buffer), std::move(dst_dram_buffer));
}
