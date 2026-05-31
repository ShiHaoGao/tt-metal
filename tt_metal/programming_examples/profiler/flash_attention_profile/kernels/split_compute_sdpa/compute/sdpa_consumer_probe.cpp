// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "tt_metal/programming_examples/profiler/flash_attention_profile/kernels/split_compute_sdpa/compute/compute_common.hpp"
#include "tt_metal/programming_examples/profiler/flash_attention_profile/kernels/split_compute_sdpa/compute/compute_streaming.hpp"

void kernel_main() {
    DeviceZoneScopedN("FAP_SPLIT_COMPUTE_V1_CONSUMER_PROBE");
    constexpr uint32_t compute_pipeline_schedule = get_compile_time_arg_val(0);
    constexpr uint32_t split_signal_expected_outputs = get_compile_time_arg_val(1);
    constexpr uint32_t consumer_probe_q_rows = get_compile_time_arg_val(2);
    constexpr uint32_t consumer_probe_k_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t consumer_probe_v_cols = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_probe_groups = get_compile_time_arg_val(5);
    constexpr bool split_signal_only = compute_pipeline_schedule == 9;
    constexpr bool split_output_stream_signal = compute_pipeline_schedule == 10;
    constexpr bool split_l1_ready_signal = compute_pipeline_schedule == 11;
    constexpr bool split_state_ready_signal = compute_pipeline_schedule == 12;
    constexpr bool split_state_consumer_probe =
        compute_pipeline_schedule == 13 || compute_pipeline_schedule == 14 || compute_pipeline_schedule == 15 ||
        compute_pipeline_schedule == 16 || compute_pipeline_schedule == 17 || compute_pipeline_schedule == 18 ||
        compute_pipeline_schedule == 19 || compute_pipeline_schedule == 20 || compute_pipeline_schedule == 21 ||
        compute_pipeline_schedule == 22 || compute_pipeline_schedule == 23 || compute_pipeline_schedule == 25;
    constexpr bool split_state_real_p_handoff =
        compute_pipeline_schedule == 18 || compute_pipeline_schedule == 19 || compute_pipeline_schedule == 20 ||
        compute_pipeline_schedule == 21 || compute_pipeline_schedule == 22 || compute_pipeline_schedule == 23 ||
        compute_pipeline_schedule == 25;
    constexpr bool split_pv_owner_output = compute_pipeline_schedule == 23 || compute_pipeline_schedule == 25;
    constexpr bool split_signal_enabled =
        split_signal_only || split_output_stream_signal || split_l1_ready_signal || split_state_ready_signal ||
        split_state_consumer_probe;
    constexpr bool split_consumer_token_enabled = !split_state_real_p_handoff;
    constexpr uint32_t cb_signal = tt::CBIndex::c_6;
    constexpr uint32_t cb_consumer_probe_p = tt::CBIndex::c_1;
    constexpr uint32_t cb_consumer_probe_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_consumer_probe_sum = tt::CBIndex::c_3;
    constexpr uint32_t cb_consumer_probe_scratch = tt::CBIndex::c_4;
    constexpr uint32_t cb_consumer_owner_output = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    constexpr uint32_t cb_consumer_probe_out = tt::CBIndex::c_16;
    constexpr uint32_t consumer_probe_p_tiles = consumer_probe_q_rows * consumer_probe_k_tiles;
    constexpr uint32_t consumer_probe_v_tiles = consumer_probe_k_tiles * consumer_probe_v_cols;
    constexpr uint32_t consumer_probe_out_tiles = consumer_probe_q_rows * consumer_probe_v_cols;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;

    if constexpr (split_signal_enabled) {
        if constexpr (split_state_consumer_probe) {
            mm_init(cb_consumer_probe_p, cb_consumer_probe_v, cb_consumer_probe_out);
        }
        if constexpr (split_consumer_token_enabled) {
            DeviceZoneScopedN("FAP_SPLIT_CONSUMER_START_TOKEN_WAIT");
            cb_wait_front(cb_signal, 1);
            cb_pop_front(cb_signal, 1);
        }
        for (uint32_t output_index = 0; output_index < split_signal_expected_outputs; ++output_index) {
            if constexpr (split_consumer_token_enabled) {
                DeviceZoneScopedN("FAP_SPLIT_CONSUMER_OUTPUT_TOKEN_WAIT");
                cb_wait_front(cb_signal, 1);
                cb_pop_front(cb_signal, 1);
            }
            if constexpr (split_state_consumer_probe) {
                {
                    DeviceZoneScopedN("FAP_SPLIT_CONSUMER_PROBE_WAIT_INPUTS");
                    cb_wait_front(cb_consumer_probe_p, consumer_probe_p_tiles);
                    cb_wait_front(cb_consumer_probe_v, consumer_probe_v_tiles);
                    if constexpr (split_pv_owner_output) {
                        cb_wait_front(cb_consumer_probe_sum, consumer_probe_q_rows);
                    }
                }
                for (uint32_t group = 0; group < consumer_probe_groups; ++group) {
                    DeviceZoneScopedN("FAP_SPLIT_CONSUMER_PROBE_PV_GROUP");
                    tile_regs_acquire();
                    uint32_t dst_index = 0;
                    for (uint32_t q_row = 0; q_row < consumer_probe_q_rows; ++q_row) {
                        for (uint32_t v_col = 0; v_col < consumer_probe_v_cols; ++v_col) {
                            for (uint32_t kt = 0; kt < consumer_probe_k_tiles; ++kt) {
                                matmul_tiles(
                                    cb_consumer_probe_p,
                                    cb_consumer_probe_v,
                                    q_row * consumer_probe_k_tiles + kt,
                                    kt * consumer_probe_v_cols + v_col,
                                    dst_index);
                            }
                            ++dst_index;
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_consumer_probe_out, consumer_probe_out_tiles);
                    for (uint32_t tile = 0; tile < consumer_probe_out_tiles; ++tile) {
                        pack_tile(tile, cb_consumer_probe_out);
                    }
                    cb_push_back(cb_consumer_probe_out, consumer_probe_out_tiles);
                    tile_regs_release();
                    if constexpr (split_pv_owner_output) {
                        DeviceZoneScopedN("FAP_SPLIT_CONSUMER_OWNER_NORMALIZE");
                        normalize_row_streaming<true, consumer_probe_v_cols, dst_size>(
                            cb_consumer_probe_sum,
                            cb_consumer_probe_out,
                            cb_col_identity,
                            cb_consumer_probe_scratch,
                            cb_consumer_owner_output,
                            consumer_probe_q_rows);
                    } else {
                        cb_pop_front(cb_consumer_probe_out, consumer_probe_out_tiles);
                    }
                }
                if constexpr (split_state_real_p_handoff) {
                    cb_pop_front(cb_consumer_probe_p, consumer_probe_p_tiles);
                }
            }
        }
        {
            DeviceZoneScopedN("FAP_SPLIT_CONSUMER_READY_AFTER_OUTPUT");
        }
    }
}
