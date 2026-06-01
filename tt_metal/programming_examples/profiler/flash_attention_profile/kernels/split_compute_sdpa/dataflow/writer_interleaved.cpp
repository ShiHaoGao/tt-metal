// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    DeviceZoneScopedN("FAP_WRITER");

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(4);
    constexpr uint32_t unpadded_Sk = get_compile_time_arg_val(5);
    constexpr uint32_t DHt = get_compile_time_arg_val(6);
    constexpr uint32_t vDHt = get_compile_time_arg_val(7);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(12);
    constexpr uint32_t scale_val = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t is_causal = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(19);
    constexpr bool use_lightweight_mask = get_compile_time_arg_val(20) == 1;
    constexpr bool use_streaming_compute = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(22);
    constexpr uint32_t k_partial_col = get_compile_time_arg_val(23);
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t qk_detail_profile_stage = get_compile_time_arg_val(25);
    constexpr uint32_t compute_pipeline_schedule = get_compile_time_arg_val(26);
    constexpr uint32_t split_signal_semaphore_id = get_compile_time_arg_val(27);
    constexpr uint32_t split_consumer_x = get_compile_time_arg_val(28);
    constexpr uint32_t split_consumer_y = get_compile_time_arg_val(29);
    constexpr uint32_t split_signal_expected_outputs = get_compile_time_arg_val(30);
    constexpr uint32_t split_state_ready_tiles = get_compile_time_arg_val(31);
    constexpr uint32_t split_state_ready_packets = get_compile_time_arg_val(32);
    constexpr uint32_t split_state_mailbox_slots = get_compile_time_arg_val(33);
    constexpr uint32_t split_state_mailbox_l1_addr = get_compile_time_arg_val(34);

    constexpr auto out_args = TensorAccessorArgs<35>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);
    const uint32_t num_phases = get_arg_val<uint32_t>(8);
    const uint32_t use_chunk_start_idx_tensor = get_arg_val<uint32_t>(9);
    uint32_t chunk_start_t_in_q_chunks_phase_1 = get_arg_val<uint32_t>(10);
    const uint32_t write_offset_phase_1 = get_arg_val<uint32_t>(11);
    uint32_t chunk_start_t_in_q_chunks_phase_2 = 0;
    uint32_t write_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunk_start_t_in_q_chunks_phase_2 = get_arg_val<uint32_t>(12);
        write_offset_phase_2 = get_arg_val<uint32_t>(13);
    }

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;  // non-streaming drain only

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_chunk_start_idx = tt::CBIndex::c_9;
    constexpr uint32_t cb_split_state_ready = tt::CBIndex::c_15;

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    const auto out_writer = TensorAccessor(out_args, out_addr);

    const auto out_tile_shape = TensorTileShape(B, NQH, valid_Sqt, vDHt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    constexpr bool split_signal_only = compute_pipeline_schedule == 9;
    constexpr bool split_output_stream_signal = compute_pipeline_schedule == 10;
    constexpr bool split_l1_ready_signal = compute_pipeline_schedule == 11;
    constexpr bool split_state_ready_signal =
        compute_pipeline_schedule == 12 || compute_pipeline_schedule == 13 || compute_pipeline_schedule == 14 ||
        compute_pipeline_schedule == 15 || compute_pipeline_schedule == 16 || compute_pipeline_schedule == 17 ||
        compute_pipeline_schedule == 18 || compute_pipeline_schedule == 19 || compute_pipeline_schedule == 20 ||
        compute_pipeline_schedule == 21 || compute_pipeline_schedule == 22 || compute_pipeline_schedule == 23 ||
        compute_pipeline_schedule == 25 || compute_pipeline_schedule == 26;
    constexpr bool split_state_mailbox_ring = compute_pipeline_schedule == 22;
    constexpr bool split_state_ready_mailbox_bridge = compute_pipeline_schedule == 26;
    constexpr bool split_pv_owner_output_no_ack = compute_pipeline_schedule == 25;
    constexpr bool split_pv_owner_output = compute_pipeline_schedule == 23 || split_pv_owner_output_no_ack;
    constexpr bool split_state_real_p_handoff =
        compute_pipeline_schedule == 18 || compute_pipeline_schedule == 19 || compute_pipeline_schedule == 20 ||
        compute_pipeline_schedule == 21 || compute_pipeline_schedule == 22 || split_pv_owner_output;
    constexpr bool split_signal_enabled =
        split_signal_only || split_output_stream_signal || split_l1_ready_signal || split_state_ready_signal;
    bool split_signal_output_sent = false;
    uint32_t split_signal_outputs_sent = 0;
    [[maybe_unused]] uint32_t split_state_mailbox_expected_seq = 0;
    auto send_split_signal = [&]() {
        const uint32_t split_signal_semaphore_addr = get_semaphore(split_signal_semaphore_id);
        const uint64_t split_signal_remote_addr =
            get_noc_addr(split_consumer_x, split_consumer_y, split_signal_semaphore_addr);
        noc_semaphore_inc(split_signal_remote_addr, 1);
        noc_async_atomic_barrier();
    };
    auto wait_split_real_p_ack = [&](uint32_t expected_ack_count) {
        const uint32_t split_signal_semaphore_addr = get_semaphore(split_signal_semaphore_id);
        auto* split_signal_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(split_signal_semaphore_addr);
        noc_semaphore_wait_min(split_signal_semaphore_ptr, expected_ack_count);
    };

    {
        DeviceZoneScopedN("FAP_WRITER_PREPARE");
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_identity_scale_in,
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW,
            dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR,
            /*compute_uses_reduce_tile=*/true>();
        generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);
    }
    if constexpr (split_signal_enabled) {
        if (core_id == 0) {
            DeviceZoneScopedN("FAP_SPLIT_SIGNAL_START_SEND");
            if constexpr (split_state_ready_mailbox_bridge) {
                auto* split_state_mailbox =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(split_state_mailbox_l1_addr);
                split_state_mailbox[0] = 0xfa320001u;
                split_state_mailbox[1] = 0;
                split_state_mailbox[2] = 0;
                split_state_mailbox[3] = 0;
                split_state_mailbox[4] = 0;
            }
            send_split_signal();
        }
    }

    // Lightweight mask: generate template tiles once, leave permanently fronted.
    // Layout: [neginf(0)] [causal_diag?(1)] [k_partial?].
    if constexpr (use_lightweight_mask) {
        DeviceZoneScopedN("FAP_WRITER_MASK_TEMPLATE");
        // is_causal handles K-partial via causal stamp; skip emitting partial tile in causal mode.
        constexpr uint32_t writer_partial_col = is_causal ? 0u : k_partial_col;
        generate_lightweight_mask_tiles<writer_partial_col, /*joint_l*/ 0u, cb_mask_in, is_causal>();
    }

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            DeviceZoneScopedN("FAP_WRITER_CHUNK_START");
            cb_wait_front(cb_chunk_start_idx, 1);
            auto chunk_start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chunk_start_idx));
            uint32_t chunk_start_idx = chunk_start_ptr[0];
            cb_pop_front(cb_chunk_start_idx, 1);
            const uint32_t q_chunk_size = Sq_chunk_t * tt::constants::TILE_HEIGHT;
            chunk_start_t_in_q_chunks_phase_1 = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunk_start_t_in_q_chunks_phase_2 = chunk_start_t_in_q_chunks_phase_1;
            }
        }
    }

    uint32_t chunk_start_t_in_q_chunks = 0;
    uint32_t write_offset = 0;
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_1;
            write_offset = write_offset_phase_1;
        } else {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_2;
            write_offset = write_offset_phase_2;
        }
        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
            const uint32_t q_batch_offset = nb * NQH * Sqt * DHt;
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                    uint32_t q_chunk = remap_q_index(local_q_start + q_iter, q_num_chunks, use_zigzag_balancing);
                    const bool split_pv_owner_output_chunk =
                        split_pv_owner_output && core_id == 0 && phase == 0 && nb == 0 && nq == 0 &&
                        q_chunk == 0;

                    // Generate mask only when user didn't provide one.
                    // Lightweight path already has a single -inf tile fronted — skip generate_mask.
                    if constexpr (!use_provided_mask && !use_lightweight_mask) {
                        DeviceZoneScopedN("FAP_WRITER_MASK_GENERATE");
                        generate_mask<is_chunked, sliding_window_size, use_padded_mask, cb_mask_in>(
                            Sq_chunk_t,
                            Sk_chunk_t,
                            q_chunk,
                            chunk_start_t_in_q_chunks,
                            true,
                            false,
                            unpadded_Sk,
                            0,
                            is_causal);
                    }

                    // Wait for compute to deliver output chunk
                    /*
                      Determine how many rows of OUT will be written. Both start and end rows are
                      capped by valid_Sqt, since Sq padding is independent of Sk padding.
                    */
                    const uint32_t out_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
                    const uint32_t out_row_end_tile = std::min(out_row_start_tile + Sq_chunk_t, valid_Sqt);
                    const uint32_t out_row_tile_count = out_row_end_tile - out_row_start_tile;
                    uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile, 0);
                    if constexpr (split_state_ready_signal) {
                        if (!split_pv_owner_output || split_pv_owner_output_chunk) {
                            if constexpr (split_state_ready_mailbox_bridge) {
                                if (core_id == 0) {
                                    auto* split_state_mailbox =
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(split_state_mailbox_l1_addr);
                                    ++split_state_mailbox_expected_seq;
                                    uint32_t mailbox_loops = 0;
                                    {
                                        DeviceZoneScopedN("FAP_SPLIT_STATE_READY_MAILBOX_WAIT");
                                        while (split_state_mailbox[1] < split_state_mailbox_expected_seq) {
                                            ++mailbox_loops;
                                        }
                                    }
                                    split_state_mailbox[3] = mailbox_loops;
                                    split_state_mailbox[4] = 0xfa320002u;
                                    DeviceZoneScopedN("FAP_SPLIT_SIGNAL_STATE_READY_SEND");
                                    send_split_signal();
                                    ++split_signal_outputs_sent;
                                }
                            } else if constexpr (split_state_mailbox_ring) {
                                uint32_t packets_drained = 0;
                                while (packets_drained < split_state_ready_packets) {
                                    const uint32_t remaining_packets = split_state_ready_packets - packets_drained;
                                    const uint32_t batch_packets =
                                        remaining_packets < split_state_mailbox_slots ? remaining_packets
                                                                                      : split_state_mailbox_slots;
                                    for (uint32_t packet_in_batch = 0; packet_in_batch < batch_packets;
                                         ++packet_in_batch) {
                                        {
                                            DeviceZoneScopedN("FAP_SPLIT_STATE_READY_LOCAL_WAIT");
                                            cb_wait_front(
                                                cb_split_state_ready,
                                                split_state_ready_tiles * (packet_in_batch + 1));
                                        }
                                        if (core_id == 0) {
                                            DeviceZoneScopedN("FAP_SPLIT_SIGNAL_STATE_READY_SEND");
                                            send_split_signal();
                                            ++split_signal_outputs_sent;
                                        }
                                    }
                                    if (core_id == 0) {
                                        DeviceZoneScopedN("FAP_SPLIT_REAL_P_ACK_WAIT");
                                        wait_split_real_p_ack(split_signal_outputs_sent);
                                    }
                                    cb_pop_front(cb_split_state_ready, split_state_ready_tiles * batch_packets);
                                    packets_drained += batch_packets;
                                }
                            } else {
                                for (uint32_t packet = 0; packet < split_state_ready_packets; ++packet) {
                                    {
                                        DeviceZoneScopedN("FAP_SPLIT_STATE_READY_LOCAL_WAIT");
                                        cb_wait_front(cb_split_state_ready, split_state_ready_tiles);
                                        if constexpr (!split_state_real_p_handoff) {
                                            cb_pop_front(cb_split_state_ready, split_state_ready_tiles);
                                        }
                                    }
                                    if (core_id == 0) {
                                        DeviceZoneScopedN("FAP_SPLIT_SIGNAL_STATE_READY_SEND");
                                        send_split_signal();
                                        if constexpr (split_state_real_p_handoff && !split_pv_owner_output_no_ack) {
                                            DeviceZoneScopedN("FAP_SPLIT_REAL_P_ACK_WAIT");
                                            wait_split_real_p_ack(split_signal_outputs_sent + 1);
                                        }
                                        ++split_signal_outputs_sent;
                                    }
                                    if constexpr (split_state_real_p_handoff) {
                                        cb_pop_front(cb_split_state_ready, split_state_ready_tiles);
                                    }
                                }
                            }
                        }
                    }
                    if constexpr (use_streaming_compute) {
                        // Streaming: drain per row-group (cb_out is a 2-slot ping-pong).
                        // Compute always pushes Sq_chunk_t rows; rows past out_row_tile_count
                        // are padding and get popped without being written.
                        if constexpr (split_l1_ready_signal) {
                            write_block_row_grouped_with_group_ready<qk_detail_profile_stage == 11>(
                                out_writer,
                                cb_out,
                                Sq_chunk_t,
                                out_row_tile_count,
                                vDHt,
                                out_tile_id,
                                tile_bytes,
                                out_subblock_h,
                                barrier_threshold,
                                [&](uint32_t, uint32_t, uint32_t) {
                                    if (core_id == 0) {
                                        DeviceZoneScopedN("FAP_SPLIT_SIGNAL_L1_READY_SEND");
                                        send_split_signal();
                                        ++split_signal_outputs_sent;
                                    }
                                });
                        } else if (split_pv_owner_output_chunk && split_pv_owner_output) {
                            write_block_row_grouped_skip_first_group<qk_detail_profile_stage == 11>(
                                out_writer,
                                cb_out,
                                Sq_chunk_t,
                                out_row_tile_count,
                                vDHt,
                                out_tile_id,
                                tile_bytes,
                                out_subblock_h,
                                barrier_threshold);
                        } else {
                            write_block_row_grouped<qk_detail_profile_stage == 11>(
                                out_writer,
                                cb_out,
                                Sq_chunk_t,
                                out_row_tile_count,
                                vDHt,
                                out_tile_id,
                                tile_bytes,
                                out_subblock_h,
                                barrier_threshold);
                        }
                    } else {
                        write_block(
                            out_writer,
                            cb_out,
                            out_chunk_tiles,
                            out_row_tile_count,
                            vDHt,
                            out_tile_id,
                            tile_bytes,
                            barrier_threshold);
                    }
                    if constexpr (split_signal_only || split_output_stream_signal) {
                        if (core_id == 0 && (split_output_stream_signal || !split_signal_output_sent)) {
                            DeviceZoneScopedN("FAP_SPLIT_SIGNAL_OUTPUT_SEND");
                            send_split_signal();
                            split_signal_output_sent = true;
                            ++split_signal_outputs_sent;
                        }
                    }
                }
            }
        }
    }
    if constexpr (split_output_stream_signal || split_l1_ready_signal || split_state_ready_signal) {
        if (core_id == 0) {
            ASSERT(split_signal_outputs_sent == split_signal_expected_outputs);
        }
    }
}
