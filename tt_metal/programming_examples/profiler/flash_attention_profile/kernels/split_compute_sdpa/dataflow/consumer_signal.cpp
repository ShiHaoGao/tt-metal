// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt_metal/programming_examples/profiler/flash_attention_profile/kernels/split_compute_sdpa/dataflow/dataflow_common.hpp"

template <uint32_t tile_bytes, typename ReaderType>
void read_consumer_probe_v_tiles(const ReaderType& reader, uint32_t cb_id, uint32_t num_tiles) {
    cb_reserve_back(cb_id, num_tiles);
    uint32_t write_ptr = get_write_ptr(cb_id);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        noc_async_read_tile(tile, reader, write_ptr + tile * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
}

template <uint32_t tile_bytes>
void read_real_p_handoff_tiles(
    uint32_t cb_dst,
    uint32_t cb_remote_mailbox,
    uint32_t producer_physical_x,
    uint32_t producer_physical_y,
    uint32_t num_tiles,
    uint32_t tile_offset = 0) {
    cb_reserve_back(cb_dst, num_tiles);
    const uint32_t dst_write_ptr = get_write_ptr(cb_dst);
    const uint32_t producer_mailbox_l1 = get_read_ptr(cb_remote_mailbox) + tile_offset * tile_bytes;
    const uint64_t producer_mailbox_noc_addr =
        get_noc_addr(producer_physical_x, producer_physical_y, producer_mailbox_l1);
    noc_async_read(producer_mailbox_noc_addr, dst_write_ptr, num_tiles * tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_dst, num_tiles);
}

template <uint32_t tile_bytes, typename WriterType>
void write_consumer_owner_output_tiles(const WriterType& writer, uint32_t cb_id, uint32_t num_tiles) {
    cb_wait_front(cb_id, num_tiles);
    const uint32_t l1_read_ptr = get_read_ptr(cb_id);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        noc_async_write_tile(tile, writer, l1_read_ptr + tile * tile_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_tiles);
}

void kernel_main() {
    constexpr uint32_t compute_pipeline_schedule = get_compile_time_arg_val(0);
    constexpr uint32_t split_signal_semaphore_id = get_compile_time_arg_val(1);
    constexpr uint32_t split_signal_expected_outputs = get_compile_time_arg_val(2);
    constexpr uint32_t consumer_probe_p_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t consumer_probe_v_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_probe_p_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t consumer_probe_v_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t producer_physical_x = get_compile_time_arg_val(7);
    constexpr uint32_t producer_physical_y = get_compile_time_arg_val(8);
    constexpr uint32_t split_state_mailbox_slots = get_compile_time_arg_val(9);
    constexpr uint32_t consumer_probe_sum_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t consumer_output_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t consumer_output_tile_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(13);
    constexpr bool split_signal_only = compute_pipeline_schedule == 9;
    constexpr bool split_output_stream_signal = compute_pipeline_schedule == 10;
    constexpr bool split_l1_ready_signal = compute_pipeline_schedule == 11;
    constexpr bool split_state_ready_signal = compute_pipeline_schedule == 12 || compute_pipeline_schedule == 26;
    constexpr bool split_state_consumer_probe =
        compute_pipeline_schedule == 13 || compute_pipeline_schedule == 14 || compute_pipeline_schedule == 15 ||
        compute_pipeline_schedule == 16 || compute_pipeline_schedule == 17 || compute_pipeline_schedule == 18 ||
        compute_pipeline_schedule == 19 || compute_pipeline_schedule == 20 || compute_pipeline_schedule == 21 ||
        compute_pipeline_schedule == 22 || compute_pipeline_schedule == 23 || compute_pipeline_schedule == 25;
    constexpr bool split_state_consumer_vprefetch =
        compute_pipeline_schedule == 16 || compute_pipeline_schedule == 18 || compute_pipeline_schedule == 20 ||
        compute_pipeline_schedule == 21 || compute_pipeline_schedule == 22 || compute_pipeline_schedule == 23 ||
        compute_pipeline_schedule == 25;
    constexpr bool split_state_consumer_vafter_state =
        compute_pipeline_schedule == 17 || compute_pipeline_schedule == 19;
    constexpr bool split_state_real_p_handoff =
        compute_pipeline_schedule == 18 || compute_pipeline_schedule == 19 || compute_pipeline_schedule == 20 ||
        compute_pipeline_schedule == 21 || compute_pipeline_schedule == 22 || compute_pipeline_schedule == 23 ||
        compute_pipeline_schedule == 25;
    constexpr bool split_state_mailbox_ring = compute_pipeline_schedule == 22;
    constexpr bool split_pv_owner_output_no_ack = compute_pipeline_schedule == 25;
    constexpr bool split_pv_owner_output = compute_pipeline_schedule == 23 || split_pv_owner_output_no_ack;
    constexpr bool split_signal_enabled =
        split_signal_only || split_output_stream_signal || split_l1_ready_signal || split_state_ready_signal ||
        split_state_consumer_probe;
    constexpr bool split_consumer_token_enabled = split_state_consumer_probe && !split_state_real_p_handoff;
    constexpr uint32_t cb_signal = tt::CBIndex::c_6;
    constexpr uint32_t cb_consumer_probe_p = tt::CBIndex::c_1;
    constexpr uint32_t cb_consumer_probe_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_consumer_probe_sum = tt::CBIndex::c_3;
    constexpr uint32_t cb_consumer_owner_output = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    constexpr uint32_t cb_split_state_real_p = tt::CBIndex::c_15;
    constexpr auto out_args = TensorAccessorArgs<14>();
    constexpr auto v_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_addr = get_arg_val<uint32_t>(1);
    const auto out_writer = TensorAccessor(out_args, out_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);

    if constexpr (split_signal_enabled) {
        const uint32_t split_signal_semaphore_addr = get_semaphore(split_signal_semaphore_id);
        auto* split_signal_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(split_signal_semaphore_addr);

        {
            DeviceZoneScopedN("FAP_SPLIT_SIGNAL_START_WAIT");
            noc_semaphore_wait_min(split_signal_semaphore_ptr, 1);
        }
        if constexpr (split_consumer_token_enabled) {
            DeviceZoneScopedN("FAP_SPLIT_SIGNAL_START_TOKEN");
            cb_reserve_back(cb_signal, 1);
            cb_push_back(cb_signal, 1);
        }
        if constexpr (split_state_consumer_probe) {
            {
                DeviceZoneScopedN("FAP_SPLIT_CONSUMER_PROBE_INPUT_FILL");
                if constexpr (split_pv_owner_output) {
                    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);
                }
                if constexpr (!split_state_real_p_handoff) {
                    cb_reserve_back(cb_consumer_probe_p, consumer_probe_p_tiles);
                    for (uint32_t tile = 0; tile < consumer_probe_p_tiles; ++tile) {
                        fill_tile_zeros<consumer_probe_p_tile_bytes, false>(cb_consumer_probe_p, tile);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_consumer_probe_p, consumer_probe_p_tiles);
                }
                if constexpr (split_state_consumer_vprefetch) {
                    DeviceZoneScopedN("FAP_SPLIT_CONSUMER_PROBE_V_PREFETCH");
                    read_consumer_probe_v_tiles<consumer_probe_v_tile_bytes>(
                        v_reader, cb_consumer_probe_v, consumer_probe_v_tiles);
                } else if constexpr (!split_state_consumer_vafter_state) {
                    cb_reserve_back(cb_consumer_probe_v, consumer_probe_v_tiles);
                    for (uint32_t tile = 0; tile < consumer_probe_v_tiles; ++tile) {
                        fill_tile_zeros<consumer_probe_v_tile_bytes, false>(cb_consumer_probe_v, tile);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_consumer_probe_v, consumer_probe_v_tiles);
                }
            }
        }

        bool consumer_probe_v_loaded = split_state_consumer_vprefetch || !split_state_consumer_vafter_state;
        for (uint32_t output_index = 0; output_index < split_signal_expected_outputs; ++output_index) {
            {
                DeviceZoneScopedN("FAP_SPLIT_SIGNAL_OUTPUT_WAIT");
                noc_semaphore_wait_min(split_signal_semaphore_ptr, 2 + output_index);
            }
            if constexpr (split_state_real_p_handoff) {
                {
                    DeviceZoneScopedN("FAP_SPLIT_REAL_P_REMOTE_READ");
                    const uint32_t mailbox_slot =
                        split_state_mailbox_ring ? (output_index % split_state_mailbox_slots) : 0;
                    read_real_p_handoff_tiles<consumer_probe_p_tile_bytes>(
                        cb_consumer_probe_p,
                        cb_split_state_real_p,
                        producer_physical_x,
                        producer_physical_y,
                        consumer_probe_p_tiles,
                        mailbox_slot * consumer_probe_p_tiles);
                    if constexpr (split_pv_owner_output) {
                        read_real_p_handoff_tiles<consumer_probe_p_tile_bytes>(
                            cb_consumer_probe_sum,
                            cb_split_state_real_p,
                            producer_physical_x,
                            producer_physical_y,
                            consumer_probe_sum_tiles,
                            consumer_probe_p_tiles);
                    }
                }
                if constexpr (!split_pv_owner_output_no_ack) {
                    DeviceZoneScopedN("FAP_SPLIT_REAL_P_ACK_SEND");
                    const uint64_t producer_signal_semaphore_noc_addr =
                        get_noc_addr(producer_physical_x, producer_physical_y, split_signal_semaphore_addr);
                    noc_semaphore_inc(producer_signal_semaphore_noc_addr, 1);
                    noc_async_atomic_barrier();
                }
            }
            if constexpr (split_state_consumer_vafter_state) {
                if (!consumer_probe_v_loaded) {
                    DeviceZoneScopedN("FAP_SPLIT_CONSUMER_PROBE_V_AFTER_STATE");
                    read_consumer_probe_v_tiles<consumer_probe_v_tile_bytes>(
                        v_reader, cb_consumer_probe_v, consumer_probe_v_tiles);
                    consumer_probe_v_loaded = true;
                }
            }
            if constexpr (split_consumer_token_enabled) {
                DeviceZoneScopedN("FAP_SPLIT_SIGNAL_OUTPUT_TOKEN");
                cb_reserve_back(cb_signal, 1);
                cb_push_back(cb_signal, 1);
            }
            if constexpr (split_pv_owner_output) {
                DeviceZoneScopedN("FAP_SPLIT_OWNER_OUTPUT_WRITE");
                write_consumer_owner_output_tiles<consumer_output_tile_bytes>(
                    out_writer, cb_consumer_owner_output, consumer_output_tiles);
            }
        }
        noc_semaphore_set(split_signal_semaphore_ptr, 0);
    }
}
