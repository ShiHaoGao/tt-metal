// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.hpp"
#include "impl/experimental/published_deployment/dispatch_plan.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt_metal.hpp>
#include "impl/buffers/semaphore.hpp"
#include <map>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include "dispatch/command_queue_common.hpp"
#include "device.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch/dispatch_core_common.hpp"
#include "dispatch_s.hpp"
#include "hal_types.hpp"
#include "prefetch.hpp"
#include "context/context_descriptor.hpp"
#include "debug/inspector/inspector.hpp"
#include <umd/device/types/xy_pair.hpp>
#include "dispatch/system_memory_manager.hpp"

#include "device/device_manager.hpp"
#include "fabric/fabric_context.hpp"
#include <dispatch/dispatch_query_manager.hpp>
#include <dispatch/dispatch_mem_map.hpp>
#include "hostdevcommon/dispatch_telemetry_types.hpp"
#include "impl/dispatch/dispatch_engine_cores.hpp"

using namespace tt::tt_metal;

DispatchKernel::DispatchKernel(
    int node_id,
    ChipId device_id,
    ChipId servicing_device_id,
    uint8_t cq_id,
    noc_selection_t noc_selection,
    bool h_variant,
    bool d_variant,
    const ContextDescriptor& descriptor,
    dispatch_core_manager& dispatch_core_manager,
    const GetControlPlaneFn& get_control_plane,
    const GetDispatchQueryManagerFn& get_dispatch_query_manager,
    const GetMaxNumEthCoresFn& get_max_num_eth_cores,
    const GetReadsDispatchCoresFn& get_reads_dispatch_cores) :
    FDKernel(
        node_id,
        device_id,
        servicing_device_id,
        cq_id,
        noc_selection,
        descriptor,
        dispatch_core_manager,
        get_control_plane,
        get_dispatch_query_manager,
        get_max_num_eth_cores,
        get_reads_dispatch_cores) {
    TT_FATAL(
        noc_selection.downstream_noc == tt_metal::k_dispatch_downstream_noc,
        "Invalid downstream NOC specified for Dispatcher kernel");
    // Quasar only has a single NOC
    if (descriptor.cluster().arch() != tt::ARCH::QUASAR) {
        TT_FATAL(
            noc_selection.upstream_noc != noc_selection.downstream_noc,
            "Dispatcher kernel cannot have identical upstream and downstream NOCs.");
    }

    static_config_.is_h_variant = h_variant;
    static_config_.is_d_variant = d_variant;
    uint16_t channel = descriptor.cluster().get_assigned_channel_for_device(device_id);

    static_config_.dispatch_telemetry_disabled = descriptor.rtoptions().get_dispatch_telemetry_disabled();

    DispatchWorkerType type = DISPATCH;
    if (h_variant && d_variant) {
        this->logical_core_ = dispatch_core_manager.dispatcher_core(device_id, channel, cq_id);
        type = DISPATCH_HD;
    } else if (h_variant) {
        channel = descriptor.cluster().get_assigned_channel_for_device(servicing_device_id);
        this->logical_core_ = dispatch_core_manager.dispatcher_core(servicing_device_id, channel, cq_id);
        type = DISPATCH_H;
    } else if (d_variant) {
        this->logical_core_ = dispatch_core_manager.dispatcher_d_core(device_id, channel, cq_id);
        type = DISPATCH_D;
    }
    this->kernel_type_ = FDKernelType::DISPATCH;
    this->send_to_brisc_ = true;
    // Log dispatch core info based on virtual core to inspector
    auto virtual_core = this->GetVirtualCore();
    Inspector::set_dispatch_core_info(virtual_core, type, cq_id, device_id, servicing_device_id);
}

void DispatchKernel::GenerateStaticConfigs() {
    uint16_t channel = descriptor_.cluster().get_assigned_channel_for_device(device_->id());
    uint8_t cq_id_ = this->cq_id_;
    const auto& my_dispatch_constants = get_dispatch_mem_map();

    // May be zero if not using dispatch on fabric
    static_config_.fabric_header_rb_base =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::FABRIC_HEADER_RB, cq_id_);
    static_config_.fabric_header_rb_entries = DispatchSettings::FABRIC_HEADER_RB_ENTRIES;
    static_config_.my_fabric_sync_status_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::FABRIC_SYNC_STATUS, cq_id_);
    static_config_.realtime_profiler_msg_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG, cq_id_);
    static_config_.dispatch_telemetry_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_TELEMETRY, cq_id_);
    static_config_.dispatch_telemetry_control_addr = my_dispatch_constants.get_device_command_queue_addr(
        CommandQueueDeviceAddrType::DISPATCH_TELEMETRY_CONTROL, cq_id_);

    if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr =
            device_->sysmem_manager().is_dram_backed()
                ? get_absolute_cq_offset(
                      channel, cq_id_, cq_size, device_->sysmem_manager().get_dram_region_base_addr())
                : get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device_->sysmem_manager().get_issue_queue_size(cq_id_);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device_->sysmem_manager().get_completion_queue_size(cq_id_);

        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base(cq_id_);
        static_config_.dispatch_cb_log_page_size = DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id = tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = command_queue_start_addr;
        static_config_.completion_queue_base_addr = completion_queue_start_addr;
        static_config_.completion_queue_size = completion_queue_size;

        static_config_.my_downstream_cb_sem_id = 0;  // unused

        static_config_.prefetch_h_max_credits = 0;                   // unused prefetch_downstream_buffer_pages

        static_config_.packed_write_max_unicast_sub_cmds =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;
        static_config_.dispatch_s_sync_sem_base_addr = my_dispatch_constants.get_device_command_queue_addr(
            CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM, cq_id_);
        static_config_.max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES;
        static_config_.max_num_go_signal_noc_data_entries = DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
        static_config_.mcast_go_signal_addr =
            descriptor_.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
        static_config_.unicast_go_signal_addr =
            (descriptor_.hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
                ? descriptor_.hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
                : 0;
        static_config_.distributed_dispatcher = get_dispatch_query_manager_ref().distributed_dispatcher();
        static_config_.first_stream_used = my_dispatch_constants.get_dispatch_stream_index(0);
        static_config_.completion_counter_offset = my_dispatch_constants.get_completion_counter_offset(cq_id_);

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR, cq_id_);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD, cq_id_);
        static_config_.dev_dispatch_progress_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_PROGRESS, cq_id_);
    } else if (static_config_.is_h_variant.value()) {
        // DISPATCH_H services a remote chip, and so has a different channel
        channel = descriptor_.cluster().get_assigned_channel_for_device(servicing_device_id_);
        uint32_t cq_start = my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED);
        uint32_t cq_size = device_->sysmem_manager().get_cq_size();
        uint32_t command_queue_start_addr =
            device_->sysmem_manager().is_dram_backed()
                ? get_absolute_cq_offset(
                      channel, cq_id_, cq_size, device_->sysmem_manager().get_dram_region_base_addr())
                : get_absolute_cq_offset(channel, cq_id_, cq_size);
        uint32_t issue_queue_start_addr = command_queue_start_addr + cq_start;
        uint32_t issue_queue_size = device_->sysmem_manager().get_issue_queue_size(cq_id_);
        uint32_t completion_queue_start_addr = issue_queue_start_addr + issue_queue_size;
        uint32_t completion_queue_size = device_->sysmem_manager().get_completion_queue_size(cq_id_);

        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base(cq_id_);
        static_config_.dispatch_cb_log_page_size = DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id = tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = command_queue_start_addr;
        static_config_.completion_queue_base_addr = completion_queue_start_addr;
        static_config_.completion_queue_size = completion_queue_size;

        static_config_.my_downstream_cb_sem_id = 0;  // Unused

        static_config_.prefetch_h_max_credits = my_dispatch_constants.prefetch_d_buffer_pages();
        static_config_.packed_write_max_unicast_sub_cmds =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;
        static_config_.dispatch_s_sync_sem_base_addr = 0;       // Unused
        static_config_.max_num_worker_sems = 1;                 // Used for array sizing, set to 1 even if unused
        static_config_.max_num_go_signal_noc_data_entries = 1;  // Used for array sizing, sset to 1 even if unused
        static_config_.mcast_go_signal_addr = 0;                // Unused
        static_config_.unicast_go_signal_addr = 0;              // Unused
        static_config_.distributed_dispatcher = 0;              // Unused
        static_config_.first_stream_used = 0;                   // Unused
        static_config_.completion_counter_offset = 0;           // Unused

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR, cq_id_);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD, cq_id_);
        static_config_.dev_dispatch_progress_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_PROGRESS, cq_id_);
    } else if (static_config_.is_d_variant.value()) {
        static_config_.dispatch_cb_base = my_dispatch_constants.dispatch_buffer_base(cq_id_);
        static_config_.dispatch_cb_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
        static_config_.dispatch_cb_pages = my_dispatch_constants.dispatch_buffer_pages();
        static_config_.my_dispatch_cb_sem_id = tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());

        static_config_.dispatch_cb_blocks = DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS;
        static_config_.command_queue_base_addr = 0;  // These are unused for DISPATCH_D
        static_config_.completion_queue_base_addr = 0;
        static_config_.completion_queue_size = 0;

        static_config_.prefetch_h_max_credits = my_dispatch_constants.prefetch_d_buffer_pages();
        static_config_.my_downstream_cb_sem_id = tt_metal::CreateSemaphore(
            *program_, logical_core_, my_dispatch_constants.prefetch_d_buffer_pages(), GetCoreType());

        static_config_.packed_write_max_unicast_sub_cmds =
            device_->compute_with_storage_grid_size().x * device_->compute_with_storage_grid_size().y;
        static_config_.dispatch_s_sync_sem_base_addr = my_dispatch_constants.get_device_command_queue_addr(
            CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM, cq_id_);
        static_config_.max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES;
        static_config_.max_num_go_signal_noc_data_entries = DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
        static_config_.mcast_go_signal_addr =
            descriptor_.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
        static_config_.unicast_go_signal_addr =
            (descriptor_.hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
                ? descriptor_.hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
                : 0;
        static_config_.distributed_dispatcher = get_dispatch_query_manager_ref().distributed_dispatcher();
        static_config_.first_stream_used = my_dispatch_constants.get_dispatch_stream_index(0);
        static_config_.completion_counter_offset = my_dispatch_constants.get_completion_counter_offset(cq_id_);

        static_config_.host_completion_q_wr_ptr =
            my_dispatch_constants.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR);
        static_config_.dev_completion_q_wr_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_WR, cq_id_);
        static_config_.dev_completion_q_rd_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::COMPLETION_Q_RD, cq_id_);
        static_config_.dev_dispatch_progress_ptr =
            my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_PROGRESS, cq_id_);
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }

    if (!is_hd()) {
        create_edm_connection_sems(edm_connection_attributes_);
        static_config_.is_2d_fabric = tt::tt_fabric::is_2d_fabric_config(get_control_plane_ref().get_fabric_config());
    } else {
        static_config_.is_2d_fabric = false;
    }
}

void DispatchKernel::InitializeRuntimeArgsValues() {
    // Initialize runtime args offsets
    int current_offset = 0;
    static_config_.offsetof_my_dev_id = current_offset++;
    static_config_.offsetof_to_dev_id = current_offset++;
    static_config_.offsetof_router_direction = current_offset++;
    // Initialize runtime args
    runtime_args_.resize(current_offset);
    runtime_args_[static_config_.offsetof_my_dev_id.value()] = dependent_config_.my_dev_id.value_or(0);
    runtime_args_[static_config_.offsetof_to_dev_id.value()] = dependent_config_.to_dev_id.value_or(0);
    runtime_args_[static_config_.offsetof_router_direction.value()] = dependent_config_.router_direction.value_or(0);
}

void DispatchKernel::GenerateDependentConfigs() {
    if (static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        // Upstream
        TT_ASSERT(upstream_kernels_.size() == 1);
        auto* prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0]);
        TT_ASSERT(prefetch_kernel);
        dependent_config_.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        dependent_config_.upstream_dispatch_cb_sem_id = prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id;
        dependent_config_.upstream_sync_sem = prefetch_kernel->GetStaticConfig().downstream_sync_sem_id;

        if (prefetch_kernel->GetStaticConfig().is_h_variant.value() &&
            prefetch_kernel->GetStaticConfig().is_d_variant.value()) {
            dependent_config_.split_prefetch = false;
            dependent_config_.prefetch_h_noc_xy = 0;
            dependent_config_.prefetch_h_local_downstream_sem_addr = 0;
        } else {
            dependent_config_.split_prefetch = true;
            dependent_config_.prefetch_h_noc_xy = descriptor_.hal().noc_xy_encoding(
                prefetch_kernel->GetVirtualCore().x, prefetch_kernel->GetVirtualCore().y);
            dependent_config_.prefetch_h_local_downstream_sem_addr =
                prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id;
        }

        // Downstream
        if (get_dispatch_query_manager_ref().dispatch_s_enabled()) {
            TT_ASSERT(downstream_kernels_.size() == 1);
            auto* dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(downstream_kernels_[0]);
            TT_ASSERT(dispatch_s_kernel);
            dependent_config_.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
            dependent_config_.dispatch_d_shutdown_sem_id =
                dispatch_s_kernel->GetStaticConfig().dispatch_d_shutdown_sem_id;
        } else {
            // If no dispatch_s, no downstream
            TT_ASSERT(downstream_kernels_.empty());
            dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
            dependent_config_.dispatch_d_shutdown_sem_id = UNUSED_SEM_ID;
        }
        dependent_config_.downstream_logical_core = UNUSED_LOGICAL_CORE;  // Unused
        dependent_config_.downstream_cb_base = 0;                         // Unused
        dependent_config_.downstream_cb_size = 0;                         // Unused
        dependent_config_.downstream_cb_sem_id = UNUSED_SEM_ID;           // Unused
        dependent_config_.num_hops = 0;
    } else if (static_config_.is_h_variant.value()) {
        // Upstream, expect connection to DISPATCH_D

        // May be overwritten below
        dependent_config_.num_hops = 0;
        TT_ASSERT(upstream_kernels_.size() == 1);
        if (auto* dispatch_d = dynamic_cast<DispatchKernel*>(upstream_kernels_[0])) {
            dependent_config_.upstream_logical_core = dispatch_d->GetLogicalCore();
            dependent_config_.upstream_dispatch_cb_sem_id = dispatch_d->GetStaticConfig().my_downstream_cb_sem_id;
            dependent_config_.upstream_sync_sem = 0;  // Unused
            dependent_config_.num_hops = tt_metal::get_num_hops(descriptor_, device_id_, dispatch_d->GetDeviceId());
            assemble_2d_fabric_packet_header_args(
                this->dependent_config_, GetDeviceId(), dispatch_d->GetDeviceId(), get_control_plane_ref());
        } else {
            TT_FATAL(false, "Unimplemented path");
        }

        // Downstream
        // PREFETCH_H || FABRIC_MUX
        // Downstream, no official downstream core but use the field to connect is to the PREFETCH_H that we need to
        // write to when resuming sending of commands post exec_buf stall.
        bool found_prefetch_h = false;
        bool found_relay_mux = false;
        for (FDKernel* ds_kernel : downstream_kernels_) {
            if (auto* prefetch_h_kernel = dynamic_cast<PrefetchKernel*>(ds_kernel)) {
                TT_ASSERT(prefetch_h_kernel && prefetch_h_kernel->GetStaticConfig().is_h_variant.value());
                TT_ASSERT(!found_prefetch_h, "DISPATCH_H has multiple downstream PREFETCH_H kernels.");
                found_prefetch_h = true;
                dependent_config_.prefetch_h_noc_xy = descriptor_.hal().noc_xy_encoding(
                    prefetch_h_kernel->GetVirtualCore().x, prefetch_h_kernel->GetVirtualCore().y);
                dependent_config_.prefetch_h_local_downstream_sem_addr =
                    prefetch_h_kernel->GetStaticConfig().my_downstream_cb_sem_id;
            } else if (auto* relay_mux = dynamic_cast<RelayMux*>(ds_kernel)) {
                TT_ASSERT(!found_relay_mux, "DISPATCH_H has multiple downstream RELAY_MUX kernels.");
                found_relay_mux = true;

                constexpr tt::tt_fabric::FabricMuxChannelType ch_type =
                    tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL;
                tt_metal::assemble_fabric_mux_client_config_args(
                    node_id_, ch_type, relay_mux, dependent_config_.fabric_mux_client_config);
            } else {
                TT_FATAL(false, "DISPATCH_H Downstream - Unimplemented path");
            }
        }

        TT_ASSERT(found_prefetch_h, "DISPATCH_H expects a PREFETCH_H downstream");

        dependent_config_.downstream_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
        dependent_config_.dispatch_d_shutdown_sem_id = UNUSED_SEM_ID;
        dependent_config_.split_prefetch = true;
        dependent_config_.downstream_cb_base = 0;    // Unused
        dependent_config_.downstream_cb_size = 0;    // Unused
        dependent_config_.downstream_cb_sem_id = 0;  // Unused
    } else if (static_config_.is_d_variant.value()) {
        // Upstream, expect a PREFETCH_D
        TT_ASSERT(upstream_kernels_.size() == 1);
        auto* prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream_kernels_[0]);
        TT_ASSERT(prefetch_kernel);
        dependent_config_.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        dependent_config_.upstream_dispatch_cb_sem_id = prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id;
        dependent_config_.upstream_sync_sem = prefetch_kernel->GetStaticConfig().downstream_sync_sem_id;
        // May be overwritten below
        dependent_config_.num_hops = 0;

        if (prefetch_kernel->GetStaticConfig().is_h_variant.value() &&
            prefetch_kernel->GetStaticConfig().is_d_variant.value()) {
            dependent_config_.split_prefetch = false;
            dependent_config_.prefetch_h_noc_xy = 0;
            dependent_config_.prefetch_h_local_downstream_sem_addr = 0;
        } else {
            dependent_config_.split_prefetch = true;
            dependent_config_.prefetch_h_noc_xy = descriptor_.hal().noc_xy_encoding(
                prefetch_kernel->GetVirtualCore().x, prefetch_kernel->GetVirtualCore().y);
            dependent_config_.prefetch_h_local_downstream_sem_addr =
                prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id;
        }

        // Downstream, expect a MUX_D
        // Or direct connection to DISPATCH_H if using fabric
        //
        // + A Dispatch_s if enabled

        bool found_dispatch_s = false;
        bool found_dispatch_h = false;
        bool found_relay_mux = false;  // fabric mux
        for (auto* ds_kernel : downstream_kernels_) {
            if (auto* dispatch_s_kernel = dynamic_cast<DispatchSKernel*>(ds_kernel)) {
                TT_ASSERT(!found_dispatch_s, "DISPATCH_D has multiple downstream DISPATCH_S kernels.");
                dependent_config_.downstream_s_logical_core = dispatch_s_kernel->GetLogicalCore();
                dependent_config_.dispatch_d_shutdown_sem_id =
                    dispatch_s_kernel->GetStaticConfig().dispatch_d_shutdown_sem_id;
                found_dispatch_s = true;
            } else if (auto* dispatch_h_kernel = dynamic_cast<DispatchKernel*>(ds_kernel)) {
                TT_ASSERT(!found_dispatch_h, "DISPATCH_D has multiple downstream DISPATCH_H kernels.");
                dependent_config_.downstream_logical_core = dispatch_h_kernel->GetLogicalCore();
                dependent_config_.downstream_cb_size = dispatch_h_kernel->GetDispatchBufferSize();
                dependent_config_.downstream_cb_base = dispatch_h_kernel->GetStaticConfig().dispatch_cb_base;
                dependent_config_.downstream_cb_sem_id = dispatch_h_kernel->GetStaticConfig().my_dispatch_cb_sem_id;
                dependent_config_.num_hops =
                    tt_metal::get_num_hops(descriptor_, dispatch_h_kernel->GetDeviceId(), device_id_);
                assemble_2d_fabric_packet_header_args(
                    this->dependent_config_, GetDeviceId(), dispatch_h_kernel->GetDeviceId(), get_control_plane_ref());
                found_dispatch_h = true;
            } else if (auto* relay_mux = dynamic_cast<RelayMux*>(ds_kernel)) {
                TT_ASSERT(!found_relay_mux, "DISPATCH_D has multiple downstream RELAY_MUX kernels.");
                found_relay_mux = true;

                constexpr tt::tt_fabric::FabricMuxChannelType ch_type =
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
                tt_metal::assemble_fabric_mux_client_config_args(
                    node_id_, ch_type, relay_mux, dependent_config_.fabric_mux_client_config);
            } else {
                TT_FATAL(false, "Unexpected downstream kernel for dispatch_d");
            }
        }

        TT_FATAL(
            !get_dispatch_query_manager_ref().dispatch_s_enabled() || found_dispatch_s,
            "dispatch_d is missing dispatch_s downstream");
        TT_FATAL(found_dispatch_h, "Path not implemented for dispatch_d. dispatch_h in downstream is required");

        if (!found_dispatch_s) {
            dependent_config_.downstream_s_logical_core = UNUSED_LOGICAL_CORE;
            dependent_config_.dispatch_d_shutdown_sem_id = UNUSED_SEM_ID;
        }
    } else {
        TT_FATAL(false, "DispatchKernel must be one of (or both) H and D variants");
    }
}

void DispatchKernel::CreateKernel() {
    // Issue #19729: Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
    // Num num_virtual_active_eth_cores is set if the user application requested virtualizing the
    // number of ethernet cores across devices (to essentially fake uniformity). This value is the
    // max number of ethernet cores across all chip in the cluster.
    // num_physical_ethernet_cores is the number of actual available ethernet cores on the current device.
    // virtualize_num_eth_cores is set if the number of virtual cores is greater than the number of actual
    // ethernet cores in the chip.
    uint32_t num_virtual_active_eth_cores = get_max_num_eth_cores();
    uint32_t num_physical_active_eth_cores =
        get_control_plane_ref().get_active_ethernet_cores(device_->id(), /*skip_reserved_tunnel_cores*/ true).size();

    const auto& compute_grid_size = device_->compute_with_storage_grid_size();
    CoreRange device_worker_cores = CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1});
    auto virtual_start = device_->virtual_core_from_logical_core(device_worker_cores.start_coord, CoreType::WORKER);
    auto virtual_end = device_->virtual_core_from_logical_core(device_worker_cores.end_coord, CoreType::WORKER);
    auto virtual_core_range = CoreRange(virtual_start, virtual_end);

    auto my_virtual_core = get_virtual_core_coord(descriptor_, logical_core_, GetCoreType());
    auto upstream_virtual_core =
        get_virtual_core_coord(descriptor_, dependent_config_.upstream_logical_core.value(), GetCoreType());
    auto downstream_virtual_core =
        get_virtual_core_coord(descriptor_, dependent_config_.downstream_logical_core.value(), GetCoreType());
    auto downstream_s_virtual_core =
        get_virtual_core_coord(descriptor_, dependent_config_.downstream_s_logical_core.value(), GetCoreType());

    auto my_virtual_noc_coords = device_->virtual_noc0_coordinate(noc_selection_.non_dispatch_noc, my_virtual_core);
    auto upstream_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.upstream_noc, upstream_virtual_core);
    auto downstream_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.downstream_noc, downstream_virtual_core);
    auto downstream_s_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.downstream_noc, downstream_s_virtual_core);

    auto configuration = resolve_dispatch_configuration();
    configuration.kernel = experimental::DispatcherConfiguration{static_config_, dependent_config_};
    auto& resolved = *configuration.resolved;
    resolved.virtual_core = my_virtual_core;
    resolved.my_noc = my_virtual_noc_coords;
    resolved.upstream_noc = upstream_virtual_noc_coords;
    resolved.downstream_noc = downstream_virtual_noc_coords;
    resolved.subordinate_noc = downstream_s_virtual_noc_coords;
    resolved.virtual_eth_cores = num_virtual_active_eth_cores;
    resolved.physical_eth_cores = num_physical_active_eth_cores;
    resolved.worker_multicast = device_->get_noc_multicast_encoding(noc_selection_.downstream_noc, virtual_core_range);
    resolved.worker_count = device_worker_cores.size();
    resolved.fabric_flow_control_sem = edm_connection_attributes_.worker_flow_control_sem;
    resolved.fabric_teardown_sem = edm_connection_attributes_.worker_teardown_sem;
    resolved.fabric_buffer_index_sem = edm_connection_attributes_.worker_buffer_index_sem;
    auto plan = experimental::plan_dispatch_kernel(device_->arch(), configuration);
    auto& kernel = plan.kernels.front();
    configure_kernel_variant(std::string(experimental::dispatch_source_path(kernel.kind)), {}, kernel.defines, kernel.opt_level);
}

void DispatchKernel::ConfigureCore() {
    TT_ASSERT(static_config_.dispatch_telemetry_addr.has_value());
    TT_ASSERT(static_config_.dispatch_telemetry_disabled.has_value());
    dispatch_telemetry_types::DispatchCoreTelemetry zero_dispatch_telemetry{};
    if (static_config_.dispatch_telemetry_disabled.value()) {
        zero_dispatch_telemetry.signature = dispatch_telemetry_types::INVALID_TELEMETRY_SIGNATURE;
    }
    detail::WriteToDeviceL1(
        device_,
        logical_core_,
        static_config_.dispatch_telemetry_addr.value(),
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(&zero_dispatch_telemetry), sizeof(zero_dispatch_telemetry)),
        GetCoreType());

    // For all dispatchers, need to clear the dispatch message
    std::vector<uint32_t> zero = {0x0};
    const auto& my_dispatch_constants = get_dispatch_mem_map();
    uint32_t dispatch_s_sync_sem_base_addr =
        my_dispatch_constants.get_device_command_queue_addr(CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM, cq_id_);
    for (uint32_t i = 0; i < DispatchSettings::DISPATCH_MESSAGE_ENTRIES; i++) {
        uint32_t dispatch_s_sync_sem_addr = dispatch_s_sync_sem_base_addr + my_dispatch_constants.get_sync_offset(i);
        detail::WriteToDeviceL1(device_, logical_core_, dispatch_s_sync_sem_addr, zero, GetCoreType());
    }

    // For DISPATCH_D, need to clear completion q events
    if (!static_config_.is_h_variant.value() && this->static_config_.is_d_variant.value()) {
        uint32_t completion_q0_last_event_ptr = my_dispatch_constants.get_device_command_queue_addr(
            CommandQueueDeviceAddrType::COMPLETION_Q0_LAST_EVENT, cq_id_);
        uint32_t completion_q1_last_event_ptr = my_dispatch_constants.get_device_command_queue_addr(
            CommandQueueDeviceAddrType::COMPLETION_Q1_LAST_EVENT, cq_id_);
        detail::WriteToDeviceL1(device_, logical_core_, completion_q0_last_event_ptr, zero, GetCoreType());
        detail::WriteToDeviceL1(device_, logical_core_, completion_q1_last_event_ptr, zero, GetCoreType());
    }
}
