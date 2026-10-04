// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_plan.hpp"
#include <reflect>
#include <stdexcept>
#include <type_traits>
namespace tt::tt_metal::experimental {
namespace {
using Defines = std::map<std::string, std::string>;
template <class T>
void require_complete(const T& value) {
    if constexpr (requires { value.has_value(); }) {
        if (!value) {
            throw std::invalid_argument("published dispatch configuration field is unresolved");
        }
    } else {
        reflect::for_each([&](auto i) { require_complete(reflect::get<i>(value)); }, value);
    }
}
Defines prefetch_defines(const DispatchKernelConfiguration& n, const PrefetchConfiguration& k) {
    const auto& s = k.static_config;
    const auto& d = k.dependent_config;
    const auto& r = *n.resolved;
    if (!s.pcie_size || !s.is_h_variant || !s.is_d_variant) {
        throw std::invalid_argument("missing active prefetch field");
    }
    std::map<std::string, std::string> defines = {
        {"MY_NOC_X", std::to_string(r.my_noc.x)},
        {"MY_NOC_Y", std::to_string(r.my_noc.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(n.nocs.upstream_noc)},  // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(r.upstream_noc.x)},
        {"UPSTREAM_NOC_Y", std::to_string(r.upstream_noc.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(r.downstream_noc.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(r.downstream_noc.y)},
        {"DOWNSTREAM_SUBORDINATE_NOC_X", std::to_string(r.subordinate_noc.x)},
        {"DOWNSTREAM_SUBORDINATE_NOC_Y", std::to_string(r.subordinate_noc.y)},

        // Direct configuration values
        {"DOWNSTREAM_CB_BASE", std::to_string(d.downstream_cb_base.value())},
        {"DOWNSTREAM_CB_LOG_PAGE_SIZE", std::to_string(d.downstream_cb_log_page_size.value())},
        {"DOWNSTREAM_CB_PAGES", std::to_string(d.downstream_cb_pages.value())},
        {"MY_DOWNSTREAM_CB_SEM_ID", std::to_string(s.my_downstream_cb_sem_id.value())},
        {"DOWNSTREAM_CB_SEM_ID", std::to_string(d.downstream_cb_sem_id.value())},
        {"IS_CQ_DRAM_BACKED", std::to_string(r.dram_backed_cq)},
        {"PCIE_BASE", std::to_string(s.pcie_base.value())},
        {"PCIE_SIZE", std::to_string(s.pcie_size.value())},
        {"PREFETCH_Q_BASE", std::to_string(s.prefetch_q_base.value())},
        {"PREFETCH_Q_SIZE", std::to_string(s.prefetch_q_size.value())},
        {"PREFETCH_Q_RD_PTR_ADDR", std::to_string(s.prefetch_q_rd_ptr_addr.value())},
        {"PREFETCH_Q_PCIE_RD_PTR_ADDR", std::to_string(s.prefetch_q_pcie_rd_ptr_addr.value())},
        {"CMDDAT_Q_BASE", std::to_string(s.cmddat_q_base.value())},
        {"CMDDAT_Q_SIZE", std::to_string(s.cmddat_q_size.value())},
        {"SCRATCH_DB_BASE", std::to_string(s.scratch_db_base.value())},
        {"SCRATCH_DB_SIZE", std::to_string(s.scratch_db_size.value())},
        {"DOWNSTREAM_SYNC_SEM_ID", std::to_string(s.downstream_sync_sem_id.value())},
        {"CMDDAT_Q_PAGES", std::to_string(s.cmddat_q_pages.value())},
        {"MY_UPSTREAM_CB_SEM_ID", std::to_string(s.my_upstream_cb_sem_id.value())},
        {"UPSTREAM_CB_SEM_ID", std::to_string(d.upstream_cb_sem_id.value())},
        {"CMDDAT_Q_LOG_PAGE_SIZE", std::to_string(s.cmddat_q_log_page_size.value())},
        {"DISPATCH_S_BUFFER_BASE", std::to_string(s.dispatch_s_buffer_base.value())},
        {"MY_DISPATCH_S_CB_SEM_ID", std::to_string(s.my_dispatch_s_cb_sem_id.value())},
        {"DOWNSTREAM_DISPATCH_S_CB_SEM_ID", std::to_string(d.downstream_dispatch_s_cb_sem_id.value())},
        {"DISPATCH_S_BUFFER_SIZE", std::to_string(s.dispatch_s_buffer_size.value())},
        {"DISPATCH_S_CB_LOG_PAGE_SIZE", std::to_string(s.dispatch_s_cb_log_page_size.value())},
        {"RINGBUFFER_SIZE", std::to_string(s.ringbuffer_size.value())},
        // Fabric configuration
        {"FABRIC_HEADER_RB_BASE", std::to_string(s.fabric_header_rb_base.value())},
        {"FABRIC_HEADER_RB_ENTRIES", std::to_string(s.fabric_header_rb_entries.value())},
        {"MY_FABRIC_SYNC_STATUS_ADDR", std::to_string(s.my_fabric_sync_status_addr.value())},
        {"DISPATCH_TELEMETRY_ADDR", std::to_string(s.dispatch_telemetry_addr.value())},
        {"DISPATCH_TELEMETRY_DISABLED", std::to_string(s.dispatch_telemetry_disabled.value_or(false))},

        {"FABRIC_MUX_X", std::to_string(d.fabric_mux_client_config.virtual_x.value_or(0))},
        {"FABRIC_MUX_Y", std::to_string(d.fabric_mux_client_config.virtual_y.value_or(0))},
        {"FABRIC_MUX_NUM_BUFFERS_PER_CHANNEL",
         std::to_string(d.fabric_mux_client_config.num_buffers_per_channel.value_or(0))},
        {"FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES",
         std::to_string(d.fabric_mux_client_config.channel_buffer_size_bytes.value_or(0))},
        {"FABRIC_MUX_CHANNEL_BASE_ADDRESS",
         std::to_string(d.fabric_mux_client_config.channel_base_address.value_or(0))},
        {"FABRIC_MUX_CONNECTION_INFO_ADDRESS",
         std::to_string(d.fabric_mux_client_config.connection_info_address.value_or(0))},
        {"FABRIC_MUX_CONNECTION_HANDSHAKE_ADDRESS",
         std::to_string(d.fabric_mux_client_config.connection_handshake_address.value_or(0))},
        {"FABRIC_MUX_FLOW_CONTROL_ADDRESS",
         std::to_string(d.fabric_mux_client_config.flow_control_address.value_or(0))},
        {"FABRIC_MUX_BUFFER_INDEX_ADDRESS",
         std::to_string(d.fabric_mux_client_config.buffer_index_address.value_or(0))},
        {"FABRIC_MUX_STATUS_ADDRESS", std::to_string(d.fabric_mux_client_config.status_address.value_or(0))},
        {"FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS",
         std::to_string(d.fabric_mux_client_config.termination_signal_address.value_or(0))},
        {"WORKER_CREDITS_STREAM_ID", std::to_string(d.fabric_mux_client_config.worker_credits_stream_id.value_or(0))},

        {"FABRIC_WORKER_FLOW_CONTROL_SEM", std::to_string(r.fabric_flow_control_sem)},
        {"FABRIC_WORKER_TEARDOWN_SEM", std::to_string(r.fabric_teardown_sem)},
        {"FABRIC_WORKER_BUFFER_INDEX_SEM", std::to_string(r.fabric_buffer_index_sem)},

        {"NUM_HOPS", std::to_string(d.num_hops.value())},

        {"EW_DIM", std::to_string(d.ew_dim.value_or(0))},
        {"TO_MESH_ID", std::to_string(d.to_mesh_id.value_or(0))},
        {"IS_D_VARIANT", std::to_string(s.is_d_variant.value())},
        {"IS_H_VARIANT", std::to_string(s.is_h_variant.value())},
    };
    defines["PREFETCH_Q_ENTRY_BITS"] = std::to_string(r.prefetch_q_entry_bits);
    if (!(s.is_h_variant.value() && s.is_d_variant.value())) {
        defines["FABRIC_RELAY"] = "1";
        if (s.is_2d_fabric.value_or(false)) {
            defines["FABRIC_2D"] = "1";
        }
    }
    if (r.dram_backed_cq) {
        defines["DRAM_BACKED_CQ_BANK_ID"] = std::to_string(r.dram_bank);
    }
    defines["OFFSETOF_MY_DEV_ID"] = std::to_string(s.offsetof_my_dev_id.value_or(0));
    defines["OFFSETOF_TO_DEV_ID"] = std::to_string(s.offsetof_to_dev_id.value_or(0));
    defines["OFFSETOF_ROUTER_DIRECTION"] = std::to_string(s.offsetof_router_direction.value_or(0));
    return defines;
}
Defines dispatcher_defines(const DispatchKernelConfiguration& n, const DispatcherConfiguration& k) {
    const auto& s = k.static_config;
    const auto& d = k.dependent_config;
    const auto& r = *n.resolved;
    if (!s.is_h_variant || !s.is_d_variant) {
        throw std::invalid_argument("missing dispatch variant");
    }
    std::map<std::string, std::string> defines = {
        {"MY_NOC_X", std::to_string(r.my_noc.x)},
        {"MY_NOC_Y", std::to_string(r.my_noc.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(n.nocs.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(r.upstream_noc.x)},
        {"UPSTREAM_NOC_Y", std::to_string(r.upstream_noc.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(r.downstream_noc.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(r.downstream_noc.y)},
        {"DOWNSTREAM_SUBORDINATE_NOC_X", std::to_string(r.subordinate_noc.x)},
        {"DOWNSTREAM_SUBORDINATE_NOC_Y", std::to_string(r.subordinate_noc.y)},

        // Add all the dispatch-specific defines
        {"DISPATCH_CB_BASE", std::to_string(s.dispatch_cb_base.value())},
        {"DISPATCH_CB_LOG_PAGE_SIZE", std::to_string(s.dispatch_cb_log_page_size.value())},
        {"DISPATCH_CB_PAGES", std::to_string(s.dispatch_cb_pages.value())},
        {"MY_DISPATCH_CB_SEM_ID", std::to_string(s.my_dispatch_cb_sem_id.value())},
        {"UPSTREAM_DISPATCH_CB_SEM_ID", std::to_string(d.upstream_dispatch_cb_sem_id.value())},
        {"DISPATCH_D_SHUTDOWN_SEM_ID", std::to_string(d.dispatch_d_shutdown_sem_id.value())},
        {"DISPATCH_CB_BLOCKS", std::to_string(s.dispatch_cb_blocks.value())},
        {"UPSTREAM_SYNC_SEM", std::to_string(d.upstream_sync_sem.value())},
        {"IS_CQ_DRAM_BACKED", std::to_string(r.dram_backed_cq)},
        {"COMMAND_QUEUE_BASE_ADDR", std::to_string(s.command_queue_base_addr.value())},
        {"COMPLETION_QUEUE_BASE_ADDR", std::to_string(s.completion_queue_base_addr.value())},
        {"COMPLETION_QUEUE_SIZE", std::to_string(s.completion_queue_size.value())},
        {"DOWNSTREAM_CB_BASE", std::to_string(d.downstream_cb_base.value())},
        {"DOWNSTREAM_CB_SIZE", std::to_string(d.downstream_cb_size.value())},
        {"MY_DOWNSTREAM_CB_SEM_ID", std::to_string(s.my_downstream_cb_sem_id.value())},
        {"DOWNSTREAM_CB_SEM_ID", std::to_string(d.downstream_cb_sem_id.value())},
        {"SPLIT_PREFETCH", std::to_string(d.split_prefetch.value())},
        {"PREFETCH_H_NOC_XY", std::to_string(d.prefetch_h_noc_xy.value())},
        {"PREFETCH_H_LOCAL_DOWNSTREAM_SEM_ADDR", std::to_string(d.prefetch_h_local_downstream_sem_addr.value())},
        {"PREFETCH_H_MAX_CREDITS", std::to_string(s.prefetch_h_max_credits.value())},
        {"PACKED_WRITE_MAX_UNICAST_SUB_CMDS", std::to_string(s.packed_write_max_unicast_sub_cmds.value())},
        {"DISPATCH_S_SYNC_SEM_BASE_ADDR", std::to_string(s.dispatch_s_sync_sem_base_addr.value())},
        {"MAX_NUM_WORKER_SEMS", std::to_string(s.max_num_worker_sems.value())},
        {"MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES", std::to_string(s.max_num_go_signal_noc_data_entries.value())},
        {"MCAST_GO_SIGNAL_ADDR", std::to_string(s.mcast_go_signal_addr.value())},
        {"UNICAST_GO_SIGNAL_ADDR", std::to_string(s.unicast_go_signal_addr.value())},
        {"DISTRIBUTED_DISPATCHER", std::to_string(s.distributed_dispatcher.value())},
        {"HOST_COMPLETION_Q_WR_PTR", std::to_string(s.host_completion_q_wr_ptr.value())},
        {"DEV_COMPLETION_Q_WR_PTR", std::to_string(s.dev_completion_q_wr_ptr.value())},
        {"DEV_COMPLETION_Q_RD_PTR", std::to_string(s.dev_completion_q_rd_ptr.value())},
        {"DEV_DISPATCH_PROGRESS_PTR", std::to_string(s.dev_dispatch_progress_ptr.value())},
        {"FIRST_STREAM_USED", std::to_string(s.first_stream_used.value())},
        {"COMPLETION_COUNTER_OFFSET", std::to_string(s.completion_counter_offset.value())},
        {"VIRTUALIZE_UNICAST_CORES", std::to_string((r.virtual_eth_cores > r.physical_eth_cores))},
        {"NUM_VIRTUAL_UNICAST_CORES", std::to_string(r.virtual_eth_cores)},
        {"NUM_PHYSICAL_UNICAST_CORES", std::to_string(r.physical_eth_cores)},
        {"FABRIC_HEADER_RB_BASE", std::to_string(s.fabric_header_rb_base.value())},
        {"FABRIC_HEADER_RB_ENTRIES", std::to_string(s.fabric_header_rb_entries.value())},
        {"MY_FABRIC_SYNC_STATUS_ADDR", std::to_string(s.my_fabric_sync_status_addr.value())},
        {"REALTIME_PROFILER_MSG_ADDR", std::to_string(s.realtime_profiler_msg_addr.value())},
        {"DISPATCH_TELEMETRY_ADDR", std::to_string(s.dispatch_telemetry_addr.value())},
        {"DISPATCH_TELEMETRY_CONTROL_ADDR", std::to_string(s.dispatch_telemetry_control_addr.value())},
        {"DISPATCH_TELEMETRY_DISABLED", std::to_string(s.dispatch_telemetry_disabled.value_or(false))},
        {"FABRIC_MUX_X", std::to_string(d.fabric_mux_client_config.virtual_x.value_or(0))},
        {"FABRIC_MUX_Y", std::to_string(d.fabric_mux_client_config.virtual_y.value_or(0))},
        {"FABRIC_MUX_NUM_BUFFERS_PER_CHANNEL",
         std::to_string(d.fabric_mux_client_config.num_buffers_per_channel.value_or(0))},
        {"FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES",
         std::to_string(d.fabric_mux_client_config.channel_buffer_size_bytes.value_or(0))},
        {"FABRIC_MUX_CHANNEL_BASE_ADDRESS",
         std::to_string(d.fabric_mux_client_config.channel_base_address.value_or(0))},
        {"FABRIC_MUX_CONNECTION_INFO_ADDRESS",
         std::to_string(d.fabric_mux_client_config.connection_info_address.value_or(0))},
        {"FABRIC_MUX_CONNECTION_HANDSHAKE_ADDRESS",
         std::to_string(d.fabric_mux_client_config.connection_handshake_address.value_or(0))},
        {"FABRIC_MUX_FLOW_CONTROL_ADDRESS",
         std::to_string(d.fabric_mux_client_config.flow_control_address.value_or(0))},
        {"FABRIC_MUX_BUFFER_INDEX_ADDRESS",
         std::to_string(d.fabric_mux_client_config.buffer_index_address.value_or(0))},
        {"FABRIC_MUX_STATUS_ADDRESS", std::to_string(d.fabric_mux_client_config.status_address.value_or(0))},
        {"FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS",
         std::to_string(d.fabric_mux_client_config.termination_signal_address.value_or(0))},
        {"WORKER_CREDITS_STREAM_ID", std::to_string(d.fabric_mux_client_config.worker_credits_stream_id.value_or(0))},
        {"FABRIC_WORKER_FLOW_CONTROL_SEM", std::to_string(r.fabric_flow_control_sem)},
        {"FABRIC_WORKER_TEARDOWN_SEM", std::to_string(r.fabric_teardown_sem)},
        {"FABRIC_WORKER_BUFFER_INDEX_SEM", std::to_string(r.fabric_buffer_index_sem)},
        {"NUM_HOPS", std::to_string(d.num_hops.value())},
        {"EW_DIM", std::to_string(d.ew_dim.value_or(0))},
        {"TO_MESH_ID", std::to_string(d.to_mesh_id.value_or(0))},
        {"WORKER_MCAST_GRID", std::to_string(r.worker_multicast)},
        {"NUM_WORKER_CORES_TO_MCAST", std::to_string(r.worker_count)},
        {"IS_D_VARIANT", std::to_string(s.is_d_variant.value())},
        {"IS_H_VARIANT", std::to_string(s.is_h_variant.value())},
    };
    if (!(s.is_h_variant.value() && s.is_d_variant.value())) {
        defines["FABRIC_RELAY"] = "1";
        if (s.is_2d_fabric.value_or(false)) {
            defines["FABRIC_2D"] = "1";
        }
    }
    if (r.dram_backed_cq) {
        defines["DRAM_BACKED_CQ_BANK_ID"] = std::to_string(r.dram_bank);
    }
    defines["OFFSETOF_MY_DEV_ID"] = std::to_string(s.offsetof_my_dev_id.value_or(0));
    defines["OFFSETOF_TO_DEV_ID"] = std::to_string(s.offsetof_to_dev_id.value_or(0));
    defines["OFFSETOF_ROUTER_DIRECTION"] = std::to_string(s.offsetof_router_direction.value_or(0));
    return defines;
}
Defines subordinate_defines(const DispatchKernelConfiguration& n, const SubordinateConfiguration& k) {
    const auto& s = k.static_config;
    const auto& d = k.dependent_config;
    const auto& r = *n.resolved;

    std::map<std::string, std::string> defines = {
        {"MY_NOC_X", std::to_string(r.my_noc.x)},
        {"MY_NOC_Y", std::to_string(r.my_noc.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(n.nocs.upstream_noc)},  // Unused, remove later
        {"UPSTREAM_NOC_X", std::to_string(r.upstream_noc.x)},
        {"UPSTREAM_NOC_Y", std::to_string(r.upstream_noc.y)},
        {"DOWNSTREAM_NOC_X", std::to_string(r.downstream_noc.x)},
        {"DOWNSTREAM_NOC_Y", std::to_string(r.downstream_noc.y)},
        {"DOWNSTREAM_SUBORDINATE_NOC_X", std::to_string(r.subordinate_noc.x)},  // Unused, remove later
        {"DOWNSTREAM_SUBORDINATE_NOC_Y", std::to_string(r.subordinate_noc.y)},  // Unused, remove later
        {"CB_BASE", std::to_string(s.cb_base.value())},
        {"CB_LOG_PAGE_SIZE", std::to_string(s.cb_log_page_size.value())},
        {"CB_SIZE", std::to_string(s.cb_size.value())},
        {"MY_DISPATCH_CB_SEM_ID", std::to_string(s.my_dispatch_cb_sem_id.value())},
        {"DISPATCH_D_SHUTDOWN_SEM_ID", std::to_string(s.dispatch_d_shutdown_sem_id.value())},
        {"UPSTREAM_DISPATCH_CB_SEM_ID", std::to_string(d.upstream_dispatch_cb_sem_id.value())},
        {"DISPATCH_S_SYNC_SEM_BASE_ADDR", std::to_string(s.dispatch_s_sync_sem_base_addr.value())},
        {"MCAST_GO_SIGNAL_ADDR", std::to_string(s.mcast_go_signal_addr.value())},
        {"UNICAST_GO_SIGNAL_ADDR", std::to_string(s.unicast_go_signal_addr.value())},
        {"DISTRIBUTED_DISPATCHER", std::to_string(s.distributed_dispatcher.value())},
        {"FIRST_STREAM_USED", std::to_string(s.first_stream_used.value())},
        {"COMPLETION_COUNTER_OFFSET", std::to_string(s.completion_counter_offset.value())},
        {"MAX_NUM_WORKER_SEMS", std::to_string(s.max_num_worker_sems.value())},
        {"MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES", std::to_string(s.max_num_go_signal_noc_data_entries.value())},
        {"VIRTUALIZE_UNICAST_CORES", std::to_string((r.virtual_eth_cores > r.physical_eth_cores))},
        {"NUM_VIRTUAL_UNICAST_CORES", std::to_string(r.virtual_eth_cores)},
        {"NUM_PHYSICAL_UNICAST_CORES", std::to_string(r.physical_eth_cores)},
        {"WORKER_MCAST_GRID", std::to_string(r.worker_multicast)},
        {"NUM_WORKER_CORES_TO_MCAST", std::to_string(r.worker_count)},
        {"REALTIME_PROFILER_MSG_ADDR", std::to_string(s.realtime_profiler_msg_addr.value())},
        {"DISPATCH_TELEMETRY_ADDR", std::to_string(s.dispatch_telemetry_addr.value())},
        {"DISPATCH_TELEMETRY_DISABLED", std::to_string(s.dispatch_telemetry_disabled.value_or(false))},
        {"DISPATCH_TELEMETRY_CONTROL_ADDR", std::to_string(s.dispatch_telemetry_control_addr.value())},
        {"DEVICE_PRINT_DISPATCH_ENABLED", std::to_string(s.device_print_dispatch_enabled.value_or(0))},
        // For each per-device dispatch_s build, MaxNocLocations equals the actual print-core count
        // for that device — passed as a compile-time #define so DevicePrintDispatch<>'s LDM arrays
        // (rw_noc_addresses, cache_buffer_offsets, cache_buffer_sizes, noc_locations_to_process)
        // are sized to actual usage instead of device_print_dispatch::DEFAULT_MAX_NOC_LOCATIONS.
        {"DEVICE_PRINT_MAX_NOC_LOCATIONS", std::to_string(s.device_print_noc_locations_count.value_or(0))},
        {"DEVICE_PRINT_NOC_LOCATIONS_ADDR", std::to_string(s.device_print_noc_locations_addr.value_or(0))},
        {"DEVICE_PRINT_NOC_LOCATIONS_COUNT", std::to_string(s.device_print_noc_locations_count.value_or(0))},
        {"DEVICE_PRINT_L1_CACHE_ADDR", std::to_string(s.device_print_l1_cache_addr.value_or(0))},
        {"DEVICE_PRINT_L1_CACHE_SIZE", std::to_string(s.device_print_l1_cache_size.value_or(0))},
        {"DEVICE_PRINT_DRAM_X", std::to_string(s.device_print_dram_x.value_or(0))},
        {"DEVICE_PRINT_DRAM_Y", std::to_string(s.device_print_dram_y.value_or(0))},
        {"DEVICE_PRINT_DRAM_RW_PTRS", std::to_string(s.device_print_dram_rw_ptrs.value_or(0)) + "ULL"},
        {"DEVICE_PRINT_DRAM_BUF_ADDR", std::to_string(s.device_print_dram_buf_addr.value_or(0)) + "ULL"},
        {"DEVICE_PRINT_DRAM_BUF_SIZE", std::to_string(s.device_print_dram_buf_size.value_or(0))},
        {"DEVICE_PRINT_CYCLES_FOR_STALL", std::to_string(s.device_print_cycles_for_stall.value_or(0)) + "ULL"},
        {"DEVICE_PRINT_CYCLES_FOR_FULL", std::to_string(s.device_print_cycles_for_full.value_or(0)) + "ULL"},
    };

    return defines;
}
Defines subordinate_compute_defines(const DispatchKernelConfiguration& n, const SubordinateConfiguration& k) {
    const auto& s = k.static_config;
    const auto& r = *n.resolved;
    std::map<std::string, std::string> compute_defines = {
        {"DISPATCH_KERNEL", "1"},
        {"FIRST_STREAM_INDEX", std::to_string(s.first_stream_used.value())},
        {"NUM_STREAMS_TO_MONITOR", std::to_string(s.max_num_worker_sems.value())},
        {"REALTIME_PROFILER_MSG_ADDR", std::to_string(s.realtime_profiler_msg_addr.value())},
        {"DISPATCH_TELEMETRY_ADDR", std::to_string(s.dispatch_telemetry_addr.value())},
        {"DISPATCH_TELEMETRY_DISABLED", std::to_string(s.dispatch_telemetry_disabled.value_or(false))},
        {"TOTAL_SUB_DEVICES", std::to_string(s.max_num_worker_sems.value())},
        {"DISPATCH_TELEMETRY_CONTROL_ADDR", std::to_string(s.dispatch_telemetry_control_addr.value())},
        {"NUM_WORKER_CORES", std::to_string(r.worker_count)},
    };
    return compute_defines;
}
}  // namespace
std::map<std::string, std::string> render_dispatch_common(const DispatchResolvedConfiguration& r) {
    Defines defines{{"DISPATCH_KERNEL", "1"}};
    if (r.force_watcher_no_inline) {
        defines["WATCHER_NOINLINE"] = "1";
    }
    if (r.watcher_dispatch_disabled) {
        defines["FORCE_WATCHER_OFF"] = "1";
    }
    if (!r.reads_dispatch_cores) {
        defines["FORCE_DPRINT_OFF"] = "1";
    }
    if (r.galaxy_cluster) {
        defines["GALAXY_CLUSTER"] = "1";
    }
    return defines;
}
std::string_view dispatch_source_path(DispatchSourceKind kind) {
    switch (kind) {
        case DispatchSourceKind::Prefetch: return "tt_metal/impl/dispatch/kernels/cq_prefetch.cpp";
        case DispatchSourceKind::Dispatcher: return "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp";
        case DispatchSourceKind::Subordinate: return "tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp";
        case DispatchSourceKind::SubordinateCompute:
            return "tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate_compute.cpp";
    }
    throw std::invalid_argument("invalid dispatch source kind");
}
std::vector<HalProcessorIdentifier> dispatch_processors(const DispatchKernelConfiguration& n) {
    std::vector<HalProcessorIdentifier> ids{n.processor};
    if (std::holds_alternative<SubordinateConfiguration>(n.kernel) &&
        n.processor.core_type == HalProgrammableCoreType::TENSIX) {
        for (int i = 0; i != 3; ++i) {
            ids.push_back({HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, i});
        }
    }
    return ids;
}
void validate_published_dispatch_configuration(const DispatchKernelConfiguration& n) {
    if (!n.resolved || !n.resolved->command_queue_size || !n.resolved->worker_count ||
        !n.resolved->prefetch_q_entry_bits || n.resolved->galaxy_cluster) {
        throw std::invalid_argument("published dispatch requires complete Blackhole nonfabric facts");
    }
    const auto expected_processor =
        n.processor.core_type == HalProgrammableCoreType::TENSIX &&
        std::holds_alternative<SubordinateConfiguration>(n.kernel) ? 1 : 0;
    if (n.processor.processor_class != HalProcessorClassType::DM ||
        n.processor.processor_type != expected_processor) {
        throw std::invalid_argument("published dispatch processor does not match kernel role");
    }
    std::visit(
        [](const auto& k) {
            using K = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<K, std::monostate>) {
                throw std::invalid_argument("missing dispatch configuration");
            } else {
                require_complete(k.static_config);
                require_complete(k.dependent_config);
                if constexpr (std::is_same_v<K, PrefetchConfiguration>) {
                    if (!*k.static_config.pcie_size || !*k.static_config.prefetch_q_size) {
                        throw std::invalid_argument("empty prefetch queue configuration");
                    }
                }
                if constexpr (!std::is_same_v<K, SubordinateConfiguration>) {
                    if (!*k.static_config.is_h_variant || !*k.static_config.is_d_variant ||
                        *k.static_config.is_2d_fabric || *k.dependent_config.num_hops) {
                        throw std::invalid_argument("published fabric dispatch is unsupported");
                    }
                }
            }
        },
        n.kernel);
}
DispatchPlan plan_dispatch_kernel(tt::ARCH arch, const DispatchKernelConfiguration& input) {
    if (!input.resolved || !input.resolved->command_queue_size || !input.resolved->prefetch_q_entry_bits) {
        throw std::invalid_argument("dispatch planner requires resolved SDK facts");
    }
    DispatchPlan plan{input, {}};
    const auto core = input.processor.core_type;
    const auto level = core == HalProgrammableCoreType::IDLE_ETH ? KernelBuildOptLevel::Os : KernelBuildOptLevel::O2;
    auto common = render_dispatch_common(*input.resolved);
    auto add = [&](DispatchSourceKind kind,
                   std::vector<HalProcessorIdentifier> processors,
                   KernelBuildOptLevel opt,
                   Defines defines) {
        for (const auto& [key, value] : common) {
            defines.try_emplace(key, value);
        }
        plan.kernels.push_back({kind, std::move(processors), opt, std::move(defines)});
    };
    std::visit(
        [&](const auto& k) {
            using K = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<K, PrefetchConfiguration>) {
                add(DispatchSourceKind::Prefetch, {input.processor}, level, prefetch_defines(input, k));
            } else if constexpr (std::is_same_v<K, DispatcherConfiguration>) {
                add(DispatchSourceKind::Dispatcher, {input.processor}, level, dispatcher_defines(input, k));
            } else if constexpr (std::is_same_v<K, SubordinateConfiguration>) {
                add(DispatchSourceKind::Subordinate,
                    {input.processor},
                    KernelBuildOptLevel::Os,
                    subordinate_defines(input, k));
                if (core == HalProgrammableCoreType::TENSIX && arch != tt::ARCH::QUASAR) {
                    auto ids = dispatch_processors(input);
                    ids.erase(ids.begin());
                    plan.kernels.push_back(
                        {DispatchSourceKind::SubordinateCompute,
                         std::move(ids),
                         KernelBuildOptLevel::O3,
                         subordinate_compute_defines(input, k)});
                }
            } else {
                throw std::invalid_argument("missing dispatch configuration");
            }
        },
        input.kernel);
    return plan;
}
}  // namespace tt::tt_metal::experimental
