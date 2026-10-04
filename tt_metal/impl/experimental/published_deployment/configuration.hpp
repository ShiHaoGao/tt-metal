// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <optional>
#include <variant>
#include <tt-metalium/experimental/published_deployment.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
namespace tt::tt_metal {
// These are the original dispatch configuration types. Source/JIT and
// publication consume the same definitions; the old headers include this owner.
struct noc_selection_t {
    tt::tt_metal::NOC non_dispatch_noc;  // For communicating with workers/DRAM/host
    tt::tt_metal::NOC upstream_noc;      // For communicating with upstream dispatch modules
    tt::tt_metal::NOC downstream_noc;    // For communicating with downstream dispatch modules
};

struct relay_mux_client_config {
    std::optional<uint32_t> virtual_x;
    std::optional<uint32_t> virtual_y;
    std::optional<uint32_t> num_buffers_per_channel;
    std::optional<uint32_t> channel_buffer_size_bytes;
    std::optional<uint32_t> channel_base_address;
    std::optional<uint32_t> connection_info_address;
    std::optional<uint32_t> connection_handshake_address;
    std::optional<uint32_t> flow_control_address;
    std::optional<uint32_t> buffer_index_address;
    std::optional<uint32_t> status_address;
    std::optional<uint32_t> termination_signal_address;
    std::optional<uint32_t> worker_credits_stream_id;
};

struct prefetch_static_config_t {
    std::optional<uint32_t> my_downstream_cb_sem_id;

    std::optional<uint32_t> pcie_base;
    std::optional<uint32_t> pcie_size;
    std::optional<uint32_t> prefetch_q_base;
    std::optional<uint32_t> prefetch_q_size;
    std::optional<uint32_t> prefetch_q_rd_ptr_addr;
    std::optional<uint32_t> prefetch_q_pcie_rd_ptr_addr;

    std::optional<uint32_t> cmddat_q_base;
    std::optional<uint32_t> cmddat_q_size;

    // Used for prefetch_h
    std::optional<uint32_t> scratch_db_base;
    std::optional<uint32_t> scratch_db_size;
    std::optional<uint32_t> downstream_sync_sem_id;
    std::optional<uint32_t> ringbuffer_size;

    // Used for prefetch_d
    std::optional<uint32_t> cmddat_q_pages;
    std::optional<uint32_t> my_upstream_cb_sem_id;
    std::optional<uint32_t> cmddat_q_log_page_size;

    // Used for prefetch_d <--> dispatch_s data path
    std::optional<uint32_t> dispatch_s_buffer_base;
    std::optional<uint32_t> my_dispatch_s_cb_sem_id;
    std::optional<uint32_t> dispatch_s_buffer_size;
    std::optional<uint32_t> dispatch_s_cb_log_page_size;

    std::optional<uint32_t> fabric_header_rb_base;
    std::optional<uint32_t> fabric_header_rb_entries;
    std::optional<uint32_t> my_fabric_sync_status_addr;

    std::optional<uint32_t> dispatch_telemetry_addr;
    std::optional<bool> dispatch_telemetry_disabled;

    std::optional<bool> is_2d_fabric;

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;

    // Offsets of runtime args
    std::optional<uint32_t> offsetof_my_dev_id;
    std::optional<uint32_t> offsetof_to_dev_id;
    std::optional<uint32_t> offsetof_router_direction;
};

struct prefetch_dependent_config_t {
    std::optional<tt_cxy_pair> upstream_logical_core;
    std::optional<tt_cxy_pair> downstream_logical_core;
    std::optional<tt_cxy_pair> downstream_s_logical_core;

    std::optional<uint32_t> downstream_cb_base;
    std::optional<uint32_t> downstream_cb_log_page_size;
    std::optional<uint32_t> downstream_cb_pages;
    std::optional<uint32_t> downstream_cb_sem_id;

    std::optional<uint32_t> upstream_cb_sem_id;

    std::optional<uint32_t> downstream_dispatch_s_cb_sem_id;

    std::optional<uint32_t> num_hops;

    tt::tt_metal::relay_mux_client_config fabric_mux_client_config;

    std::optional<uint32_t> my_dev_id;
    std::optional<uint32_t> ew_dim;
    std::optional<uint32_t> to_mesh_id;
    std::optional<uint32_t> to_dev_id;
    std::optional<uint32_t> router_direction;
};

struct dispatch_static_config_t {
    std::optional<uint32_t> dispatch_cb_base;  // 0
    std::optional<uint32_t> dispatch_cb_log_page_size;
    std::optional<uint32_t> dispatch_cb_pages;
    std::optional<uint32_t> my_dispatch_cb_sem_id;

    std::optional<uint32_t> dispatch_cb_blocks;  // 5
    std::optional<uint32_t> command_queue_base_addr;
    std::optional<uint32_t> completion_queue_base_addr;
    std::optional<uint32_t> completion_queue_size;

    std::optional<uint32_t> my_downstream_cb_sem_id;

    std::optional<uint32_t> prefetch_h_max_credits;             // Used if split_prefetch is true

    std::optional<uint32_t> packed_write_max_unicast_sub_cmds;  // 19
    std::optional<uint32_t> dispatch_s_sync_sem_base_addr;
    std::optional<uint32_t> max_num_worker_sems;
    std::optional<uint32_t> max_num_go_signal_noc_data_entries;
    std::optional<uint32_t> mcast_go_signal_addr;
    std::optional<uint32_t> unicast_go_signal_addr;
    std::optional<uint32_t> distributed_dispatcher;
    std::optional<uint32_t> first_stream_used;
    std::optional<uint32_t> completion_counter_offset;

    std::optional<uint32_t> host_completion_q_wr_ptr;  // 26
    std::optional<uint32_t> dev_completion_q_wr_ptr;
    std::optional<uint32_t> dev_completion_q_rd_ptr;
    std::optional<uint32_t> dev_dispatch_progress_ptr;

    std::optional<uint32_t> fabric_header_rb_base;
    std::optional<uint32_t> fabric_header_rb_entries;
    std::optional<uint32_t> my_fabric_sync_status_addr;
    std::optional<bool> is_2d_fabric;

    // Dispatch-core-local L1 address of the realtime_profiler_msg_t block (state, ping-pong
    // timestamps, host<->device sync, and the program-id handoff FIFO between cq_dispatch BRISC
    // and cq_dispatch_subordinate NCRISC). Assigned by DispatchMemMap via
    // CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG. The same address must be passed to
    // DispatchSKernel and to the RT-profiler core kernels.
    std::optional<uint32_t> realtime_profiler_msg_addr;

    std::optional<uint32_t> dispatch_telemetry_addr;
    std::optional<uint32_t> dispatch_telemetry_control_addr;
    std::optional<bool> dispatch_telemetry_disabled;

    std::optional<bool> is_d_variant;
    std::optional<bool> is_h_variant;

    // Offsets of runtime args
    std::optional<uint32_t> offsetof_my_dev_id;
    std::optional<uint32_t> offsetof_to_dev_id;
    std::optional<uint32_t> offsetof_router_direction;
};

struct dispatch_dependent_config_t {
    std::optional<tt_cxy_pair> upstream_logical_core;      // Dependent
    std::optional<tt_cxy_pair> downstream_logical_core;    // Dependent
    std::optional<tt_cxy_pair> downstream_s_logical_core;  // Dependent

    std::optional<uint32_t> upstream_dispatch_cb_sem_id;  // Dependent

    std::optional<uint32_t> upstream_sync_sem;  // Dependent
    std::optional<uint32_t> dispatch_d_shutdown_sem_id;

    std::optional<uint32_t> downstream_cb_base;    // 10, dependent
    std::optional<uint32_t> downstream_cb_size;    // Dependent
    std::optional<uint32_t> downstream_cb_sem_id;  // Dependent

    std::optional<uint32_t> split_prefetch;                        // If upstream is NOT a prefetch_HD
    std::optional<uint32_t> prefetch_h_noc_xy;                     // Dependent. Used if split_prefetch is true
    std::optional<uint32_t> prefetch_h_local_downstream_sem_addr;  // Dependent. Used if split_prefetch is true

    std::optional<uint32_t> num_hops;

    tt::tt_metal::relay_mux_client_config fabric_mux_client_config;

    std::optional<uint32_t> my_dev_id;
    std::optional<uint32_t> ew_dim;
    std::optional<uint32_t> to_mesh_id;
    std::optional<uint32_t> to_dev_id;
    std::optional<uint32_t> router_direction;
};

struct dispatch_s_static_config_t {
    std::optional<uint32_t> cb_base;
    std::optional<uint32_t> cb_log_page_size;
    std::optional<uint32_t> cb_size;
    std::optional<uint32_t> my_dispatch_cb_sem_id;
    std::optional<uint32_t> dispatch_d_shutdown_sem_id;
    std::optional<uint32_t> dispatch_s_sync_sem_base_addr;

    std::optional<uint32_t> mcast_go_signal_addr;
    std::optional<uint32_t> unicast_go_signal_addr;
    std::optional<uint32_t> distributed_dispatcher;
    std::optional<uint32_t> first_stream_used;
    std::optional<uint32_t> completion_counter_offset;
    std::optional<uint32_t> max_num_worker_sems;
    std::optional<uint32_t> max_num_go_signal_noc_data_entries;

    // Dispatch-core-local L1 address of the realtime_profiler_msg_t block (includes the
    // program-id handoff FIFO consumed by this kernel). Assigned by DispatchMemMap via
    // CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG. Must match the value passed to the
    // co-located DispatchKernel and to the RT-profiler core kernels.
    std::optional<uint32_t> realtime_profiler_msg_addr;

    std::optional<uint32_t> dispatch_telemetry_addr;
    std::optional<bool> dispatch_telemetry_disabled;
    std::optional<uint32_t> dispatch_telemetry_control_addr;

    // Configuration for DEVICE_PRINT dispatch. Populated only when the dprint server
    // exists and dispatch_s_enabled() is true. enabled stays 0 otherwise and the kernel
    // compiles the feature out via #if DEVICE_PRINT_DISPATCH_ENABLED.
    std::optional<uint32_t> device_print_dispatch_enabled;
    std::optional<uint32_t> device_print_noc_locations_addr;
    std::optional<uint32_t> device_print_noc_locations_count;
    std::optional<uint32_t> device_print_l1_cache_addr;
    std::optional<uint32_t> device_print_l1_cache_size;
    std::optional<uint32_t> device_print_dram_x;
    std::optional<uint32_t> device_print_dram_y;
    std::optional<uint64_t> device_print_dram_rw_ptrs;
    std::optional<uint64_t> device_print_dram_buf_addr;
    std::optional<uint32_t> device_print_dram_buf_size;
    std::optional<uint64_t> device_print_cycles_for_stall;
    std::optional<uint64_t> device_print_cycles_for_full;
};

struct dispatch_s_dependent_config_t {
    std::optional<tt_cxy_pair> upstream_logical_core;     // Dependent
    std::optional<tt_cxy_pair> downstream_logical_core;   // Dependent
    std::optional<uint32_t> upstream_dispatch_cb_sem_id;  // Dependent
};
namespace experimental {
struct PrefetchConfiguration { prefetch_static_config_t static_config; prefetch_dependent_config_t dependent_config; };
struct DispatcherConfiguration { dispatch_static_config_t static_config; dispatch_dependent_config_t dependent_config; };
struct SubordinateConfiguration { dispatch_s_static_config_t static_config; dispatch_s_dependent_config_t dependent_config; };
// Values computed by SDK adapters before define rendering. The pure planner
// consumes values resolved by actual topology/CQ/observation owners.
struct DispatchResolvedConfiguration {
    CoreCoord virtual_core;
    CoreCoord my_noc;
    CoreCoord upstream_noc;
    CoreCoord downstream_noc;
    CoreCoord subordinate_noc;
    uint32_t command_queue_size = 0;
    uint32_t prefetch_q_entry_bits = 0;
    bool dram_backed_cq = false;
    uint32_t dram_bank = 0;
    uint32_t physical_eth_cores = 0;
    uint32_t virtual_eth_cores = 0;
    uint32_t worker_multicast = 0;
    uint32_t worker_count = 0;
    uint32_t fabric_flow_control_sem = 0;
    uint32_t fabric_teardown_sem = 0;
    uint32_t fabric_buffer_index_sem = 0;
    bool force_watcher_no_inline = false;
    bool watcher_dispatch_disabled = false;
    bool reads_dispatch_cores = false;
    bool galaxy_cluster = false;
};
struct DispatchKernelConfiguration {
    uint32_t node_id = NoDispatchNode;
    ChipId device_id = 0;
    ChipId servicing_device_id = 0;
    uint8_t cq_id = 0;
    HalProcessorIdentifier processor{};
    CoreCoord logical_core;
    noc_selection_t nocs{};
    std::optional<DispatchResolvedConfiguration> resolved;
    std::variant<std::monostate, PrefetchConfiguration, DispatcherConfiguration, SubordinateConfiguration> kernel;
};
} // namespace experimental
} // namespace tt::tt_metal
