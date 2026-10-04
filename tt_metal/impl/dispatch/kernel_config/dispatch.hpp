// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "impl/experimental/published_deployment/configuration.hpp"

#include <stdint.h>
#include <optional>

#include "core_coord.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "fd_kernel.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "impl/context/context_descriptor.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {





class DispatchKernel : public FDKernel {
public:
    DispatchKernel(
        int node_id,
        ChipId device_id,
        ChipId servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        bool h_variant,
        bool d_variant,
        const ContextDescriptor& descriptor,
        dispatch_core_manager& dispatch_core_manager,
        const GetControlPlaneFn& get_control_plane = {},
        const GetDispatchQueryManagerFn& get_dispatch_query_manager = {},
        const GetMaxNumEthCoresFn& get_max_num_eth_cores = {},
        const GetReadsDispatchCoresFn& get_reads_dispatch_cores = {});

    void CreateKernel() override;

    void GenerateStaticConfigs() override;

    void GenerateDependentConfigs() override;

    void InitializeRuntimeArgsValues() override;

    void ConfigureCore() override;

    uint32_t GetDispatchBufferSize() const {
        return (1 << static_config_.dispatch_cb_log_page_size.value()) * static_config_.dispatch_cb_pages.value();
    }
    const dispatch_static_config_t& GetStaticConfig() { return static_config_; }

private:
    dispatch_static_config_t static_config_;
    dispatch_dependent_config_t dependent_config_;
    FDKernelEdmConnectionAttributes edm_connection_attributes_;

    bool is_hd() const { return static_config_.is_h_variant.value() && static_config_.is_d_variant.value(); }
};

}  // namespace tt::tt_metal
