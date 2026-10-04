// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "impl/experimental/published_deployment/configuration.hpp"
#include <stdint.h>
#include <optional>

#include "fd_kernel.hpp"
#include "impl/context/context_descriptor.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {





class DispatchSKernel : public FDKernel {
public:
    DispatchSKernel(
        int node_id,
        ChipId device_id,
        ChipId servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        const ContextDescriptor& descriptor,
        dispatch_core_manager& dispatch_core_manager,
        const GetControlPlaneFn& get_control_plane = {},
        const GetDispatchQueryManagerFn& get_dispatch_query_manager = {},
        const GetMaxNumEthCoresFn& get_max_num_eth_cores = {},
        const GetReadsDispatchCoresFn& get_reads_dispatch_cores = {});

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
    const dispatch_s_static_config_t& GetStaticConfig() { return static_config_; }

private:
    dispatch_s_static_config_t static_config_;
    dispatch_s_dependent_config_t dependent_config_;
};

}  // namespace tt::tt_metal
