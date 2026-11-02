// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llrt/hal.hpp"

namespace tt::tt_metal::wormhole {

// If you are trying to include this file and you aren't hal...you are doing something wrong

HalCoreInfoType create_tensix_mem_map();
HalCoreInfoType create_active_eth_mem_map(bool is_base_routing_fw_enabled);
HalCoreInfoType create_idle_eth_mem_map();
std::optional<HalProcessorImageRegions> get_processor_image_regions(HalProcessorIdentifier processor);
std::optional<HalProcessorKernelRegions> get_processor_kernel_regions(
    HalProcessorIdentifier processor, uint64_t firmware_text_end, uint64_t firmware_data_end, bool erisc_iram);

}  // namespace tt::tt_metal::wormhole
