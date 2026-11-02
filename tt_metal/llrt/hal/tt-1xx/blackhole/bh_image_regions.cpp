// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "llrt/hal.hpp"
#include "../wormhole/wh_hal.hpp"
#include <limits>
namespace tt::tt_metal::blackhole::image_regions {
#include "hw/inc/internal/tt-1xx/blackhole/core_config.h"
}
#include "hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h"

namespace tt::tt_metal {
std::optional<HalProcessorImageRegions> Hal::get_processor_image_regions(HalProcessorIdentifier processor) const {
    using namespace blackhole::image_regions;
    if (get_arch() != tt::ARCH::BLACKHOLE && get_arch() != tt::ARCH::WORMHOLE_B0) {
        return std::nullopt;
    }
    bool present = false;
    for (uint32_t core = 0; core != get_programmable_core_type_count(); ++core) {
        const auto type = get_programmable_core_type(core);
        if (processor.core_type != type) {
            continue;
        }
        for (uint32_t cls = 0; cls != get_processor_classes_count(type); ++cls) {
            if (processor.processor_class == static_cast<HalProcessorClassType>(cls) && processor.processor_type >= 0 &&
                static_cast<uint32_t>(processor.processor_type) < get_processor_class_num_fw_binaries(core, cls)) {
                present = true;
            }
        }
    }
    if (!present) {
        return std::nullopt;
    }
    if (get_arch() == tt::ARCH::WORMHOLE_B0) {
        return wormhole::get_processor_image_regions(processor);
    }
    // Match hw/toolchain/main.ld TEXT_START/TEXT_SIZE, DATA_START/DATA_SIZE,
    // STACK_MIN_SIZE and MEM_PAD. These are the same macro owner as the HAL maps.
    auto regions = [](DeviceAddr text,
                      uint32_t text_size,
                      DeviceAddr data,
                      uint32_t data_size,
                      uint32_t stack,
                      DeviceAddr init,
                      uint32_t init_size,
                      uint32_t kernel_size,
                      HalKernelTextLimit limit,
                      uint32_t pad) {
        return HalProcessorImageRegions{
            {text, text_size}, {data, data_size - stack}, {init, init_size}, kernel_size, limit, pad};
    };
    switch (processor.core_type) {
        case HalProgrammableCoreType::TENSIX:
            if (processor.processor_class == HalProcessorClassType::DM) {
                if (processor.processor_type == 0) {
                    return regions(
                        MEM_BRISC_FIRMWARE_BASE,
                        MEM_BRISC_FIRMWARE_SIZE,
                        MEM_LOCAL_BASE,
                        MEM_BRISC_LOCAL_SIZE,
                        MEM_BRISC_STACK_MIN_SIZE,
                        MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH,
                        MEM_BRISC_LOCAL_SIZE,
                        MEM_BRISC_KERNEL_SIZE,
                        HalKernelTextLimit::FromFirmwareBase,
                        0);
                }
                return regions(
                    MEM_NCRISC_FIRMWARE_BASE,
                    MEM_NCRISC_FIRMWARE_SIZE,
                    MEM_LOCAL_BASE,
                    MEM_NCRISC_LOCAL_SIZE,
                    MEM_NCRISC_STACK_MIN_SIZE,
                    MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH,
                    MEM_NCRISC_LOCAL_SIZE,
                    MEM_NCRISC_KERNEL_SIZE,
                    HalKernelTextLimit::FromFirmwareBase,
                    0);
            }
            if (processor.processor_type == 0) {
                return regions(
                    MEM_TRISC0_FIRMWARE_BASE,
                    MEM_TRISC0_FIRMWARE_SIZE,
                    MEM_LOCAL_BASE,
                    MEM_TRISC_LOCAL_SIZE,
                    MEM_TRISC0_STACK_MIN_SIZE,
                    MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH,
                    MEM_TRISC_LOCAL_SIZE,
                    MEM_TRISC0_KERNEL_SIZE,
                    HalKernelTextLimit::FromFirmwareBase,
                    0);
            }
            if (processor.processor_type == 1) {
                return regions(
                    MEM_TRISC1_FIRMWARE_BASE,
                    MEM_TRISC1_FIRMWARE_SIZE,
                    MEM_LOCAL_BASE,
                    MEM_TRISC_LOCAL_SIZE,
                    MEM_TRISC1_STACK_MIN_SIZE,
                    MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH,
                    MEM_TRISC_LOCAL_SIZE,
                    MEM_TRISC1_KERNEL_SIZE,
                    HalKernelTextLimit::FromFirmwareBase,
                    0);
            }
            return regions(
                MEM_TRISC2_FIRMWARE_BASE,
                MEM_TRISC2_FIRMWARE_SIZE,
                MEM_LOCAL_BASE,
                MEM_TRISC_LOCAL_SIZE,
                MEM_TRISC2_STACK_MIN_SIZE,
                MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH,
                MEM_TRISC_LOCAL_SIZE,
                MEM_TRISC2_KERNEL_SIZE,
                HalKernelTextLimit::FromFirmwareBase,
                0);
        case HalProgrammableCoreType::ACTIVE_ETH:
            if (processor.processor_type == 0) {
                return regions(
                    MEM_AERISC_FIRMWARE_BASE,
                    MEM_AERISC_FIRMWARE_SIZE,
                    MEM_AERISC_LOCAL_BASE,
                    MEM_AERISC_LOCAL_SIZE,
                    0,
                    MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH,
                    MEM_AERISC_LOCAL_SIZE,
                    MEM_AERISC_KERNEL_SIZE,
                    HalKernelTextLimit::AfterFirmware,
                    MEM_IERISC_KERNEL_PAD);
            }
            return regions(
                MEM_SUBORDINATE_AERISC_FIRMWARE_BASE,
                MEM_SUBORDINATE_AERISC_FIRMWARE_SIZE,
                MEM_SUBORDINATE_AERISC_LOCAL_BASE,
                MEM_SUBORDINATE_AERISC_LOCAL_SIZE,
                MEM_SUBORDINATE_AERISC_STACK_MIN_SIZE,
                MEM_SUBORDINATE_AERISC_INIT_LOCAL_L1_BASE_SCRATCH,
                MEM_SUBORDINATE_AERISC_LOCAL_SIZE,
                MEM_AERISC_KERNEL_SIZE,
                HalKernelTextLimit::AfterFirmware,
                MEM_IERISC_KERNEL_PAD);
        case HalProgrammableCoreType::IDLE_ETH:
            if (processor.processor_type == 0) {
                return regions(
                    MEM_IERISC_FIRMWARE_BASE,
                    MEM_IERISC_FIRMWARE_SIZE,
                    MEM_LOCAL_BASE,
                    MEM_IERISC_LOCAL_SIZE,
                    MEM_IERISC_STACK_MIN_SIZE,
                    MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH,
                    MEM_IERISC_LOCAL_SIZE,
                    MEM_IERISC_KERNEL_SIZE,
                    HalKernelTextLimit::FromFirmwareBase,
                    MEM_IERISC_KERNEL_PAD);
            }
            return regions(
                MEM_SUBORDINATE_IERISC_FIRMWARE_BASE,
                MEM_SUBORDINATE_IERISC_FIRMWARE_SIZE,
                MEM_LOCAL_BASE,
                MEM_SUBORDINATE_IERISC_LOCAL_SIZE,
                MEM_SUBORDINATE_IERISC_STACK_MIN_SIZE,
                MEM_SUBORDINATE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH,
                MEM_SUBORDINATE_IERISC_LOCAL_SIZE,
                MEM_IERISC_KERNEL_SIZE,
                HalKernelTextLimit::FromFirmwareBase,
                MEM_IERISC_KERNEL_PAD);
        case HalProgrammableCoreType::DRAM:
            return regions(
                MEM_DRISC_FIRMWARE_BASE,
                MEM_DRISC_FIRMWARE_SIZE,
                MEM_LOCAL_BASE,
                MEM_DRISC_LOCAL_SIZE,
                MEM_DRISC_STACK_MIN_SIZE,
                MEM_DRISC_INIT_LOCAL_L1_BASE_SCRATCH,
                MEM_DRISC_INIT_LOCAL_L1_SCRATCH_SIZE,
                MEM_DRISC_FIRMWARE_SIZE,
                HalKernelTextLimit::FromFirmwareBase,
                0);
        default: return std::nullopt;
    }
}
std::optional<HalProcessorKernelRegions> Hal::get_processor_kernel_regions(
    HalProcessorIdentifier processor, uint64_t firmware_text_end, uint64_t firmware_data_end, bool erisc_iram) const {
    const auto regions = get_processor_image_regions(processor);
    if (!regions) {
        return std::nullopt;
    }
    if (get_arch() == tt::ARCH::WORMHOLE_B0) {
        return wormhole::get_processor_kernel_regions(processor, firmware_text_end, firmware_data_end, erisc_iram);
    }
    const auto base = regions->kernel_text_limit == HalKernelTextLimit::FromFirmwareBase ? regions->firmware_text.base
                                                                                         : firmware_text_end;
    if (base > std::numeric_limits<uint64_t>::max() - regions->kernel_text_size) {
        return std::nullopt;
    }
    const uint64_t text_limit = base + regions->kernel_text_size;
    const uint64_t data_limit = regions->local_data.base + regions->local_data.size;
    if (firmware_text_end < regions->firmware_text.base || firmware_text_end > text_limit ||
        firmware_data_end < regions->local_data.base || firmware_data_end > data_limit) {
        return std::nullopt;
    }
    return HalProcessorKernelRegions{
        {firmware_text_end, static_cast<uint32_t>(text_limit - firmware_text_end)},
        {firmware_data_end, static_cast<uint32_t>(data_limit - firmware_data_end)}};
}
}  // namespace tt::tt_metal
