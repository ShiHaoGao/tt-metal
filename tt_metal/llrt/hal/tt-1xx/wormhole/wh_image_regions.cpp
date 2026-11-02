// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "wh_hal.hpp"

namespace tt::tt_metal::wormhole::image_regions {
#include "hw/inc/internal/tt-1xx/wormhole/core_config.h"
}
#include "hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h"

namespace tt::tt_metal::wormhole {
namespace {
using namespace image_regions;

HalProcessorImageRegions main_linker_regions(
    DeviceAddr text,
    uint32_t text_size,
    uint32_t local_size,
    uint32_t stack_size,
    DeviceAddr initialization,
    uint32_t kernel_size,
    uint32_t pad) {
    // main.ld TEXT_SIZE and DATA_SIZE - STACK_MIN_SIZE, with the exact
    // architecture initialization scratch allocation from dev_mem_map.h.
    return {
        {text, text_size},
        {MEM_LOCAL_BASE, local_size - stack_size},
        {initialization, local_size},
        kernel_size,
        HalKernelTextLimit::FromFirmwareBase,
        pad};
}
}  // namespace

std::optional<HalProcessorImageRegions> get_processor_image_regions(HalProcessorIdentifier processor) {
    if (processor.core_type == HalProgrammableCoreType::TENSIX) {
        if (processor.processor_class == HalProcessorClassType::DM) {
            switch (processor.processor_type) {
                case 0:
                    return main_linker_regions(
                        MEM_BRISC_FIRMWARE_BASE,
                        MEM_BRISC_FIRMWARE_SIZE,
                        MEM_BRISC_LOCAL_SIZE,
                        MEM_BRISC_STACK_MIN_SIZE,
                        MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH,
                        MEM_BRISC_KERNEL_SIZE,
                        0);
                case 1:
                    return main_linker_regions(
                        MEM_NCRISC_FIRMWARE_BASE,
                        MEM_NCRISC_FIRMWARE_SIZE,
                        MEM_NCRISC_LOCAL_SIZE,
                        MEM_NCRISC_STACK_MIN_SIZE,
                        MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH,
                        MEM_NCRISC_KERNEL_SIZE,
                        0);
                default: return std::nullopt;
            }
        }
        if (processor.processor_class == HalProcessorClassType::COMPUTE) {
            switch (processor.processor_type) {
                case 0:
                    return main_linker_regions(
                        MEM_TRISC0_FIRMWARE_BASE,
                        MEM_TRISC0_FIRMWARE_SIZE,
                        MEM_TRISC_LOCAL_SIZE,
                        MEM_TRISC0_STACK_MIN_SIZE,
                        MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH,
                        MEM_TRISC0_KERNEL_SIZE,
                        0);
                case 1:
                    return main_linker_regions(
                        MEM_TRISC1_FIRMWARE_BASE,
                        MEM_TRISC1_FIRMWARE_SIZE,
                        MEM_TRISC_LOCAL_SIZE,
                        MEM_TRISC1_STACK_MIN_SIZE,
                        MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH,
                        MEM_TRISC1_KERNEL_SIZE,
                        0);
                case 2:
                    return main_linker_regions(
                        MEM_TRISC2_FIRMWARE_BASE,
                        MEM_TRISC2_FIRMWARE_SIZE,
                        MEM_TRISC_LOCAL_SIZE,
                        MEM_TRISC2_STACK_MIN_SIZE,
                        MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH,
                        MEM_TRISC2_KERNEL_SIZE,
                        0);
                default: return std::nullopt;
            }
        }
        return std::nullopt;
    }
    if (processor.processor_class != HalProcessorClassType::DM || processor.processor_type != 0) {
        return std::nullopt;
    }
    if (processor.core_type == HalProgrammableCoreType::IDLE_ETH) {
        return main_linker_regions(
            MEM_IERISC_FIRMWARE_BASE,
            MEM_IERISC_FIRMWARE_SIZE,
            MEM_IERISC_LOCAL_SIZE,
            MEM_IERISC_STACK_MIN_SIZE,
            MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH,
            MEM_IERISC_KERNEL_SIZE,
            MEM_IERISC_KERNEL_PAD);
    }
    if (processor.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
        // erisc-b0-app.ld uses ERISC_APP_CODE and ERISC_APP_DATA from
        // erisc-b0-memory.ld. Both are resident L1 regions; no local relocation.
        return HalProcessorImageRegions{
            {0x9040, 6 * 1024}, {0x103e0, 3104}, {0x103e0, 3104}, 17312, HalKernelTextLimit::AfterFirmware, 0};
    }
    return std::nullopt;
}

std::optional<HalProcessorKernelRegions> get_processor_kernel_regions(
    HalProcessorIdentifier processor, uint64_t firmware_text_end, uint64_t firmware_data_end, bool erisc_iram) {
    const auto regions = get_processor_image_regions(processor);
    if (!regions) {
        return std::nullopt;
    }
    const uint64_t firmware_text_limit =
        (regions->firmware_text.base + regions->firmware_text.size + regions->firmware_kernel_pad + 15) & ~uint64_t{15};
    const uint64_t data_limit = regions->local_data.base + regions->local_data.size;
    if (firmware_text_end < regions->firmware_text.base || firmware_text_end > firmware_text_limit ||
        firmware_data_end < regions->local_data.base || firmware_data_end > data_limit) {
        return std::nullopt;
    }
    if (processor.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
        // erisc-b0-kernel.ld selects ERISC_APP_IRAM_CODE or
        // ERISC_APP_KERNEL_CODE, and always ERISC_APP_KERNEL_DATA.
        return HalProcessorKernelRegions{
            erisc_iram ? HalImageRegion{MEM_NCRISC_IRAM_BASE, 16 * 1024} : HalImageRegion{0xa840, 17312},
            {0xebe0, 6144}};
    }
    const HalImageRegion data{firmware_data_end, static_cast<uint32_t>(data_limit - firmware_data_end)};
    if (processor.core_type == HalProgrammableCoreType::TENSIX &&
        processor.processor_class == HalProcessorClassType::DM && processor.processor_type == 1) {
        // main.ld fixes WH NCRISC kernel TEXT_START at its private IRAM base.
        return HalProcessorKernelRegions{{MEM_NCRISC_KERNEL_BASE, MEM_NCRISC_KERNEL_SIZE}, data};
    }
    const uint64_t text_limit = regions->firmware_text.base + regions->kernel_text_size;
    if (firmware_text_end < regions->firmware_text.base || firmware_text_end > text_limit) {
        return std::nullopt;
    }
    return HalProcessorKernelRegions{{firmware_text_end, static_cast<uint32_t>(text_limit - firmware_text_end)}, data};
}
}  // namespace tt::tt_metal::wormhole
