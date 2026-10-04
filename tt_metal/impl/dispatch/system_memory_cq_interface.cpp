// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "system_memory_cq_interface.hpp"
#include "command_queue_common.hpp"
#include "dispatch_settings.hpp"
#include <limits>
#include <stdexcept>

namespace tt::tt_metal {
namespace {
const SystemMemoryCQLayout& checked_layout(const SystemMemoryCQLayout& layout) {
    if (!layout.command_issue_region_size || !layout.command_completion_region_size ||
        layout.device_offset % 16 || layout.cq_start % 16 ||
        layout.command_issue_region_size % 16 || layout.command_completion_region_size % 16 ||
        uint64_t(layout.device_offset) + layout.cq_start + layout.command_issue_region_size +
            layout.command_completion_region_size > (uint64_t{1} << 32))
        throw std::invalid_argument("invalid CQ layout or device address range");
    return layout;
}
} // namespace
SystemMemoryCQLayout plan_system_memory_cq(
    uint16_t channel, uint8_t cq_id, uint32_t cq_size, uint32_t cq_start, uint32_t alignment, uint32_t base) {
    if (!alignment || cq_size <= cq_start)
        throw std::invalid_argument("CQ size must exceed its reserved prefix with nonzero alignment");
    const uint32_t completion =
        (((cq_size - cq_start) / DispatchSettings::TRANSFER_PAGE_SIZE) / 4) * DispatchSettings::TRANSFER_PAGE_SIZE;
    const uint32_t issue = cq_size - cq_start - completion;
    if (!completion)
        throw std::invalid_argument("CQ must contain a nonempty completion region");
    if (completion % alignment || issue % alignment || cq_start % 16 || cq_size % 16)
        throw std::invalid_argument("CQ issue and completion regions are misaligned");
    // Check in wide arithmetic before using the canonical SDK address helper.
    const uint64_t channel_base = base ? base :
        uint64_t(DispatchSettings::MAX_HUGEPAGE_SIZE) * get_umd_channel(channel) +
        uint64_t(channel >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;
    if (channel_base % 16 || channel_base + (uint64_t(cq_id) + 1) * cq_size >
        (uint64_t{1} << 32))
        throw std::invalid_argument("CQ exceeds its device address range");
    return {cq_id, cq_start, issue, completion, get_absolute_cq_offset(channel, cq_id, cq_size, base)};
}

SystemMemoryCQInterface::SystemMemoryCQInterface(
    uint16_t channel, uint8_t cq_id, uint32_t cq_size, uint32_t cq_start, uint32_t alignment, uint32_t base)
    : SystemMemoryCQInterface(plan_system_memory_cq(channel, cq_id, cq_size, cq_start, alignment, base)) {}

SystemMemoryCQInterface::SystemMemoryCQInterface(const SystemMemoryCQLayout& layout)
    : cq_start(checked_layout(layout).cq_start),
      command_completion_region_size(layout.command_completion_region_size),
      command_issue_region_size(layout.command_issue_region_size),
      id(layout.id),
      issue_fifo_size(command_issue_region_size >> 4),
      issue_fifo_limit(static_cast<uint32_t>((uint64_t(layout.device_offset) + cq_start + command_issue_region_size) >> 4)),
      offset(layout.device_offset),
      issue_fifo_wr_ptr((cq_start + offset) >> 4),
      completion_fifo_size(command_completion_region_size >> 4),
      completion_fifo_limit(issue_fifo_limit + completion_fifo_size),
      completion_fifo_rd_ptr(issue_fifo_limit) {}
} // namespace tt::tt_metal
