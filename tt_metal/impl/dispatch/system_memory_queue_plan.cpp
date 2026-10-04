// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "system_memory_queue_plan.hpp"
#include "dispatch_settings.hpp"
#include <limits>
#include <stdexcept>
namespace tt::tt_metal {
SystemMemoryQueuePlan plan_system_memory_queues(const SystemMemoryQueueInputs& input) {
    if (!input.num_hw_cqs || !input.alignment)
        throw std::invalid_argument("CQ count and alignment must be nonzero");
    uint32_t gross_cq_size = 0;
    uint32_t reduction = 2 * DispatchSettings::TRANSFER_PAGE_SIZE;
    uint16_t channel = 0;
    uint32_t device_base = 0;
    if (const auto* host = std::get_if<HostQueueBacking>(&input.backing)) {
        channel = host->channel;
        const uint32_t backing_size = host->galaxy
            ? host->channel_size / DispatchSettings::DEVICES_PER_UMD_CHANNEL
            : host->channel_size;
        if (host->cq_size_override) gross_cq_size = *host->cq_size_override;
        else gross_cq_size = backing_size / input.num_hw_cqs;
        if (uint64_t(gross_cq_size) * input.num_hw_cqs > backing_size)
            throw std::invalid_argument("CQ override exceeds the actual host backing");
        if (host->d2h_hugepage_fallback) {
            const uint32_t reserve_per_cq =
                (DispatchSettings::HUGEPAGE_D2H_FALLBACK_RESERVE_BYTES + input.num_hw_cqs - 1) / input.num_hw_cqs;
            const uint32_t page = DispatchSettings::TRANSFER_PAGE_SIZE;
            reduction += ((reserve_per_cq + page - 1) / page) * page;
        }
    } else {
        const auto& dram = std::get<DramQueueBacking>(input.backing);
        if (!dram.region_size || dram.region_size % input.num_hw_cqs)
            throw std::invalid_argument("DRAM CQ region must divide evenly across queues");
        if (uint64_t(dram.device_base) + dram.region_size > (uint64_t{1} << 32))
            throw std::invalid_argument("DRAM CQ backing exceeds the device address range");
        gross_cq_size = dram.region_size / input.num_hw_cqs;
        if (gross_cq_size % input.alignment)
            throw std::invalid_argument("DRAM CQ region is misaligned");
        device_base = dram.device_base;
    }
    if (gross_cq_size <= reduction)
        throw std::invalid_argument("CQ size is too small for the auxiliary reservation");
    SystemMemoryQueuePlan plan{};
    plan.cq_size = gross_cq_size - reduction;
    const uint64_t total_cq_bytes = uint64_t(input.num_hw_cqs) * plan.cq_size;
    const uint64_t total_auxiliary_bytes = uint64_t(input.num_hw_cqs) * reduction;
    if (total_cq_bytes + total_auxiliary_bytes > (uint64_t{1} << 32))
        throw std::invalid_argument("CQ backing exceeds the addressable region");
    plan.auxiliary_offset = static_cast<uint32_t>(total_cq_bytes);
    plan.auxiliary_size = static_cast<uint32_t>(total_auxiliary_bytes);
    plan.queues.reserve(input.num_hw_cqs);
    for (uint8_t cq_id = 0; cq_id < input.num_hw_cqs; ++cq_id) {
        const auto queue = plan_system_memory_cq(
            channel, cq_id, plan.cq_size, input.cq_start, input.alignment, device_base);
        if (queue.command_issue_region_size < input.minimum_issue_size)
            throw std::invalid_argument("CQ issue region cannot cover the prefetch window");
        plan.queues.push_back(queue);
    }
    if (std::holds_alternative<HostQueueBacking>(input.backing)) {
        plan.host_view_offset = (uint32_t(channel) >> 2) * DispatchSettings::MAX_DEV_CHANNEL_SIZE;
        plan.channel_offset = plan.queues.front().device_offset;
        if (uint64_t(plan.channel_offset) + total_cq_bytes + total_auxiliary_bytes >
            (uint64_t{1} << 32))
            throw std::invalid_argument("CQ auxiliary region exceeds the device address range");
    }
    return plan;
}
} // namespace tt::tt_metal
