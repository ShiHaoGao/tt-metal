// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "system_memory_cq_interface.hpp"
#include <optional>
#include <variant>
#include <vector>

namespace tt::tt_metal {
struct HostQueueBacking {
    uint16_t channel;
    uint32_t channel_size;
    bool galaxy;
    bool d2h_hugepage_fallback;
    std::optional<uint32_t> cq_size_override;
};
struct DramQueueBacking {
    uint32_t region_size;
    uint32_t device_base;
};
struct SystemMemoryQueueInputs {
    std::variant<HostQueueBacking, DramQueueBacking> backing;
    uint8_t num_hw_cqs;
    uint32_t cq_start;
    uint32_t alignment;
    uint64_t minimum_issue_size;
};
struct SystemMemoryQueuePlan {
    uint32_t cq_size;
    // Offset into the UMD host DMA mapping; zero for a host mirror of DRAM.
    uint32_t host_view_offset;
    // Existing SDK sysmem channel address encoding; zero for DRAM.
    uint32_t channel_offset;
    // Offsets in the selected host view/mirror, not host virtual addresses.
    uint32_t auxiliary_offset;
    uint32_t auxiliary_size;
    std::vector<SystemMemoryCQLayout> queues;
};
// No environment, context, device, host pointers, mappings or I/O windows.
SystemMemoryQueuePlan plan_system_memory_queues(const SystemMemoryQueueInputs&);
} // namespace tt::tt_metal
