// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_program_adapter.hpp"
#include "impl/context/context_descriptor.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_query_manager.hpp"
#include "impl/dispatch/system_memory_manager.hpp"
#include "impl/dispatch/topology.hpp"
#include "hostdev/realtime_profiler_msgs.h"
#include <tt_metal.hpp>
#include <set>
#include <limits>

namespace tt::tt_metal::experimental {
namespace {
uint32_t u32(uint64_t value) {
    if (value > std::numeric_limits<uint32_t>::max())
        throw DispatchProgramError(DispatchProgramReject::Overflow, "dispatch HAL value exceeds 32 bits");
    return static_cast<uint32_t>(value);
}
}
DispatchMemoryFacts capture_dispatch_memory(const DispatchMemMap& m, uint8_t cq) {
    auto a = [&](CommandQueueDeviceAddrType slot) { return m.get_device_command_queue_addr(slot, cq); };
    using A = CommandQueueDeviceAddrType;
    DispatchMemoryFacts result{
        a(A::UNRESERVED), m.prefetch_q_size(), m.prefetch_q_entry_size_bytes(), a(A::PREFETCH_Q_RD), a(A::PREFETCH_Q_PCIE_RD),
        m.cmddat_q_base(cq), m.cmddat_q_size(), m.scratch_db_base(cq), m.scratch_db_size(), m.ringbuffer_size(),
        m.prefetch_d_buffer_pages(), m.dispatch_buffer_base(cq), m.dispatch_buffer_pages(),
        m.dispatch_s_buffer_size(), m.dispatch_s_buffer_pages(), a(A::COMPLETION_Q_WR), a(A::COMPLETION_Q_RD),
        a(A::COMPLETION_Q0_LAST_EVENT), a(A::COMPLETION_Q1_LAST_EVENT), m.get_host_command_queue_addr(CommandQueueHostAddrType::COMPLETION_Q_WR),
        a(A::DISPATCH_PROGRESS), a(A::DISPATCH_S_SYNC_SEM), {}, a(A::FABRIC_HEADER_RB), a(A::FABRIC_SYNC_STATUS),
        a(A::DISPATCH_TELEMETRY), a(A::DISPATCH_TELEMETRY_CONTROL), a(A::REALTIME_PROFILER_MSG),
        m.get_dispatch_stream_index(0), m.get_completion_counter_offset(cq)};
    for (uint32_t index = 0; index != DispatchSettings::DISPATCH_MESSAGE_ENTRIES; ++index)
        result.sync_offsets.push_back(m.get_sync_offset(index));
    return result;
}
DispatchHalFacts capture_dispatch_hal(const Hal& h) {
    using Message = realtime_profiler_msgs::realtime_profiler_msg_t;
    using Field = Message::Field;
    const auto& factory = h.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    const auto offset = [&](Field field) { return u32(factory.offset_of<Message>(field)); };
    const uint32_t timestamp_id = offsetof(realtime_profiler_timestamp_t, id);
    return {h.get_alignment(HalMemType::L1), h.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE),
        u32(h.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG)),
        h.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) == -1 ? 0 :
            u32(h.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)),
        {offset(Field::realtime_profiler_core_noc_xy), offset(Field::realtime_profiler_remote_state_addr),
         offset(Field::realtime_profiler_state), offset(Field::program_id_fifo_start), offset(Field::program_id_fifo_end),
         offset(Field::kernel_start_a) + timestamp_id, offset(Field::kernel_end_a) + timestamp_id,
         offset(Field::kernel_start_b) + timestamp_id, offset(Field::kernel_end_b) + timestamp_id}};
}
DispatchProgramInputs capture_dispatch_program(
    tt::tt_metal::Device& device, const ContextDescriptor& descriptor, std::span<const DispatchKernelNode> graph,
    std::span<const tt_cxy_pair> placed_nodes, tt_cxy_pair completion_writer,
    uint32_t virtual_eth_cores, bool reads_dispatch_cores) {
    if (graph.size() != placed_nodes.size())
        throw DispatchProgramError(DispatchProgramReject::Placement, "missing actual dispatch placement");
    auto& context = descriptor.metal_context();
    if (context.get_context_id() != device.get_context_id())
        throw DispatchProgramError(DispatchProgramReject::Placement, "dispatch capture device belongs to another context");
    const auto& cluster = descriptor.cluster();
    const auto& query = context.get_dispatch_query_manager();
    const auto& options = descriptor.rtoptions();
    DispatchProgramInputs input{};
    auto& t = input.topology;
    t.arch = device.arch(); t.device = device.id(); t.mmio = device.is_mmio_capable();
    t.galaxy = cluster.is_galaxy_cluster(); t.fabric = descriptor.fabric_config() != tt_fabric::FabricConfig::DISABLED;
    t.mock = cluster.is_mock_or_emulated(); t.core_type = context.get_dispatch_core_manager().get_dispatch_core_type();
    t.num_hw_cqs = device.num_hw_cqs(); t.subordinate_enabled = query.dispatch_s_enabled();
    t.distributed_dispatcher = query.distributed_dispatcher();
    if (t.arch != tt::ARCH::BLACKHOLE || t.core_type != CoreType::WORKER || t.mock || t.num_hw_cqs != 1 || t.fabric || !t.mmio)
        throw DispatchProgramError(DispatchProgramReject::Topology, "unsupported live dispatch family");
    std::set<ChipId> devices;
    for (const auto& n : graph) devices.insert(n.device_id);
    t.deployment_devices.assign(devices.begin(), devices.end());
    const auto serviced = cluster.get_devices_controlled_by_mmio_device(device.id());
    t.serviced_devices.assign(serviced.begin(), serviced.end());
    auto project = [&](tt_cxy_pair actual) -> DispatchCorePlacement {
        if (actual.chip != static_cast<size_t>(device.id()))
            throw DispatchProgramError(DispatchProgramReject::Placement, "foreign actual dispatch core");
        CoreCoord logical{actual.x, actual.y};
        auto v = device.virtual_core_from_logical_core(logical, t.core_type);
        return {logical, v, {device.virtual_noc0_coordinate(NOC_0, v), device.virtual_noc0_coordinate(NOC_1, v)}};
    };
    for (size_t index = 0; index != graph.size(); ++index) {
        const auto& n = graph[index];
        DispatchPlacedNode value{static_cast<uint32_t>(n.id), n.device_id, n.servicing_device_id, n.cq_id,
            n.kernel_type, {}, {}, n.noc_selection, project(placed_nodes[index]), n.tunnel_index};
        for (int id : n.upstream_ids) if (id >= 0) value.upstream.push_back(id);
        for (int id : n.downstream_ids) if (id >= 0) value.downstream.push_back(id);
        t.nodes.push_back(std::move(value));
    }
    t.completion_writer = project(completion_writer); t.unused_core = project(tt_cxy_pair(device.id(), 0, 0));
    const auto grid = device.compute_with_storage_grid_size();
    if (!grid.x || !grid.y) throw DispatchProgramError(DispatchProgramReject::Placement, "empty actual worker grid");
    CoreRange workers(device.virtual_core_from_logical_core({0, 0}, CoreType::WORKER),
        device.virtual_core_from_logical_core({grid.x - 1, grid.y - 1}, CoreType::WORKER));
    input.workers = {static_cast<uint32_t>(grid.x * grid.y),
        {device.get_noc_multicast_encoding(NOC_0, workers), device.get_noc_multicast_encoding(NOC_1, workers)},
        static_cast<uint32_t>(device.get_active_ethernet_cores(true).size()), virtual_eth_cores};
    input.observation = {options.get_watcher_enabled(), options.get_watcher_noinline(), options.watcher_dispatch_disabled(),
        reads_dispatch_cores, options.get_dispatch_telemetry_disabled(), static_cast<bool>(context.dprint_server())};
    auto& sysmem = device.sysmem_manager();
    input.queue = {sysmem.queue_inputs(), sysmem.queue_plan(), std::nullopt};
    if (sysmem.is_dram_backed()) input.queue.dram_bank = sysmem.get_dram_region_bank_id();
    input.memory = capture_dispatch_memory(context.dispatch_mem_map(), 0);
    input.hal = capture_dispatch_hal(descriptor.hal());
    return input;
}
void initialize_dispatch_program(tt::tt_metal::Device& device, const DispatchProgramPlan& plan) {
    if (device.id() != plan.device_id() || device.arch() != tt::ARCH::BLACKHOLE)
        throw DispatchProgramError(DispatchProgramReject::Placement, "dispatch initialization device mismatch");
    for (const auto& write : plan.initialization()) {
        detail::WriteToDeviceL1(&device, write.logical, write.address,
            std::span(reinterpret_cast<const uint8_t*>(write.bytes.data()), write.bytes.size()), CoreType::WORKER);
    }
}
} // namespace tt::tt_metal::experimental
