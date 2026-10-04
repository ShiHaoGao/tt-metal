// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_program_plan.hpp"
#include "record_writer.hpp"
#include "impl/dispatch/dispatch_settings.hpp"
#include "impl/buffers/semaphore.hpp"
#include <hostdevcommon/dispatch_telemetry_types.hpp>
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <set>

namespace tt::tt_metal::experimental {
DispatchProgramError::DispatchProgramError(DispatchProgramReject reason, const char* message) :
    std::invalid_argument(message), reason_(reason) {}
namespace {
using Reject = DispatchProgramReject;
[[noreturn]] void fail(Reject reason, const char* message) { throw DispatchProgramError(reason, message); }
void require(bool condition, Reject reason, const char* message) { if (!condition) fail(reason, message); }
uint32_t checked(uint64_t value) {
    require(value <= std::numeric_limits<uint32_t>::max(), Reject::Overflow, "dispatch address or size overflow");
    return static_cast<uint32_t>(value);
}
template<class T> bool equal_record(const T& a, const T& b) {
    deployment_detail::RecordWriter left, right; left.append(a); right.append(b); return left.bytes == right.bytes;
}
template<class T> std::vector<std::byte> bytes_of(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::endian::native == std::endian::little);
    std::vector<std::byte> bytes(sizeof(value)); std::memcpy(bytes.data(), &value, sizeof(value)); return bytes;
}
const DispatchPlacedNode& node(const DispatchProgramInputs& i, DispatchWorkerType role) {
    auto at = std::ranges::find(i.topology.nodes, role, &DispatchPlacedNode::role);
    require(at != i.topology.nodes.end(), Reject::Topology, "missing dispatch role"); return *at;
}
void validate(const DispatchProgramInputs& i) {
    const auto& t = i.topology;
    require(t.arch == tt::ARCH::BLACKHOLE, Reject::Architecture, "dispatch family requires Blackhole");
    require(t.mmio && !t.galaxy && !t.fabric && !t.mock && t.core_type == CoreType::WORKER &&
        t.num_hw_cqs == 1 && t.subordinate_enabled && !t.distributed_dispatcher &&
        t.deployment_devices == std::vector<ChipId>{t.device} && t.serviced_devices == std::vector<ChipId>{t.device} &&
        t.nodes.size() == 3, Reject::Topology, "unsupported dispatch topology family");
    require(!i.observation.device_print_enabled && !i.observation.watcher_enabled && !i.observation.watcher_noinline,
        Reject::Observation, "unsupported dispatch instrumentation");
    const auto& p = node(i, PREFETCH_HD); const auto& d = node(i, DISPATCH_HD); const auto& s = node(i, DISPATCH_S);
    std::set<uint32_t> ids;
    for (const auto& n : t.nodes) {
        require(n.id != NoDispatchNode && ids.insert(n.id).second && n.device == t.device &&
            n.servicing_device == t.device && n.cq == 0 && n.tunnel_index == -1,
            Reject::Topology, "foreign, duplicate or tunneled dispatch node");
    }
    auto same_ids = [](auto a, auto b) { std::ranges::sort(a); std::ranges::sort(b); return a == b; };
    require(p.upstream.empty() && same_ids(p.downstream, std::vector{s.id, d.id}) &&
        d.upstream == std::vector{p.id} && d.downstream == std::vector{s.id} &&
        s.upstream == std::vector{p.id} && s.downstream == std::vector{d.id},
        Reject::Topology, "dispatch graph does not match the supported family");
    require(equal_record(p.nocs, noc_selection_t{NOC_0, NOC_0, NOC_0}) &&
        equal_record(d.nocs, noc_selection_t{NOC_0, NOC_1, NOC_0}) &&
        equal_record(s.nocs, noc_selection_t{NOC_1, NOC_1, NOC_1}),
        Reject::Placement, "dispatch NoC allocation differs from supported family");
    require(equal_record(d.core, s.core) && equal_record(d.core, t.completion_writer) && p.core.logical != d.core.logical &&
        p.core.virtual_core != d.core.virtual_core && t.unused_core.logical == CoreCoord{0, 0} &&
        i.workers.count && i.workers.virtual_eth_cores >= i.workers.physical_eth_cores,
        Reject::Placement, "invalid dispatch placement or worker facts");
    require(i.queue.inputs.num_hw_cqs == 1 && i.queue.plan.queues.size() == 1 &&
        i.queue.plan.queues.front().id == 0, Reject::Queue, "dispatch queue count mismatch");
    try {
        require(equal_record(plan_system_memory_queues(i.queue.inputs), i.queue.plan), Reject::Queue,
            "dispatch queue differs from the owning queue plan");
    } catch (const DispatchProgramError&) { throw; }
    catch (const std::exception&) { fail(Reject::Queue, "invalid dispatch queue input"); }
    const bool dram = std::holds_alternative<DramQueueBacking>(i.queue.inputs.backing);
    require(dram == i.queue.dram_bank.has_value(), Reject::Queue, "queue bank and backing differ");
    require(!std::holds_alternative<HostQueueBacking>(i.queue.inputs.backing) ||
        !std::get<HostQueueBacking>(i.queue.inputs.backing).galaxy, Reject::Topology, "Galaxy CQ unsupported");
    const auto& m = i.memory; const auto& h = i.hal;
    require(h.l1_alignment && std::has_single_bit(h.l1_alignment) && h.l1_size &&
        m.prefetch_q_entry_bytes == 4 && m.prefetch_q_size && m.prefetch_q_size % m.prefetch_q_entry_bytes == 0 &&
        m.dispatch_buffer_pages && m.subordinate_buffer_pages && m.subordinate_buffer_size &&
        m.sync_offsets.size() == DispatchSettings::DISPATCH_MESSAGE_ENTRIES,
        Reject::MemoryMap, "invalid dispatch memory geometry");
    require(std::set(m.sync_offsets.begin(), m.sync_offsets.end()).size() == m.sync_offsets.size() &&
        std::set(h.realtime_profiler_reset_offsets.begin(), h.realtime_profiler_reset_offsets.end()).size() == 9,
        Reject::MemoryMap, "duplicate dispatch initialization slot");
    for (size_t index = 0; index != m.sync_offsets.size(); ++index)
        require(m.sync_offsets[index] == uint64_t{index} * h.l1_alignment,
            Reject::MemoryMap, "dispatch sync offsets disagree with HAL alignment");
    auto region = [&](uint32_t address, uint64_t size) {
        auto end = checked(uint64_t{address} + size);
        require(address % 4 == 0 && end <= h.l1_size, Reject::MemoryMap, "dispatch region escapes L1");
    };
    region(m.prefetch_q_base, m.prefetch_q_size);
    region(m.cmddat_q_base, m.cmddat_q_size); region(m.scratch_db_base, m.scratch_db_size);
    region(m.dispatch_buffer_base, (uint64_t{m.dispatch_buffer_pages} << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE) + m.subordinate_buffer_size);
    require(m.subordinate_buffer_size == (uint64_t{m.subordinate_buffer_pages} << DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE),
        Reject::MemoryMap, "subordinate page count and buffer size differ");
}
relay_mux_client_config inactive_fabric() {
    return {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}
} // namespace

struct DispatchProgramPlan::Storage {
    DispatchProgramInputs inputs;
    std::vector<DispatchKernelConfiguration> nodes;
    std::vector<DispatchPlan> kernels;
    std::vector<DispatchSemaphorePlan> semaphores;
    std::vector<DispatchRuntimeArguments> arguments;
    std::vector<DispatchInitialization> initialization;
    std::vector<DispatchCoreRegistration> registrations;
    std::vector<std::byte> canonical;
};
DispatchProgramPlan::DispatchProgramPlan(std::shared_ptr<const Storage> storage) : storage_(std::move(storage)) {}
DispatchProgramFamily DispatchProgramPlan::family() const { return DispatchProgramFamily::BlackholeSingleMmioWorkerOneCq; }
const DispatchProgramInputs& DispatchProgramPlan::inputs() const { return storage_->inputs; }
ChipId DispatchProgramPlan::device_id() const { return storage_->inputs.topology.device; }
std::span<const DispatchKernelConfiguration> DispatchProgramPlan::dispatch_nodes() const { return storage_->nodes; }
std::span<const DispatchPlan> DispatchProgramPlan::kernel_plans() const { return storage_->kernels; }
std::span<const DispatchSemaphorePlan> DispatchProgramPlan::semaphores() const { return storage_->semaphores; }
std::span<const DispatchRuntimeArguments> DispatchProgramPlan::runtime_arguments() const { return storage_->arguments; }
std::span<const DispatchInitialization> DispatchProgramPlan::initialization() const { return storage_->initialization; }
std::span<const DispatchCoreRegistration> DispatchProgramPlan::core_registrations() const { return storage_->registrations; }
std::span<const std::byte> DispatchProgramPlan::canonical_record() const { return storage_->canonical; }
std::vector<DispatchSemaphoreAddress> DispatchProgramPlan::semaphore_addresses(uint32_t config_base, uint32_t sem_offset) const {
    const auto& h = inputs().hal;
    require(config_base % h.l1_alignment == 0 && sem_offset % h.l1_alignment == 0,
        Reject::Semaphore, "unaligned finalized semaphore base");
    std::vector<DispatchSemaphoreAddress> result;
    for (const auto& s : semaphores()) {
        auto address = checked(uint64_t{config_base} + sem_offset + s.slot_offset);
        require(uint64_t{address} + 4 <= h.l1_size, Reject::Semaphore, "semaphore address escapes L1");
        result.push_back({s.node, s.role, s.logical, address});
    }
    return result;
}

DispatchProgramPlan plan_dispatch_program(const DispatchProgramInputs& input) {
    validate(input);
    auto owner = std::make_shared<DispatchProgramPlan::Storage>(); owner->inputs = input;
    const auto& i = owner->inputs; const auto& t = i.topology; const auto& m = i.memory;
    const auto& h = i.hal; const auto& o = i.observation; const auto& q = i.queue.plan.queues.front();
    const auto& prefetch = node(i, PREFETCH_HD); const auto& dispatch = node(i, DISPATCH_HD);
    const auto& subordinate = node(i, DISPATCH_S);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> used;
    std::map<DispatchSemaphoreRole, uint32_t> sem_ids;
    auto allocate = [&](const DispatchPlacedNode& n, DispatchSemaphoreRole role, uint32_t initial) {
        auto& mask = used[{n.core.logical.x, n.core.logical.y}];
        uint32_t id = 0; while (id < NUM_SEMAPHORES && (mask & (uint32_t{1} << id))) ++id;
        require(id < NUM_SEMAPHORES, Reject::Semaphore, "dispatch semaphore capacity exhausted");
        mask |= uint32_t{1} << id; sem_ids.emplace(role, id);
        owner->semaphores.push_back({n.id, n.core.logical, role, id, checked(uint64_t{id} * h.l1_alignment), initial});
    };
    // Fresh isolated CQ Program: traversal and actual core incidence jointly own IDs.
    for (const auto& n : t.nodes) {
        switch (n.role) {
        case PREFETCH_HD:
            allocate(n, DispatchSemaphoreRole::PrefetchDownstream, m.dispatch_buffer_pages);
            allocate(n, DispatchSemaphoreRole::PrefetchSync, 0);
            allocate(n, DispatchSemaphoreRole::PrefetchSubordinate, m.subordinate_buffer_pages); break;
        case DISPATCH_HD: allocate(n, DispatchSemaphoreRole::DispatcherBuffer, 0); break;
        case DISPATCH_S:
            allocate(n, DispatchSemaphoreRole::SubordinateBuffer, 0);
            allocate(n, DispatchSemaphoreRole::SubordinateShutdown, 0); break;
        default: fail(Reject::Topology, "unsupported dispatch role");
        }
    }
    auto sem = [&](DispatchSemaphoreRole role) { return sem_ids.at(role); };
    auto logical = [&](const DispatchPlacedNode& n) { return tt_cxy_pair(t.device, n.core.logical.x, n.core.logical.y); };
    const tt_cxy_pair unused(t.device, 0, 0);
    const uint32_t issue_base = checked(uint64_t{q.device_offset} + q.cq_start);
    const uint32_t completion_base = checked(uint64_t{issue_base} + q.command_issue_region_size);
    const uint32_t s_base = checked(uint64_t{m.dispatch_buffer_base} +
        (uint64_t{m.dispatch_buffer_pages} << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE));
    PrefetchConfiguration p;
    p.static_config = {
        .my_downstream_cb_sem_id = sem(DispatchSemaphoreRole::PrefetchDownstream),
        .pcie_base = issue_base, .pcie_size = q.command_issue_region_size,
        .prefetch_q_base = m.prefetch_q_base, .prefetch_q_size = m.prefetch_q_size,
        .prefetch_q_rd_ptr_addr = m.prefetch_q_rd, .prefetch_q_pcie_rd_ptr_addr = m.prefetch_q_pcie_rd,
        .cmddat_q_base = m.cmddat_q_base, .cmddat_q_size = m.cmddat_q_size,
        .scratch_db_base = m.scratch_db_base, .scratch_db_size = m.scratch_db_size,
        .downstream_sync_sem_id = sem(DispatchSemaphoreRole::PrefetchSync), .ringbuffer_size = m.ringbuffer_size,
        .cmddat_q_pages = m.prefetch_d_pages, .my_upstream_cb_sem_id = 0,
        .cmddat_q_log_page_size = DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE,
        .dispatch_s_buffer_base = s_base, .my_dispatch_s_cb_sem_id = sem(DispatchSemaphoreRole::PrefetchSubordinate),
        .dispatch_s_buffer_size = m.subordinate_buffer_size, .dispatch_s_cb_log_page_size = DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE,
        .fabric_header_rb_base = m.fabric_header_base, .fabric_header_rb_entries = DispatchSettings::FABRIC_HEADER_RB_ENTRIES,
        .my_fabric_sync_status_addr = m.fabric_sync_status, .dispatch_telemetry_addr = m.telemetry,
        .dispatch_telemetry_disabled = o.telemetry_disabled, .is_2d_fabric = false, .is_d_variant = true, .is_h_variant = true,
        .offsetof_my_dev_id = 0, .offsetof_to_dev_id = 1, .offsetof_router_direction = 2};
    p.dependent_config = {
        .upstream_logical_core = unused, .downstream_logical_core = logical(dispatch), .downstream_s_logical_core = logical(subordinate),
        .downstream_cb_base = m.dispatch_buffer_base, .downstream_cb_log_page_size = DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE,
        .downstream_cb_pages = m.dispatch_buffer_pages, .downstream_cb_sem_id = sem(DispatchSemaphoreRole::DispatcherBuffer),
        .upstream_cb_sem_id = 0, .downstream_dispatch_s_cb_sem_id = sem(DispatchSemaphoreRole::SubordinateBuffer), .num_hops = 0,
        .fabric_mux_client_config = inactive_fabric(), .my_dev_id = 0, .ew_dim = 0, .to_mesh_id = 0, .to_dev_id = 0, .router_direction = 0};
    DispatcherConfiguration d;
    d.static_config = {
        .dispatch_cb_base = m.dispatch_buffer_base, .dispatch_cb_log_page_size = DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE,
        .dispatch_cb_pages = m.dispatch_buffer_pages, .my_dispatch_cb_sem_id = sem(DispatchSemaphoreRole::DispatcherBuffer),
        .dispatch_cb_blocks = DispatchSettings::DISPATCH_BUFFER_SIZE_BLOCKS, .command_queue_base_addr = q.device_offset,
        .completion_queue_base_addr = completion_base, .completion_queue_size = q.command_completion_region_size,
        .my_downstream_cb_sem_id = 0, .prefetch_h_max_credits = 0, .packed_write_max_unicast_sub_cmds = i.workers.count,
        .dispatch_s_sync_sem_base_addr = m.sync_sem_base, .max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES,
        .max_num_go_signal_noc_data_entries = DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES,
        .mcast_go_signal_addr = h.worker_go_message, .unicast_go_signal_addr = h.ethernet_go_message,
        .distributed_dispatcher = 0, .first_stream_used = m.first_stream, .completion_counter_offset = m.completion_counter_offset,
        .host_completion_q_wr_ptr = m.host_completion_q_wr, .dev_completion_q_wr_ptr = m.completion_q_wr,
        .dev_completion_q_rd_ptr = m.completion_q_rd, .dev_dispatch_progress_ptr = m.dispatch_progress,
        .fabric_header_rb_base = m.fabric_header_base, .fabric_header_rb_entries = DispatchSettings::FABRIC_HEADER_RB_ENTRIES,
        .my_fabric_sync_status_addr = m.fabric_sync_status, .is_2d_fabric = false,
        .realtime_profiler_msg_addr = m.realtime_profiler, .dispatch_telemetry_addr = m.telemetry,
        .dispatch_telemetry_control_addr = m.telemetry_control, .dispatch_telemetry_disabled = o.telemetry_disabled,
        .is_d_variant = true, .is_h_variant = true, .offsetof_my_dev_id = 0, .offsetof_to_dev_id = 1, .offsetof_router_direction = 2};
    d.dependent_config = {
        .upstream_logical_core = logical(prefetch), .downstream_logical_core = unused, .downstream_s_logical_core = logical(subordinate),
        .upstream_dispatch_cb_sem_id = sem(DispatchSemaphoreRole::PrefetchDownstream), .upstream_sync_sem = sem(DispatchSemaphoreRole::PrefetchSync),
        .dispatch_d_shutdown_sem_id = sem(DispatchSemaphoreRole::SubordinateShutdown), .downstream_cb_base = 0,
        .downstream_cb_size = 0, .downstream_cb_sem_id = 0, .split_prefetch = 0, .prefetch_h_noc_xy = 0,
        .prefetch_h_local_downstream_sem_addr = 0, .num_hops = 0, .fabric_mux_client_config = inactive_fabric(),
        .my_dev_id = 0, .ew_dim = 0, .to_mesh_id = 0, .to_dev_id = 0, .router_direction = 0};
    SubordinateConfiguration s;
    s.static_config = {
        .cb_base = s_base, .cb_log_page_size = DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE,
        .cb_size = m.subordinate_buffer_size, .my_dispatch_cb_sem_id = sem(DispatchSemaphoreRole::SubordinateBuffer),
        .dispatch_d_shutdown_sem_id = sem(DispatchSemaphoreRole::SubordinateShutdown), .dispatch_s_sync_sem_base_addr = m.sync_sem_base,
        .mcast_go_signal_addr = h.worker_go_message, .unicast_go_signal_addr = h.ethernet_go_message,
        .distributed_dispatcher = 0, .first_stream_used = m.first_stream, .completion_counter_offset = m.completion_counter_offset,
        .max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES,
        .max_num_go_signal_noc_data_entries = DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES,
        .realtime_profiler_msg_addr = m.realtime_profiler, .dispatch_telemetry_addr = m.telemetry,
        .dispatch_telemetry_disabled = o.telemetry_disabled, .dispatch_telemetry_control_addr = m.telemetry_control,
        .device_print_dispatch_enabled = 0, .device_print_noc_locations_addr = 0, .device_print_noc_locations_count = 0,
        .device_print_l1_cache_addr = 0, .device_print_l1_cache_size = 0, .device_print_dram_x = 0, .device_print_dram_y = 0,
        .device_print_dram_rw_ptrs = 0, .device_print_dram_buf_addr = 0, .device_print_dram_buf_size = 0,
        .device_print_cycles_for_stall = 0, .device_print_cycles_for_full = 0};
    s.dependent_config = {logical(prefetch), logical(dispatch), sem(DispatchSemaphoreRole::PrefetchSubordinate)};
    for (const auto& n : t.nodes) {
        const DispatchCorePlacement* upstream = &prefetch.core;
        const DispatchCorePlacement* downstream = &dispatch.core;
        const DispatchCorePlacement* subordinate_core = &subordinate.core;
        if (n.role == PREFETCH_HD) upstream = &t.unused_core;
        if (n.role == DISPATCH_HD) downstream = &t.unused_core;
        if (n.role == DISPATCH_S) subordinate_core = &t.unused_core;
        DispatchResolvedConfiguration r{
            .virtual_core = n.core.virtual_core, .my_noc = n.core.noc.at(n.nocs.non_dispatch_noc),
            .upstream_noc = upstream->noc.at(n.nocs.upstream_noc), .downstream_noc = downstream->noc.at(n.nocs.downstream_noc),
            .subordinate_noc = subordinate_core->noc.at(n.nocs.downstream_noc),
            .command_queue_size = i.queue.plan.cq_size, .prefetch_q_entry_bits = m.prefetch_q_entry_bytes * 8,
            .dram_backed_cq = i.queue.dram_bank.has_value(), .dram_bank = i.queue.dram_bank.value_or(0),
            .physical_eth_cores = i.workers.physical_eth_cores, .virtual_eth_cores = i.workers.virtual_eth_cores,
            .worker_multicast = i.workers.multicast.at(n.nocs.downstream_noc), .worker_count = i.workers.count,
            .force_watcher_no_inline = false, .watcher_dispatch_disabled = o.watcher_dispatch_disabled,
            .reads_dispatch_cores = o.reads_dispatch_cores, .galaxy_cluster = false};
        DispatchKernelConfiguration configuration{
            .node_id = n.id, .device_id = t.device, .servicing_device_id = t.device, .cq_id = 0,
            .processor = {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, n.role == DISPATCH_S ? 1 : 0},
            .logical_core = n.core.logical, .nocs = n.nocs, .resolved = r};
        if (n.role == PREFETCH_HD) configuration.kernel = p;
        else if (n.role == DISPATCH_HD) configuration.kernel = d;
        else configuration.kernel = s;
        validate_published_dispatch_configuration(configuration);
        owner->nodes.push_back(configuration); owner->kernels.push_back(plan_dispatch_kernel(t.arch, configuration));
        if (n.role != DISPATCH_S)
            owner->arguments.push_back({n.id, configuration.processor, n.core.logical, {0, 0, 0}});
        if (std::ranges::none_of(owner->registrations, [&](const auto& existing) { return existing.virtual_core == n.core.virtual_core; }))
            owner->registrations.push_back({n.core.virtual_core, CoreType::WORKER});
    }
    auto write = [&](DispatchInitializationKind kind, CoreCoord core, uint32_t address, std::vector<std::byte> bytes) {
        auto end = checked(uint64_t{address} + bytes.size());
        require(!bytes.empty() && address % 4 == 0 && bytes.size() % 4 == 0,
            Reject::MemoryMap, "unaligned dispatch initialization");
        require(end <= h.l1_size, Reject::MemoryMap, "dispatch initialization escapes L1");
        owner->initialization.push_back({kind, core, address, std::move(bytes)});
    };
    auto u32 = [&](DispatchInitializationKind kind, CoreCoord core, uint32_t address, uint32_t value) {
        write(kind, core, address, bytes_of(value));
    };
    using K = DispatchInitializationKind;
    const auto completion = t.completion_writer.logical;
    require(completion_base % 16 == 0, Reject::Queue, "completion queue pointer is not 16-byte aligned");
    u32(K::CompletionRead, completion, m.completion_q_rd, completion_base >> 4);
    u32(K::CompletionWrite, completion, m.completion_q_wr, completion_base >> 4);
    u32(K::CompletionEvent0, completion, m.completion_q0_last_event, 0);
    u32(K::CompletionEvent1, completion, m.completion_q1_last_event, 0);
    for (const auto& n : t.nodes) {
        const auto core = n.core.logical;
        if (n.role == PREFETCH_HD) {
            dispatch_telemetry_types::PrefetchCoreTelemetry telemetry{};
            if (o.telemetry_disabled) telemetry.signature = dispatch_telemetry_types::INVALID_TELEMETRY_SIGNATURE;
            write(K::PrefetchTelemetry, core, m.telemetry, bytes_of(telemetry));
            u32(K::PrefetchRead, core, m.prefetch_q_rd, checked(uint64_t{m.prefetch_q_base} + m.prefetch_q_size));
            u32(K::PrefetchPcieRead, core, m.prefetch_q_pcie_rd, issue_base);
            write(K::PrefetchQueue, core, m.prefetch_q_base, std::vector<std::byte>(m.prefetch_q_size));
        } else if (n.role == DISPATCH_HD) {
            dispatch_telemetry_types::DispatchCoreTelemetry telemetry{};
            if (o.telemetry_disabled) telemetry.signature = dispatch_telemetry_types::INVALID_TELEMETRY_SIGNATURE;
            write(K::DispatcherTelemetry, core, m.telemetry, bytes_of(telemetry));
            for (auto offset : m.sync_offsets) u32(K::DispatchSync, core, checked(uint64_t{m.sync_sem_base} + offset), 0);
        } else {
            for (auto offset : h.realtime_profiler_reset_offsets)
                u32(K::RealtimeProfiler, core, checked(uint64_t{m.realtime_profiler} + offset), 0);
            write(K::TelemetryControl, core, m.telemetry_control, bytes_of(dispatch_telemetry_types::DispatchTelemetryControl{}));
        }
    }
    deployment_detail::RecordWriter record;
    record.append(uint32_t{1}); record.append(DispatchProgramFamily::BlackholeSingleMmioWorkerOneCq);
    record.append(owner->inputs); record.append(owner->nodes); record.append(owner->semaphores);
    record.append(owner->arguments); record.append(owner->initialization); record.append(owner->registrations);
    owner->canonical = std::move(record.bytes);
    return DispatchProgramPlan(std::move(owner));
}
} // namespace tt::tt_metal::experimental
