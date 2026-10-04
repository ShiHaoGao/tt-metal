// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "dispatch_plan.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "impl/dispatch/system_memory_queue_plan.hpp"
#include <array>
#include <memory>
#include <span>
#include <stdexcept>

namespace tt::tt_metal::experimental {
enum class DispatchProgramFamily : uint8_t { BlackholeSingleMmioWorkerOneCq };
enum class DispatchProgramReject : uint8_t {
    Architecture, Topology, Placement, Observation, Queue, MemoryMap, Semaphore, Overflow
};
class DispatchProgramError : public std::invalid_argument {
public:
    DispatchProgramError(DispatchProgramReject reason, const char* message);
    DispatchProgramReject reason() const noexcept { return reason_; }
private:
    DispatchProgramReject reason_;
};

struct DispatchCorePlacement {
    CoreCoord logical;
    CoreCoord virtual_core;
    std::array<CoreCoord, 2> noc;
};
struct DispatchPlacedNode {
    uint32_t id;
    ChipId device;
    ChipId servicing_device;
    uint8_t cq;
    DispatchWorkerType role;
    std::vector<uint32_t> upstream;
    std::vector<uint32_t> downstream;
    noc_selection_t nocs;
    DispatchCorePlacement core;
    int tunnel_index = -1;
};
struct DispatchTopologyFacts {
    tt::ARCH arch;
    ChipId device;
    std::vector<ChipId> deployment_devices;
    std::vector<ChipId> serviced_devices;
    bool mmio;
    bool galaxy;
    bool fabric;
    bool mock;
    CoreType core_type;
    uint8_t num_hw_cqs;
    bool subordinate_enabled;
    bool distributed_dispatcher;
    std::vector<DispatchPlacedNode> nodes; // Actual static-configuration traversal order.
    DispatchCorePlacement completion_writer;
    DispatchCorePlacement unused_core; // Existing SDK logical (0, 0) sentinel projection.
};
struct DispatchWorkerFacts {
    uint32_t count;
    std::array<uint32_t, 2> multicast;
    uint32_t physical_eth_cores;
    uint32_t virtual_eth_cores;
};
struct DispatchObservationFacts {
    bool watcher_enabled;
    bool watcher_noinline;
    bool watcher_dispatch_disabled;
    bool reads_dispatch_cores;
    bool telemetry_disabled;
    bool device_print_enabled;
};
struct DispatchQueueFacts {
    SystemMemoryQueueInputs inputs;
    SystemMemoryQueuePlan plan;
    std::optional<uint32_t> dram_bank;
};
// Values copied from the owning actual DispatchMemMap; no Hal/context accessor.
struct DispatchMemoryFacts {
    uint32_t prefetch_q_base;
    uint32_t prefetch_q_size;
    uint32_t prefetch_q_entry_bytes;
    uint32_t prefetch_q_rd;
    uint32_t prefetch_q_pcie_rd;
    uint32_t cmddat_q_base;
    uint32_t cmddat_q_size;
    uint32_t scratch_db_base;
    uint32_t scratch_db_size;
    uint32_t ringbuffer_size;
    uint32_t prefetch_d_pages;
    uint32_t dispatch_buffer_base;
    uint32_t dispatch_buffer_pages;
    uint32_t subordinate_buffer_size;
    uint32_t subordinate_buffer_pages;
    uint32_t completion_q_wr;
    uint32_t completion_q_rd;
    uint32_t completion_q0_last_event;
    uint32_t completion_q1_last_event;
    uint32_t host_completion_q_wr;
    uint32_t dispatch_progress;
    uint32_t sync_sem_base;
    std::vector<uint32_t> sync_offsets;
    uint32_t fabric_header_base;
    uint32_t fabric_sync_status;
    uint32_t telemetry;
    uint32_t telemetry_control;
    uint32_t realtime_profiler;
    uint32_t first_stream;
    uint32_t completion_counter_offset;
};
struct DispatchHalFacts {
    uint32_t l1_alignment;
    uint32_t l1_size;
    uint32_t worker_go_message;
    uint32_t ethernet_go_message;
    // Actual HAL factory offsets, in the original ordered nine-word reset sequence.
    std::array<uint32_t, 9> realtime_profiler_reset_offsets;
};
struct DispatchProgramInputs {
    DispatchTopologyFacts topology;
    DispatchWorkerFacts workers;
    DispatchObservationFacts observation;
    DispatchQueueFacts queue;
    DispatchMemoryFacts memory;
    DispatchHalFacts hal;
};
enum class DispatchSemaphoreRole : uint8_t {
    PrefetchDownstream, PrefetchSync, PrefetchSubordinate, DispatcherBuffer,
    SubordinateBuffer, SubordinateShutdown
};
struct DispatchSemaphorePlan {
    uint32_t node;
    CoreCoord logical;
    DispatchSemaphoreRole role;
    uint32_t id;
    uint32_t slot_offset;
    uint32_t initial_value;
};
struct DispatchRuntimeArguments {
    uint32_t node;
    HalProcessorIdentifier processor;
    CoreCoord logical;
    std::vector<uint32_t> words;
};
enum class DispatchInitializationKind : uint8_t {
    CompletionRead, CompletionWrite, CompletionEvent0, CompletionEvent1,
    PrefetchTelemetry, PrefetchRead, PrefetchPcieRead, PrefetchQueue,
    DispatcherTelemetry, DispatchSync, RealtimeProfiler, TelemetryControl
};
struct DispatchInitialization {
    DispatchInitializationKind kind;
    CoreCoord logical;
    uint32_t address;
    std::vector<std::byte> bytes;
};
struct DispatchCoreRegistration {
    CoreCoord virtual_core;
    CoreType type;
};
struct DispatchSemaphoreAddress {
    uint32_t node;
    DispatchSemaphoreRole role;
    CoreCoord logical;
    uint32_t address;
};
class DispatchProgramPlan {
public:
    DispatchProgramFamily family() const;
    const DispatchProgramInputs& inputs() const;
    ChipId device_id() const;
    std::span<const DispatchKernelConfiguration> dispatch_nodes() const;
    std::span<const DispatchPlan> kernel_plans() const;
    std::span<const DispatchSemaphorePlan> semaphores() const;
    std::span<const DispatchRuntimeArguments> runtime_arguments() const;
    std::span<const DispatchInitialization> initialization() const;
    std::span<const DispatchCoreRegistration> core_registrations() const;
    std::span<const std::byte> canonical_record() const;
    // Program finalization owns the base. This function performs no live query.
    std::vector<DispatchSemaphoreAddress> semaphore_addresses(
        uint32_t kernel_config_base, uint32_t finalized_sem_offset) const;
private:
    struct Storage;
    std::shared_ptr<const Storage> storage_;
    explicit DispatchProgramPlan(std::shared_ptr<const Storage>);
    friend DispatchProgramPlan plan_dispatch_program(const DispatchProgramInputs&);
};
DispatchProgramPlan plan_dispatch_program(const DispatchProgramInputs&);
} // namespace tt::tt_metal::experimental
