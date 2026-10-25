// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_core_manager.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <list>
#include <optional>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include "context/metal_env_accessor.hpp"
#include "core_coord.hpp"
#include "core_descriptor.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "impl/context/metal_env_impl.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <internal/service/service_core_manager.hpp>
#include "impl/internal/service/service_core_manager_impl.hpp"
#include "impl/dispatch/dispatch_engine_cores.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

const tt_cxy_pair& dispatch_core_manager::prefetcher_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.prefetcher.has_value()) {
        validate_worker_core_locked(guard, assignment.prefetcher.value());
        return assignment.prefetcher.value();
    }
    // Issue queue interface is on the MMIO device
    ChipId mmio_device_id = env_.get_cluster().get_associated_mmio_device(device_id);
    CoreCoord issue_queue_coord = this->get_next_available_dispatch_core(guard, mmio_device_id);
    assignment.prefetcher = tt_cxy_pair(mmio_device_id, issue_queue_coord.x, issue_queue_coord.y);
    log_dispatch_assignment("Prefetcher", assignment.prefetcher.value(), device_id, channel, cq_id);
    return assignment.prefetcher.value();
}

bool dispatch_core_manager::is_prefetcher_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.prefetcher.has_value();
}

const tt_cxy_pair& dispatch_core_manager::prefetcher_d_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.prefetcher_d.has_value()) {
        validate_worker_core_locked(guard, assignment.prefetcher_d.value());
        return assignment.prefetcher_d.value();
    }
    CoreCoord prefetch_d_coord = this->get_next_available_dispatch_core(guard, device_id);
    assignment.prefetcher_d = tt_cxy_pair(device_id, prefetch_d_coord.x, prefetch_d_coord.y);
    log_dispatch_assignment("Prefetcher D", assignment.prefetcher_d.value(), device_id, channel, cq_id);
    return assignment.prefetcher_d.value();
}

bool dispatch_core_manager::is_prefetcher_d_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.prefetcher_d.has_value();
}

const tt_cxy_pair& dispatch_core_manager::completion_queue_writer_core(
    ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.completion_queue_writer.has_value()) {
        validate_worker_core_locked(guard, assignment.completion_queue_writer.value());
        return assignment.completion_queue_writer.value();
    }
    // Completion queue interface is on the MMIO device
    ChipId mmio_device_id = env_.get_cluster().get_associated_mmio_device(device_id);
    CoreCoord completion_queue_coord = this->get_next_available_dispatch_core(guard, mmio_device_id);
    assignment.completion_queue_writer =
        tt_cxy_pair(mmio_device_id, completion_queue_coord.x, completion_queue_coord.y);
    TT_ASSERT(
        not assignment.dispatcher.has_value(),
        "Command dispatcher core {} must match completion queue interface core for MMIO device {}",
        assignment.dispatcher.value().str(),
        device_id);
    assignment.dispatcher = assignment.completion_queue_writer;
    log_dispatch_assignment(
        "Completion Queue Writer", assignment.completion_queue_writer.value(), device_id, channel, cq_id);
    return assignment.completion_queue_writer.value();
}

bool dispatch_core_manager::is_completion_queue_writer_core_allocated(
    ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.completion_queue_writer.has_value();
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    return this->dispatcher_core_locked(guard, device_id, channel, cq_id);
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_core_locked(
    const NiuAdmission::Guard& guard, ChipId device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher.has_value()) {
        validate_worker_core_locked(guard, assignment.dispatcher.value());
        return assignment.dispatcher.value();
    }
    ChipId mmio_device_id = env_.get_cluster().get_associated_mmio_device(device_id);
    CoreCoord dispatcher_coord = this->get_next_available_dispatch_core(guard, mmio_device_id);
    assignment.dispatcher = tt_cxy_pair(mmio_device_id, dispatcher_coord.x, dispatcher_coord.y);
    TT_ASSERT(
        not assignment.completion_queue_writer.has_value(),
        "Command dispatcher core must match completion queue interface core for MMIO device {}",
        device_id);
    assignment.completion_queue_writer = assignment.dispatcher;
    log_dispatch_assignment("Dispatcher", assignment.dispatcher.value(), device_id, channel, cq_id);
    return assignment.dispatcher.value();
}

bool dispatch_core_manager::is_dispatcher_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.dispatcher.has_value();
}

bool dispatch_core_manager::is_dispatcher_s_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.dispatcher_s.has_value();
}

bool dispatch_core_manager::is_dispatcher_d_core_allocated(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.dispatcher_d.has_value();
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_d_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    return this->dispatcher_d_core_locked(guard, device_id, channel, cq_id);
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_d_core_locked(
    const NiuAdmission::Guard& guard, ChipId device_id, uint16_t channel, uint8_t cq_id) {
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher_d.has_value()) {
        validate_worker_core_locked(guard, assignment.dispatcher_d.value());
        return assignment.dispatcher_d.value();
    }
    CoreCoord dispatcher_d_coord = this->get_next_available_dispatch_core(guard, device_id);
    assignment.dispatcher_d = tt_cxy_pair(device_id, dispatcher_d_coord.x, dispatcher_d_coord.y);
    log_dispatch_assignment("Dispatcher D", assignment.dispatcher_d.value(), device_id, channel, cq_id);
    return assignment.dispatcher_d.value();
}

const tt_cxy_pair& dispatch_core_manager::fabric_mux_core(
    ChipId device_id, uint16_t channel, uint8_t cq_id, int tunnel) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (!assignment.fabric_mux.contains(tunnel)) {
        // Allocate the map node before consuming a dispatch pool slot.
        auto [entry, inserted] = assignment.fabric_mux.emplace(tunnel, tt_cxy_pair(device_id, 0, 0));
        try {
            CoreCoord coord = this->get_next_available_dispatch_core(guard, device_id);
            entry->second = tt_cxy_pair(device_id, coord.x, coord.y);
        } catch (...) {
            assignment.fabric_mux.erase(entry);
            throw;
        }
        log_dispatch_assignment("FabricMux", entry->second, device_id, channel, cq_id);
    }
    validate_worker_core_locked(guard, assignment.fabric_mux.at(tunnel));
    return assignment.fabric_mux.at(tunnel);
}

bool dispatch_core_manager::is_fabric_mux_core_allocated(
    ChipId device_id, uint16_t channel, uint8_t cq_id, int tunnel) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    return assignment.fabric_mux.contains(tunnel);
}

const tt_cxy_pair& dispatch_core_manager::dispatcher_s_core(ChipId device_id, uint16_t channel, uint8_t cq_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    dispatch_core_placement_t& assignment = this->dispatch_core_assignments[device_id][channel][cq_id];
    if (assignment.dispatcher_s.has_value()) {
        validate_worker_core_locked(guard, assignment.dispatcher_s.value());
        return assignment.dispatcher_s.value();
    }
    CoreCoord dispatcher_s_coord;
    if (this->get_dispatch_core_type() == CoreType::WORKER) {
        ChipId mmio_device_id = env_.get_cluster().get_associated_mmio_device(device_id);
        if (mmio_device_id == device_id) {
            // dispatch_s is on the same tensix core as dispatch_hd
            dispatcher_s_coord = this->dispatcher_core_locked(guard, device_id, channel, cq_id);
        } else {
            // dispatch_s is on the same tensix as dispatch_d
            dispatcher_s_coord = this->dispatcher_d_core_locked(guard, device_id, channel, cq_id);
        }
    } else if (this->get_dispatch_core_type() == CoreType::DISPATCH) {
        // Dispatch-engine 1CQ FD: prefetch (DM0), dispatch (DM1), and dispatch_s (DM2) share one tile.
        dispatcher_s_coord = this->dispatcher_core_locked(guard, device_id, channel, cq_id);
    } else {
        dispatcher_s_coord = this->get_next_available_dispatch_core(guard, device_id);
    }
    assignment.dispatcher_s = tt_cxy_pair(device_id, dispatcher_s_coord.x, dispatcher_s_coord.y);
    log_dispatch_assignment("Dispatcher S", assignment.dispatcher_s.value(), device_id, channel, cq_id);
    return assignment.dispatcher_s.value();
}

CoreType dispatch_core_manager::get_dispatch_core_type() const {
    // Quasar needs resolve (DISPATCH vs WORKER). WH/BH keep config-only (BH Galaxy regression).
    if (env_.get_cluster().arch() != tt::ARCH::QUASAR) {
        return get_core_type_from_config(this->dispatch_core_config_);
    }
    const auto& cluster = env_.get_cluster();
    if (cluster.all_chip_ids().empty()) {
        return get_core_type_from_config(this->dispatch_core_config_);
    }
    const ChipId device_id = *cluster.all_chip_ids().begin();
    return resolve_dispatch_core_type(env_, device_id, this->dispatch_core_config_);
}

DispatchCoreConfig dispatch_core_manager::get_dispatch_core_config() { return this->dispatch_core_config_; }

void dispatch_core_manager::add_dispatch_core_to_device(ChipId device_id, const CoreCoord& core) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    this->add_dispatch_core_to_device_locked(device_id, core);
}

void dispatch_core_manager::add_dispatch_core_to_device_locked(ChipId device_id, const CoreCoord& core) {
    // TODO: remove this API, we should read the core descriptor once, should not have backdoors like this to add cores
    auto& dispatch_cores = available_dispatch_cores_by_device.at(device_id);
    if (std::find(dispatch_cores.begin(), dispatch_cores.end(), core) == dispatch_cores.end()) {
        dispatch_cores.push_back(core);
    }
}

std::vector<CoreCoord> dispatch_core_manager::get_all_logical_dispatch_cores(ChipId device_id) {
    return tt::get_logical_dispatch_cores(this->env_, device_id, MAX_NUM_HW_CQS, this->dispatch_core_config_);
}

std::optional<tt_cxy_pair> dispatch_core_manager::get_reserved_realtime_profiler_core(ChipId device_id) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    auto it = reserved_realtime_profiler_core_by_device_.find(device_id);
    if (it == reserved_realtime_profiler_core_by_device_.end()) {
        return std::nullopt;
    }
    validate_worker_core_locked(guard, it->second);
    return it->second;
}

// private methods

dispatch_core_manager::dispatch_core_manager(
    const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, MetalEnvImpl& env, MetalContext& ctx) :
    env_(env), ctx_(ctx) {
    try {
        this->reset_dispatch_core_manager(dispatch_core_config, num_hw_cqs, env);
    } catch (...) {
        // No assignment from an unsuccessful constructor was published.
        auto guard = ctx_.niu_admission().lock();
        for (const auto& [chip, reservations] : worker_reservations_) {
            for (const auto& [core, token] : reservations) {
                std::string ignored;
                ctx_.niu_admission().cancel_unissued_locked(guard, token, ignored);
            }
        }
        throw;
    }
}

void dispatch_core_manager::reset_dispatch_core_manager(
    const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, MetalEnvImpl& env) {
    auto guard = ctx_.niu_admission().lock();
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    this->dispatch_core_assignments.clear();
    this->available_dispatch_cores_by_device.clear();
    this->reserved_realtime_profiler_core_by_device_.clear();
    this->dispatch_core_config_ = dispatch_core_config;
    for (ChipId device_id : env.get_cluster().all_chip_ids()) {
        if (env.get_cluster().arch() == tt::ARCH::QUASAR && env.get_rtoptions().get_fast_dispatch()) {
            tt::tt_metal::detail::validate_quasar_dispatch_cores_for_fd(
                env, device_id, num_hw_cqs, dispatch_core_config);
        }

        std::list<CoreCoord>& logical_dispatch_cores = this->available_dispatch_cores_by_device[device_id];
        for (const CoreCoord& logical_dispatch_core :
             tt::get_logical_dispatch_cores(env, device_id, MAX_NUM_HW_CQS, dispatch_core_config)) {
            logical_dispatch_cores.push_back(logical_dispatch_core);
        }

        this->num_hw_cqs = num_hw_cqs;

        // When running Multiple CQs using Ethernet Dispatch, we may need more dispatch cores than those allocated in
        // the core descriptor (ex: 2 CQs on N300 need 10 dispatch cores and the core descriptor only allocates 6).
        // Infer the remaining dispatch cores from the idle eth core list (this is device dependent).
        if (get_core_type_from_config(dispatch_core_config) == CoreType::ETH) {
            for (const auto& idle_eth_core : env_.get_control_plane().get_inactive_ethernet_cores(device_id)) {
                add_dispatch_core_to_device_locked(device_id, idle_eth_core);
            }
        }

        // Reserve a tensix for the real-time profiler from the back of the dispatch pool
        // (dispatch consumes from the front). Skipped when:
        //   - chip is not MMIO-capable (RT profiler is gated to MMIO chips upstream);
        //   - dispatch core type is ETH (pool holds ethernet cores, not tensixes);
        //   - fabric tensix datamover (MUX or UDM) is enabled (it claims dispatch-pool slots
        //     at fabric-init time and shrinking the pool further can starve fabric_mux_core);
        //   - chip is Quasar (FD kernels are placed on the same core, a significant architectural
        //     change from WH/BH, so the RT profiler isn't currently supported on Quasar).
        const bool is_mmio = env.get_cluster().get_associated_mmio_device(device_id) == device_id;
        const bool fabric_tensix_datamover_enabled =
            env.get_fabric_tensix_config() != tt_fabric::FabricTensixConfig::DISABLED;
        const bool is_quasar = env.get_cluster().arch() == tt::ARCH::QUASAR;
        const auto claimed = ctx_.get_service_core_manager().impl().claimed_cores_locked(guard, device_id);
        const auto profiler_slot = std::find_if(
            logical_dispatch_cores.rbegin(), logical_dispatch_cores.rend(),
            [&claimed](const CoreCoord& core) { return !claimed.contains(core); });
        if (is_mmio && get_core_type_from_config(dispatch_core_config) == CoreType::WORKER &&
            !fabric_tensix_datamover_enabled && !is_quasar && profiler_slot != logical_dispatch_cores.rend()) {
            CoreCoord rt_core = *profiler_slot;
            reserve_worker_core_locked(guard, tt_cxy_pair(device_id, rt_core.x, rt_core.y));
            logical_dispatch_cores.erase(std::prev(profiler_slot.base()));
            this->reserved_realtime_profiler_core_by_device_.emplace(
                device_id, tt_cxy_pair(device_id, rt_core.x, rt_core.y));
        }

        // Service claims are filtered under the same admission guard by every
        // snapshot/allocation. Keep their pool entries so unissued release can
        // make them available again after a dispatch-manager reconstruction.
    }
}

std::vector<CoreCoord> dispatch_core_manager::get_available_dispatch_cores(ChipId device_id) {
    auto guard = ctx_.niu_admission().lock();
    return get_available_dispatch_cores_locked(guard, device_id);
}

std::vector<CoreCoord> dispatch_core_manager::get_available_dispatch_cores_locked(
    const NiuAdmission::Guard& guard, ChipId device_id) {
    ctx_.niu_admission().validate_guard(guard);
    std::lock_guard<std::mutex> lock(this->dispatch_core_assignments_mutex);
    auto it = this->available_dispatch_cores_by_device.find(device_id);
    if (it == this->available_dispatch_cores_by_device.end()) {
        return {};
    }
    std::vector<CoreCoord> available(it->second.begin(), it->second.end());
    if (get_dispatch_core_type() == CoreType::WORKER) {
        const auto claimed = ctx_.get_service_core_manager().impl().claimed_cores_locked(guard, device_id);
        std::erase_if(available, [&claimed](const CoreCoord& core) { return claimed.contains(core); });
    }
    return available;
}

void dispatch_core_manager::reserve_worker_core_locked(const NiuAdmission::Guard& guard, const tt_cxy_pair& core) {
    auto& admission = ctx_.niu_admission();
    admission.validate_guard(guard);
    if (get_dispatch_core_type() != CoreType::WORKER) return;
    auto& reservations = worker_reservations_[core.chip];
    if (reservations.contains(core)) {
        validate_worker_core_locked(guard, core);
        return;
    }
    const ChipId chip = core.chip;
    const auto physical = env_.get_cluster().get_soc_desc(chip).get_physical_tensix_core_from_logical(core);
    const std::array keys{detail::NiuKey{chip, physical, detail::NiuNoc::Noc0},
                          detail::NiuKey{chip, physical, detail::NiuNoc::Noc1}};
    std::string error;
    auto token = admission.reserve_ordinary_locked(guard, keys, error);
    TT_FATAL(token.has_value(), "Dispatch worker NIU reservation failed: {}", error);
    try {
        reservations.emplace(core, *token);
    } catch (...) {
        admission.cancel_unissued_locked(guard, *token, error);
        throw;
    }
}

void dispatch_core_manager::validate_worker_core_locked(const NiuAdmission::Guard& guard, const tt_cxy_pair& core) {
    auto& admission = ctx_.niu_admission();
    admission.validate_guard(guard);
    if (get_dispatch_core_type() != CoreType::WORKER) return;
    std::string error;
    const auto& reservation = worker_reservations_.at(core.chip).at(core);
    TT_FATAL(admission.validate_ordinary_locked(guard, reservation, error),
             "Dispatch worker NIU reservation is no longer admissible: {}", error);
}

CoreCoord dispatch_core_manager::get_next_available_dispatch_core(const NiuAdmission::Guard& guard, ChipId device_id) {
    ctx_.niu_admission().validate_guard(guard);
    auto found = available_dispatch_cores_by_device.find(device_id);
    TT_FATAL(found != available_dispatch_cores_by_device.end(), "Invalid device ID to assign dispatch cores {}", device_id);
    auto& available = found->second;
    const auto claimed = get_dispatch_core_type() == CoreType::WORKER
        ? ctx_.get_service_core_manager().impl().claimed_cores_locked(guard, device_id)
        : std::unordered_set<CoreCoord>{};
    auto next = std::find_if(available.begin(), available.end(),
                             [&claimed](const CoreCoord& core) { return !claimed.contains(core); });
    TT_FATAL(next != available.end(),
             "No more available dispatch cores on device {} to assign. Expand dispatch cores specified in core descriptor YAML",
             device_id);
    const CoreCoord core = *next;
    reserve_worker_core_locked(guard, tt_cxy_pair(device_id, core.x, core.y));
    available.erase(next);
    return core;
}

void dispatch_core_manager::log_dispatch_assignment(
    [[maybe_unused]] std::string name,
    [[maybe_unused]] tt_cxy_pair& cxy,
    [[maybe_unused]] ChipId device_id,
    [[maybe_unused]] uint16_t channel,
    [[maybe_unused]] uint8_t cq_id,
    [[maybe_unused]] bool force_ethernet) {
    log_debug(
        tt::LogMetal,
        "Allocated {} Core: {}({}) for Device {} Channel {} CQ ID {}",
        name,
        cxy.str(),
        env_.get_cluster()
            .get_virtual_coordinate_from_logical_coordinates(
                cxy, force_ethernet ? CoreType::ETH : get_dispatch_core_type())
            .str(),
        device_id,
        channel,
        cq_id);
}

}  // namespace tt::tt_metal
