// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "program_execution.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>
#include "program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/experimental/native_kernel/native_image.hpp"
#include "impl/experimental/native_kernel/native_kernel.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/internal/service/service_core_manager_impl.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::detail {

ProgramExecution::FailureScope::FailureScope(std::shared_ptr<ProgramExecution> execution) :
    execution(std::move(execution)), exceptions(std::uncaught_exceptions()) {}
ProgramExecution::FailureScope::~FailureScope() {
    if (execution && execution->managed && std::uncaught_exceptions() > exceptions) {
        execution->abandon_managed(true);
    }
}

ProgramExecution::Preparation::Preparation(Preparation&& other) noexcept :
    execution_(std::move(other.execution_)), borrowed_(std::exchange(other.borrowed_, false)) {}

ProgramExecution::Preparation::~Preparation() {
    if (!execution_ || !borrowed_) {
        return;
    }
    auto admission = execution_->owner.lock();
    if (!admission) {
        return;
    }
    NiuAdmission::Retirement retired;
    bool cancel_ready = false;
    {
        auto guard = admission->lock();
        execution_->preparation.release();
        cancel_ready = execution_->preparation.ready_to_cancel();
        if (!execution_->preparation.occupied() && execution_->state == State::Withdrawn &&
            execution_->attached_chips.empty() && !execution_->resources_retired) {
            admission->retire_managed_locked(*execution_, retired);
        }
    }
    if (cancel_ready) {
        execution_->abandon_managed();
    }
}

ProgramExecution::Preparation ProgramExecution::reserve_direct(Program& program, IDevice& device) {
    // A rejected caller has no preparation or failure-unwind authority over
    // the real invocation. Validate its complete identity before borrowing.
    auto execution = reserve(program, std::array{&device});
    if (!execution->managed) {
        return Preparation(std::move(execution), false);
    }
    auto admission = execution->owner.lock();
    TT_FATAL(admission, "Managed NIU invocation context has closed");
    const auto cores = program_cores(program, std::array{&device});
    {
        auto guard = admission->lock();
        TT_FATAL(
            execution->origin == Origin::Direct,
            "Managed queue invocation cannot use a direct configuration or launch receiver");
        // Revalidate in the same critical section that grants the borrow:
        // close, rebind or an intervening submission cannot authorize it.
        execution->validate_member_locked(*admission, guard, program, std::array{&device}, cores);
        TT_FATAL(execution->preparation.acquire(), "Managed invocation already has another host preparation owner");
    }
    return Preparation(std::move(execution), true);
}

ProgramExecution::Preparation ProgramExecution::reserve_workload(
    distributed::MeshDevice& mesh, distributed::MeshWorkload& workload) {
    auto execution = validate_workload(mesh, workload);
    if (!execution) {
        return Preparation(nullptr, false);
    }
    auto admission = execution->owner.lock();
    TT_FATAL(admission, "Managed workload preparation context has closed");
    {
        auto guard = admission->lock();
        execution->validate_group_locked(execution->group_key(workload));
        std::string error;
        TT_FATAL(
            admission->validate_managed_locked(guard, *execution->token, execution->boots, error),
            "Managed workload preparation binding changed: {}", error);
        TT_FATAL(execution->preparation.acquire(), "Managed invocation already has another host preparation owner");
    }
    return Preparation(std::move(execution), true);
}

ProgramExecution::Cores ProgramExecution::program_cores(Program& program, std::span<IDevice* const> devices) {
    Cores result;
    for (auto* device : devices) {
        TT_FATAL(dynamic_cast<Device*>(device), "NIU execution requires actual local Devices, not a MeshDevice id");
        auto& ctx = MetalContext::instance(extract_context_id(device));
        const auto& soc = ctx.get_cluster().get_soc_desc(device->id());
        const auto logical = program.impl().logical_cores();
        for (size_t index = 0; index != logical.size(); ++index) {
            if (ctx.hal().get_core_type(index) != CoreType::WORKER) {
                continue;
            }
            for (const auto& core : logical[index]) {
                result[device->id()].insert(soc.get_physical_tensix_core_from_logical(core));
            }
        }
    }
    return result;
}

ProgramExecution::ProgramExecution(
    MetalContext& context, Origin origin, Cores cores, std::optional<NiuAdmission::Allocation> allocation) :
    owner(context.niu_admission_owner()),
    simulated(
        context.get_cluster().get_target_device_type() == tt::TargetDevice::Mock ||
        context.get_cluster().get_target_device_type() == tt::TargetDevice::Emule),
    origin(origin),
    footprint(cores),
    pending(std::move(cores)),
    managed(std::move(allocation)) {
    if (managed) {
        token = managed->token_;
        return;
    }
    std::vector<NiuKey> keys;
    for (const auto& [chip, workers] : pending) {
        for (const auto& core : workers) {
            for (NiuNoc noc : {NiuNoc::Noc0, NiuNoc::Noc1}) {
                keys.push_back({chip, core, noc});
            }
        }
    }
    if (keys.empty()) {
        return;
    }
    std::string error;
    token = context.niu_admission().reserve_ordinary(keys, error);
    TT_FATAL(token, "Program NIU admission failed: {}", error);
}

std::shared_ptr<ProgramExecution> ProgramExecution::reserve(Program& program, std::span<IDevice* const> devices) {
    TT_FATAL(!devices.empty(), "Program NIU admission requires actual local devices");
    auto& context = MetalContext::instance(extract_context_id(devices.front()));
    for (auto* device : devices) {
        TT_FATAL(&MetalContext::instance(extract_context_id(device)) == &context, "Program NIU devices cross contexts");
    }
    if (program.impl().managed_niu_bound_) {
        auto execution = program.impl().managed_niu_invocation_.lock();
        TT_FATAL(execution, "Managed NIU invocation owner has expired");
        const auto cores = program_cores(program, devices);
        auto admission = execution->owner.lock();
        TT_FATAL(
            admission && admission == context.niu_admission_owner().lock(), "Managed NIU invocation crosses contexts");
        auto guard = admission->lock();
        execution->validate_member_locked(*admission, guard, program, devices, cores);
        return execution;
    }
    auto result = std::shared_ptr<ProgramExecution>(
        new ProgramExecution(context, Origin::Direct, program_cores(program, devices)));
    for (auto* device : devices) {
        TT_FATAL(device->is_initialized(), "Program NIU execution requires an open actual Device");
    }
    return result;
}

ProgramExecution::Cores ProgramExecution::dispatch_cores(
    distributed::MeshDevice& mesh, std::span<const SubDeviceId> sub_devices) {
    auto& context = MetalContext::instance(mesh.impl().get_context_id());
    Cores cores;
    std::set<CoreCoord> logical_workers;
    for (auto subdevice : sub_devices) {
        const auto workers = mesh.worker_cores(HalProgrammableCoreType::TENSIX, subdevice);
        for (const auto& range : workers.ranges()) {
            for (const auto& core : range) {
                logical_workers.insert(core);
            }
        }
    }
    // FD sends GO (and receives worker NoC acknowledgements) for the entire
    // selected subdevice, including local chips without a placed program.
    for (auto* device : mesh.impl().get_devices()) {
        TT_FATAL(device->is_initialized(), "Program NIU dispatch requires open actual Devices");
        const auto& soc = context.get_cluster().get_soc_desc(device->id());
        for (const auto& core : logical_workers) {
            cores[device->id()].insert(soc.get_physical_tensix_core_from_logical(core));
        }
    }
    return cores;
}

std::shared_ptr<ProgramExecution> ProgramExecution::reserve_dispatch(
    distributed::MeshDevice& mesh, std::span<const SubDeviceId> sub_devices) {
    return std::shared_ptr<ProgramExecution>(new ProgramExecution(
        MetalContext::instance(mesh.impl().get_context_id()), Origin::Queue, dispatch_cores(mesh, sub_devices)));
}

ProgramExecution::Member ProgramExecution::freeze_member(Program& program, std::span<IDevice* const> devices) {
    Member member;
    member.program = program.impl().shared_from_this();
    member.cores = program_cores(program, devices);
    for (auto* device : devices) {
        member.devices.emplace(device->id(), device);
    }
    member.logical = program.impl().logical_cores();
    for (size_t index = 0; index != member.logical.size(); ++index) {
        for (const auto& [handle, kernel] : program.impl().get_kernels(index)) {
            member.kernels.emplace(std::make_pair(index, handle), kernel);
        }
    }
    return member;
}

ProgramExecution::Endpoints ProgramExecution::freeze_managed_endpoints(MetalContext& context, const Cores& cores) {
    // This uses actual SDK topology and service-safe logical grids. It must run
    // outside the admission lock: Device coordinate queries can enter context
    // and service owners. Only the original invocation's reserved physical
    // workers are admitted, never arbitrary Tensix/DRAM/PCIe coordinates.
    Endpoints result;
    for (const auto& [chip, workers] : cores) {
        auto* device = dynamic_cast<Device*>(context.device_manager()->get_active_device(chip));
        TT_FATAL(device && device->is_initialized(), "Managed endpoints require the original active SDK Device");
        const auto& soc = context.get_cluster().get_soc_desc(chip);
        const auto grid = device->compute_with_storage_grid_size();
        std::set<CoreCoord> covered;
        auto& endpoints = result[chip];
        for (uint32_t x = 0; x != grid.x; ++x) {
            for (uint32_t y = 0; y != grid.y; ++y) {
                const CoreCoord logical{x, y};
                const auto physical = soc.get_physical_tensix_core_from_logical(logical);
                if (!workers.contains(physical)) continue;
                covered.insert(physical);
                const auto translated = device->worker_core_from_logical_core(logical);
                for (uint8_t noc = 0; noc != TT_NATIVE_NOC_COUNT; ++noc) {
                    const auto coordinate = device->virtual_noc0_coordinate(noc, translated);
                    TT_FATAL(coordinate.x < TT_NATIVE_NOC_COORDINATE_LIMIT &&
                                 coordinate.y < TT_NATIVE_NOC_COORDINATE_LIMIT &&
                                 device->get_noc_unicast_encoding(noc, translated) ==
                                     context.hal().noc_xy_encoding(coordinate.x, coordinate.y),
                             "Managed endpoint differs from actual SDK NoC coordinate encoding");
                    endpoints.push_back({noc, static_cast<uint8_t>(coordinate.x),
                                         static_cast<uint8_t>(coordinate.y), 0});
                }
            }
        }
        TT_FATAL(covered == workers, "Managed footprint includes workers outside the actual safe compute grid");
    }
    return result;
}

void ProgramExecution::validate_managed_traffic(
    const Member& member, const NiuAdmission::Boots& boots, const Endpoints& endpoints,
    distributed::MeshDevice* mesh) {
    auto program = member.program.lock();
    TT_FATAL(program && member.kernels.size() == program->num_kernels(),
             "Managed native Program lost its complete algorithm inventory");
    std::vector<std::shared_ptr<Kernel>> retained;
    std::vector<const NativeElfKernel*> native;
    for (const auto& [identity, frozen] : member.kernels) {
        auto kernel = frozen.lock();
        const auto& current = program->get_kernels(identity.first);
        const auto found = current.find(identity.second);
        TT_FATAL(kernel && found != current.end() && found->second == kernel,
                 "Managed native Program algorithm identity changed");
        const auto* image = dynamic_cast<const NativeElfKernel*>(kernel.get());
        TT_FATAL(image, "Managed NIU invocation requires verified native algorithm traffic");
        for (const auto& [chip, device] : member.devices) {
            const auto boot = boots.find(chip);
            TT_FATAL(boot != boots.end() && boot->second && boot->second->live(),
                     "Managed native traffic requires its original live firmware boot");
            for (const auto& role : image->native_images().images)
                TT_FATAL(role.firmware().matches(boot->second->bundle()),
                         "Managed native traffic declaration differs from actual firmware boot");
            const auto* actual = dynamic_cast<const Device*>(device);
            TT_FATAL(actual && device->allocator(), "Managed native traffic requires the actual SDK allocator");
            image->validate_managed_traffic(device->allocator()->impl().get_config());
            const auto allowed = endpoints.find(chip);
            TT_FATAL(allowed != endpoints.end(), "Managed native traffic lost its original endpoint projection");
            image->validate_managed_endpoints(allowed->second);
        }
        if (mesh) {
            TT_FATAL(mesh->allocator_impl(), "Managed native traffic requires the actual mesh allocator");
            image->validate_managed_traffic(mesh->allocator_impl()->get_config());
        }
        retained.push_back(std::move(kernel));
        native.push_back(image);
    }
    NativeElfKernel::validate_managed_group(native);
}

std::unique_ptr<ProgramExecution::ManagedInvocation> ProgramExecution::acquire_managed(
    Program& program, std::span<IDevice* const> devices) {
    TT_FATAL(!devices.empty(), "Managed NIU invocation requires actual Devices and their boot");
    auto& context = MetalContext::instance(extract_context_id(devices.front()));
    for (auto* device : devices) {
        TT_FATAL(
            dynamic_cast<Device*>(device) && device->is_initialized() &&
                extract_context_id(device) == context.get_context_id(),
            "Managed NIU invocation requires open actual Devices in one context");
    }
    auto member = freeze_member(program, devices);
    auto cores = member.cores;
    Group group;
    group.members.push_back(0);
    group.cores = cores;
    std::vector<Member> members;
    members.push_back(std::move(member));
    std::vector<Group> groups;
    groups.push_back(std::move(group));
    return acquire_managed(context, Origin::Direct, std::move(cores), std::move(members), std::move(groups));
}

std::unique_ptr<ProgramExecution::ManagedInvocation> ProgramExecution::acquire_managed(
    distributed::MeshDevice& mesh, std::span<distributed::MeshWorkload* const> workloads) {
    TT_FATAL(!workloads.empty(), "Managed invocation requires a nonempty workload collection");
    auto api_lock = mesh.impl().lock_api();
    auto& context = MetalContext::instance(mesh.impl().get_context_id());
    const auto origin = context.rtoptions().get_fast_dispatch() ? Origin::Queue : Origin::Direct;
    std::vector<Member> members;
    std::vector<Group> groups;
    std::vector<std::shared_ptr<distributed::MeshWorkloadImpl>> retained_workloads;
    std::set<const distributed::MeshWorkloadImpl*> workload_identities;
    std::set<const ProgramImpl*> program_identities;
    Cores cores;
    for (auto* workload : workloads) {
        TT_FATAL(workload, "Managed invocation received a null workload");
        TT_FATAL(workload->pimpl_, "Managed invocation received a moved-from workload");
        TT_FATAL(
            workload_identities.insert(workload->pimpl_.get()).second,
            "Managed invocation received a duplicate workload owner");
        TT_FATAL(!workload->get_programs().empty(), "Managed invocation received an empty workload");
        Group group;
        group.workload = workload->pimpl_;
        for (auto& [range, program] : workload->get_programs()) {
            TT_FATAL(
                std::none_of(group.placements.begin(), group.placements.end(), [&](const auto& previous) {
                    return previous.first.intersects(range);
                }),
                "Managed workload has overlapping Program mesh ranges");
            TT_FATAL(
                program_identities.insert(&program.impl()).second,
                "Managed invocation requires distinct Program owners across its workload collection");
            TT_FATAL(
                program.impl().get_context_id() == context.get_context_id(),
                "Managed Program belongs to another context");
            group.placements.emplace_back(range, program.impl().shared_from_this());
            const auto& devices = mesh.impl().get_local_devices(range);
            if (devices.empty()) {
                continue;
            }
            auto member = freeze_member(program, devices);
            member.group = GroupKey{groups.size()};
            TT_FATAL(!member.cores.empty(), "Managed workload requires nonempty Worker Program members");
            for (const auto& [chip, workers] : member.cores) {
                group.cores[chip].insert(workers.begin(), workers.end());
            }
            group.members.push_back(members.size());
            members.push_back(std::move(member));
        }
        TT_FATAL(!group.members.empty(), "Managed workload requires nonempty local Worker Program members");
        if (origin == Origin::Queue) {
            const auto& ids = workload->impl().determine_sub_device_ids(&mesh);
            group.subdevices.assign(ids.begin(), ids.end());
            TT_FATAL(group.subdevices.size() == 1, "Managed FD workload requires one dispatch subdevice batch");
            group.cores = dispatch_cores(mesh, group.subdevices);
        }
        for (const auto& [chip, workers] : group.cores) {
            cores[chip].insert(workers.begin(), workers.end());
        }
        retained_workloads.push_back(workload->pimpl_);
        groups.push_back(std::move(group));
    }
    if (origin == Origin::Queue) {
        for (uint32_t cq = 0; cq != mesh.num_hw_cqs(); ++cq) {
            TT_FATAL(!mesh.mesh_command_queue(cq).trace_id(), "Managed NIU invocation does not support trace capture");
        }
    }
    return acquire_managed(
        context, origin, std::move(cores), std::move(members), std::move(groups), &mesh,
        std::move(retained_workloads));
}

std::unique_ptr<ProgramExecution::ManagedInvocation> ProgramExecution::acquire_managed(
    MetalContext& context,
    Origin origin,
    Cores cores,
    std::vector<Member> members,
    std::vector<Group> groups,
    distributed::MeshDevice* mesh,
    std::vector<std::shared_ptr<distributed::MeshWorkloadImpl>> workloads) {
    TT_FATAL(!members.empty() && !cores.empty(), "Managed NIU invocation requires a nonempty Worker footprint");
    // Validate every member and retained workload before consulting firmware or
    // reserving any physical selector. A malformed later group cannot publish
    // a partial invocation or be hidden by an earlier chip's missing boot.
    std::set<const ProgramImpl*> identities;
    for (const auto& member : members) {
        auto program = member.program.lock();
        TT_FATAL(
            program && !member.cores.empty() && identities.insert(program.get()).second,
            "Managed invocation requires distinct nonempty Worker Program members");
        TT_FATAL(program->get_context_id() == context.get_context_id(), "Managed Program belongs to another context");
        {
            auto guard = context.niu_admission().lock();
            if (auto previous = program->managed_niu_invocation_.lock()) {
                TT_FATAL(
                    previous->state == State::Withdrawn && previous->resources_retired,
                    "Managed Program still belongs to a single-shot invocation or retained Device generation");
            }
        }
        for (const auto& [chip, workers] : member.cores) {
            const auto& group_cores = groups.at(member.group.index).cores;
            const auto found = group_cores.find(chip);
            TT_FATAL(
                found != group_cores.end() &&
                    std::includes(found->second.begin(), found->second.end(), workers.begin(), workers.end()),
                "Managed dispatch footprint does not cover its Program members");
        }
    }
    {
        auto guard = context.niu_admission().lock();
        for (const auto& workload : workloads) {
            if (auto previous = workload->managed_niu_invocation_.lock()) {
                TT_FATAL(
                    previous->state == State::Withdrawn && previous->resources_retired,
                    "Managed workload still belongs to a single-shot invocation or retained Device generation");
            }
        }
    }
    NiuAdmission::Boots boots;
    std::vector<NiuRequest> requests;
    for (const auto& [chip, workers] : cores) {
        auto boot = context.native_firmware(chip);
        TT_FATAL(boot && boot->live(), "Managed NIU invocation requires a successful actual SDK boot");
        TT_FATAL(
            context.device_manager() && context.device_manager()->is_device_active(chip),
            "Managed invocation requires the actual SDK DeviceManager owner");
        boots.emplace(chip, std::move(boot));
        for (const auto& core : workers) {
            for (auto noc : {NiuNoc::Noc0, NiuNoc::Noc1}) {
                requests.push_back({{chip, core, noc}, niu_selector_count});
            }
        }
    }
    for (const auto& member : members) {
        for (const auto& [chip, device] : member.devices) {
            TT_FATAL(
                context.device_manager()->get_active_device(chip) == device,
                "Managed invocation requires actual SDK-owned Device identity");
        }
    }
    // Static algorithm eligibility and actual configured load-region bounds
    // precede reservation/publication. They do not prove NIU bootstrap or clean.
    auto endpoints = freeze_managed_endpoints(context, cores);
    for (const auto& member : members) validate_managed_traffic(member, boots, endpoints, mesh);
    std::string error;
    auto allocation = context.niu_admission().acquire_managed(boots, requests, error);
    TT_FATAL(allocation, "Managed NIU invocation admission failed: {}", error);
    const auto rollback_token = allocation->token_;
    try {
        auto execution = std::shared_ptr<ProgramExecution>(
            new ProgramExecution(context, origin, std::move(cores), std::move(allocation)));
        execution->boots = std::move(boots);
        execution->managed_endpoints = std::move(endpoints);
        execution->members = std::move(members);
        execution->groups = std::move(groups);
        execution->retained_workloads = std::move(workloads);
        for (const auto& member : execution->members) {
            execution->retained_programs.push_back(member.program.lock());
        }
        for (const auto& [chip, workers] : execution->footprint) {
            execution->attached_chips.insert(chip);
            execution->retirement_chips.insert(chip);
        }
        if (mesh) {
            execution->mesh_owner = mesh->shared_from_this();
            execution->subdevice_manager = mesh->get_active_sub_device_manager_id();
        }
        for (size_t index = 0; index != execution->members.size(); ++index) {
            for (const auto& [chip, workers] : execution->members[index].cores) {
                execution->leaves.emplace(LeafKey{index, chip}, Leaf{State::Reserved, workers});
            }
        }
        // Reserve the lifetime registry before publishing any Program binding.
        auto admission = execution->owner.lock();
        auto guard = admission->lock();
        TT_FATAL(
            admission->validate_managed_locked(guard, *execution->token, execution->boots, error),
            "Managed invocation changed before publication: {}",
            error);
        for (const auto& member : execution->members) {
            auto program = member.program.lock();
            if (auto previous = program->managed_niu_invocation_.lock()) {
                TT_FATAL(
                    previous->state == State::Withdrawn && previous->resources_retired,
                    "Managed Program was rebound or retains its previous Device generation");
            }
        }
        for (const auto& workload : execution->retained_workloads) {
            if (auto previous = workload->managed_niu_invocation_.lock()) {
                TT_FATAL(
                    previous->state == State::Withdrawn && previous->resources_retired,
                    "Managed workload was rebound or retains its previous Device generation");
            }
        }
        for (auto chip : execution->attached_chips) {
            admission->retirements_.try_emplace(chip, 0);
        }
        admission->executions_.reserve(admission->executions_.size() + 1);
        // The handle destructor needs these preallocated cancellation gates.
        // Construct it only after their allocation can no longer fail.
        auto result = std::unique_ptr<ManagedInvocation>(new ManagedInvocation(execution));
        admission->executions_.push_back(execution);
        // All remaining publication operations are nonthrowing.
        for (auto& member : execution->members) {
            auto program = member.program.lock();
            program->managed_niu_invocation_ = execution;
            program->managed_niu_bound_ = true;
        }
        for (const auto& workload : execution->retained_workloads) {
            workload->managed_niu_invocation_ = execution;
            workload->managed_niu_bound_ = true;
        }
        return result;
    } catch (...) {
        std::string ignored;
        context.niu_admission().cancel_unissued(rollback_token, ignored);
        throw;
    }
}

ProgramExecution::LeafKey ProgramExecution::leaf_key(Program& program, ChipId chip) const {
    for (size_t index = 0; index != members.size(); ++index) {
        if (members[index].program.lock().get() == &program.impl() && leaves.contains({index, chip})) {
            return {index, chip};
        }
    }
    TT_THROW("Managed invocation received an unsealed Program/device member");
}

ProgramExecution::GroupKey ProgramExecution::group_key(distributed::MeshWorkload& workload) const {
    TT_FATAL(workload.pimpl_, "Managed invocation received a moved-from workload");
    for (size_t index = 0; index != groups.size(); ++index) {
        if (groups[index].workload.lock() == workload.pimpl_) {
            return GroupKey{index};
        }
    }
    TT_THROW("Managed invocation received another workload owner");
}

void ProgramExecution::validate_group_locked(GroupKey key) const {
    TT_FATAL(
        managed && (state == State::Reserved || state == State::Issued) && !preparation.cancelled(),
        "Managed invocation is cancelled, retired or quarantined");
    // Group identity and one union reservation are implemented, but no real
    // SDK NIU-clean receiver yet certifies selector reuse between groups.
    // Reject before any compilation, configuration, binary load or GO. This
    // gate must be replaced by the actual LoadedFirmware capability/receiver,
    // never a caller claim or a per-group release/reacquire workaround.
    TT_FATAL(groups.size() == 1, "Managed ordered workload execution requires the actual SDK NIU-clean receiver");
    const auto& group = groups.at(key.index);
    TT_FATAL(
        group.state == GroupState::Reserved || group.state == GroupState::Issued,
        "Managed workload group was already stopped or retired");
    for (size_t index = 0; index != key.index; ++index) {
        TT_FATAL(groups[index].state == GroupState::Clean, "Managed workload group precedes actual NIU-clean reception");
    }
}

void ProgramExecution::validate_leaf_overlap_locked(LeafKey leaf) const {
    if (origin != Origin::Direct) {
        return;
    }
    const auto& workers = members[leaf.first].cores.at(leaf.second);
    for (const auto& [key, other] : leaves) {
        if (key == leaf || key.second != leaf.second ||
            (other.state != State::Issued && other.state != State::Stopped) ||
            groups[members[key.first].group.index].state == GroupState::Clean) {
            continue;
        }
        const auto& other_workers = members[key.first].cores.at(key.second);
        TT_FATAL(
            std::none_of(workers.begin(), workers.end(), [&](auto core) { return other_workers.contains(core); }),
            "Managed direct members cannot overlap before actual NIU-clean reception");
    }
}

void ProgramExecution::validate_member_locked(
    NiuAdmission& admission,
    const NiuAdmission::Guard& guard,
    Program& program,
    std::span<IDevice* const> devices,
    const Cores& cores) const {
    TT_FATAL(
        managed && (state == State::Reserved || state == State::Issued) && owner.lock().get() == &admission,
        "Managed invocation is retired or quarantined");
    TT_FATAL(
        program.impl().managed_niu_bound_ && program.impl().managed_niu_invocation_.lock().get() == this,
        "Managed Program invocation owner changed");
    for (auto* device : devices) {
        TT_FATAL(
            extract_context_id(device) == program.impl().get_context_id(),
            "Managed NIU invocation crosses contexts");
        TT_FATAL(device->is_initialized(), "Managed NIU invocation requires open Devices");
        const auto key = leaf_key(program, device->id());
        TT_FATAL(
            members[key.first].devices.at(device->id()) == device,
            "Managed invocation received another Device object for its frozen chip");
    }
    TT_FATAL(!cores.empty(), "Managed Program lost its frozen Worker placement");
    for (const auto& [chip, workers] : cores) {
        const auto key = leaf_key(program, chip);
        const auto& member = members[key.first];
        validate_group_locked(member.group);
        TT_FATAL(
            member.cores.at(chip) == workers && member.logical == program.impl().logical_cores() &&
                member.kernels.size() == program.impl().num_kernels(),
            "Managed Program placement or kernel inventory changed");
        for (const auto& [identity, kernel] : member.kernels) {
            const auto& kernels = program.impl().get_kernels(identity.first);
            const auto found = kernels.find(identity.second);
            TT_FATAL(
                found != kernels.end() && found->second == kernel.lock(), "Managed Program kernel identity changed");
        }
        validate_managed_traffic(member, boots, managed_endpoints, mesh_owner.lock().get());
        TT_FATAL(leaves.at(key).state == State::Reserved, "Managed Program/device member was already submitted");
        validate_leaf_overlap_locked(key);
    }
    std::string error;
    TT_FATAL(
        admission.validate_managed_locked(guard, *token, boots, error),
        "Managed NIU invocation is stale: {}", error);
}

std::shared_ptr<ProgramExecution> ProgramExecution::validate_workload(
    distributed::MeshDevice& mesh, distributed::MeshWorkload& workload) {
    TT_FATAL(workload.pimpl_, "Cannot prepare a moved-from workload");
    if (!workload.pimpl_->managed_niu_bound_) {
        for (const auto& [range, program] : workload.get_programs()) {
            TT_FATAL(!program.impl().managed_niu_bound_, "Managed Program moved to another workload owner");
        }
        return nullptr;
    }
    auto execution = workload.pimpl_->managed_niu_invocation_.lock();
    TT_FATAL(execution, "Managed workload invocation owner has expired");
    auto admission = execution->owner.lock();
    TT_FATAL(
        admission && admission == MetalContext::instance(mesh.impl().get_context_id()).niu_admission_owner().lock(),
        "Managed workload crosses contexts");
    struct CurrentMember {
        Program* program;
        std::vector<IDevice*> devices;
        Cores cores;
    };
    std::vector<CurrentMember> current_members;
    for (auto& [range, program] : workload.get_programs()) {
        const auto& devices = mesh.impl().get_local_devices(range);
        if (!devices.empty()) {
            // SOC/context lookup must stay outside the NIU admission lock.
            current_members.push_back({&program, devices, program_cores(program, devices)});
        }
    }
    auto guard = admission->lock();
    const auto key = execution->group_key(workload);
    execution->validate_group_locked(key);
    const auto& group = execution->groups.at(key.index);
    TT_FATAL(
        execution->mesh_owner.lock().get() == &mesh &&
            execution->subdevice_manager == mesh.get_active_sub_device_manager_id() &&
            group.placements.size() == workload.get_programs().size(),
        "Managed workload changed its mesh, manager or complete member set");
    size_t local_members = 0;
    for (const auto& [range, frozen_program] : group.placements) {
        auto found = workload.get_programs().find(range);
        TT_FATAL(
            found != workload.get_programs().end() && &found->second.impl() == frozen_program.lock().get(),
            "Managed workload changed its Program identity or mesh placement");
        auto& program = found->second;
        const auto current = std::find_if(current_members.begin(), current_members.end(), [&](const auto& member) {
            return member.program == &program;
        });
        if (current == current_members.end()) {
            continue;
        }
        const auto& devices = current->devices;
        ++local_members;
        TT_FATAL(
            program.impl().managed_niu_bound_ && program.impl().managed_niu_invocation_.lock() == execution,
            "Managed workload mixes invocation members");
        const auto member_key = execution->leaf_key(program, devices.front()->id());
        const auto& member = execution->members.at(member_key.first);
        const auto& cores = current->cores;
        TT_FATAL(
            member.group == key && member.cores == cores,
            "Managed workload changed its complete member placement");
        for (auto* device : devices) {
            TT_FATAL(
                device->is_initialized() && member.devices.at(device->id()) == device,
                "Managed workload requires its original open Device objects");
        }
        execution->validate_member_locked(*admission, guard, program, devices, cores);
    }
    TT_FATAL(local_members == group.members.size(), "Managed workload changed its complete local member set");
    if (execution->origin == Origin::Queue) {
        for (uint32_t cq = 0; cq != mesh.num_hw_cqs(); ++cq) {
            TT_FATAL(!mesh.mesh_command_queue(cq).trace_id(), "Managed NIU invocation does not support trace capture");
        }
    }
    std::string error;
    TT_FATAL(
        admission->validate_managed_locked(guard, *execution->token, execution->boots, error),
        "Managed workload binding changed: {}", error);
    return execution;
}

std::shared_ptr<ProgramExecution> ProgramExecution::reserve_dispatch(
    distributed::MeshDevice& mesh, std::span<const SubDeviceId> subdevices, distributed::MeshWorkload& workload) {
    auto execution = validate_workload(mesh, workload);
    if (!execution) {
        return reserve_dispatch(mesh, subdevices);
    }
    const auto cores = dispatch_cores(mesh, subdevices);
    auto admission = execution->owner.lock();
    TT_FATAL(admission, "Managed FD invocation context has closed");
    auto guard = admission->lock();
    const auto key = execution->group_key(workload);
    execution->validate_group_locked(key);
    const auto& group = execution->groups.at(key.index);
    TT_FATAL(
        execution->origin == Origin::Queue && group.cores == cores &&
            std::set<SubDeviceId>(group.subdevices.begin(), group.subdevices.end()) ==
                std::set<SubDeviceId>(subdevices.begin(), subdevices.end()),
        "Managed FD invocation changed its GO footprint");
    return execution;
}

const NiuAdmission::Allocation& ProgramExecution::ManagedInvocation::allocation() const { return *execution->managed; }
ProgramExecution::ManagedInvocation::~ManagedInvocation() { execution->abandon_managed(); }

void ProgramExecution::abandon_managed(bool preparation_owner_only) noexcept {
    auto admission = owner.lock();
    NiuAdmission::Retirement retired;
    if (!admission || !token) {
        return;
    }
    auto guard = admission->lock();
    std::string ignored;
    if (preparation_owner_only && preparation.occupied() && !preparation.owned()) {
        return;
    }
    if (state == State::Reserved) {
        preparation.cancel();
        // Invalidate immediately, but the reservation still excludes new owners
        // until the last nested preparation has left all device writes.
        if (preparation.occupied()) {
            return;
        }
        admission->cancel_unissued_locked(guard, *token, ignored);
        state = State::Withdrawn;
        attached_chips.clear();
        admission->retire_managed_locked(*this, retired);
    } else if (state == State::Issued) {
        admission->quarantine_locked(guard, *token, ignored);
        state = State::Quarantined;
    }
}

ProgramExecution::~ProgramExecution() {
    if (state != State::Reserved) {
        return;
    }
    auto admission = owner.lock();
    if (!admission || !token) {
        return;
    }
    auto guard = admission->lock();
    if (state != State::Reserved) {
        return;
    }
    std::string ignored;
    admission->cancel_unissued_locked(guard, *token, ignored);
}

ProgramExecution::Submission::Submission(
    ProgramExecution& execution,
    internal::ServiceCoreManagerImpl* services,
    std::optional<LeafKey> leaf,
    std::optional<GroupKey> group) :
    execution(execution) {
    auto admission = execution.owner.lock();
    TT_FATAL(admission, "Program NIU admission context has closed");
    auto& owner = *admission;
    if (!execution.token || execution.simulated) {
        return;
    }
    auto guard = owner.lock();
    const bool first_issue = execution.state == State::Reserved;
    if (execution.managed) {
        TT_FATAL(group, "Managed invocation requires its actual workload group");
        execution.validate_group_locked(*group);
        for (auto member : execution.groups.at(group->index).members)
            validate_managed_traffic(execution.members.at(member), execution.boots, execution.managed_endpoints,
                                     execution.mesh_owner.lock().get());
        TT_FATAL(execution.preparation.owned(), "Managed issue requires its actual host preparation owner");
        if (execution.origin == Origin::Direct) {
            TT_FATAL(
                leaf && execution.leaves.at(*leaf).state == State::Reserved &&
                    execution.members.at(leaf->first).group == *group,
                "Managed invocation requires one previously unsubmitted actual member");
            execution.validate_leaf_overlap_locked(*leaf);
        } else {
            TT_FATAL(
                !leaf && execution.groups.at(group->index).state == GroupState::Reserved,
                "Managed FD invocation can only submit each complete workload group once");
        }
    }
    std::vector<internal::ServiceCoreManagerImpl::CoreState*> service_claims;
    if (services) {
        service_claims = services->prepare_launch_locked(guard, execution);
    }
    if (first_issue && !execution.managed) {
        owner.executions_.reserve(owner.executions_.size() + 1);
    }
    auto self = execution.shared_from_this();
    std::string error;
    auto* record = owner.find_locked(*execution.token, error);
    TT_FATAL(record, "Program NIU issue failed: {}", error);
    NiuAdmission::Boots captured;
    for (const auto& [chip, binding] : record->bindings) {
        captured.emplace(chip, binding.boot.firmware);
    }
    TT_FATAL(
        !execution.replay_boots || *execution.replay_boots == captured,
        "Program NIU replay belongs to another firmware boot");
    TT_FATAL(
        !execution.managed || execution.boots == captured, "Managed invocation firmware boot changed before issue");
    if (first_issue) {
        TT_FATAL(owner.mark_issued_locked(guard, *execution.token, error), "Program NIU issue failed: {}", error);
    } else {
        TT_FATAL(
            execution.managed && record->state == NiuAdmission::State::Issued &&
                owner.validate_managed_locked(guard, *execution.token, execution.boots, error),
            "Managed invocation binding changed during submission: {}",
            error);
    }
    execution.boots = std::move(captured);
    execution.state = State::Issued;
    if (execution.managed) {
        execution.groups.at(group->index).state = GroupState::Issued;
        if (leaf) {
            execution.leaves.at(*leaf).state = State::Issued;
        } else {
            for (auto member : execution.groups.at(group->index).members) {
                for (const auto& [chip, workers] : execution.members[member].cores) {
                    execution.leaves.at({member, chip}).state = State::Issued;
                }
            }
        }
    }
    if (services) {
        services->commit_launch_locked(service_claims);
    }
    if (first_issue && !execution.managed) {
        owner.executions_.push_back(std::move(self));
    }
}

ProgramExecution::Submission::~Submission() {
    if (!committed) {
        execution.quarantine();
    }
}

ProgramExecution::Submission ProgramExecution::issue(internal::ServiceCoreManagerImpl* services) {
    return Submission(*this, services);
}

ProgramExecution::Submission ProgramExecution::issue(GroupKey group) {
    return Submission(*this, nullptr, std::nullopt, group);
}

ProgramExecution::Submission ProgramExecution::issue(
    Program& program, IDevice& device, internal::ServiceCoreManagerImpl* services) {
    const auto leaf = managed ? std::optional(leaf_key(program, device.id())) : std::nullopt;
    return Submission(*this, services, leaf, leaf ? std::optional(members[leaf->first].group) : std::nullopt);
}

std::shared_ptr<const ProgramExecution::DispatchFootprint> ProgramExecution::snapshot(
    std::span<const SubDeviceId> subdevices) const {
    TT_FATAL(!managed, "Managed NIU invocation does not support trace capture or replay");
    auto admission = owner.lock();
    TT_FATAL(admission, "Program NIU trace context has closed");
    auto guard = admission->lock();
    auto result = std::shared_ptr<DispatchFootprint>(new DispatchFootprint);
    result->owner = owner;
    result->cores = pending;
    result->subdevices.assign(subdevices.begin(), subdevices.end());
    for (const auto& [chip, cores] : pending) {
        result->epochs.emplace(chip, admission->device_epochs_[chip]);
    }
    if (token) {
        std::string error;
        TT_FATAL(admission->validate_ordinary_locked(guard, *token, error), "Program NIU capture failed: {}", error);
        auto* record = admission->find_locked(*token, error);
        for (const auto& [chip, binding] : record->bindings) {
            result->boots.emplace(chip, binding.boot.firmware);
        }
    }
    return result;
}

std::shared_ptr<ProgramExecution> ProgramExecution::reserve_replay(
    distributed::MeshDevice& mesh,
    std::span<const std::shared_ptr<const DispatchFootprint>> footprints,
    std::span<const SubDeviceId> subdevices) {
    TT_FATAL(!footprints.empty(), "Program NIU replay requires its captured physical dispatch footprint");
    auto result = reserve_dispatch(mesh, subdevices);
    auto admission = result->owner.lock();
    TT_FATAL(admission, "Program NIU replay context has closed");
    auto guard = admission->lock();
    Cores captured;
    NiuAdmission::Boots boots;
    std::set<SubDeviceId> captured_subdevices;
    for (const auto& footprint : footprints) {
        TT_FATAL(footprint && footprint->owner.lock() == admission, "Program NIU trace belongs to another context");
        captured_subdevices.insert(footprint->subdevices.begin(), footprint->subdevices.end());
        for (const auto& [chip, cores] : footprint->cores) {
            TT_FATAL(
                footprint->epochs.at(chip) == admission->device_epochs_[chip],
                "Program NIU trace belongs to a closed Device generation");
            captured[chip].insert(cores.begin(), cores.end());
        }
        for (const auto& [chip, boot] : footprint->boots) {
            auto [it, inserted] = boots.emplace(chip, boot);
            TT_FATAL(inserted || it->second == boot, "Program NIU trace mixes firmware boots");
        }
    }
    TT_FATAL(
        captured_subdevices == std::set<SubDeviceId>(subdevices.begin(), subdevices.end()) &&
            captured == result->pending,
        "Program NIU trace dispatch recipients changed after capture");
    if (result->token) {
        std::string error;
        auto* record = admission->find_locked(*result->token, error);
        TT_FATAL(
            record && admission->validate_boots_locked(*record, boots, error), "Program NIU replay failed: {}", error);
    }
    result->replay_boots = std::move(boots);
    return result;
}

void ProgramExecution::quarantine() noexcept {
    auto admission = owner.lock();
    if (!admission || !token) {
        return;
    }
    auto guard = admission->lock();
    if (state != State::Issued) {
        return;
    }
    std::string ignored;
    admission->quarantine_locked(guard, *token, ignored);
    state = State::Quarantined;
}

void ProgramExecution::stop() {
    auto admission = this->owner.lock();
    TT_FATAL(admission, "Program NIU completion context has closed");
    auto& owner = *admission;
    auto guard = owner.lock();
    if (!token || state != State::Issued) {
        return;
    }
    TT_FATAL(!managed, "Managed completion must name its actual workload group or Program/device member");
    std::string error;
    TT_FATAL(owner.record_stop_locked(*token, boots, error), "Program NIU stop failed: {}", error);
    state = State::Stopped;
    std::erase_if(owner.executions_, [](const auto& execution) {
        return !execution->managed && execution->state == State::Stopped;
    });
}

void ProgramExecution::stop(GroupKey key) {
    auto admission = owner.lock();
    TT_FATAL(admission, "Managed queue completion context has closed");
    auto guard = admission->lock();
    if (!token || state != State::Issued) {
        return;
    }
    TT_FATAL(managed && origin == Origin::Queue, "Managed group completion requires its original queue invocation");
    auto& group = groups.at(key.index);
    TT_FATAL(group.state == GroupState::Issued, "Managed queue completion has no issued workload group");
    for (auto member : group.members) {
        for (const auto& [chip, workers] : members[member].cores) {
            auto& leaf = leaves.at({member, chip});
            TT_FATAL(leaf.state == State::Issued, "Managed queue completion omitted an issued member");
            leaf.state = State::Stopped;
            leaf.pending.clear();
        }
    }
    receive_managed_stop_locked(*admission);
}

void ProgramExecution::receive_managed_stop_locked(NiuAdmission& admission) {
    for (auto& group : groups) {
        if (group.state != GroupState::Issued) {
            continue;
        }
        const bool stopped = std::all_of(group.members.begin(), group.members.end(), [&](auto member) {
            const auto& cores = members[member].cores;
            return std::all_of(cores.begin(), cores.end(), [&](const auto& entry) {
                return leaves.at({member, entry.first}).state == State::Stopped;
            });
        });
        if (stopped) {
            group.state = GroupState::Stopped;
        }
    }
    if (!std::all_of(groups.begin(), groups.end(), [](const auto& group) {
            return group.state == GroupState::Stopped || group.state == GroupState::Clean;
        })) {
        return;
    }
    std::string error;
    TT_FATAL(admission.record_stop_locked(*token, boots, error), "Managed NIU stop failed: {}", error);
    state = State::Stopped;
    // The same registry retains the stopped managed token until an actual
    // clean receiver or boot withdrawal. Stopped is never permission to reuse.
}

void ProgramExecution::stop(Program& program, IDevice& device) {
    if (!managed) {
        stop();
        return;
    }
    auto admission = owner.lock();
    TT_FATAL(admission, "Managed completion context has closed");
    auto guard = admission->lock();
    TT_FATAL(state == State::Issued && origin == Origin::Direct, "Managed direct completion has no issued invocation");
    auto& leaf = leaves.at(leaf_key(program, device.id()));
    TT_FATAL(leaf.state == State::Issued, "Managed direct completion has no issued member");
    leaf.state = State::Stopped;
    leaf.pending.clear();
    receive_managed_stop_locked(*admission);
}

ProgramExecution::Completion ProgramExecution::observe_cores(
    MetalContext& context, ChipId chip, std::set<CoreCoord> physical) {
    Completion result;
    result.owner = context.niu_admission_owner();
    result.chip = chip;
    result.physical = std::move(physical);
    auto target = context.get_cluster().get_target_device_type();
    if (target == tt::TargetDevice::Mock || target == tt::TargetDevice::Emule) {
        return result;
    }
    auto& owner = context.niu_admission();
    auto guard = owner.lock();
    for (auto& execution : owner.executions_) {
        if (execution->origin != Origin::Direct || execution->state != State::Issued) {
            continue;
        }
        if (execution->managed) {
            for (const auto& [key, leaf] : execution->leaves) {
                if (key.second != chip || leaf.state != State::Issued) {
                    continue;
                }
                if (std::any_of(result.physical.begin(), result.physical.end(), [&](auto core) {
                        return leaf.pending.contains(core);
                    })) {
                    result.managed_leaves.emplace_back(execution, key);
                }
            }
            continue;
        }
        auto found = execution->pending.find(chip);
        if (found == execution->pending.end()) {
            continue;
        }
        if (std::none_of(result.physical.begin(), result.physical.end(), [&](auto core) {
                return found->second.contains(core);
            })) {
            continue;
        }
        result.executions.push_back(execution);
    }
    return result;
}

void ProgramExecution::Completion::receive() {
    auto admission = this->owner.lock();
    TT_FATAL(admission, "Program NIU completion context has closed");
    auto& owner = *admission;
    auto guard = owner.lock();
    for (auto& [execution, key] : managed_leaves) {
        if (execution->state != State::Issued) {
            continue;
        }
        auto& leaf = execution->leaves.at(key);
        if (leaf.state != State::Issued) {
            continue;
        }
        for (auto core : physical) {
            leaf.pending.erase(core);
        }
        if (leaf.pending.empty()) {
            leaf.state = State::Stopped;
        }
        execution->receive_managed_stop_locked(owner);
    }
    for (auto& execution : executions) {
        if (execution->state != State::Issued) {
            continue;
        }
        auto found = execution->pending.find(chip);
        if (found == execution->pending.end()) {
            continue;
        }
        for (auto core : physical) {
            found->second.erase(core);
        }
        if (found->second.empty()) {
            execution->pending.erase(found);
        }
        if (!execution->pending.empty()) {
            continue;
        }
        std::string error;
        TT_FATAL(
            owner.record_stop_locked(*execution->token, execution->boots, error), "Program NIU stop failed: {}", error);
        execution->state = State::Stopped;
    }
    std::erase_if(owner.executions_, [](const auto& execution) {
        return !execution->managed && execution->state == State::Stopped;
    });
    received = true;
}

ProgramExecution::Completion::~Completion() {
    if (!received) {
        for (auto& [execution, key] : managed_leaves) {
            execution->quarantine();
        }
        for (auto& execution : executions) {
            execution->quarantine();
        }
    }
}

ProgramExecution::Completion ProgramExecution::observe_logical_cores(
    MetalContext& context, ChipId chip, const std::vector<std::vector<CoreCoord>>& logical) {
    std::set<CoreCoord> physical;
    const auto& soc = context.get_cluster().get_soc_desc(chip);
    for (size_t index = 0; index != logical.size(); ++index) {
        if (context.hal().get_core_type(index) != CoreType::WORKER) {
            continue;
        }
        for (auto core : logical[index]) {
            physical.insert(soc.get_physical_tensix_core_from_logical(core));
        }
    }
    return observe_cores(context, chip, std::move(physical));
}

ProgramExecution::Completion ProgramExecution::observe_program(IDevice& device, Program& program) {
    auto& context = MetalContext::instance(extract_context_id(&device));
    return observe_logical_cores(context, device.id(), program.impl().logical_cores());
}
}  // namespace tt::tt_metal::detail
