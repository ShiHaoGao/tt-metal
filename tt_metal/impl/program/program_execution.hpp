// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <atomic>
#include <memory>
#include <set>
#include <span>
#include <vector>
#include <thread>
#include <limits>

#include "impl/context/niu_admission.hpp"
#include "hostdev/native_kernel_contract.h"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {
class Device;
class DispatchKernelInitializer;
class MetalContext;
class Program;
class IDevice;
class Kernel;
namespace experimental {
void DispatchCompiledProgramToDevice(IDevice*, Program&);
}
namespace distributed {
class MeshDevice;
class MeshWorkload;
class MeshWorkloadImpl;
class FDMeshCommandQueue;
class SDMeshCommandQueue;
}  // namespace distributed
namespace detail {
class ProgramImpl;
void LaunchProgram(IDevice*, Program&, bool, bool);
void WaitProgramDone(IDevice*, Program&, bool);

// Used under the admission mutex. This is host preparation ownership only,
// never device issue/completion evidence.
class ProgramPreparation final {
public:
    bool acquire() noexcept {
        if (cancelled_) {
            return false;
        }
        const auto caller = std::this_thread::get_id();
        if (depth_ && owner_ != caller) {
            return false;
        }
        if (depth_ == std::numeric_limits<size_t>::max()) {
            return false;
        }
        owner_ = caller;
        ++depth_;
        return true;
    }
    bool owned() const noexcept { return depth_ && owner_ == std::this_thread::get_id(); }
    bool occupied() const noexcept { return depth_ != 0; }
    void cancel() noexcept { cancelled_ = true; }
    bool cancelled() const noexcept { return cancelled_; }
    bool ready_to_cancel() const noexcept { return cancelled_ && !occupied(); }
    void release() noexcept {
        if (owned() && --depth_ == 0) {
            owner_ = {};
        }
    }

private:
    std::thread::id owner_;
    size_t depth_ = 0;
    bool cancelled_ = false;
};

// An execution reservation, never a firmware or NIU-clean receipt. Only the
// actual SDK submission/wait receivers below can advance its state.
class ProgramExecution final : public std::enable_shared_from_this<ProgramExecution> {
public:
    class Preparation final {
    public:
        ~Preparation();
        Preparation(Preparation&& other) noexcept;
        Preparation(const Preparation&) = delete;
        std::shared_ptr<ProgramExecution> execution() const { return execution_; }

    private:
        friend class ProgramExecution;
        explicit Preparation(std::shared_ptr<ProgramExecution> execution, bool borrowed) :
            execution_(std::move(execution)), borrowed_(borrowed) {}
        std::shared_ptr<ProgramExecution> execution_;
        bool borrowed_;
    };
    static Preparation reserve_direct(Program& program, IDevice& device);
    static Preparation reserve_workload(distributed::MeshDevice& mesh, distributed::MeshWorkload& workload);
    class FailureScope final {
    public:
        ~FailureScope();
        FailureScope(FailureScope&&) noexcept = default;
        FailureScope(const FailureScope&) = delete;

    private:
        friend class ProgramExecution;
        explicit FailureScope(std::shared_ptr<ProgramExecution> execution);
        std::shared_ptr<ProgramExecution> execution;
        int exceptions;
    };
    FailureScope watch_failure() { return FailureScope(shared_from_this()); }
    class ManagedInvocation final {
    public:
        ~ManagedInvocation();
        ManagedInvocation(const ManagedInvocation&) = delete;
        ManagedInvocation& operator=(const ManagedInvocation&) = delete;
        const NiuAdmission::Allocation& allocation() const;

    private:
        friend class ProgramExecution;
        explicit ManagedInvocation(std::shared_ptr<ProgramExecution> execution) : execution(std::move(execution)) {}
        std::shared_ptr<ProgramExecution> execution;
    };
    static std::unique_ptr<ManagedInvocation> acquire_managed(Program& program, std::span<IDevice* const> devices);
    static std::unique_ptr<ManagedInvocation> acquire_managed(
        distributed::MeshDevice& mesh, std::span<distributed::MeshWorkload* const> workloads);
    class DispatchFootprint final {
    private:
        friend class ProgramExecution;
        DispatchFootprint() = default;
        std::weak_ptr<NiuAdmission> owner;
        std::map<ChipId, std::set<CoreCoord>> cores;
        std::map<ChipId, uint64_t> epochs;
        NiuAdmission::Boots boots;
        std::vector<SubDeviceId> subdevices;
    };
    static std::shared_ptr<ProgramExecution> reserve(Program& program, std::span<IDevice* const> devices);
    static std::shared_ptr<ProgramExecution> reserve_dispatch(
        distributed::MeshDevice& mesh, std::span<const SubDeviceId> sub_devices);
    static std::shared_ptr<ProgramExecution> reserve_dispatch(
        distributed::MeshDevice& mesh, std::span<const SubDeviceId> sub_devices, distributed::MeshWorkload& workload);
    static std::shared_ptr<ProgramExecution> validate_workload(
        distributed::MeshDevice& mesh, distributed::MeshWorkload& workload);
    ~ProgramExecution();
    ProgramExecution(const ProgramExecution&) = delete;
    ProgramExecution& operator=(const ProgramExecution&) = delete;

private:
    friend class ::tt::tt_metal::NiuAdmission;
    friend class ::tt::tt_metal::Device;
    friend class ::tt::tt_metal::DispatchKernelInitializer;
    friend class ::tt::tt_metal::internal::ServiceCoreManagerImpl;
    friend class ::tt::tt_metal::distributed::FDMeshCommandQueue;
    friend class ::tt::tt_metal::distributed::SDMeshCommandQueue;
    friend void LaunchProgram(IDevice*, Program&, bool, bool);
    friend void WaitProgramDone(IDevice*, Program&, bool);
    friend void ::tt::tt_metal::experimental::DispatchCompiledProgramToDevice(IDevice*, Program&);

    enum class Origin { Direct, Queue };
    enum class State { Reserved, Issued, Stopped, Quarantined, Withdrawn };
    enum class GroupState { Reserved, Issued, Stopped, Clean };
    using Cores = std::map<ChipId, std::set<CoreCoord>>;
    using Endpoints = std::map<ChipId, std::vector<tt_native_noc_endpoint>>;
    using LeafKey = std::pair<size_t, ChipId>;
    struct GroupKey {
        size_t index;
        bool operator==(const GroupKey&) const = default;
    };
    struct Member {
        GroupKey group{0};
        std::weak_ptr<ProgramImpl> program;
        Cores cores;
        std::map<ChipId, IDevice*> devices;
        std::vector<std::vector<CoreCoord>> logical;
        std::map<std::pair<size_t, uint64_t>, std::weak_ptr<Kernel>> kernels;
    };
    struct Leaf {
        State state = State::Reserved;
        std::set<CoreCoord> pending;
    };
    struct Group {
        std::weak_ptr<distributed::MeshWorkloadImpl> workload;
        std::vector<std::pair<distributed::MeshCoordinateRange, std::weak_ptr<ProgramImpl>>> placements;
        std::vector<size_t> members;
        Cores cores;
        std::vector<SubDeviceId> subdevices;
        GroupState state = GroupState::Reserved;
    };
    ProgramExecution(
        MetalContext& context,
        Origin origin,
        Cores cores,
        std::optional<NiuAdmission::Allocation> managed = std::nullopt);
    static Cores dispatch_cores(distributed::MeshDevice& mesh, std::span<const SubDeviceId> subdevices);
    static Member freeze_member(Program& program, std::span<IDevice* const> devices);
    static Endpoints freeze_managed_endpoints(MetalContext& context, const Cores& cores);
    static void validate_managed_traffic(
        const Member& member, const NiuAdmission::Boots& boots, const Endpoints& endpoints,
        distributed::MeshDevice* mesh);
    static std::unique_ptr<ManagedInvocation> acquire_managed(
        MetalContext& context,
        Origin origin,
        Cores cores,
        std::vector<Member> members,
        std::vector<Group> groups,
        distributed::MeshDevice* mesh = nullptr,
        std::vector<std::shared_ptr<distributed::MeshWorkloadImpl>> workloads = {});
    void validate_member_locked(
        NiuAdmission& admission,
        const NiuAdmission::Guard& guard,
        Program& program,
        std::span<IDevice* const> devices,
        const Cores& cores) const;
    void validate_group_locked(GroupKey group) const;
    GroupKey group_key(distributed::MeshWorkload& workload) const;
    LeafKey leaf_key(Program& program, ChipId chip) const;
    void validate_leaf_overlap_locked(LeafKey leaf) const;
    void abandon_managed(bool preparation_owner_only = false) noexcept;
    void receive_managed_stop_locked(NiuAdmission& admission);
    class Submission final {
    public:
        ~Submission();
        Submission(const Submission&) = delete;
        Submission& operator=(const Submission&) = delete;
        Submission(Submission&& other) noexcept : execution(other.execution), committed(other.committed) {
            other.committed = true;
        }
        void commit() noexcept { committed = true; }

    private:
        friend class ProgramExecution;
        Submission(
            ProgramExecution& execution,
            internal::ServiceCoreManagerImpl* services,
            std::optional<LeafKey> leaf = std::nullopt,
            std::optional<GroupKey> group = std::nullopt);
        ProgramExecution& execution;
        bool committed = false;
    };
    Submission issue(internal::ServiceCoreManagerImpl* services = nullptr);
    Submission issue(GroupKey group);
    Submission issue(Program& program, IDevice& device, internal::ServiceCoreManagerImpl* services = nullptr);
    std::shared_ptr<const DispatchFootprint> snapshot(std::span<const SubDeviceId> subdevices) const;
    static std::shared_ptr<ProgramExecution> reserve_replay(
        distributed::MeshDevice& mesh,
        std::span<const std::shared_ptr<const DispatchFootprint>> footprints,
        std::span<const SubDeviceId> subdevices);
    void stop();
    void stop(GroupKey group);
    void stop(Program& program, IDevice& device);
    void quarantine() noexcept;
    class Completion final {
    public:
        Completion(Completion&&) = default;
        Completion(const Completion&) = delete;
        ~Completion();
        void receive();

    private:
        friend class ProgramExecution;
        Completion() = default;
        bool received = false;
        std::weak_ptr<NiuAdmission> owner;
        ChipId chip;
        std::set<CoreCoord> physical;
        std::vector<std::shared_ptr<ProgramExecution>> executions;
        std::vector<std::pair<std::shared_ptr<ProgramExecution>, LeafKey>> managed_leaves;
    };
    static Completion observe_cores(MetalContext& context, ChipId chip, std::set<CoreCoord> physical);
    static Completion observe_program(IDevice& device, Program& program);
    static Completion observe_logical_cores(
        MetalContext& context, ChipId chip, const std::vector<std::vector<CoreCoord>>& logical);
    static Cores program_cores(Program& program, std::span<IDevice* const> devices);
    std::weak_ptr<NiuAdmission> owner;
    bool simulated;
    Origin origin;
    const Cores footprint;
    Cores pending;
    std::optional<NiuAdmission::Token> token;
    NiuAdmission::Boots boots;
    Endpoints managed_endpoints;
    std::optional<NiuAdmission::Boots> replay_boots;
    std::atomic<State> state{State::Reserved};
    std::optional<NiuAdmission::Allocation> managed;
    std::vector<Member> members;
    std::vector<Group> groups;
    std::vector<std::shared_ptr<ProgramImpl>> retained_programs;
    std::vector<std::shared_ptr<distributed::MeshWorkloadImpl>> retained_workloads;
    std::set<ChipId> attached_chips;
    std::set<ChipId> retirement_chips;
    ProgramPreparation preparation;
    bool resources_retired = false;
    std::map<LeafKey, Leaf> leaves;
    std::weak_ptr<distributed::MeshDevice> mesh_owner;
    std::optional<SubDeviceManagerId> subdevice_manager;
};
}  // namespace detail
}  // namespace tt::tt_metal
