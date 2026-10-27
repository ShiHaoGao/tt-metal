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
#include "impl/context/niu_handoff_receiver.hpp"
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
class Buffer;
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
    // Stable mailbox bytes matched against this invocation's frozen request
    // and exact stopped footprint. These are not bootstrap or clean receipts.
    struct HandoffObservation {
        ChipId chip;
        CoreCoord physical_worker;
        NiuHandoffObservation mailbox;
    };
    class ManagedInvocation final {
        struct ItemWitnessStorage;
    public:
        class SelectorLease final {
        public:
            SelectorLease(SelectorLease&&) noexcept = default;
            SelectorLease& operator=(SelectorLease&&) noexcept = default;
            SelectorLease(const SelectorLease&) = delete;
            SelectorLease& operator=(const SelectorLease&) = delete;
            detail::NiuKey niu() const { return selector_.niu(); }
            uint8_t selector() const { return selector_.selector(); }
            // Handle population only; a retained handle is not a new live-boot
            // admission, issue epoch, or completion capability.
            bool valid() const noexcept { return owner_ && selector_.valid(); }
            uint64_t boot_cookie() const { return boot_cookie_; }
            uint64_t invocation_id() const { return invocation_id_; }

        private:
            friend class ManagedInvocation;
            SelectorLease(
                std::shared_ptr<ProgramExecution> owner,
                NiuAdmission::SelectorLease selector,
                uint64_t boot_cookie,
                uint64_t invocation_id) :
                owner_(std::move(owner)),
                selector_(std::move(selector)),
                boot_cookie_(boot_cookie),
                invocation_id_(invocation_id) {}
            // Retain the original managed execution, its allocation, and its
            // original boot set. These correlation values never grant issue,
            // completion, visibility, or clean-state authority.
            std::shared_ptr<ProgramExecution> owner_;
            NiuAdmission::SelectorLease selector_;
            uint64_t boot_cookie_;
            uint64_t invocation_id_;
        };
        // This is the narrow handoff that the SDK can currently prove for a
        // managed invocation.  It is projected directly from the immutable
        // NiuAdmission allocation and the original firmware boot.  The
        // correlation values identify this invocation/boot only; they are not
        // NIU or NoC transaction IDs.
        struct CompletionBinding {
            detail::NiuKey niu;
            detail::NiuSelectorMask selectors;
            uint64_t boot_cookie = 0;
            uint64_t invocation_id = 0;
        };
        // An aggregate mailbox receipt correlated with one allocation row.
        // The mailbox is shared by the two NoCs on a Worker and therefore
        // does not establish per-item payload completion or a TRID lease.
        struct CompletionObservation {
            CompletionBinding binding;
            HandoffObservation handoff;
        };
        // Minted only after all original issuers stopped and their retained
        // request-correlated hardware selector counters are both zero. This
        // is a terminal observation, never online item publication or payload
        // visibility. A retained value cannot authorize later selector reuse.
        class SelectorObservation final {
        public:
            detail::NiuKey niu() const { return niu_; }
            uint8_t selector() const { return selector_; }
            uint64_t boot_cookie() const { return boot_cookie_; }
            uint64_t invocation_id() const { return invocation_id_; }

        private:
            friend class ManagedInvocation;
            SelectorObservation(
                std::shared_ptr<ProgramExecution> owner, detail::NiuKey niu, uint8_t selector,
                uint64_t boot_cookie, uint64_t invocation_id) :
                owner_(std::move(owner)), niu_(niu), selector_(selector),
                boot_cookie_(boot_cookie), invocation_id_(invocation_id) {}
            std::shared_ptr<ProgramExecution> owner_;
            detail::NiuKey niu_;
            uint8_t selector_;
            uint64_t boot_cookie_;
            uint64_t invocation_id_;
        };
        enum class ItemCompletionFact : uint8_t {
            ReadDestinationLocal = TT_NIU_ITEM_READ_DESTINATION_LOCAL,
            NonPostedWriteAcknowledged = TT_NIU_ITEM_NONPOSTED_WRITE_ACKNOWLEDGED,
        };
        struct ItemWitnessRequirement {
            std::shared_ptr<Kernel> kernel;
            CoreCoord logical_core;
            uint64_t item;
            uint64_t generation;
            uint64_t occurrence;
            uint32_t leaf;
            ItemCompletionFact completion;
        };
        class ItemWitnessLease final {
        public:
            ItemWitnessLease(ItemWitnessLease&&) noexcept = default;
            ItemWitnessLease& operator=(ItemWitnessLease&&) noexcept = default;
            ItemWitnessLease(const ItemWitnessLease&) = delete;
            ItemWitnessLease& operator=(const ItemWitnessLease&) = delete;
            uint32_t address() const;
            const tt_niu_item_witness_request& request() const;

        private:
            friend class ManagedInvocation;
            ItemWitnessLease(std::shared_ptr<ProgramExecution> owner, std::shared_ptr<const ItemWitnessStorage> storage) :
                owner_(std::move(owner)), storage_(std::move(storage)) {}
            std::shared_ptr<ProgramExecution> owner_;
            std::shared_ptr<const ItemWitnessStorage> storage_;
        };
        class ItemWitnessObservation final {
        public:
            const tt_niu_item_witness_request& request() const;
            detail::NiuKey niu() const;
            ItemCompletionFact completion() const;

        private:
            friend class ManagedInvocation;
            ItemWitnessObservation(
                std::shared_ptr<ProgramExecution> owner, std::shared_ptr<const ItemWitnessStorage> storage) :
                owner_(std::move(owner)), storage_(std::move(storage)) {}
            std::shared_ptr<ProgramExecution> owner_;
            std::shared_ptr<const ItemWitnessStorage> storage_;
        };
        ~ManagedInvocation();
        ManagedInvocation(const ManagedInvocation&) = delete;
        ManagedInvocation& operator=(const ManagedInvocation&) = delete;
        const NiuAdmission::Allocation& allocation() const;
        std::vector<CompletionBinding> completion_bindings() const;
        // Selects one real hardware TRID from this invocation's admitted NIU
        // allocation. No item ordinal, group id, or mailbox observation is
        // accepted as a selector source.
        std::optional<SelectorLease> acquire_selector(const NiuKey& niu) const;
        std::vector<HandoffObservation> handoff_observations() const;
        std::vector<CompletionObservation> completion_observations() const;
        std::optional<SelectorObservation> observe_selector(const SelectorLease& selector) const;
        // Revalidate every admitted selector against the original stopped
        // invocation. Nonzero/uncertain counters preserve the full exclusion.
        // No caller-supplied counters, receipt, bool or generation can retire.
        bool retire_selectors() const;
        // The SDK allocates and initializes one real Worker L1 slot before
        // issue, binding the exact frozen native kernel and original selector.
        // No caller-supplied address or copied receipt can produce a lease.
        std::optional<ItemWitnessLease> acquire_item_witness(
            const SelectorLease&, const ItemWitnessRequirement&) const;
        // Online observation reads the original allocation and then revalidates
        // the original boot/token/Device epoch. It reports physical completion
        // only, never consumer visibility, release or rolling-slot reuse.
        std::optional<ItemWitnessObservation> observe_item(const ItemWitnessLease&) const;

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
    // Actual stop receivers call this only after the complete managed
    // invocation has stopped. It never releases NIUs or grants clean state.
    void observe_handoff();
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
    enum class State { Reserved, Issued, Stopped, SelectorsClean, Quarantined, Withdrawn };
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
    struct HandoffSnapshot {
        NiuAdmission::Token token;
        NiuAdmission::Boots boots;
        Cores footprint;
        std::map<ChipId, uint64_t> epochs;
        std::map<ChipId, std::map<CoreCoord, NiuHandoffExpectedExecution>> expected;
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
    void validate_handoff_locked(
        NiuAdmission& admission, const NiuAdmission::Guard& guard, const HandoffSnapshot& snapshot) const;
    bool validate_item_witness_locked(
        NiuAdmission&, const NiuAdmission::Guard&, const ManagedInvocation::ItemWitnessStorage&, bool issued) const;
    void validate_request_locked(
        NiuAdmission& admission, const NiuAdmission::Guard& guard, const HandoffSnapshot& snapshot) const;
    void prepare_handoff();
    std::vector<ManagedInvocation::CompletionBinding> completion_bindings() const;
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
    uint64_t invocation_identity = 0;
    NiuAdmission::Boots boots;
    std::shared_ptr<const HandoffSnapshot> handoff_snapshot;
    std::shared_ptr<const HandoffSnapshot> issued_handoff;
    std::vector<HandoffObservation> observed_handoffs;
    Endpoints managed_endpoints;
    std::optional<NiuAdmission::Boots> replay_boots;
    std::atomic<State> state{State::Reserved};
    std::optional<NiuAdmission::Allocation> managed;
    std::vector<Member> members;
    std::vector<Group> groups;
    std::vector<std::shared_ptr<ProgramImpl>> retained_programs;
    std::vector<std::shared_ptr<distributed::MeshWorkloadImpl>> retained_workloads;
    std::vector<std::shared_ptr<Buffer>> retained_witness_backing;
    std::vector<std::shared_ptr<const ManagedInvocation::ItemWitnessStorage>> item_witnesses;
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
