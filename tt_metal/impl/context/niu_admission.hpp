// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal { class NiuAdmission; }
namespace tt::tt_metal::distributed { class MeshWorkloadImpl; }

namespace tt::tt_metal::detail {
class ProgramExecution;
class ProgramImpl;

enum class NiuNoc : uint8_t { Noc0, Noc1 };

// Blackhole's four-bit transaction selector namespace. The 255 outstanding
// transactions per selector are a separate issue-window contract.
inline constexpr size_t niu_selector_count = 16;
using NiuSelectorMask = std::bitset<niu_selector_count>;

struct NiuKey {
    ChipId chip;
    CoreCoord physical_worker;
    NiuNoc noc;
    bool operator==(const NiuKey&) const = default;
    bool operator<(const NiuKey& other) const noexcept;
};

struct NiuRequest {
    NiuKey niu;
    uint8_t selectors;
};

struct NiuSelection {
    NiuKey niu;
    NiuSelectorMask selectors;
};

class NiuLedger;

// A bookkeeping token, not a firmware receipt or an execution capability.
// Destruction cannot cancel an issued reservation or certify NIU cleanup.
class NiuToken {
public:
    NiuToken() = default;
private:
    struct Owner {};
    struct Lease {};
    friend class NiuLedger;
    NiuToken(std::shared_ptr<const Owner> owner, std::shared_ptr<const Lease> lease, uint64_t id) :
        owner_(std::move(owner)), lease_(std::move(lease)), id_(id) {}
    std::shared_ptr<const Owner> owner_;
    std::shared_ptr<const Lease> lease_;
    uint64_t id_ = 0;
};

struct NiuAllocation {
    NiuToken token;
    std::vector<NiuSelection> selections;
};

// Pure allocation/state algorithm. It has no LoadedFirmware or activation
// authority. Production access to a boot-owned instance is serialized and
// admitted by NiuAdmission; direct instances are suitable for CPU tests only.
// There is no second mutex in this layer.
class NiuLedger {
public:
    NiuLedger();
    NiuLedger(const NiuLedger&) = delete;
    NiuLedger& operator=(const NiuLedger&) = delete;

    std::optional<NiuAllocation> acquire_managed(std::span<const NiuRequest> requests, std::string& error);
    std::optional<NiuToken> reserve_ordinary(std::span<const NiuKey> nius, std::string& error);
    bool mark_issued(const NiuToken& token, std::string& error);
    bool cancel_unissued(const NiuToken& token, std::string& error);
    // Kernel/CQ completion alone leaves the selector exclusion in place.
    bool record_stop(const NiuToken& token, std::string& error);
    // The caller must own a separate NIU-clean handoff proof. Production does
    // not expose this algorithm method through LoadedFirmware.
    bool record_clean(const NiuToken& token, std::string& error);
    bool quarantine(const NiuToken& token, std::string& error);
    void withdraw();

private:
    friend class ::tt::tt_metal::NiuAdmission;
    enum class Kind { Managed, Ordinary };
    enum class State { Reserved, Issued, Stopped, Quarantined };
    struct Record {
        Kind kind;
        State state;
        std::vector<NiuSelection> selections;
        std::weak_ptr<const NiuToken::Lease> lease;
    };
    struct Occupancy {
        NiuSelectorMask managed;
        NiuSelectorMask quarantined;
        size_t ordinary = 0;
    };

    Record* find(const NiuToken& token, std::string& error);
    Record* prepare_issue(const NiuToken& token, std::string& error);
    static void commit_issue(Record& record) noexcept;
    void quarantine_abandoned();
    void commit_occupancy(std::map<NiuKey, Occupancy>& pending);
    void erase(uint64_t id);
    std::shared_ptr<const NiuToken::Owner> owner_;
    std::map<NiuKey, Occupancy> nius_;
    std::map<uint64_t, Record> records_;
    uint64_t next_id_ = 1;
    bool withdrawn_ = false;
};

}  // namespace tt::tt_metal::detail

namespace tt::tt_metal {
class RiscFirmwareInitializer;
class dispatch_core_manager;
namespace internal { class ServiceCoreManagerImpl; }
namespace experimental::native_detail { class LoadedFirmware; }

// One context owns the only admission mutex. If other SDK locks are needed,
// acquire queue API -> admission -> dispatch assignments. Never wait on device
// I/O, enter MetalContext::instance(), or call another locking admission method
// while holding it. The pure NiuLedger algorithm has no lock of its own.
class NiuAdmission {
public:
    class Guard {
    public:
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    private:
        friend class NiuAdmission;
        explicit Guard(NiuAdmission& owner) : owner_(&owner), lock_(owner.mutex_) {}
        NiuAdmission* owner_;
        std::unique_lock<std::mutex> lock_;
    };
    using Boots = std::map<ChipId, std::shared_ptr<const experimental::native_detail::LoadedFirmware>>;
    class Token {
    public:
        Token() = default;
    private:
        friend class NiuAdmission;
        struct Owner {};
        struct Lease {};
        Token(std::shared_ptr<const Owner> owner, std::shared_ptr<const Lease> lease, uint64_t id) :
            owner_(std::move(owner)), lease_(std::move(lease)), id_(id) {}
        std::shared_ptr<const Owner> owner_;
        std::shared_ptr<const Lease> lease_;
        uint64_t id_ = 0;
    };
    class Allocation {
    public:
        Allocation(const Allocation&) = default;
        Allocation(Allocation&&) noexcept = default;
        Allocation& operator=(const Allocation&) = delete;
        Allocation& operator=(Allocation&&) = delete;
        std::span<const detail::NiuSelection> selections() const { return selections_; }
    private:
        friend class NiuAdmission;
        friend class detail::ProgramExecution;
        Allocation(Token token, std::vector<detail::NiuSelection> selections) :
            token_(std::move(token)), selections_(std::move(selections)) {}
        Token token_;
        std::vector<detail::NiuSelection> selections_;
    };

    NiuAdmission();
    NiuAdmission(const NiuAdmission&) = delete;
    NiuAdmission& operator=(const NiuAdmission&) = delete;

    // Lookup shares the publication/withdrawal lock and returns only the
    // retained boot published by a completed real firmware launch phase.
    std::shared_ptr<const experimental::native_detail::LoadedFirmware> native_firmware(ChipId chip) const;

    // Also used before boot by infrastructure planning, and by ordinary SDK
    // deployments which do not publish native firmware. Such intents are
    // imported atomically when an actual native boot is published.
    std::optional<Token> reserve_ordinary(std::span<const detail::NiuKey> nius, std::string& error);
    std::optional<Allocation> acquire_managed(
        const Boots& boots, std::span<const detail::NiuRequest> requests, std::string& error);
    bool mark_issued(const Token& token, std::string& error);
    bool cancel_unissued(const Token& token, std::string& error);
    bool quarantine(const Token& token, std::string& error);

private:
    friend class detail::ProgramExecution;
    friend class Device;
    friend class RiscFirmwareInitializer;
    friend class dispatch_core_manager;
    friend class internal::ServiceCoreManagerImpl;
    Guard lock() { return Guard(*this); }
    void validate_guard(const Guard& guard) const;
    std::optional<Token> reserve_ordinary_locked(
        const Guard& guard, std::span<const detail::NiuKey> nius, std::string& error);
    bool validate_ordinary_locked(const Guard& guard, const Token& token, std::string& error);
    bool validate_managed_locked(const Guard& guard, const Token& token, const Boots& boots, std::string& error);
    bool cancel_unissued_locked(const Guard& guard, const Token& token, std::string& error);
    bool quarantine_locked(const Guard& guard, const Token& token, std::string& error);
    bool mark_issued_locked(const Guard& guard, const Token& token, std::string& error);
    void withdraw_device(ChipId chip);
    void validate_device_open(ChipId chip);
    using FirmwareSet = std::unordered_map<ChipId, std::shared_ptr<const experimental::native_detail::LoadedFirmware>>;
    using LedgerSet = std::unordered_map<ChipId, std::shared_ptr<detail::NiuLedger>>;
    enum class Kind { Managed, Ordinary };
    enum class State { Reserved, Issued, Stopped, Quarantined };
    struct BootState {
        std::shared_ptr<const experimental::native_detail::LoadedFirmware> firmware;
        std::shared_ptr<detail::NiuLedger> ledger;
    };
    struct Binding {
        BootState boot;
        detail::NiuToken token;
    };
    struct Record {
        Kind kind;
        State state;
        std::vector<detail::NiuKey> nius;
        std::map<ChipId, Binding> bindings;
        std::weak_ptr<const Token::Lease> lease;
    };

    // Only the existing initializer's successful real boot seam can publish.
    // It holds mutex_ while changing both LoadedFirmware::live and these maps.
    void publish_boots_locked(const FirmwareSet& firmware, const LedgerSet& ledgers);
    struct Retirement {
        Retirement() = default;
        ~Retirement();
        Retirement(Retirement&& other) noexcept;
        Retirement& operator=(Retirement&& other) noexcept;
        Retirement(const Retirement&) = delete;
        void release() noexcept;
        NiuAdmission* owner = nullptr;
        std::set<ChipId> chips;
        std::vector<std::shared_ptr<detail::ProgramExecution>> executions;
        std::vector<std::vector<std::shared_ptr<detail::ProgramImpl>>> programs;
        std::vector<std::shared_ptr<detail::ProgramImpl>> single_programs;
        std::vector<std::vector<std::shared_ptr<distributed::MeshWorkloadImpl>>> workloads;
        std::vector<std::shared_ptr<distributed::MeshWorkloadImpl>> single_workloads;
        std::shared_ptr<detail::ProgramExecution> single_execution;
    };
    Retirement retire_executions_locked();
    void retire_managed_locked(detail::ProgramExecution& execution, Retirement& retired) noexcept;
    Retirement withdraw_boots_locked(const FirmwareSet& firmware);
    Record* find_locked(const Token& token, std::string& error);
    bool validate_bindings_locked(const Record& record, detail::NiuLedger::State state, std::string& error);
    bool validate_boots_locked(const Record& record, const Boots& boots, std::string& error);
    void quarantine_abandoned_locked();
    // Reserved for actual SDK completion/NIU-clean receivers. No public
    // "completed" flag or caller-supplied bool can release a real boot lease.
    // The receiver retains the exact boot set from the operation being proved;
    // a completion from an old boot cannot retire a reimported ordinary intent.
    bool record_stop_locked(const Token& token, const Boots& boots, std::string& error);
    bool record_clean_locked(const Token& token, const Boots& boots, std::string& error);

    std::map<ChipId, size_t> retirements_;
    mutable std::mutex mutex_;
    std::shared_ptr<const Token::Owner> owner_;
    std::map<ChipId, BootState> boots_;
    std::map<uint64_t, Record> records_;
    std::vector<std::shared_ptr<detail::ProgramExecution>> executions_;
    std::map<ChipId, uint64_t> device_epochs_;
    uint64_t next_id_ = 1;
};

}  // namespace tt::tt_metal
