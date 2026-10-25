// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "niu_admission.hpp"
#include "impl/program/program_execution.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

#include "impl/experimental/native_kernel/native_image.hpp"

namespace tt::tt_metal::detail {

bool NiuKey::operator<(const NiuKey& other) const noexcept {
    return std::tie(chip, physical_worker.x, physical_worker.y, noc) <
           std::tie(other.chip, other.physical_worker.x, other.physical_worker.y, other.noc);
}

NiuLedger::NiuLedger() : owner_(std::make_shared<const NiuToken::Owner>()) {}

namespace {
bool valid_niu(const NiuKey& key) {
    return key.chip >= 0 && (key.noc == NiuNoc::Noc0 || key.noc == NiuNoc::Noc1);
}
bool reject(std::string& error, const char* message) {
    error = message;
    return false;
}
}  // namespace

void NiuLedger::quarantine_abandoned() {
    for (auto& [id, record] : records_) {
        if (record.state != State::Issued || !record.lease.expired()) continue;
        record.state = State::Quarantined;
        for (const auto& selection : record.selections)
            nius_.at(selection.niu).quarantined |= selection.selectors;
    }
}

void NiuLedger::commit_occupancy(std::map<NiuKey, Occupancy>& pending) {
    // Every new map node and the record were allocated before this commit.
    // Moving map nodes and assigning Occupancy cannot allocate or throw.
    while (!pending.empty()) {
        auto node = pending.extract(pending.begin());
        auto current = nius_.find(node.key());
        if (current == nius_.end()) nius_.insert(std::move(node));
        else current->second = node.mapped();
    }
}

std::optional<NiuAllocation> NiuLedger::acquire_managed(
    std::span<const NiuRequest> requests, std::string& error) {
    quarantine_abandoned();
    if (withdrawn_ || requests.empty() || next_id_ == std::numeric_limits<uint64_t>::max()) {
        reject(error, "NIU managed admission requires a live ledger and nonempty batch");
        return std::nullopt;
    }
    std::map<NiuKey, Occupancy> pending;
    std::vector<NiuSelection> selections;
    selections.reserve(requests.size());
    for (const auto& request : requests) {
        if (!valid_niu(request.niu) || request.selectors == 0 || request.selectors > niu_selector_count ||
            pending.contains(request.niu)) {
            reject(error, "NIU managed request has invalid or repeated selectors");
            return std::nullopt;
        }
        auto found = nius_.find(request.niu);
        Occupancy occupancy = found == nius_.end() ? Occupancy{} : found->second;
        if (occupancy.ordinary != 0 || occupancy.quarantined.any()) {
            reject(error, "NIU managed request overlaps an ordinary or quarantined owner");
            return std::nullopt;
        }
        NiuSelectorMask selected;
        for (size_t index = 0; index != niu_selector_count && selected.count() < request.selectors; ++index)
            if (!occupancy.managed.test(index)) selected.set(index);
        if (selected.count() != request.selectors) {
            reject(error, "NIU selector capacity is exhausted");
            return std::nullopt;
        }
        occupancy.managed |= selected;
        pending.emplace(request.niu, occupancy);
        selections.push_back({request.niu, selected});
    }
    auto lease = std::make_shared<const NiuToken::Lease>();
    records_.emplace(next_id_, Record{Kind::Managed, State::Reserved, selections, lease});
    commit_occupancy(pending);
    NiuAllocation result{NiuToken(owner_, std::move(lease), next_id_++), std::move(selections)};
    error.clear();
    return result;
}

std::optional<NiuToken> NiuLedger::reserve_ordinary(std::span<const NiuKey> nius, std::string& error) {
    quarantine_abandoned();
    if (withdrawn_ || nius.empty() || next_id_ == std::numeric_limits<uint64_t>::max()) {
        reject(error, "NIU ordinary reservation requires a live ledger and nonempty batch");
        return std::nullopt;
    }
    std::map<NiuKey, Occupancy> pending;
    std::vector<NiuSelection> selections;
    selections.reserve(nius.size());
    for (const auto& niu : nius) {
        if (!valid_niu(niu) || pending.contains(niu)) {
            reject(error, "NIU ordinary reservation has invalid or repeated physical NIUs");
            return std::nullopt;
        }
        auto found = nius_.find(niu);
        Occupancy occupancy = found == nius_.end() ? Occupancy{} : found->second;
        if (occupancy.managed.any() || occupancy.quarantined.any() ||
            occupancy.ordinary == std::numeric_limits<size_t>::max()) {
            reject(error, "NIU ordinary reservation overlaps a managed or quarantined owner");
            return std::nullopt;
        }
        ++occupancy.ordinary;
        pending.emplace(niu, occupancy);
        selections.push_back({niu, NiuSelectorMask{}.set()});
    }
    auto lease = std::make_shared<const NiuToken::Lease>();
    records_.emplace(next_id_, Record{Kind::Ordinary, State::Reserved, std::move(selections), lease});
    commit_occupancy(pending);
    error.clear();
    return NiuToken(owner_, std::move(lease), next_id_++);
}

NiuLedger::Record* NiuLedger::find(const NiuToken& token, std::string& error) {
    if (withdrawn_ || token.owner_ != owner_ || !token.lease_) {
        reject(error, "NIU token belongs to a foreign or withdrawn ledger");
        return nullptr;
    }
    auto found = records_.find(token.id_);
    if (found == records_.end() || found->second.lease.lock() != token.lease_) {
        reject(error, "NIU token has already retired");
        return nullptr;
    }
    return &found->second;
}

NiuLedger::Record* NiuLedger::prepare_issue(const NiuToken& token, std::string& error) {
    quarantine_abandoned();
    auto* record = find(token, error);
    if (!record) return nullptr;
    if (record->state != State::Reserved) {
        reject(error, "NIU reservation was already issued or quarantined");
        return nullptr;
    }
    for (const auto& selection : record->selections)
        if (nius_.at(selection.niu).quarantined.any()) {
            reject(error, "NIU issue overlaps a quarantined owner");
            return nullptr;
        }
    return record;
}

void NiuLedger::commit_issue(Record& record) noexcept { record.state = State::Issued; }

bool NiuLedger::mark_issued(const NiuToken& token, std::string& error) {
    auto* record = prepare_issue(token, error);
    if (!record) return false;
    commit_issue(*record);
    error.clear();
    return true;
}

void NiuLedger::erase(uint64_t id) {
    const auto& record = records_.at(id);
    for (const auto& selection : record.selections) {
        auto found = nius_.find(selection.niu);
        auto& occupancy = found->second;
        if (record.kind == Kind::Ordinary) --occupancy.ordinary;
        else occupancy.managed &= ~selection.selectors;
        if (!occupancy.ordinary && !occupancy.managed.any() && !occupancy.quarantined.any()) nius_.erase(found);
    }
    records_.erase(id);
}

bool NiuLedger::cancel_unissued(const NiuToken& token, std::string& error) {
    auto* record = find(token, error);
    if (!record) return false;
    if (record->state != State::Reserved) return reject(error, "Issued NIU reservation cannot be cancelled");
    erase(token.id_);
    error.clear();
    return true;
}

bool NiuLedger::record_stop(const NiuToken& token, std::string& error) {
    auto* record = find(token, error);
    if (!record) return false;
    if (record->state != State::Issued) return reject(error, "NIU stop requires its issued reservation");
    record->state = State::Stopped;
    error.clear();
    return true;
}

bool NiuLedger::record_clean(const NiuToken& token, std::string& error) {
    auto* record = find(token, error);
    if (!record) return false;
    if (record->state != State::Stopped) return reject(error, "NIU clean handoff requires a stopped reservation");
    erase(token.id_);
    error.clear();
    return true;
}

bool NiuLedger::quarantine(const NiuToken& token, std::string& error) {
    auto* record = find(token, error);
    if (!record) return false;
    record->state = State::Quarantined;
    for (const auto& selection : record->selections)
        nius_.at(selection.niu).quarantined |= selection.selectors;
    error.clear();
    return true;
}

void NiuLedger::withdraw() {
    withdrawn_ = true;
    for (auto& [id, record] : records_) {
        record.state = State::Quarantined;
        for (const auto& selection : record.selections)
            nius_.at(selection.niu).quarantined |= selection.selectors;
    }
}

}  // namespace tt::tt_metal::detail

namespace tt::tt_metal {

namespace {
bool same_boot(
    const std::shared_ptr<const experimental::native_detail::LoadedFirmware>& first,
    const std::shared_ptr<const experimental::native_detail::LoadedFirmware>& second) {
    return first.get() == second.get() && !first.owner_before(second) && !second.owner_before(first);
}
}  // namespace

NiuAdmission::NiuAdmission() : owner_(std::make_shared<const Token::Owner>()) {}

std::shared_ptr<const experimental::native_detail::LoadedFirmware>
NiuAdmission::native_firmware(ChipId chip) const {
    std::lock_guard lock(mutex_);
    auto found = boots_.find(chip);
    return found == boots_.end() ? nullptr : found->second.firmware;
}

NiuAdmission::Record* NiuAdmission::find_locked(const Token& token, std::string& error) {
    if (token.owner_ != owner_ || !token.lease_) {
        detail::reject(error, "NIU reservation belongs to another admission owner");
        return nullptr;
    }
    auto found = records_.find(token.id_);
    if (found == records_.end() || found->second.lease.lock() != token.lease_) {
        detail::reject(error, "NIU reservation has already retired");
        return nullptr;
    }
    return &found->second;
}

void NiuAdmission::quarantine_abandoned_locked() {
    for (auto& [id, record] : records_) {
        // Known-stopped owners remain ordinary-compatible even if their last
        // caller disappears; they still exclude managed selectors until clean.
        if (record.state != State::Issued || !record.lease.expired()) continue;
        record.state = State::Quarantined;
        for (auto& [chip, binding] : record.bindings) {
            std::string error;
            binding.boot.ledger->quarantine(binding.token, error);
        }
    }
}

bool NiuAdmission::validate_bindings_locked(
    const Record& record, detail::NiuLedger::State state, std::string& error) {
    for (const auto& [chip, binding] : record.bindings) {
        auto current = boots_.find(chip);
        if (current == boots_.end() || !same_boot(current->second.firmware, binding.boot.firmware) ||
            !binding.boot.firmware->live())
            return detail::reject(error, "NIU reservation belongs to a withdrawn firmware boot");
        auto* allocation = binding.boot.ledger->find(binding.token, error);
        if (!allocation) return false;
        if (allocation->state != state)
            return detail::reject(error, "NIU reservation and boot ledger transitions disagree");
    }
    return true;
}

bool NiuAdmission::validate_boots_locked(const Record& record, const Boots& boots, std::string& error) {
    if (boots.size() != record.bindings.size())
        return detail::reject(error, "NIU completion requires its complete captured boot set");
    for (const auto& [chip, binding] : record.bindings) {
        auto expected = boots.find(chip);
        if (expected == boots.end() || !same_boot(expected->second, binding.boot.firmware))
            return detail::reject(error, "NIU completion belongs to another firmware boot");
    }
    return true;
}

std::optional<NiuAdmission::Token> NiuAdmission::reserve_ordinary(
    std::span<const detail::NiuKey> nius, std::string& error) {
    auto guard = lock();
    return reserve_ordinary_locked(guard, nius, error);
}

void NiuAdmission::validate_guard(const Guard& guard) const {
    if (guard.owner_ != this || !guard.lock_.owns_lock() || guard.lock_.mutex() != &mutex_)
        throw std::logic_error("NIU guard belongs to another admission owner");
}

std::optional<NiuAdmission::Token> NiuAdmission::reserve_ordinary_locked(
    const Guard& guard, std::span<const detail::NiuKey> nius, std::string& error) {
    validate_guard(guard);
    quarantine_abandoned_locked();
    if (nius.empty() || next_id_ == std::numeric_limits<uint64_t>::max()) {
        detail::reject(error, "NIU reservation requires a nonempty physical NIU set");
        return std::nullopt;
    }
    std::set<detail::NiuKey> unique;
    std::map<ChipId, std::vector<detail::NiuKey>> grouped;
    for (const auto& niu : nius) {
        if (!detail::valid_niu(niu) || !unique.insert(niu).second) {
            detail::reject(error, "NIU reservation contains invalid or repeated physical NIUs");
            return std::nullopt;
        }
        grouped[niu.chip].push_back(niu);
    }
    // Preboot/ordinary-only contexts have no native ledger yet, but unknown
    // issued owners must still exclude another ordinary launch on these NIUs.
    for (const auto& [id, record] : records_)
        if (record.state == State::Quarantined)
            for (const auto& niu : record.nius)
                if (unique.contains(niu)) {
                    detail::reject(error, "NIU reservation overlaps quarantined infrastructure");
                    return std::nullopt;
                }

    auto lease = std::make_shared<const Token::Lease>();
    Record pending{Kind::Ordinary, State::Reserved, {nius.begin(), nius.end()}, {}, lease};
    for (const auto& [chip, keys] : grouped) {
        const auto boot = boots_.find(chip);
        if (boot == boots_.end()) continue;
        if (!boot->second.firmware->live()) {
            detail::reject(error, "NIU reservation requires a current firmware boot");
            return std::nullopt;
        }
        pending.bindings.emplace(chip, Binding{boot->second, {}});
    }
    std::vector<Binding*> allocated;
    allocated.reserve(pending.bindings.size());
    auto [position, inserted] = records_.emplace(next_id_, std::move(pending));
    auto rollback = [&] {
        for (auto* binding : allocated) {
            std::string ignored;
            binding->boot.ledger->cancel_unissued(binding->token, ignored);
        }
        records_.erase(position);
    };
    try {
        for (auto& [chip, binding] : position->second.bindings) {
            auto token = binding.boot.ledger->reserve_ordinary(grouped.at(chip), error);
            if (!token) {
                rollback();
                return std::nullopt;
            }
            binding.token = std::move(*token);
            allocated.push_back(&binding);
        }
    } catch (...) {
        rollback();
        throw;
    }
    error.clear();
    return Token(owner_, std::move(lease), next_id_++);
}

std::optional<NiuAdmission::Allocation> NiuAdmission::acquire_managed(
    const Boots& boots, std::span<const detail::NiuRequest> requests, std::string& error) {
    std::lock_guard lock(mutex_);
    quarantine_abandoned_locked();
    if (requests.empty() || next_id_ == std::numeric_limits<uint64_t>::max()) {
        detail::reject(error, "NIU managed admission requires a nonempty request");
        return std::nullopt;
    }
    std::map<ChipId, std::vector<detail::NiuRequest>> grouped;
    std::set<detail::NiuKey> unique;
    std::vector<detail::NiuKey> nius;
    nius.reserve(requests.size());
    for (const auto& request : requests) {
        if (!detail::valid_niu(request.niu) || !unique.insert(request.niu).second ||
            request.selectors == 0 || request.selectors > detail::niu_selector_count) {
            detail::reject(error, "NIU managed admission contains an invalid or repeated request");
            return std::nullopt;
        }
        grouped[request.niu.chip].push_back(request);
        nius.push_back(request.niu);
    }
    if (boots.size() != grouped.size()) {
        detail::reject(error, "NIU managed admission requires its complete actual boot set");
        return std::nullopt;
    }
    auto lease = std::make_shared<const Token::Lease>();
    Record pending{Kind::Managed, State::Reserved, std::move(nius), {}, lease};
    for (const auto& [chip, batch] : grouped) {
        auto requested = boots.find(chip);
        auto current = boots_.find(chip);
        if (requested == boots.end() || !requested->second || !requested->second->live() ||
            current == boots_.end() || !same_boot(requested->second, current->second.firmware)) {
            detail::reject(error, "NIU managed admission requires its current successful SDK boot");
            return std::nullopt;
        }
        pending.bindings.emplace(chip, Binding{current->second, {}});
    }
    std::vector<detail::NiuSelection> selections;
    selections.reserve(requests.size());
    std::vector<Binding*> allocated;
    allocated.reserve(pending.bindings.size());
    auto [position, inserted] = records_.emplace(next_id_, std::move(pending));
    auto rollback = [&] {
        for (auto* binding : allocated) {
            std::string ignored;
            binding->boot.ledger->cancel_unissued(binding->token, ignored);
        }
        records_.erase(position);
    };
    try {
        for (auto& [chip, binding] : position->second.bindings) {
            auto allocation = binding.boot.ledger->acquire_managed(grouped.at(chip), error);
            if (!allocation) {
                rollback();
                return std::nullopt;
            }
            binding.token = std::move(allocation->token);
            allocated.push_back(&binding);
            for (const auto& selection : allocation->selections) selections.push_back(selection);
        }
    } catch (...) {
        rollback();
        throw;
    }
    error.clear();
    return Allocation{Token(owner_, std::move(lease), next_id_++), std::move(selections)};
}

bool NiuAdmission::mark_issued(const Token& token, std::string& error) {
    auto guard = lock();
    return mark_issued_locked(guard, token, error);
}

bool NiuAdmission::mark_issued_locked(const Guard& guard, const Token& token, std::string& error) {
    validate_guard(guard);
    quarantine_abandoned_locked();
    auto* record = find_locked(token, error);
    if (!record) return false;
    if (record->state != State::Reserved)
        return detail::reject(error, "NIU admission was already issued or quarantined");
    // Preboot contexts have no boot ledger, so their physical intents must
    // independently reject an owner quarantined after this reservation.
    for (const auto& [id, occupant] : records_)
        if (occupant.state == State::Quarantined)
            for (const auto& niu : occupant.nius)
                if (std::find(record->nius.begin(), record->nius.end(), niu) != record->nius.end())
                    return detail::reject(error, "NIU issue overlaps quarantined infrastructure");
    if (!validate_bindings_locked(*record, detail::NiuLedger::State::Reserved, error)) return false;
    std::vector<detail::NiuLedger::Record*> prepared;
    prepared.reserve(record->bindings.size());
    for (auto& [chip, binding] : record->bindings) {
        auto* allocation = binding.boot.ledger->prepare_issue(binding.token, error);
        if (!allocation) return false;
        prepared.push_back(allocation);
    }
    // Every binding and every physical NIU passed before any reservation is
    // issued. The single admission lock protects this no-fail commit.
    for (auto* allocation : prepared) detail::NiuLedger::commit_issue(*allocation);
    record->state = State::Issued;
    error.clear();
    return true;
}

bool NiuAdmission::cancel_unissued(const Token& token, std::string& error) {
    auto guard = lock();
    return cancel_unissued_locked(guard, token, error);
}

bool NiuAdmission::cancel_unissued_locked(const Guard& guard, const Token& token, std::string& error) {
    validate_guard(guard);
    auto* record = find_locked(token, error);
    if (!record) return false;
    if (record->state != State::Reserved)
        return detail::reject(error, "Issued NIU admission cannot be cancelled");
    if (!validate_bindings_locked(*record, detail::NiuLedger::State::Reserved, error)) return false;
    for (auto& [chip, binding] : record->bindings) binding.boot.ledger->cancel_unissued(binding.token, error);
    records_.erase(token.id_);
    error.clear();
    return true;
}

bool NiuAdmission::quarantine(const Token& token, std::string& error) {
    auto guard = lock();
    return quarantine_locked(guard, token, error);
}

bool NiuAdmission::quarantine_locked(const Guard& guard, const Token& token, std::string& error) {
    validate_guard(guard);
    auto* record = find_locked(token, error);
    if (!record) return false;
    for (auto& [chip, binding] : record->bindings) {
        std::string ignored;
        binding.boot.ledger->quarantine(binding.token, ignored);
    }
    record->state = State::Quarantined;
    error.clear();
    return true;
}

bool NiuAdmission::validate_ordinary_locked(const Guard& guard, const Token& token, std::string& error) {
    validate_guard(guard);
    quarantine_abandoned_locked();
    auto* record = find_locked(token, error);
    if (!record) return false;
    if (record->kind != Kind::Ordinary || record->state == State::Quarantined)
        return detail::reject(error, "NIU infrastructure reservation is not admissible");
    for (const auto& [id, occupant] : records_)
        if (occupant.state == State::Quarantined)
            for (const auto& niu : occupant.nius)
                if (std::find(record->nius.begin(), record->nius.end(), niu) != record->nius.end())
                    return detail::reject(error, "NIU infrastructure overlaps quarantine");
    const auto state = record->state == State::Reserved ? detail::NiuLedger::State::Reserved :
        record->state == State::Issued ? detail::NiuLedger::State::Issued : detail::NiuLedger::State::Stopped;
    if (!validate_bindings_locked(*record, state, error)) return false;
    error.clear();
    return true;
}

bool NiuAdmission::record_stop_locked(const Token& token, const Boots& boots, std::string& error) {
    auto* record = find_locked(token, error);
    if (!record) return false;
    if (record->state != State::Issued)
        return detail::reject(error, "NIU completion requires its issued admission");
    if (!validate_boots_locked(*record, boots, error)) return false;
    if (!validate_bindings_locked(*record, detail::NiuLedger::State::Issued, error)) return false;
    for (auto& [chip, binding] : record->bindings) binding.boot.ledger->record_stop(binding.token, error);
    record->state = State::Stopped;
    error.clear();
    return true;
}

bool NiuAdmission::validate_managed_locked(const Guard& guard, const Token& token, const Boots& boots, std::string& error) {
    validate_guard(guard);
    quarantine_abandoned_locked();
    auto* record = find_locked(token, error);
    if (!record) return false;
    if (record->kind != Kind::Managed || (record->state != State::Reserved && record->state != State::Issued))
        return detail::reject(error, "Managed NIU invocation is stopped or quarantined");
    return validate_boots_locked(*record, boots, error) &&
           validate_bindings_locked(*record, record->state == State::Reserved ? detail::NiuLedger::State::Reserved :
                                    detail::NiuLedger::State::Issued, error);
}

bool NiuAdmission::record_clean_locked(const Token& token, const Boots& boots, std::string& error) {
    auto* record = find_locked(token, error);
    if (!record) return false;
    if (record->state != State::Stopped)
        return detail::reject(error, "NIU clean handoff requires its stopped admission");
    if (!validate_boots_locked(*record, boots, error)) return false;
    if (!validate_bindings_locked(*record, detail::NiuLedger::State::Stopped, error)) return false;
    for (auto& [chip, binding] : record->bindings) binding.boot.ledger->record_clean(binding.token, error);
    records_.erase(token.id_);
    error.clear();
    return true;
}

void NiuAdmission::publish_boots_locked(const FirmwareSet& firmware, const LedgerSet& ledgers) {
    quarantine_abandoned_locked();
    if (firmware.size() != ledgers.size()) throw std::runtime_error("NIU boot ledger set is incomplete");
    auto pending_boots = boots_;
    for (const auto& [chip, boot] : firmware) {
        auto ledger = ledgers.find(chip);
        if (!boot || !boot->live() || ledger == ledgers.end() || !ledger->second || boots_.contains(chip))
            throw std::runtime_error("NIU publication requires fresh actual firmware boots");
        pending_boots.emplace(chip, BootState{boot, ledger->second});
    }

    std::map<uint64_t, std::map<ChipId, Binding>> pending_bindings;
    // Import all compatible ordinary reservations before applying quarantine;
    // several already-issued ordinary owners may share a physical NIU.
    for (const auto& [id, record] : records_) {
        if (record.kind != Kind::Ordinary) continue;
        std::map<ChipId, std::vector<detail::NiuKey>> grouped;
        for (const auto& niu : record.nius)
            if (firmware.contains(niu.chip)) grouped[niu.chip].push_back(niu);
        if (grouped.empty()) continue;
        auto& bindings = pending_bindings.emplace(id, record.bindings).first->second;
        for (const auto& [chip, nius] : grouped) {
            if (bindings.contains(chip)) throw std::runtime_error("NIU preboot reservation retained an old boot");
            auto& boot = pending_boots.at(chip);
            std::string error;
            auto token = boot.ledger->reserve_ordinary(nius, error);
            if (!token) throw std::runtime_error(error);
            bindings.emplace(chip, Binding{boot, std::move(*token)});
        }
    }
    for (const auto& [id, bindings] : pending_bindings) {
        const auto state = records_.at(id).state;
        if (state != State::Issued && state != State::Stopped) continue;
        for (const auto& [chip, binding] : bindings) {
            if (!firmware.contains(chip)) continue;
            std::string error;
            if (!binding.boot.ledger->mark_issued(binding.token, error)) throw std::runtime_error(error);
            if (state == State::Stopped && !binding.boot.ledger->record_stop(binding.token, error))
                throw std::runtime_error(error);
        }
    }
    // These are historical states, not new launches: all previously issued
    // compatible ordinary owners must be imported before any quarantine.
    // Record ID order cannot decide whether their boot publication succeeds.
    for (const auto& [id, bindings] : pending_bindings) {
        if (records_.at(id).state != State::Quarantined) continue;
        for (const auto& [chip, binding] : bindings) {
            if (!firmware.contains(chip)) continue;
            std::string error;
            if (!binding.boot.ledger->quarantine(binding.token, error)) throw std::runtime_error(error);
        }
    }
    boots_.swap(pending_boots);
    for (auto& [id, bindings] : pending_bindings) records_.at(id).bindings.swap(bindings);
}

NiuAdmission::Retirement::Retirement(Retirement&& other) noexcept :
    owner(std::exchange(other.owner, nullptr)), chips(std::move(other.chips)),
    executions(std::move(other.executions)), programs(std::move(other.programs)),
    single_programs(std::move(other.single_programs)), workloads(std::move(other.workloads)),
    single_workloads(std::move(other.single_workloads)), single_execution(std::move(other.single_execution)) {}

NiuAdmission::Retirement& NiuAdmission::Retirement::operator=(Retirement&& other) noexcept {
    if (this == &other) return *this;
    release();
    owner = std::exchange(other.owner, nullptr);
    chips = std::move(other.chips);
    executions = std::move(other.executions);
    programs = std::move(other.programs);
    single_programs = std::move(other.single_programs);
    workloads = std::move(other.workloads);
    single_workloads = std::move(other.single_workloads);
    single_execution = std::move(other.single_execution);
    return *this;
}

NiuAdmission::Retirement::~Retirement() { release(); }

void NiuAdmission::Retirement::release() noexcept {
    // Resource destructors may acquire Device/allocator locks or reenter
    // admission. Reopen stays blocked until those destructors have finished.
    single_programs.clear();
    programs.clear();
    single_workloads.clear();
    workloads.clear();
    if (owner) {
        auto guard = owner->lock();
        for (auto chip : chips) --owner->retirements_.at(chip);
        for (const auto& execution : executions) execution->resources_retired = true;
        if (single_execution) single_execution->resources_retired = true;
        owner = nullptr;
    }
    single_execution.reset();
    executions.clear();
}

void NiuAdmission::retire_managed_locked(detail::ProgramExecution& execution, Retirement& retired) noexcept {
    // All cancellation/last-preparation destruction paths are nonallocating.
    // The nodes and chip set were created before invocation publication.
    retired.chips.swap(execution.retirement_chips);
    retired.single_programs.swap(execution.retained_programs);
    retired.single_workloads.swap(execution.retained_workloads);
    retired.single_execution = execution.shared_from_this();
    for (auto chip : retired.chips) ++retirements_.at(chip);
    retired.owner = this;
    if (execution.token) records_.erase(execution.token->id_);
    std::erase_if(executions_, [&](const auto& entry) { return entry.get() == &execution; });
}

NiuAdmission::Retirement NiuAdmission::retire_executions_locked() {
    Retirement retired;
    retired.executions.reserve(executions_.size());
    retired.programs.reserve(executions_.size());
    retired.workloads.reserve(executions_.size());
    for (const auto& execution : executions_)
        if (execution->state == detail::ProgramExecution::State::Withdrawn &&
            (!execution->managed || (execution->attached_chips.empty() && !execution->preparation.occupied())))
            for (const auto& [chip, workers] : execution->footprint) retired.chips.insert(chip);
    if (retired.chips.empty()) return retired;
    auto pending_retirements = retirements_;
    for (auto chip : retired.chips) ++pending_retirements[chip];
    // Every potentially allocating step precedes the ownership transfer.
    retirements_.swap(pending_retirements);
    retired.owner = this;
    for (auto& execution : executions_) {
        if (execution->state != detail::ProgramExecution::State::Withdrawn ||
            (execution->managed && (!execution->attached_chips.empty() || execution->preparation.occupied()))) continue;
        if (execution->managed && execution->token) records_.erase(execution->token->id_);
        retired.programs.push_back(std::move(execution->retained_programs));
        retired.workloads.push_back(std::move(execution->retained_workloads));
        retired.executions.push_back(std::move(execution));
    }
    std::erase(executions_, nullptr);
    return retired;
}

NiuAdmission::Retirement NiuAdmission::withdraw_boots_locked(const FirmwareSet& firmware) {
    for (auto& execution : executions_) {
        if (!execution->managed) continue;
        for (const auto& [chip, boot] : execution->boots) {
            auto retired = firmware.find(chip);
            if (retired != firmware.end() && same_boot(retired->second, boot)) {
                execution->state = detail::ProgramExecution::State::Withdrawn;
                execution->attached_chips.erase(chip);
            }
        }
    }
    for (auto& [id, record] : records_) {
        bool affected = false;
        for (const auto& [chip, binding] : record.bindings) {
            auto retired = firmware.find(chip);
            affected |= retired != firmware.end() && same_boot(retired->second, binding.boot.firmware);
        }
        if (!affected) continue;
        if (record.kind == Kind::Managed || record.state == State::Issued) {
            record.state = State::Quarantined;
            for (auto& [chip, binding] : record.bindings) {
                std::string ignored;
                binding.boot.ledger->quarantine(binding.token, ignored);
            }
        }
        if (record.kind == Kind::Ordinary) {
            for (auto it = record.bindings.begin(); it != record.bindings.end();) {
                auto retired = firmware.find(it->first);
                if (retired != firmware.end() && same_boot(retired->second, it->second.boot.firmware))
                    it = record.bindings.erase(it);
                else ++it;
            }
        }
    }
    for (const auto& [chip, boot] : firmware) {
        auto current = boots_.find(chip);
        if (current == boots_.end() || !same_boot(current->second.firmware, boot)) continue;
        current->second.ledger->withdraw();
        boots_.erase(current);
    }
    return retire_executions_locked();
}

void NiuAdmission::validate_device_open(ChipId chip) {
    auto guard = lock();
    if (auto found = retirements_.find(chip); found != retirements_.end() && found->second != 0)
        throw std::runtime_error("Device generation resources are still being retired");
    for (const auto& execution : executions_)
        if (execution->managed && execution->state == detail::ProgramExecution::State::Withdrawn &&
            execution->footprint.contains(chip) && !execution->attached_chips.contains(chip))
            throw std::runtime_error("Managed NIU resources from this Device generation await the remaining actual withdrawals");
}

void NiuAdmission::withdraw_device(ChipId chip) {
    Retirement retired;
    auto guard = lock();
    ++device_epochs_[chip];
    for (auto it = records_.begin(); it != records_.end();) {
        auto& record = it->second;
        if (std::none_of(record.nius.begin(), record.nius.end(),
                         [chip](const auto& niu) { return niu.chip == chip; })) {
            ++it;
            continue;
        }
        // Closing an actual device retires the caller identity even without a
        // NativeImage boot. It does not certify any live NIU as clean.
        record.lease.reset();
        if (record.state == State::Reserved) {
            for (auto& [bound_chip, binding] : record.bindings) {
                std::string ignored;
                binding.boot.ledger->cancel_unissued(binding.token, ignored);
            }
            it = records_.erase(it);
            continue;
        }
        if (record.kind == Kind::Managed || record.state == State::Issued || record.state == State::Quarantined) {
            record.state = State::Quarantined;
            for (auto& [bound_chip, binding] : record.bindings) {
                std::string ignored;
                binding.boot.ledger->quarantine(binding.token, ignored);
            }
        }
        if (record.kind == Kind::Ordinary) record.bindings.erase(chip);
        ++it;
    }
    for (auto& execution : executions_) {
        if (execution->footprint.contains(chip)) {
            execution->state = detail::ProgramExecution::State::Withdrawn;
            execution->attached_chips.erase(chip);
        }
    }
    retired = retire_executions_locked();
    if (auto found = boots_.find(chip); found != boots_.end()) {
        found->second.firmware->withdraw();
        found->second.ledger->withdraw();
        boots_.erase(found);
    }
}

}  // namespace tt::tt_metal
