// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "semaphore.hpp"
#include <cstdint>

#include <limits>
#include <stdexcept>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

Semaphore::Semaphore(const CoreRangeSet& core_range_set, uint32_t id, uint32_t initial_value) :
    core_range_set_(core_range_set), id_(id), initial_value_(initial_value), core_type_(CoreType::WORKER) {}

Semaphore::Semaphore(const CoreRangeSet& core_range_set, uint32_t id, uint32_t initial_value, CoreType core_type) :
    core_range_set_(core_range_set), id_(id), initial_value_(initial_value), core_type_(core_type) {}

Semaphore::Semaphore(const Semaphore& other) = default;

Semaphore& Semaphore::operator=(const Semaphore& other) = default;

Semaphore::Semaphore(Semaphore&& other) noexcept = default;

Semaphore& Semaphore::operator=(Semaphore&& other) noexcept = default;

bool Semaphore::initialized_on_logical_core(const CoreCoord& logical_core) const {
    return this->core_range_set_.contains(logical_core);
}

uint32_t Semaphore::offset(uint32_t l1_alignment) const {
    if (!l1_alignment || (l1_alignment & (l1_alignment - 1)) ||
        l1_alignment < sizeof(uint32_t) || id_ >= NUM_SEMAPHORES ||
        uint64_t(l1_alignment) * id_ > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("invalid semaphore slot or L1 alignment");
    return l1_alignment * id_;
}

}  // namespace tt::tt_metal
