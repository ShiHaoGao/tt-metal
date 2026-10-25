// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_event.hpp>

namespace tt::tt_metal::experimental {
class PinnedMemoryImpl;
}

namespace tt::tt_metal::distributed {

class MeshDevice;
class FDMeshCommandQueue;
class MeshEventCompletion;

class MeshEventImpl {
public:
    MeshEventImpl(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range);

    uint32_t id() const { return id_; }
    MeshDevice* device() const { return device_; }
    uint32_t mesh_cq_id() const { return mesh_cq_id_; }
    const MeshCoordinateRange& device_range() const { return device_range_; }

private:
    friend class FDMeshCommandQueue;
    friend void EventSynchronize(const MeshEvent& event);
    friend bool EventQuery(const MeshEvent& event);
    friend class ::tt::tt_metal::experimental::PinnedMemoryImpl;
    // Only an actual SDK record can attach this receiver. Publicly constructed
    // numeric events, including mock event zero, carry no execution authority.
    std::optional<bool> query_completion() const;
    bool completion_observed() const noexcept;
    std::shared_ptr<MeshEventCompletion> completion_;
    uint32_t id_ = 0;
    MeshDevice* device_ = nullptr;
    uint32_t mesh_cq_id_ = 0;
    MeshCoordinateRange device_range_;
};

}  // namespace tt::tt_metal::distributed
