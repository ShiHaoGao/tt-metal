// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include <mutex>

namespace tt::tt_metal {
class MetalContext;
// Shared short admission transition for exclusive client acquisition and
// LightMetal capture begin/end. Never held while executing a Program.
class WorkerStreamStateAdmission final {
public:
    WorkerStreamStateAdmission();
    WorkerStreamStateAdmission(const WorkerStreamStateAdmission&) = delete;
    WorkerStreamStateAdmission& operator=(const WorkerStreamStateAdmission&) = delete;
private:
    std::unique_lock<std::mutex> lock_;
};
class WorkerStreamStateClient final {
public:
    WorkerStreamStateClient(const WorkerStreamStateClient&) = delete;
    WorkerStreamStateClient& operator=(const WorkerStreamStateClient&) = delete;
private:
    friend class MetalContext;
    WorkerStreamStateClient() = default;
};
// Scoped host lifecycle access; Program submission retains its own binding and
// therefore does not depend on propagation of thread-local state into SDK tasks.
class WorkerStreamStateAccess final {
public:
    // These definitions live in libtt_metal.  Keeping the TLS transition
    // out-of-line is required when a client (such as TReX's runtime DSO)
    // includes this header: an inline static thread_local would otherwise be
    // instantiated once per shared object, so SDK code would observe a null
    // access even while the client DSO held its own copy.
    explicit WorkerStreamStateAccess(std::shared_ptr<const WorkerStreamStateClient> client);
    ~WorkerStreamStateAccess();
    WorkerStreamStateAccess(const WorkerStreamStateAccess&) = delete;
    WorkerStreamStateAccess& operator=(const WorkerStreamStateAccess&) = delete;
private:
    friend class MetalContext;
    std::shared_ptr<const WorkerStreamStateClient> client_;
    const WorkerStreamStateClient* previous_;
    inline static thread_local const WorkerStreamStateClient* current_ = nullptr;
};
}  // namespace tt::tt_metal
