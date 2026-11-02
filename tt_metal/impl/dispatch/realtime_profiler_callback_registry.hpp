// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/realtime_profiler.hpp>

namespace tt {

using ProgramRealtimeRecord = tt::tt_metal::experimental::ProgramRealtimeRecord;
using ProgramRealtimeRecordBatch = tt::tt_metal::experimental::ProgramRealtimeRecordBatch;
using ProgramRealtimeProfilerCallback = tt::tt_metal::experimental::ProgramRealtimeProfilerCallback;
using ProgramRealtimeProfilerCallbackHandle = tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;

// Abort is internal setup rollback before a registration or attachment returns.
// Unregistering a returned handle always drains its published host prefix.
enum class RealtimeProfilerCallbackStopMode : uint8_t { DrainPublishedPrefix, Abort };

class RealtimeProfilerCallbackListener {
public:
    virtual ~RealtimeProfilerCallbackListener() = default;
    virtual void on_callback_registered(
        ProgramRealtimeProfilerCallbackHandle handle, const ProgramRealtimeProfilerCallback& callback) = 0;
    virtual void on_callback_unregistered(
        ProgramRealtimeProfilerCallbackHandle handle, RealtimeProfilerCallbackStopMode mode) = 0;
};

}  // namespace tt

namespace tt::tt_metal {

class RealtimeProfilerCallbackRegistry {
public:
    tt::ProgramRealtimeProfilerCallbackHandle register_callback(tt::ProgramRealtimeProfilerCallback callback) {
        std::lock_guard lock(mutex_);
        const auto handle = next_handle_++;
        registrations_.push_back({handle, std::move(callback)});
        const auto& stored_callback = registrations_.back().callback;
        size_t started = 0;
        try {
            for (auto* listener : listeners_) {
                ++started;
                listener->on_callback_registered(handle, stored_callback);
            }
        } catch (...) {
            const auto failure = std::current_exception();
            const auto rollback_failure = rollback(listeners_.data(), started, handle);
            registrations_.pop_back();
            std::rethrow_exception(rollback_failure ? rollback_failure : failure);
        }
        return handle;
    }

    void unregister_callback(tt::ProgramRealtimeProfilerCallbackHandle handle) {
        std::lock_guard lock(mutex_);
        auto registration = std::find_if(registrations_.begin(), registrations_.end(), [handle](const auto& entry) {
            return entry.handle == handle;
        });
        if (registration == registrations_.end()) {
            return;
        }
        registrations_.erase(registration);
        for (auto* listener : listeners_) {
            listener->on_callback_unregistered(handle, tt::RealtimeProfilerCallbackStopMode::DrainPublishedPrefix);
        }
    }

    void attach(tt::RealtimeProfilerCallbackListener& listener) {
        std::lock_guard lock(mutex_);
        listeners_.push_back(&listener);
        size_t started = 0;
        try {
            for (const auto& registration : registrations_) {
                ++started;
                listener.on_callback_registered(registration.handle, registration.callback);
            }
        } catch (...) {
            const auto failure = std::current_exception();
            std::exception_ptr rollback_failure;
            while (started != 0) {
                try {
                    listener.on_callback_unregistered(
                        registrations_[--started].handle, tt::RealtimeProfilerCallbackStopMode::Abort);
                } catch (...) {
                    if (!rollback_failure) {
                        rollback_failure = std::current_exception();
                    }
                }
            }
            listeners_.pop_back();
            std::rethrow_exception(rollback_failure ? rollback_failure : failure);
        }
    }

    void detach(tt::RealtimeProfilerCallbackListener& listener) {
        std::lock_guard lock(mutex_);
        std::erase(listeners_, &listener);
    }

private:
    static std::exception_ptr rollback(
        tt::RealtimeProfilerCallbackListener* const* listeners,
        size_t started,
        tt::ProgramRealtimeProfilerCallbackHandle handle) {
        std::exception_ptr failure;
        while (started != 0) {
            try {
                listeners[--started]->on_callback_unregistered(handle, tt::RealtimeProfilerCallbackStopMode::Abort);
            } catch (...) {
                if (!failure) {
                    failure = std::current_exception();
                }
            }
        }
        return failure;
    }

    struct Registration {
        tt::ProgramRealtimeProfilerCallbackHandle handle;
        tt::ProgramRealtimeProfilerCallback callback;
    };

    std::mutex mutex_;
    std::vector<Registration> registrations_;
    std::vector<tt::RealtimeProfilerCallbackListener*> listeners_;
    tt::ProgramRealtimeProfilerCallbackHandle next_handle_{0};
};

}  // namespace tt::tt_metal
