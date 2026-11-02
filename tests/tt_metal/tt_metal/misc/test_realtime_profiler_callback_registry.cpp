// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/realtime_profiler_callback_registry.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace tt::tt_metal {
namespace {

class CallbackListener final : public tt::RealtimeProfilerCallbackListener {
public:
    void on_callback_registered(
        tt::ProgramRealtimeProfilerCallbackHandle handle,
        const tt::ProgramRealtimeProfilerCallback& callback) override {
        ++registrations;
        callbacks.emplace(handle, callback);
        if (registrations == fail_registration) {
            throw std::runtime_error("listener setup failed after callback ownership");
        }
    }

    void on_callback_unregistered(
        tt::ProgramRealtimeProfilerCallbackHandle handle, tt::RealtimeProfilerCallbackStopMode mode) override {
        stop_modes.push_back(mode);
        callbacks.erase(handle);
    }

    unsigned registrations = 0;
    unsigned fail_registration = 0;
    std::map<tt::ProgramRealtimeProfilerCallbackHandle, tt::ProgramRealtimeProfilerCallback> callbacks;
    std::vector<tt::RealtimeProfilerCallbackStopMode> stop_modes;
};

TEST(RealtimeProfilerCallbackRegistry, RegistrationFailureRollsBackEveryStartedListener) {
    RealtimeProfilerCallbackRegistry registry;
    CallbackListener first;
    CallbackListener second;
    CallbackListener later;
    registry.attach(first);
    registry.attach(second);
    second.fail_registration = 1;
    auto owner = std::make_shared<int>(1);
    std::weak_ptr<int> retained = owner;
    auto callback = [owner = std::move(owner)](const tt::ProgramRealtimeRecordBatch&) {};
    EXPECT_THROW(registry.register_callback(std::move(callback)), std::runtime_error);
    EXPECT_TRUE(first.callbacks.empty());
    EXPECT_TRUE(second.callbacks.empty());
    EXPECT_TRUE(retained.expired());
    registry.attach(later);
    EXPECT_TRUE(later.callbacks.empty());
    EXPECT_EQ(first.stop_modes, (std::vector{tt::RealtimeProfilerCallbackStopMode::Abort}));
    EXPECT_EQ(second.stop_modes, (std::vector{tt::RealtimeProfilerCallbackStopMode::Abort}));

    second.fail_registration = 0;
    auto handle = registry.register_callback([](const tt::ProgramRealtimeRecordBatch&) {});
    EXPECT_EQ(first.callbacks.size(), 1u);
    EXPECT_EQ(second.callbacks.size(), 1u);
    EXPECT_EQ(later.callbacks.size(), 1u);
    registry.unregister_callback(handle);
    EXPECT_TRUE(first.callbacks.empty());
    EXPECT_TRUE(second.callbacks.empty());
    EXPECT_TRUE(later.callbacks.empty());
}

TEST(RealtimeProfilerCallbackRegistry, FailedListenerAttachmentRollsBackReplayAndStaysDetached) {
    RealtimeProfilerCallbackRegistry registry;
    const auto first = registry.register_callback([](const tt::ProgramRealtimeRecordBatch&) {});
    const auto second = registry.register_callback([](const tt::ProgramRealtimeRecordBatch&) {});
    CallbackListener failed;
    failed.fail_registration = 2;
    EXPECT_THROW(registry.attach(failed), std::runtime_error);
    EXPECT_TRUE(failed.callbacks.empty());
    const auto third = registry.register_callback([](const tt::ProgramRealtimeRecordBatch&) {});
    EXPECT_EQ(failed.registrations, 2u);

    CallbackListener healthy;
    registry.attach(healthy);
    EXPECT_EQ(healthy.callbacks.size(), 3u);
    for (auto handle : {first, second, third}) {
        registry.unregister_callback(handle);
    }
    EXPECT_TRUE(healthy.callbacks.empty());
    EXPECT_TRUE(failed.callbacks.empty());
}

TEST(RealtimeProfilerCallbackRegistry, FailedRegistrationPreservesEarlierCallbacks) {
    RealtimeProfilerCallbackRegistry registry;
    CallbackListener first;
    CallbackListener second;
    registry.attach(first);
    registry.attach(second);
    const auto retained = registry.register_callback([](const tt::ProgramRealtimeRecordBatch&) {});
    second.fail_registration = 2;
    EXPECT_THROW(registry.register_callback([](const tt::ProgramRealtimeRecordBatch&) {}), std::runtime_error);
    EXPECT_EQ(first.callbacks.size(), 1u);
    EXPECT_EQ(second.callbacks.size(), 1u);
    EXPECT_TRUE(first.callbacks.contains(retained));
    EXPECT_TRUE(second.callbacks.contains(retained));
    registry.unregister_callback(retained);
    EXPECT_TRUE(first.callbacks.empty());
    EXPECT_TRUE(second.callbacks.empty());
}

TEST(RealtimeProfilerCallbackRegistry, UnregisterCarriesTypedDispositionAndRemovesOwnership) {
    RealtimeProfilerCallbackRegistry registry;
    CallbackListener listener;
    registry.attach(listener);
    const auto finished = registry.register_callback([](const tt::ProgramRealtimeRecordBatch&) {});
    registry.unregister_callback(finished);
    registry.unregister_callback(finished);
    EXPECT_TRUE(listener.callbacks.empty());
    EXPECT_EQ(listener.stop_modes, (std::vector{tt::RealtimeProfilerCallbackStopMode::DrainPublishedPrefix}));
}

}  // namespace
}  // namespace tt::tt_metal
