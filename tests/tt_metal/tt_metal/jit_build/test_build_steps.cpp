// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <exception>
#include <future>
#include <stdexcept>
#include <vector>

#include "jit_build/build.hpp"

namespace tt::tt_metal {
namespace {

using namespace std::chrono_literals;

// Capture the exception only after the real synchronization routine returns.
// Promise-backed events let each test hold a sibling task pending explicitly.
std::future<std::exception_ptr> synchronize_in_background(
    std::vector<std::shared_future<void>>& events, std::promise<void>& started) {
    return std::async(std::launch::async, [&events, &started] {
        started.set_value();
        try {
            sync_build_steps(events);
        } catch (...) {
            return std::current_exception();
        }
        return std::exception_ptr{};
    });
}

TEST(BuildSteps, FailureWaitsForPendingSiblingBeforeRethrowing) {
    std::promise<void> failed_step;
    std::promise<void> pending_step;
    std::vector<std::shared_future<void>> events{
        failed_step.get_future().share(), pending_step.get_future().share()};
    const auto failure = std::make_exception_ptr(std::runtime_error("compile failed"));
    failed_step.set_exception(failure);

    std::promise<void> started;
    auto synchronization = synchronize_in_background(events, started);
    started.get_future().wait();

    // Do not use ASSERT here: the pending event must be released even if the
    // implementation incorrectly returns early, so teardown cannot hang.
    EXPECT_EQ(synchronization.wait_for(100ms), std::future_status::timeout);
    pending_step.set_value();
    EXPECT_EQ(synchronization.get(), failure);
}

TEST(BuildSteps, MultipleFailuresDrainRemainingStepsAndPreserveFirstException) {
    std::promise<void> first_step;
    std::promise<void> second_step;
    std::promise<void> pending_step;
    std::vector<std::shared_future<void>> events{
        first_step.get_future().share(), second_step.get_future().share(), pending_step.get_future().share()};
    const auto first_failure = std::make_exception_ptr(std::runtime_error("first compile failed"));
    first_step.set_exception(first_failure);
    second_step.set_exception(std::make_exception_ptr(std::logic_error("second compile failed")));

    std::promise<void> started;
    auto synchronization = synchronize_in_background(events, started);
    started.get_future().wait();

    EXPECT_EQ(synchronization.wait_for(100ms), std::future_status::timeout);
    pending_step.set_value();
    EXPECT_EQ(synchronization.get(), first_failure);
}

TEST(BuildSteps, SuccessWaitsForAllSteps) {
    std::promise<void> completed_step;
    std::promise<void> pending_step;
    std::vector<std::shared_future<void>> events{
        completed_step.get_future().share(), pending_step.get_future().share()};
    completed_step.set_value();

    std::promise<void> started;
    auto synchronization = synchronize_in_background(events, started);
    started.get_future().wait();

    EXPECT_EQ(synchronization.wait_for(100ms), std::future_status::timeout);
    pending_step.set_value();
    EXPECT_FALSE(synchronization.get());
}

TEST(BuildSteps, EmptyBatchSucceeds) {
    std::vector<std::shared_future<void>> events;
    EXPECT_NO_THROW(sync_build_steps(events));
}

}  // namespace
}  // namespace tt::tt_metal
