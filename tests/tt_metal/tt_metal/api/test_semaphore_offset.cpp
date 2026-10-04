// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <limits>
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
namespace tt::tt_metal {
TEST(SemaphoreOffsetTest, UsesExplicitAlignmentWithoutCreatingContext) {
    const CoreRangeSet cores(CoreRange({0, 0}, {0, 0}));
    const Semaphore semaphore(cores, 3, 11);
    EXPECT_FALSE(MetalContext::instance_exists());
    EXPECT_EQ(semaphore.offset(16), 48u);
    EXPECT_EQ(semaphore.offset(64), 192u);
    EXPECT_EQ(semaphore.initial_value(), 11u);
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST(SemaphoreOffsetTest, RejectsInvalidAlignmentSlotAndOverflowWithoutContext) {
    const CoreRangeSet cores(CoreRange({0, 0}, {0, 0}));
    const Semaphore semaphore(cores, 3, 11);
    EXPECT_THROW(semaphore.offset(0), std::invalid_argument);
    EXPECT_THROW(semaphore.offset(2), std::invalid_argument);
    EXPECT_THROW(semaphore.offset(24), std::invalid_argument);
    EXPECT_THROW(semaphore.offset(0x80000000u), std::invalid_argument);
    const Semaphore invalid(cores, NUM_SEMAPHORES, 0);
    EXPECT_THROW(invalid.offset(16), std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}
}
