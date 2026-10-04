// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "hostdev/native_kernel_contract.h"
#include "hostdev/worker_stream_state_contract.h"

#if defined(ARCH_BLACKHOLE)
namespace {
__attribute__((used, section(TT_NATIVE_IMAGE_SECTION)))
const tt_native_image_record native_firmware_contract{
    TT_NATIVE_IMAGE_MAGIC,
    TT_NATIVE_IMAGE_ABI_VERSION,
    TT_NATIVE_IMAGE_FIRMWARE,
#if defined(COMPILE_FOR_BRISC)
    TT_NATIVE_ROLE_BRISC,
#elif defined(COMPILE_FOR_NCRISC)
    TT_NATIVE_ROLE_NCRISC,
#else
    TT_NATIVE_ROLE_TRISC0 + COMPILE_FOR_TRISC,
#endif
    TT_NATIVE_ARCH_BLACKHOLE,
    TT_NATIVE_LOADING_DISCRETE,
#if defined(PROFILE_KERNEL) && !defined(PROFILE_STREAMING)
    TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM,
#else
    TT_NATIVE_PROFILE_ABSENT,
#endif
    TT_NATIVE_PRINT_TENSIX_SHARED_BUFFER,
    tt::worker_stream_state::kVersion,
    TT_NATIVE_WORKER_STREAM_SDK,
    0};
}  // namespace
#endif
