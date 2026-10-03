// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "internal/debug/fw_debug.h"
#include "llk_param_structs.h"

using namespace ckernel;

#ifdef UCK_CHLKC_MATH
#include "chlkc_descriptors.h"
#include "chlkc_math.cpp"
#endif

#ifdef UCK_CHLKC_PACK
#include "chlkc_descriptors.h"
#include "chlkc_pack.cpp"
#endif

#ifdef UCK_CHLKC_UNPACK
#include "chlkc_descriptors.h"
#include "chlkc_unpack.cpp"
#endif

uint run_kernel() {
// The generated role source supplies this selector from KernelBodyMode.
// Native descriptors retain storage metadata without any LLK math policy.
#if !TT_METAL_NATIVE_BODY
#ifdef UCK_CHLKC_MATH
    zeroacc();
#endif

#ifdef UCK_CHLKC_UNPACK
    zerosrc();
#endif
#endif

    kernel_main();

    return 0;
}
