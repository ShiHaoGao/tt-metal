// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Pure preprocessor constants: dev_mem_map.h is also included by linker
// scripts. The typed wire layout independently asserts these exact sizes.
#define TT_NIU_HANDOFF_L1_BYTES 608
#define TT_NIU_HANDOFF_ALIGNMENT 32
#define TT_NIU_HANDOFF_COUNTER_BYTES 80
