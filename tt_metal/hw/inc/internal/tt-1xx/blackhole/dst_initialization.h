// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "cfg_defines.h"
#include "internal/vptr_uint.h"

namespace blackhole {

// Called by BRISC during quiescent device startup, before releasing subordinate
// RISCs. This initializes physical Dst data, not just ZEROACC's row-valid bits.
// It does not replace kernel accumulator initialization or device hang recovery.
inline void initialize_dst(vptr_uint instructions, vptr_uint configuration, vptr_uint debug_features) {
    // Dst.md requires the normal Dst16b mapping. Read back this MMIO write before
    // using a different MMIO region: a RISC fence alone does not acknowledge it.
    constexpr uint32_t dst_debug_mapping = 1u << 11;
    debug_features[0] &= ~dst_debug_mapping;
    while (debug_features[0] & dst_debug_mapping) {}

    instructions[0] = TT_OP_SETC16(CFG_STATE_ID_StateID_ADDR32, 0);
    instructions[0] = TT_OP_SFPENCC(3, 0, 0, 10);
    instructions[0] = TT_OP_SFPCONFIG(0, 15, 1);
    instructions[0] = TT_OP_SETC16(ADDR_MOD_AB_SEC0_SrcAIncr_ADDR32, 0);
    instructions[0] = TT_OP_SETC16(ADDR_MOD_DST_SEC0_DestIncr_ADDR32, 0);
    instructions[0] = TT_OP_SETC16(ADDR_MOD_BIAS_SEC0_BiasIncr_ADDR32, 0);

    // ThreadConfig is readable, but not writable, through RISC MMIO. Observe a
    // nonzero marker first so the final zero cannot be confused with reset state.
    constexpr uint32_t dst_offset = 3 * CFG_STATE_SIZE * 4 + DEST_TARGET_REG_CFG_MATH_Offset_ADDR32;
    instructions[0] = TT_OP_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 2);
    while ((configuration[dst_offset] & DEST_TARGET_REG_CFG_MATH_Offset_MASK) != 2) {}

    // Blackhole SFPSTORE mode 11 writes zero without reading an LReg. Each store
    // covers four rows and eight alternating columns. Sweeping the even 10-bit
    // addresses covers both column parities, including wrapping base offsets
    // and Dst16b row remapping. Keep this as a loop in the firmware binary.
#pragma GCC unroll 0
    for (uint32_t address = 0; address < 1024; address += 2) {
        instructions[0] = TT_OP_SFPSTORE(0, 11, 0, address);
    }

    // Block CFG and SFPU until the SFPU drains, then acknowledge through the
    // restored offset. Merely enqueueing STALLWAIT would not wait on BRISC.
    instructions[0] = TT_OP_STALLWAIT(0x180, 0x800);
    instructions[0] = TT_OP_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
    while ((configuration[dst_offset] & DEST_TARGET_REG_CFG_MATH_Offset_MASK) != 0) {}
}

}  // namespace blackhole
