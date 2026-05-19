// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    const uint32_t in_tile_size_bytes = get_tile_size(cb_in0);
    const InterleavedAddrGenFast<true> in0 = {
        .bank_base_address = in0_addr,
        .page_size = in_tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    cb_reserve_back(cb_in0, 1);
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);
    noc_async_read_tile(0, in0, cb_in0_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in0, 1);

    const uint32_t out_tile_size_bytes = get_tile_size(cb_out0);
    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = out_tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    cb_wait_front(cb_out0, 1);
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);
    noc_async_write_tile(0, dst, cb_out0_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_out0, 1);
}
