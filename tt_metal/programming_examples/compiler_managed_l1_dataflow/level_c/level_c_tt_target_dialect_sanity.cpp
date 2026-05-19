// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

struct CaseResult {
    const char* name = "";
    bool passed = false;
};

struct L1Ring {
    uint32_t base_addr = 0;
    uint32_t slot_count = 1;
    uint32_t slot_bytes = 1;
};

constexpr std::string_view kTileAddPseudoIr = R"mlir(
func.func @tile_add(%src0: !axe.tensor, %src1: !axe.tensor, %dst: !axe.tensor) {
  %src0_ring = tt.l1.alloc "src0_ring" {base = 0x10000, slots = 2, page_bytes = 2048}
  %src1_ring = tt.l1.alloc "src1_ring" {base = 0x12000, slots = 2, page_bytes = 2048}
  %dst_ring = tt.l1.alloc "dst_ring" {base = 0x14000, slots = 2, page_bytes = 2048}

  tt.schedule.stage @reader {
    %slot = tt.target.addr.ring_slot %i, 2
    %src0_l1 = tt.target.addr.add %src0_ring.base, %slot * 2048
    %src1_l1 = tt.target.addr.add %src1_ring.base, %slot * 2048
    %src0_noc = axe.lower_addr %src0[%i]
    %src1_noc = axe.lower_addr %src1[%i]
    tt.target.wait.ge %input_consumed0, %i_minus_slots
    tt.target.noc.read %src0_noc, %src0_l1, 2048
    tt.target.noc.read %src1_noc, %src1_l1, 2048
    tt.target.noc.barrier read
    tt.target.reg.write %input_ready0, %i_plus_1
    tt.dep.produce input_ready0, %i_plus_1
    tt.target.reg.write %input_ready1, %i_plus_1
    tt.dep.produce input_ready1, %i_plus_1
  }

  tt.schedule.stage @compute {
    %slot = tt.target.addr.ring_slot %i, 2
    %src0_l1 = tt.target.addr.add %src0_ring.base, %slot * 2048
    %src1_l1 = tt.target.addr.add %src1_ring.base, %slot * 2048
    %dst_l1 = tt.target.addr.add %dst_ring.base, %slot * 2048
    tt.target.wait.eq %input_ready0, %i_plus_1
    tt.dep.consume input_ready0, %i_plus_1
    tt.target.wait.eq %input_ready1, %i_plus_1
    tt.dep.consume input_ready1, %i_plus_1
    tt.target.tile.acquire
    tt.target.unpack.ab %src0_l1, %src1_l1
    tt.target.math.add
    tt.target.tile.commit
    tt.target.wait.ge %output_consumed, %i_minus_slots
    tt.dep.consume output_consumed, %i_minus_slots
    tt.target.tile.wait
    tt.target.pack %dst_l1
    tt.target.tile.release
    tt.target.reg.write %input_consumed0, %i_plus_1
    tt.dep.produce input_consumed0, %i_plus_1
    tt.target.reg.write %input_consumed1, %i_plus_1
    tt.dep.produce input_consumed1, %i_plus_1
    tt.target.reg.write %output_ready, %i_plus_1
    tt.dep.produce output_ready, %i_plus_1
  }

  tt.schedule.stage @writer {
    %slot = tt.target.addr.ring_slot %i, 2
    %dst_l1 = tt.target.addr.add %dst_ring.base, %slot * 2048
    %dst_noc = axe.lower_addr %dst[%i]
    tt.target.wait.ge %output_ready, %i_plus_1
    tt.dep.consume output_ready, %i_plus_1
    tt.target.noc.write %dst_l1, %dst_noc, 2048
    tt.target.noc.barrier write
    tt.target.reg.write %output_consumed, %i_plus_1
    tt.dep.produce output_consumed, %i_plus_1
  }
}
)mlir";

constexpr std::string_view kMatmulPseudoIr = R"mlir(
func.func @matmul_reuse(%a: !axe.tensor, %b: !axe.tensor, %c: !axe.tensor) {
  %a_ring = tt.l1.alloc "a_ring" {base = 0x20000, slots = 2, page_bytes = 8192}
  %b_ring = tt.l1.alloc "b_ring" {base = 0x24000, slots = 2, page_bytes = 8192}
  %out_ring = tt.l1.alloc "out_ring" {base = 0x28000, slots = 2, page_bytes = 8192}

  tt.schedule.stage @reader {
    %gen = tt.target.addr.add %block, 1
    %slot = tt.target.addr.ring_slot %block, 2
    %a_l1 = tt.target.addr.add %a_ring.base, %slot * 8192
    %b_l1 = tt.target.addr.add %b_ring.base, %slot * 8192
    tt.target.wait.ge %input_consumed0, %gen_minus_slots
    tt.target.noc.read_tile %a_tile_id, %a, %a_l1
    tt.target.noc.read_tile %b_tile_id, %b, %b_l1
    tt.target.noc.barrier read
    tt.target.reg.write %input_ready0, %gen
    tt.dep.produce input_ready0, %gen
    tt.target.reg.write %input_ready1, %gen
    tt.dep.produce input_ready1, %gen
  }

  tt.schedule.stage @compute {
    %gen = tt.target.addr.add %block, 1
    %slot = tt.target.addr.ring_slot %block, 2
    %a_l1 = tt.target.addr.add %a_ring.base, %slot * 8192
    %b_l1 = tt.target.addr.add %b_ring.base, %slot * 8192
    tt.target.wait.ge %input_ready0, %gen
    tt.dep.consume input_ready0, %gen
    tt.target.wait.ge %input_ready1, %gen
    tt.dep.consume input_ready1, %gen
    tt.target.tile.acquire
    tt.target.unpack.ab_matmul %a_l1, %b_l1, %a_tile_index, %b_tile_index
    tt.target.math.matmul %dst_index
    tt.target.tile.commit
    tt.target.wait.ge %output_consumed, %output_gen_minus_slots
    tt.dep.consume output_consumed, %output_gen_minus_slots
    tt.target.tile.wait
    tt.target.pack %out_l1
    tt.target.tile.release
    tt.target.reg.write %input_consumed0, %gen
    tt.dep.produce input_consumed0, %gen
    tt.target.reg.write %input_consumed1, %gen
    tt.dep.produce input_consumed1, %gen
    tt.target.reg.write %output_ready, %output_gen
    tt.dep.produce output_ready, %output_gen
  }

  tt.schedule.stage @writer {
    tt.target.wait.ge %output_ready, %output_gen
    tt.dep.consume output_ready, %output_gen
    tt.target.noc.write_tile %out_tile_id, %c, %out_l1
    tt.target.noc.barrier write
    tt.target.reg.write %output_consumed, %output_gen
    tt.dep.produce output_consumed, %output_gen
  }
}
)mlir";

constexpr std::array<std::string_view, 18> kForbiddenLegacyTokens = {
    "TensorAccessor",
    "TensorAccessorArgs",
    "CircularBuffer",
    "CircularBufferConfig",
    "CreateCircularBuffer",
    "LocalCBInterface",
    "CBInterface",
    "get_local_cb_interface",
    "get_cb_tiles_received_ptr",
    "get_cb_tiles_acked_ptr",
    "cb_wait_front",
    "cb_reserve_back",
    "cb_push_back",
    "cb_pop_front",
    "get_read_ptr",
    "get_write_ptr",
    "get_tile_size",
    "CBIndex",
};

bool contains(std::string_view text, std::string_view token) {
    return text.find(token) != std::string_view::npos;
}

uint32_t count_token(std::string_view text, std::string_view token) {
    uint32_t count = 0;
    std::size_t pos = 0;
    while ((pos = text.find(token, pos)) != std::string_view::npos) {
        ++count;
        pos += token.size();
    }
    return count;
}

bool has_no_legacy_tokens(std::string_view text) {
    for (const auto token : kForbiddenLegacyTokens) {
        if (contains(text, token)) {
            return false;
        }
    }
    return true;
}

bool has_all_tokens(std::string_view text, const std::vector<std::string_view>& tokens) {
    for (const auto token : tokens) {
        if (!contains(text, token)) {
            return false;
        }
    }
    return true;
}

bool has_balanced_dependency(std::string_view text, std::string_view name) {
    const std::string produce = "tt.dep.produce ";
    const std::string consume = "tt.dep.consume ";
    return count_token(text, produce + std::string(name)) == count_token(text, consume + std::string(name));
}

constexpr uint32_t ring_slot(const L1Ring& ring, uint32_t iteration) {
    return iteration % ring.slot_count;
}

constexpr uint32_t ring_addr(const L1Ring& ring, uint32_t iteration) {
    return ring.base_addr + ring_slot(ring, iteration) * ring.slot_bytes;
}

void add_result(std::vector<CaseResult>& results, const char* name, bool passed) {
    results.push_back(CaseResult{name, passed});
}

bool tile_add_ring_schedule_is_correct() {
    constexpr L1Ring src0{0x10000, 2, 2048};
    constexpr L1Ring src1{0x12000, 2, 2048};
    constexpr L1Ring dst{0x14000, 2, 2048};

    return ring_slot(src0, 0) == 0 && ring_slot(src0, 1) == 1 && ring_slot(src0, 2) == 0 &&
           ring_addr(src0, 0) == ring_addr(src0, 2) && ring_addr(src1, 1) == ring_addr(src1, 3) &&
           ring_addr(dst, 0) == 0x14000 && ring_addr(dst, 1) == 0x14800;
}

bool matmul_ring_schedule_is_correct() {
    constexpr L1Ring in0{0x20000, 2, 8192};
    constexpr L1Ring in1{0x24000, 2, 8192};
    constexpr L1Ring out{0x28000, 2, 8192};

    return ring_slot(in0, 0) == 0 && ring_slot(in0, 1) == 1 && ring_slot(in0, 2) == 0 &&
           ring_addr(in0, 2) == ring_addr(in0, 0) && ring_addr(in1, 3) == ring_addr(in1, 1) &&
           ring_addr(out, 0) == 0x28000 && ring_addr(out, 1) == 0x2a000;
}

}  // namespace

int main() {
    std::vector<CaseResult> results;

    const std::vector<std::string_view> stage_tokens = {
        "tt.schedule.stage @reader",
        "tt.schedule.stage @compute",
        "tt.schedule.stage @writer",
    };
    const std::vector<std::string_view> tile_add_tokens = {
        "tt.l1.alloc",
        "tt.target.addr.ring_slot",
        "tt.target.addr.add",
        "tt.target.wait.eq",
        "tt.target.wait.ge",
        "tt.target.reg.write",
        "tt.target.noc.read",
        "tt.target.noc.write",
        "tt.target.noc.barrier",
        "tt.target.tile.acquire",
        "tt.target.tile.commit",
        "tt.target.tile.wait",
        "tt.target.tile.release",
        "tt.target.unpack.ab",
        "tt.target.math.add",
        "tt.target.pack",
    };
    const std::vector<std::string_view> matmul_tokens = {
        "tt.l1.alloc",
        "tt.target.addr.ring_slot",
        "tt.target.addr.add",
        "tt.target.wait.ge",
        "tt.target.reg.write",
        "tt.target.noc.read_tile",
        "tt.target.noc.write_tile",
        "tt.target.noc.barrier",
        "tt.target.tile.acquire",
        "tt.target.tile.commit",
        "tt.target.tile.wait",
        "tt.target.tile.release",
        "tt.target.unpack.ab_matmul",
        "tt.target.math.matmul",
        "tt.target.pack",
    };

    add_result(results, "tile-add pseudo IR has no legacy CB/TensorAccessor tokens", has_no_legacy_tokens(kTileAddPseudoIr));
    add_result(results, "matmul pseudo IR has no legacy CB/TensorAccessor tokens", has_no_legacy_tokens(kMatmulPseudoIr));
    add_result(results, "tile-add has reader/compute/writer stages", has_all_tokens(kTileAddPseudoIr, stage_tokens));
    add_result(results, "matmul has reader/compute/writer stages", has_all_tokens(kMatmulPseudoIr, stage_tokens));
    add_result(results, "tile-add covers tt.target v0 op families", has_all_tokens(kTileAddPseudoIr, tile_add_tokens));
    add_result(results, "matmul covers tt.target v0 op families", has_all_tokens(kMatmulPseudoIr, matmul_tokens));
    add_result(results, "tile-add dependencies are balanced", has_balanced_dependency(kTileAddPseudoIr, "input_ready0") &&
            has_balanced_dependency(kTileAddPseudoIr, "input_ready1") &&
            has_balanced_dependency(kTileAddPseudoIr, "output_ready") &&
            has_balanced_dependency(kTileAddPseudoIr, "output_consumed"));
    add_result(results, "matmul dependencies are balanced", has_balanced_dependency(kMatmulPseudoIr, "input_ready0") &&
            has_balanced_dependency(kMatmulPseudoIr, "input_ready1") &&
            has_balanced_dependency(kMatmulPseudoIr, "output_ready") &&
            has_balanced_dependency(kMatmulPseudoIr, "output_consumed"));
    add_result(results, "tile-add ring slot evaluator matches two-slot reuse", tile_add_ring_schedule_is_correct());
    add_result(results, "matmul ring slot evaluator matches two-slot reuse", matmul_ring_schedule_is_correct());

    bool ok = true;
    for (const auto& result : results) {
        ok = ok && result.passed;
        std::cout << result.name << ": " << (result.passed ? "covered" : "failed") << "\n";
    }

    if (!ok) {
        std::cerr << "TT target dialect sanity failed\n";
        return 1;
    }

    std::cout << "TT target dialect sanity passed\n";
    std::cout << "tt_target_cases=" << results.size() << "\n";
    std::cout << "canonical_bottom_ir=NoC+stream-register+L1-address+tile-register+unpack/math/pack\n";
    return 0;
}
