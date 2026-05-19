// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compiler_managed_l1_dataflow/include/level_c_ir.hpp"
#include "compiler_managed_l1_dataflow/include/level_c_lowering_descriptors.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

namespace level_c = tt::tt_metal::experimental::compiler_managed_l1::level_c;

struct CoverageCase {
    const char* name;
    level_c::AxeLayout layout;
    std::array<level_c::LayoutAxisKind, 6> required_d_axes{};
    uint32_t required_d_axis_count = 0;
    bool requires_replica = false;
    bool requires_offset = false;
};

level_c::AxeAxis axis(level_c::LayoutAxisKind kind) {
    switch (kind) {
        case level_c::LayoutAxisKind::row: return level_c::AxeAxis{"row", kind};
        case level_c::LayoutAxisKind::col: return level_c::AxeAxis{"col", kind};
        case level_c::LayoutAxisKind::height: return level_c::AxeAxis{"height", kind};
        case level_c::LayoutAxisKind::width: return level_c::AxeAxis{"width", kind};
        case level_c::LayoutAxisKind::depth: return level_c::AxeAxis{"depth", kind};
        case level_c::LayoutAxisKind::page: return level_c::AxeAxis{"page", kind};
        case level_c::LayoutAxisKind::tile_row: return level_c::AxeAxis{"tile_row", kind};
        case level_c::LayoutAxisKind::tile_col: return level_c::AxeAxis{"tile_col", kind};
        case level_c::LayoutAxisKind::face_row: return level_c::AxeAxis{"face_row", kind};
        case level_c::LayoutAxisKind::face_col: return level_c::AxeAxis{"face_col", kind};
        case level_c::LayoutAxisKind::bank: return level_c::AxeAxis{"bank", kind};
        case level_c::LayoutAxisKind::core_x: return level_c::AxeAxis{"core_x", kind};
        case level_c::LayoutAxisKind::core_y: return level_c::AxeAxis{"core_y", kind};
        case level_c::LayoutAxisKind::mesh_x: return level_c::AxeAxis{"mesh_x", kind};
        case level_c::LayoutAxisKind::mesh_y: return level_c::AxeAxis{"mesh_y", kind};
        case level_c::LayoutAxisKind::storage: return level_c::AxeAxis{"storage", kind};
        case level_c::LayoutAxisKind::none: break;
    }
    return level_c::AxeAxis{"none", level_c::LayoutAxisKind::none};
}

level_c::AxeIter iter(uint32_t extent, int32_t stride, level_c::LayoutAxisKind axis_kind) {
    return level_c::AxeIter{extent, stride, axis(axis_kind)};
}

level_c::AxeCoordinateEntry coordinate_entry(int32_t value, level_c::LayoutAxisKind axis_kind) {
    return level_c::AxeCoordinateEntry{value, axis(axis_kind)};
}

level_c::AxeLayout make_row_major_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(64, 64, level_c::LayoutAxisKind::row),
        iter(64, 1, level_c::LayoutAxisKind::col),
    };
    return d;
}

level_c::AxeLayout make_tiled_face_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(2, 2048, level_c::LayoutAxisKind::tile_row),
        iter(2, 32, level_c::LayoutAxisKind::tile_col),
        iter(2, 16, level_c::LayoutAxisKind::face_row),
        iter(2, 1, level_c::LayoutAxisKind::face_col),
    };
    return d;
}

level_c::AxeLayout make_interleaved_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(8, 8, level_c::LayoutAxisKind::storage),
        iter(3, 0, level_c::LayoutAxisKind::bank),
        iter(8, 1, level_c::LayoutAxisKind::col),
    };
    return d;
}

level_c::AxeLayout make_height_sharded_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(8, 0, level_c::LayoutAxisKind::core_y),
        iter(32, 256, level_c::LayoutAxisKind::height),
        iter(256, 1, level_c::LayoutAxisKind::width),
    };
    return d;
}

level_c::AxeLayout make_width_sharded_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(4, 0, level_c::LayoutAxisKind::core_x),
        iter(64, 128, level_c::LayoutAxisKind::height),
        iter(128, 1, level_c::LayoutAxisKind::width),
    };
    return d;
}

level_c::AxeLayout make_block_sharded_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(4, 0, level_c::LayoutAxisKind::core_y),
        iter(4, 0, level_c::LayoutAxisKind::core_x),
        iter(64, 64, level_c::LayoutAxisKind::height),
        iter(64, 1, level_c::LayoutAxisKind::width),
    };
    return d;
}

level_c::AxeLayout make_nd_sharded_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(2, 0, level_c::LayoutAxisKind::core_y),
        iter(2, 0, level_c::LayoutAxisKind::core_x),
        iter(2, 8, level_c::LayoutAxisKind::storage),
        iter(2, 16, level_c::LayoutAxisKind::row),
        iter(2, 4, level_c::LayoutAxisKind::col),
        iter(2, 1, level_c::LayoutAxisKind::depth),
    };
    return d;
}

level_c::AxeLayout make_mesh_replicated_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(64, 128, level_c::LayoutAxisKind::row),
        iter(128, 1, level_c::LayoutAxisKind::col),
    };
    d.r_iters = {
        iter(2, 0, level_c::LayoutAxisKind::mesh_x),
        iter(2, 0, level_c::LayoutAxisKind::mesh_y),
    };
    return d;
}

level_c::AxeLayout make_mesh_sharded_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(2, 32, level_c::LayoutAxisKind::mesh_y),
        iter(32, 128, level_c::LayoutAxisKind::row),
        iter(2, 64, level_c::LayoutAxisKind::mesh_x),
        iter(64, 1, level_c::LayoutAxisKind::col),
    };
    return d;
}

level_c::AxeLayout make_offset_view_case() {
    level_c::AxeLayout d{};
    d.d_iters = {
        iter(16, 64, level_c::LayoutAxisKind::row),
        iter(16, 1, level_c::LayoutAxisKind::col),
    };
    d.o_entries = {
        coordinate_entry(65, level_c::LayoutAxisKind::storage),
    };
    return d;
}

int32_t tensor_accessor_linear_offset(const level_c::AxeLayout& d, uint32_t row, uint32_t col) {
    return static_cast<int32_t>(row) * d.d_iters[0].stride + static_cast<int32_t>(col) * d.d_iters[1].stride +
           d.o_entries.front().value;
}

bool has_d_axis(const level_c::AxeLayout& d, level_c::LayoutAxisKind axis_kind) {
    for (const auto& it : d.d_iters) {
        if (it.axis.kind == axis_kind) {
            return true;
        }
    }
    return false;
}

bool is_valid_iter(const level_c::AxeIter& it) {
    return it.extent > 0 && it.axis.kind != level_c::LayoutAxisKind::none && !it.axis.name.empty();
}

bool is_valid_layout(const level_c::AxeLayout& d) {
    if (d.d_iters.empty()) {
        return false;
    }
    for (const auto& it : d.d_iters) {
        if (!is_valid_iter(it)) {
            return false;
        }
    }
    for (const auto& it : d.r_iters) {
        if (!is_valid_iter(it)) {
            return false;
        }
    }
    for (const auto& coordinate_entry_value : d.o_entries) {
        if (coordinate_entry_value.axis.kind == level_c::LayoutAxisKind::none ||
            coordinate_entry_value.axis.name.empty()) {
            return false;
        }
    }
    return true;
}

bool has_required_d_axes(const CoverageCase& c) {
    for (uint32_t i = 0; i < c.required_d_axis_count; ++i) {
        if (!has_d_axis(c.layout, c.required_d_axes[i])) {
            return false;
        }
    }
    return true;
}

bool covers_case(const CoverageCase& c) {
    const bool replica_ok = !c.requires_replica || !c.layout.r_iters.empty();
    const bool offset_ok = !c.requires_offset || !c.layout.o_entries.empty();
    return is_valid_layout(c.layout) && has_required_d_axes(c) && replica_ok && offset_ok;
}

int main() {
    using Axis = level_c::LayoutAxisKind;
    const std::array<CoverageCase, 10> cases = {{
        {"row-major", make_row_major_case(), {Axis::row, Axis::col}, 2, false, false},
        {"tiled-face", make_tiled_face_case(), {Axis::tile_row, Axis::tile_col, Axis::face_row, Axis::face_col}, 4, false, false},
        {"interleaved-bank", make_interleaved_case(), {Axis::storage, Axis::bank, Axis::col}, 3, false, false},
        {"height-sharded", make_height_sharded_case(), {Axis::core_y, Axis::height, Axis::width}, 3, false, false},
        {"width-sharded", make_width_sharded_case(), {Axis::core_x, Axis::height, Axis::width}, 3, false, false},
        {"block-sharded", make_block_sharded_case(), {Axis::core_y, Axis::core_x, Axis::height, Axis::width}, 4, false, false},
        {"nd-sharded", make_nd_sharded_case(), {Axis::core_y, Axis::core_x, Axis::storage, Axis::row, Axis::col, Axis::depth}, 6, false, false},
        {"mesh-replicated", make_mesh_replicated_case(), {Axis::row, Axis::col}, 2, true, false},
        {"mesh-sharded", make_mesh_sharded_case(), {Axis::mesh_y, Axis::row, Axis::mesh_x, Axis::col}, 4, false, false},
        {"offset-view", make_offset_view_case(), {Axis::row, Axis::col}, 2, false, true},
    }};

    bool ok = true;
    for (const auto& c : cases) {
        const bool case_ok = covers_case(c);
        ok = ok && case_ok;
        std::cout << c.name << ": " << (case_ok ? "covered" : "failed") << "\n";
    }

    const auto offset_view = make_offset_view_case();
    const bool accessor_ok = tensor_accessor_linear_offset(offset_view, 3, 7) == 264;
    ok = ok && accessor_ok;
    std::cout << "tensor-accessor-address: " << (accessor_ok ? "covered" : "failed") << "\n";

    if (!ok) {
        std::cerr << "Level C Axe layout coverage sanity failed\n";
        return 1;
    }

    std::cout << "Level C Axe layout coverage sanity passed\n";
    std::cout << "AxeLayout IR uses vectors for D/R and one sparse coordinate O\n";
    std::cout << "TensorLayoutDescriptor=" << sizeof(level_c::lowering::TensorLayoutDescriptor) << " bytes\n";
    return 0;
}
