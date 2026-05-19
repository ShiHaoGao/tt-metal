// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compiler_managed_l1_dataflow/include/level_c_ir.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace level_c = tt::tt_metal::experimental::compiler_managed_l1::level_c;

enum class ModeledLayout {
    row_major,
    tiled,
};

enum class ModeledMemoryLayout {
    interleaved,
    height_sharded,
    width_sharded,
    block_sharded,
    nd_sharded,
    mesh_replicated,
    mesh_sharded,
};

enum class ModeledBufferType {
    dram,
    l1,
    system_memory,
};

struct Placement {
    uint32_t bank = 0;
    uint32_t core_x = 0;
    uint32_t core_y = 0;
    uint32_t page_offset = 0;

    bool operator==(const Placement& other) const = default;
};

struct Shard2DSpec {
    uint32_t pages_per_shard_y = 1;
    uint32_t pages_per_shard_x = 1;
    uint32_t grid_h = 1;
    uint32_t grid_w = 1;
    bool transposed_grid = false;
};

struct NdShardSpecModel {
    std::vector<uint32_t> tensor_shape{};
    std::vector<uint32_t> shard_shape{};
    uint32_t num_banks = 1;
};

struct CaseResult {
    const char* name = "";
    bool covered = false;
};

level_c::AxeAxis axis(std::string name, level_c::LayoutAxisKind kind) {
    return level_c::AxeAxis{std::move(name), kind};
}

level_c::AxeIter iter(uint32_t extent, int32_t stride, std::string name, level_c::LayoutAxisKind kind) {
    return level_c::AxeIter{extent, stride, axis(std::move(name), kind)};
}

level_c::AxeCoordinateEntry offset_entry(int32_t value, std::string name, level_c::LayoutAxisKind kind) {
    return level_c::AxeCoordinateEntry{value, axis(std::move(name), kind)};
}

level_c::AxeMemorySpace to_memory_space(ModeledBufferType buffer_type) {
    switch (buffer_type) {
        case ModeledBufferType::dram: return level_c::AxeMemorySpace::dram;
        case ModeledBufferType::l1: return level_c::AxeMemorySpace::l1;
        case ModeledBufferType::system_memory: return level_c::AxeMemorySpace::system_memory;
    }
    return level_c::AxeMemorySpace::unknown;
}

level_c::AxeStorageKind default_storage_kind(ModeledBufferType buffer_type) {
    if (buffer_type == ModeledBufferType::system_memory) {
        return level_c::AxeStorageKind::borrowed_host_storage;
    }
    return level_c::AxeStorageKind::owned_device_allocation;
}

level_c::AxeStorageOwnership default_storage_ownership(ModeledBufferType buffer_type) {
    if (buffer_type == ModeledBufferType::system_memory) {
        return level_c::AxeStorageOwnership::borrowed;
    }
    return level_c::AxeStorageOwnership::owned;
}

level_c::AxeStorage make_storage(
    std::string name,
    ModeledBufferType buffer_type,
    level_c::AxeStorageKind kind,
    level_c::AxeStorageOwnership ownership,
    uint64_t size_bytes = 0,
    uint64_t base_address = 0,
    std::string root_storage_name = {},
    std::string device_scope = {}) {
    level_c::AxeStorage storage;
    storage.name = std::move(name);
    storage.kind = kind;
    storage.memory_space = to_memory_space(buffer_type);
    storage.ownership = ownership;
    storage.size_bytes = size_bytes;
    storage.base_address = base_address;
    storage.root_storage_name = std::move(root_storage_name);
    storage.device_scope = std::move(device_scope);
    return storage;
}

level_c::AxeStorage make_default_storage(std::string name, ModeledBufferType buffer_type) {
    return make_storage(
        std::move(name),
        buffer_type,
        default_storage_kind(buffer_type),
        default_storage_ownership(buffer_type));
}

bool has_axis_name(const level_c::AxeLayout& layout, std::string_view name) {
    for (const auto& it : layout.d_iters) {
        if (it.axis.name == name) {
            return true;
        }
    }
    for (const auto& it : layout.r_iters) {
        if (it.axis.name == name) {
            return true;
        }
    }
    for (const auto& entry : layout.o_entries) {
        if (entry.axis.name == name) {
            return true;
        }
    }
    return false;
}

const level_c::AxeIter* find_iter(const level_c::AxeLayout& layout, std::string_view name) {
    for (const auto& it : layout.d_iters) {
        if (it.axis.name == name) {
            return &it;
        }
    }
    return nullptr;
}

bool tensor_has_storage(const level_c::AxeTensor& tensor, level_c::AxeMemorySpace memory_space) {
    return !tensor.name.empty() && !tensor.storage.name.empty() && tensor.storage.memory_space == memory_space;
}

bool tensor_has_storage_kind(
    const level_c::AxeTensor& tensor,
    level_c::AxeStorageKind kind,
    level_c::AxeStorageOwnership ownership) {
    return !tensor.name.empty() && !tensor.storage.name.empty() && tensor.storage.kind == kind &&
           tensor.storage.ownership == ownership;
}

int32_t layout_offset(
    const level_c::AxeLayout& layout,
    const std::vector<std::pair<std::string_view, uint32_t>>& coords) {
    int32_t result = 0;
    for (const auto& coord : coords) {
        const auto* it = find_iter(layout, coord.first);
        if (it == nullptr) {
            return -1;
        }
        result += static_cast<int32_t>(coord.second) * it->stride;
    }
    for (const auto& entry : layout.o_entries) {
        result += entry.value;
    }
    return result;
}

level_c::AxeTensor make_tensor(
    std::string name,
    level_c::AxeElementType element_type,
    ModeledBufferType buffer_type,
    level_c::AxeLayout layout) {
    level_c::AxeTensor tensor;
    tensor.name = std::move(name);
    tensor.element_type = element_type;
    tensor.storage = make_default_storage(tensor.name + "_storage", buffer_type);
    tensor.layout = std::move(layout);
    return tensor;
}

level_c::AxeTensor adapt_row_major_tensor(uint32_t height, uint32_t width, ModeledBufferType buffer_type) {
    level_c::AxeLayout layout;
    layout.d_iters = {
        iter(height, static_cast<int32_t>(width), "row", level_c::LayoutAxisKind::row),
        iter(width, 1, "col", level_c::LayoutAxisKind::col),
    };
    return make_tensor("tt_row_major_tensor", level_c::AxeElementType::bfloat16, buffer_type, std::move(layout));
}

level_c::AxeTensor adapt_tiled_tensor(
    uint32_t height,
    uint32_t width,
    uint32_t tile_h,
    uint32_t tile_w,
    uint32_t face_h,
    uint32_t face_w,
    bool transpose_tile,
    ModeledBufferType buffer_type) {
    const uint32_t tile_rows = height / tile_h;
    const uint32_t tile_cols = width / tile_w;
    const uint32_t face_rows = tile_h / face_h;
    const uint32_t face_cols = tile_w / face_w;
    const uint32_t tile_hw = tile_h * tile_w;
    const uint32_t face_hw = face_h * face_w;

    level_c::AxeLayout layout;
    layout.d_iters = {
        iter(tile_rows, static_cast<int32_t>(tile_cols * tile_hw), "tile_row", level_c::LayoutAxisKind::tile_row),
        iter(tile_cols, static_cast<int32_t>(tile_hw), "tile_col", level_c::LayoutAxisKind::tile_col),
        iter(
            face_rows,
            static_cast<int32_t>((transpose_tile ? 1 : face_cols) * face_hw),
            "face_row",
            level_c::LayoutAxisKind::face_row),
        iter(
            face_cols,
            static_cast<int32_t>((transpose_tile ? face_rows : 1) * face_hw),
            "face_col",
            level_c::LayoutAxisKind::face_col),
        iter(face_h, static_cast<int32_t>(transpose_tile ? 1 : face_w), "row_in_face", level_c::LayoutAxisKind::row),
        iter(face_w, static_cast<int32_t>(transpose_tile ? face_h : 1), "col_in_face", level_c::LayoutAxisKind::col),
    };
    return make_tensor("tt_tiled_tensor", level_c::AxeElementType::bfloat16, buffer_type, std::move(layout));
}

level_c::AxeTensor adapt_interleaved_tensor(
    uint32_t page_count,
    uint32_t num_banks,
    ModeledBufferType buffer_type) {
    level_c::AxeLayout layout;
    layout.d_iters = {
        iter((page_count + num_banks - 1) / num_banks, static_cast<int32_t>(num_banks), "bank_page", level_c::LayoutAxisKind::storage),
        iter(num_banks, 1, "bank", level_c::LayoutAxisKind::bank),
        iter(page_count, 1, "page", level_c::LayoutAxisKind::page),
    };
    return make_tensor("tt_interleaved_tensor", level_c::AxeElementType::bfloat16, buffer_type, std::move(layout));
}

Placement tt_interleaved_page(uint32_t page_id, uint32_t num_banks) {
    return Placement{.bank = page_id % num_banks, .page_offset = page_id / num_banks};
}

Placement axe_interleaved_page(const level_c::AxeLayout& layout, uint32_t page_id) {
    const auto* bank = find_iter(layout, "bank");
    if (bank == nullptr || bank->extent == 0) {
        return Placement{.bank = 0xFFFFFFFFu};
    }
    return Placement{.bank = page_id % bank->extent, .page_offset = page_id / bank->extent};
}

level_c::AxeTensor adapt_height_sharded_tensor(const Shard2DSpec& spec) {
    const uint32_t pages_per_shard = spec.pages_per_shard_y * spec.pages_per_shard_x;
    level_c::AxeLayout layout;
    layout.d_iters = {
        iter(spec.grid_w, static_cast<int32_t>(spec.grid_h * pages_per_shard), "core_x", level_c::LayoutAxisKind::core_x),
        iter(spec.grid_h, static_cast<int32_t>(pages_per_shard), "core_y", level_c::LayoutAxisKind::core_y),
        iter(spec.pages_per_shard_y, static_cast<int32_t>(spec.pages_per_shard_x), "local_page_row", level_c::LayoutAxisKind::row),
        iter(spec.pages_per_shard_x, 1, "local_page_col", level_c::LayoutAxisKind::col),
    };
    return make_tensor("tt_height_sharded_tensor", level_c::AxeElementType::bfloat16, ModeledBufferType::l1, std::move(layout));
}

level_c::AxeTensor adapt_width_sharded_tensor(const Shard2DSpec& spec) {
    const uint32_t pages_per_tensor_x = spec.pages_per_shard_x * spec.grid_h * spec.grid_w;
    level_c::AxeLayout layout;
    layout.d_iters = {
        iter(spec.pages_per_shard_y, static_cast<int32_t>(pages_per_tensor_x), "local_page_row", level_c::LayoutAxisKind::row),
        iter(spec.grid_h, static_cast<int32_t>(spec.grid_w * spec.pages_per_shard_x), "core_y", level_c::LayoutAxisKind::core_y),
        iter(spec.grid_w, static_cast<int32_t>(spec.pages_per_shard_x), "core_x", level_c::LayoutAxisKind::core_x),
        iter(spec.pages_per_shard_x, 1, "local_page_col", level_c::LayoutAxisKind::col),
    };
    return make_tensor("tt_width_sharded_tensor", level_c::AxeElementType::bfloat16, ModeledBufferType::l1, std::move(layout));
}

level_c::AxeTensor adapt_block_sharded_tensor(const Shard2DSpec& spec) {
    const uint32_t pages_per_tensor_x = spec.pages_per_shard_x * spec.grid_w;
    level_c::AxeLayout layout;
    layout.d_iters = {
        iter(spec.grid_h, static_cast<int32_t>(spec.pages_per_shard_y * pages_per_tensor_x), "core_y", level_c::LayoutAxisKind::core_y),
        iter(spec.grid_w, static_cast<int32_t>(spec.pages_per_shard_x), "core_x", level_c::LayoutAxisKind::core_x),
        iter(spec.pages_per_shard_y, static_cast<int32_t>(pages_per_tensor_x), "local_page_row", level_c::LayoutAxisKind::row),
        iter(spec.pages_per_shard_x, 1, "local_page_col", level_c::LayoutAxisKind::col),
    };
    return make_tensor("tt_block_sharded_tensor", level_c::AxeElementType::bfloat16, ModeledBufferType::l1, std::move(layout));
}

std::pair<uint32_t, uint32_t> flat_index_to_2d(uint32_t index, uint32_t inner_dim_size) {
    return {index % inner_dim_size, index / inner_dim_size};
}

Placement tt_height_sharded_page(const Shard2DSpec& spec, uint32_t page_id) {
    const uint32_t pages_per_shard = spec.pages_per_shard_y * spec.pages_per_shard_x;
    const uint32_t global_shard_index = page_id / pages_per_shard;
    const uint32_t page_offset = page_id - (global_shard_index * pages_per_shard);
    const auto [inner, outer] = flat_index_to_2d(global_shard_index, spec.transposed_grid ? spec.grid_w : spec.grid_h);
    return Placement{
        .core_x = spec.transposed_grid ? inner : outer,
        .core_y = spec.transposed_grid ? outer : inner,
        .page_offset = page_offset};
}

Placement tt_width_sharded_page(const Shard2DSpec& spec, uint32_t page_id) {
    const uint32_t pages_per_tensor_x = spec.pages_per_shard_x * spec.grid_h * spec.grid_w;
    const uint32_t page_global_outer_dim = page_id / pages_per_tensor_x;
    const uint32_t page_global_inner_dim = page_id - (page_global_outer_dim * pages_per_tensor_x);
    const uint32_t global_shard_index = page_global_inner_dim / spec.pages_per_shard_x;
    const auto [inner, outer] = flat_index_to_2d(global_shard_index, spec.transposed_grid ? spec.grid_h : spec.grid_w);
    const uint32_t page_in_shard_x = page_global_inner_dim - (global_shard_index * spec.pages_per_shard_x);
    const uint32_t page_offset = (page_global_outer_dim * spec.pages_per_shard_x) + page_in_shard_x;
    return Placement{
        .core_x = spec.transposed_grid ? outer : inner,
        .core_y = spec.transposed_grid ? inner : outer,
        .page_offset = page_offset};
}

Placement tt_block_sharded_page(const Shard2DSpec& spec, uint32_t page_id) {
    const uint32_t pages_per_tensor_x = spec.pages_per_shard_x * spec.grid_w;
    const uint32_t page_global_outer_dim = page_id / pages_per_tensor_x;
    const uint32_t page_global_inner_dim = page_id - (page_global_outer_dim * pages_per_tensor_x);
    const uint32_t shard_grid_x = page_global_inner_dim / spec.pages_per_shard_x;
    const uint32_t shard_grid_y = page_global_outer_dim / spec.pages_per_shard_y;
    const uint32_t page_offset_x = page_global_inner_dim - (shard_grid_x * spec.pages_per_shard_x);
    const uint32_t page_offset_y = page_global_outer_dim - (shard_grid_y * spec.pages_per_shard_y);
    return Placement{
        .core_x = spec.transposed_grid ? shard_grid_y : shard_grid_x,
        .core_y = spec.transposed_grid ? shard_grid_x : shard_grid_y,
        .page_offset = page_offset_y * spec.pages_per_shard_x + page_offset_x};
}

Placement axe_height_sharded_page(const level_c::AxeLayout& layout, uint32_t page_id) {
    const auto* core_y = find_iter(layout, "core_y");
    const auto* local_y = find_iter(layout, "local_page_row");
    const auto* local_x = find_iter(layout, "local_page_col");
    const uint32_t pages_per_shard = local_y->extent * local_x->extent;
    const uint32_t global_shard_index = page_id / pages_per_shard;
    return Placement{
        .core_x = global_shard_index / core_y->extent,
        .core_y = global_shard_index % core_y->extent,
        .page_offset = page_id % pages_per_shard};
}

Placement axe_width_sharded_page(const level_c::AxeLayout& layout, uint32_t page_id) {
    const auto* core_x = find_iter(layout, "core_x");
    const auto* core_y = find_iter(layout, "core_y");
    const auto* local_x = find_iter(layout, "local_page_col");
    const uint32_t pages_per_tensor_x = core_y->extent * core_x->extent * local_x->extent;
    const uint32_t page_global_outer_dim = page_id / pages_per_tensor_x;
    const uint32_t page_global_inner_dim = page_id - (page_global_outer_dim * pages_per_tensor_x);
    const uint32_t global_shard_index = page_global_inner_dim / local_x->extent;
    const uint32_t page_in_shard_x = page_global_inner_dim - (global_shard_index * local_x->extent);
    return Placement{
        .core_x = global_shard_index % core_x->extent,
        .core_y = global_shard_index / core_x->extent,
        .page_offset = page_global_outer_dim * local_x->extent + page_in_shard_x};
}

Placement axe_block_sharded_page(const level_c::AxeLayout& layout, uint32_t page_id) {
    const auto* core_x = find_iter(layout, "core_x");
    const auto* local_y = find_iter(layout, "local_page_row");
    const auto* local_x = find_iter(layout, "local_page_col");
    const uint32_t pages_per_tensor_x = core_x->extent * local_x->extent;
    const uint32_t page_global_outer_dim = page_id / pages_per_tensor_x;
    const uint32_t page_global_inner_dim = page_id - (page_global_outer_dim * pages_per_tensor_x);
    const uint32_t shard_grid_x = page_global_inner_dim / local_x->extent;
    const uint32_t shard_grid_y = page_global_outer_dim / local_y->extent;
    const uint32_t page_offset_x = page_global_inner_dim - (shard_grid_x * local_x->extent);
    const uint32_t page_offset_y = page_global_outer_dim - (shard_grid_y * local_y->extent);
    return Placement{
        .core_x = shard_grid_x,
        .core_y = shard_grid_y,
        .page_offset = page_offset_y * local_x->extent + page_offset_x};
}

std::vector<uint32_t> row_major_strides(const std::vector<uint32_t>& shape) {
    std::vector<uint32_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

uint32_t volume(const std::vector<uint32_t>& shape) {
    uint32_t result = 1;
    for (const auto dim : shape) {
        result *= dim;
    }
    return result;
}

std::vector<uint32_t> page_id_to_coord(uint32_t page_id, const std::vector<uint32_t>& shape) {
    std::vector<uint32_t> coord(shape.size(), 0);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        coord[i] = page_id % shape[i];
        page_id /= shape[i];
    }
    return coord;
}

level_c::AxeTensor adapt_nd_sharded_tensor(const NdShardSpecModel& spec) {
    const auto shard_grid_shape = [&] {
        std::vector<uint32_t> result(spec.tensor_shape.size(), 1);
        for (size_t i = 0; i < spec.tensor_shape.size(); ++i) {
            result[i] = (spec.tensor_shape[i] + spec.shard_shape[i] - 1) / spec.shard_shape[i];
        }
        return result;
    }();
    const auto shard_grid_strides = row_major_strides(shard_grid_shape);
    const auto shard_strides = row_major_strides(spec.shard_shape);
    const uint32_t total_shards = volume(shard_grid_shape);

    level_c::AxeLayout layout;
    layout.d_iters.push_back(iter(spec.num_banks, 1, "bank", level_c::LayoutAxisKind::bank));
    layout.d_iters.push_back(iter((total_shards + spec.num_banks - 1) / spec.num_banks, static_cast<int32_t>(spec.num_banks), "bank_shard", level_c::LayoutAxisKind::storage));
    for (size_t i = 0; i < shard_grid_shape.size(); ++i) {
        layout.d_iters.push_back(iter(
            shard_grid_shape[i],
            static_cast<int32_t>(shard_grid_strides[i]),
            "shard_dim_" + std::to_string(i),
            i == 0 ? level_c::LayoutAxisKind::depth : (i == 1 ? level_c::LayoutAxisKind::row : level_c::LayoutAxisKind::col)));
    }
    for (size_t i = 0; i < spec.shard_shape.size(); ++i) {
        layout.d_iters.push_back(iter(
            spec.shard_shape[i],
            static_cast<int32_t>(shard_strides[i]),
            "local_dim_" + std::to_string(i),
            i == 0 ? level_c::LayoutAxisKind::depth : (i == 1 ? level_c::LayoutAxisKind::row : level_c::LayoutAxisKind::col)));
    }
    return make_tensor("tt_nd_sharded_tensor", level_c::AxeElementType::bfloat16, ModeledBufferType::l1, std::move(layout));
}

Placement tt_nd_page(const NdShardSpecModel& spec, uint32_t page_id) {
    const auto shard_grid_shape = [&] {
        std::vector<uint32_t> result(spec.tensor_shape.size(), 1);
        for (size_t i = 0; i < spec.tensor_shape.size(); ++i) {
            result[i] = (spec.tensor_shape[i] + spec.shard_shape[i] - 1) / spec.shard_shape[i];
        }
        return result;
    }();
    const auto shard_grid_strides = row_major_strides(shard_grid_shape);
    const auto shard_strides = row_major_strides(spec.shard_shape);
    const auto coord = page_id_to_coord(page_id, spec.tensor_shape);

    uint32_t flattened_shard_id = 0;
    uint32_t page_offset_within_shard = 0;
    for (size_t i = 0; i < coord.size(); ++i) {
        flattened_shard_id += (coord[i] / spec.shard_shape[i]) * shard_grid_strides[i];
        page_offset_within_shard += (coord[i] % spec.shard_shape[i]) * shard_strides[i];
    }
    return Placement{
        .bank = flattened_shard_id % spec.num_banks,
        .page_offset = (flattened_shard_id / spec.num_banks) * volume(spec.shard_shape) + page_offset_within_shard};
}

Placement axe_nd_page(const level_c::AxeLayout& layout, uint32_t page_id) {
    const auto* bank = find_iter(layout, "bank");
    std::vector<uint32_t> tensor_shape;
    std::vector<uint32_t> shard_shape;
    for (uint32_t i = 0;; ++i) {
        const auto* shard_dim = find_iter(layout, "shard_dim_" + std::to_string(i));
        const auto* local_dim = find_iter(layout, "local_dim_" + std::to_string(i));
        if (shard_dim == nullptr || local_dim == nullptr) {
            break;
        }
        tensor_shape.push_back(shard_dim->extent * local_dim->extent);
        shard_shape.push_back(local_dim->extent);
    }
    return tt_nd_page(NdShardSpecModel{tensor_shape, shard_shape, bank->extent}, page_id);
}

level_c::AxeTensor adapt_mesh_replicated_tensor(uint32_t rows, uint32_t cols, uint32_t mesh_x, uint32_t mesh_y) {
    auto tensor = adapt_row_major_tensor(rows, cols, ModeledBufferType::dram);
    tensor.name = "tt_mesh_replicated_tensor";
    tensor.layout.r_iters = {
        iter(mesh_x, 0, "mesh_x", level_c::LayoutAxisKind::mesh_x),
        iter(mesh_y, 0, "mesh_y", level_c::LayoutAxisKind::mesh_y),
    };
    return tensor;
}

level_c::AxeTensor adapt_mesh_sharded_tensor(uint32_t rows_per_mesh, uint32_t cols_per_mesh, uint32_t mesh_x, uint32_t mesh_y) {
    level_c::AxeLayout layout;
    layout.d_iters = {
        iter(mesh_y, static_cast<int32_t>(rows_per_mesh * cols_per_mesh * mesh_x), "mesh_y", level_c::LayoutAxisKind::mesh_y),
        iter(mesh_x, static_cast<int32_t>(cols_per_mesh), "mesh_x", level_c::LayoutAxisKind::mesh_x),
        iter(rows_per_mesh, static_cast<int32_t>(cols_per_mesh * mesh_x), "row", level_c::LayoutAxisKind::row),
        iter(cols_per_mesh, 1, "col", level_c::LayoutAxisKind::col),
    };
    return make_tensor("tt_mesh_sharded_tensor", level_c::AxeElementType::bfloat16, ModeledBufferType::dram, std::move(layout));
}

bool placements_match(
    const std::vector<uint32_t>& page_ids,
    Placement (*tt_fn)(const Shard2DSpec&, uint32_t),
    Placement (*axe_fn)(const level_c::AxeLayout&, uint32_t),
    const Shard2DSpec& spec,
    const level_c::AxeLayout& layout) {
    for (const auto page_id : page_ids) {
        if (!(tt_fn(spec, page_id) == axe_fn(layout, page_id))) {
            return false;
        }
    }
    return true;
}

void add_result(std::vector<CaseResult>& results, const char* name, bool covered) {
    results.push_back(CaseResult{name, covered});
}

int main() {
    std::vector<CaseResult> layout_results;
    std::vector<CaseResult> resource_results;

    const auto row_major = adapt_row_major_tensor(64, 64, ModeledBufferType::dram);
    add_result(
        layout_results,
        "TensorLayout + RowMajorPageConfig -> AxeTensor/AxeLayout",
        tensor_has_storage(row_major, level_c::AxeMemorySpace::dram) &&
            layout_offset(row_major.layout, {{"row", 3}, {"col", 5}}) == 197);

    const auto tiled = adapt_tiled_tensor(64, 64, 32, 32, 16, 16, false, ModeledBufferType::dram);
    add_result(
        layout_results,
        "TensorLayout + TilePageConfig + default faces -> AxeLayout nested tile/face axes",
        tensor_has_storage(tiled, level_c::AxeMemorySpace::dram) && has_axis_name(tiled.layout, "tile_row") &&
            has_axis_name(tiled.layout, "face_row") &&
            layout_offset(tiled.layout, {{"tile_row", 1}, {"tile_col", 0}, {"face_row", 1}, {"face_col", 0}, {"row_in_face", 2}, {"col_in_face", 3}}) == 2595);

    const auto transposed_tile = adapt_tiled_tensor(64, 64, 16, 32, 16, 16, true, ModeledBufferType::dram);
    add_result(
        layout_results,
        "Tile(transpose_tile=true) -> distinct face/value strides",
        has_axis_name(transposed_tile.layout, "face_col") &&
            layout_offset(transposed_tile.layout, {{"tile_row", 1}, {"tile_col", 0}, {"face_row", 0}, {"face_col", 1}, {"row_in_face", 2}, {"col_in_face", 3}}) == 1330);

    const auto interleaved_dram = adapt_interleaved_tensor(17, 4, ModeledBufferType::dram);
    bool interleaved_dram_ok = tensor_has_storage(interleaved_dram, level_c::AxeMemorySpace::dram);
    for (const auto page_id : {0u, 3u, 4u, 16u}) {
        interleaved_dram_ok = interleaved_dram_ok && tt_interleaved_page(page_id, 4) == axe_interleaved_page(interleaved_dram.layout, page_id);
    }
    add_result(layout_results, "MemoryConfig(INTERLEAVED, DRAM) / InterleavedAddrGen -> bank/page axes", interleaved_dram_ok);

    const auto interleaved_l1 = adapt_interleaved_tensor(9, 2, ModeledBufferType::l1);
    add_result(
        layout_results,
        "MemoryConfig(INTERLEAVED, L1) -> same interleaved layout with L1 storage",
        tensor_has_storage(interleaved_l1, level_c::AxeMemorySpace::l1) &&
            tt_interleaved_page(7, 2) == axe_interleaved_page(interleaved_l1.layout, 7));

    const Shard2DSpec height_spec{.pages_per_shard_y = 3, .pages_per_shard_x = 4, .grid_h = 2, .grid_w = 4};
    const auto height_tensor = adapt_height_sharded_tensor(height_spec);
    add_result(
        layout_results,
        "TensorMemoryLayout::HEIGHT_SHARDED + ShardSpec/ShardSpecBuffer -> core_y/core_x + local page axes",
        placements_match({0, 11, 12, 25, 71, 95}, tt_height_sharded_page, axe_height_sharded_page, height_spec, height_tensor.layout));

    const Shard2DSpec width_spec{.pages_per_shard_y = 3, .pages_per_shard_x = 4, .grid_h = 2, .grid_w = 4};
    const auto width_tensor = adapt_width_sharded_tensor(width_spec);
    add_result(
        layout_results,
        "TensorMemoryLayout::WIDTH_SHARDED + ShardSpec/ShardSpecBuffer -> width-fractured placement",
        placements_match({0, 3, 4, 31, 32, 95}, tt_width_sharded_page, axe_width_sharded_page, width_spec, width_tensor.layout));

    const Shard2DSpec block_spec{.pages_per_shard_y = 2, .pages_per_shard_x = 3, .grid_h = 2, .grid_w = 3};
    const auto block_tensor = adapt_block_sharded_tensor(block_spec);
    add_result(
        layout_results,
        "TensorMemoryLayout::BLOCK_SHARDED + ShardSpec/ShardSpecBuffer -> 2D block placement",
        placements_match({0, 2, 3, 8, 9, 35}, tt_block_sharded_page, axe_block_sharded_page, block_spec, block_tensor.layout));

    const NdShardSpecModel nd_regular{{4, 4, 4}, {2, 2, 2}, 4};
    const auto nd_regular_tensor = adapt_nd_sharded_tensor(nd_regular);
    bool nd_regular_ok = tensor_has_storage(nd_regular_tensor, level_c::AxeMemorySpace::l1);
    for (const auto page_id : {0u, 1u, 7u, 8u, 21u, 42u, 63u}) {
        nd_regular_ok = nd_regular_ok && tt_nd_page(nd_regular, page_id) == axe_nd_page(nd_regular_tensor.layout, page_id);
    }
    add_result(layout_results, "TensorMemoryLayout::ND_SHARDED regular 4x4x4 -> round-robin bank/storage axes", nd_regular_ok);

    const NdShardSpecModel nd_uneven{{4, 4, 3}, {2, 2, 3}, 3};
    const auto nd_uneven_tensor = adapt_nd_sharded_tensor(nd_uneven);
    bool nd_uneven_ok = true;
    for (const auto page_id : {0u, 5u, 6u, 23u, 24u, 47u}) {
        nd_uneven_ok = nd_uneven_ok && tt_nd_page(nd_uneven, page_id) == axe_nd_page(nd_uneven_tensor.layout, page_id);
    }
    add_result(layout_results, "NdShardSpec uneven shard distribution -> bank_shard + bank factorization", nd_uneven_ok);

    const NdShardSpecModel nd_single_dim{{3, 6, 4}, {3, 2, 4}, 4};
    const auto nd_single_dim_tensor = adapt_nd_sharded_tensor(nd_single_dim);
    bool nd_single_dim_ok = true;
    for (const auto page_id : {0u, 7u, 8u, 23u, 24u, 71u}) {
        nd_single_dim_ok = nd_single_dim_ok && tt_nd_page(nd_single_dim, page_id) == axe_nd_page(nd_single_dim_tensor.layout, page_id);
    }
    add_result(layout_results, "NdShardSpec single-dimension sharding -> only sharded axis has shard_dim extent > 1", nd_single_dim_ok);

    const auto mesh_replicated = adapt_mesh_replicated_tensor(64, 128, 2, 2);
    add_result(
        layout_results,
        "MeshBuffer replicated tensor -> R={mesh_x, mesh_y}",
        tensor_has_storage(mesh_replicated, level_c::AxeMemorySpace::dram) && mesh_replicated.layout.r_iters.size() == 2 &&
            has_axis_name(mesh_replicated.layout, "mesh_x") && has_axis_name(mesh_replicated.layout, "mesh_y"));

    const auto mesh_sharded = adapt_mesh_sharded_tensor(32, 64, 2, 2);
    add_result(
        layout_results,
        "MeshBuffer sharded tensor -> mesh placement axes in D",
        tensor_has_storage(mesh_sharded, level_c::AxeMemorySpace::dram) &&
            layout_offset(mesh_sharded.layout, {{"mesh_y", 1}, {"mesh_x", 1}, {"row", 3}, {"col", 7}}) == 4551);

    level_c::AxeLayout offset_layout = row_major.layout;
    offset_layout.o_entries = {offset_entry(65, "storage_offset", level_c::LayoutAxisKind::storage)};
    const auto view_tensor = make_tensor("tt_buffer_view_tensor", level_c::AxeElementType::bfloat16, ModeledBufferType::dram, offset_layout);
    add_result(
        layout_results,
        "Buffer::view / padded-slice style offset -> O sparse coordinate offset",
        tensor_has_storage(view_tensor, level_c::AxeMemorySpace::dram) &&
            layout_offset(view_tensor.layout, {{"row", 3}, {"col", 7}}) == 264);

    auto owned_buffer_tensor = row_major;
    owned_buffer_tensor.name = "tt_buffer_owned_tensor";
    owned_buffer_tensor.storage = make_storage(
        "tt_buffer_owned_storage",
        ModeledBufferType::dram,
        level_c::AxeStorageKind::owned_device_allocation,
        level_c::AxeStorageOwnership::owned,
        8192,
        0x200000,
        {},
        "device0");
    add_result(
        resource_results,
        "Buffer owning allocation -> AxeStorage owned_device_allocation",
        tensor_has_storage_kind(
            owned_buffer_tensor,
            level_c::AxeStorageKind::owned_device_allocation,
            level_c::AxeStorageOwnership::owned) &&
            owned_buffer_tensor.storage.base_address == 0x200000 &&
            owned_buffer_tensor.storage.device_scope == "device0");

    auto view_resource_tensor = view_tensor;
    view_resource_tensor.storage = make_storage(
        "tt_buffer_view_storage",
        ModeledBufferType::dram,
        level_c::AxeStorageKind::view_of_storage,
        level_c::AxeStorageOwnership::view,
        4096,
        0x202000,
        "tt_buffer_owned_storage",
        "device0");
    add_result(
        resource_results,
        "Buffer::view ownership -> AxeStorage view_of_storage + root_storage_name",
        tensor_has_storage_kind(
            view_resource_tensor,
            level_c::AxeStorageKind::view_of_storage,
            level_c::AxeStorageOwnership::view) &&
            view_resource_tensor.storage.root_storage_name == "tt_buffer_owned_storage" &&
            layout_offset(view_resource_tensor.layout, {{"row", 3}, {"col", 7}}) == 264);

    auto mesh_owned_tensor = mesh_sharded;
    mesh_owned_tensor.storage = make_storage(
        "tt_mesh_buffer_storage",
        ModeledBufferType::dram,
        level_c::AxeStorageKind::owned_mesh_allocation,
        level_c::AxeStorageOwnership::owned,
        32768,
        0x400000,
        {},
        "mesh[2x2]");
    add_result(
        resource_results,
        "MeshBuffer owning allocation -> AxeStorage owned_mesh_allocation + mesh device_scope",
        tensor_has_storage_kind(
            mesh_owned_tensor,
            level_c::AxeStorageKind::owned_mesh_allocation,
            level_c::AxeStorageOwnership::owned) &&
            mesh_owned_tensor.storage.device_scope == "mesh[2x2]" &&
            has_axis_name(mesh_owned_tensor.layout, "mesh_x") &&
            has_axis_name(mesh_owned_tensor.layout, "mesh_y"));

    auto borrowed_host_tensor = adapt_row_major_tensor(8, 16, ModeledBufferType::system_memory);
    borrowed_host_tensor.name = "tt_host_tensor";
    borrowed_host_tensor.storage = make_storage(
        "tt_host_borrowed_storage",
        ModeledBufferType::system_memory,
        level_c::AxeStorageKind::borrowed_host_storage,
        level_c::AxeStorageOwnership::borrowed,
        256,
        0,
        {},
        "host");
    add_result(
        resource_results,
        "HostTensor / from_borrowed_data -> AxeStorage borrowed_host_storage",
        tensor_has_storage_kind(
            borrowed_host_tensor,
            level_c::AxeStorageKind::borrowed_host_storage,
            level_c::AxeStorageOwnership::borrowed) &&
            borrowed_host_tensor.storage.memory_space == level_c::AxeMemorySpace::system_memory &&
            layout_offset(borrowed_host_tensor.layout, {{"row", 1}, {"col", 2}}) == 18);

    auto borrowed_device_tensor = row_major;
    borrowed_device_tensor.name = "tt_external_device_tensor";
    borrowed_device_tensor.storage = make_storage(
        "tt_external_device_storage",
        ModeledBufferType::l1,
        level_c::AxeStorageKind::borrowed_device_address,
        level_c::AxeStorageOwnership::borrowed,
        4096,
        0x12000,
        {},
        "device0/core(0,0)");
    add_result(
        resource_results,
        "externally owned device address -> AxeStorage borrowed_device_address",
        tensor_has_storage_kind(
            borrowed_device_tensor,
            level_c::AxeStorageKind::borrowed_device_address,
            level_c::AxeStorageOwnership::borrowed) &&
            borrowed_device_tensor.storage.memory_space == level_c::AxeMemorySpace::l1 &&
            borrowed_device_tensor.storage.base_address == 0x12000);

    level_c::AxeLayout fragment_layout;
    fragment_layout.d_iters = {
        iter(32, 1, "fragment_lane", level_c::LayoutAxisKind::storage),
    };
    level_c::AxeTensor register_fragment;
    register_fragment.name = "tt_register_fragment";
    register_fragment.element_type = level_c::AxeElementType::float32;
    register_fragment.storage = level_c::AxeStorage{
        .name = "tt_register_fragment_storage",
        .kind = level_c::AxeStorageKind::register_fragment,
        .memory_space = level_c::AxeMemorySpace::register_file,
        .ownership = level_c::AxeStorageOwnership::temporary,
        .size_bytes = 128,
        .device_scope = "trisc0"};
    register_fragment.layout = fragment_layout;
    add_result(
        resource_results,
        "compute/register fragment -> AxeStorage register_fragment",
        tensor_has_storage_kind(
            register_fragment,
            level_c::AxeStorageKind::register_fragment,
            level_c::AxeStorageOwnership::temporary) &&
            register_fragment.storage.memory_space == level_c::AxeMemorySpace::register_file &&
            layout_offset(register_fragment.layout, {{"fragment_lane", 7}}) == 7);

    bool ok = true;
    for (const auto& result : layout_results) {
        ok = ok && result.covered;
        std::cout << result.name << ": " << (result.covered ? "covered" : "failed") << "\n";
    }
    for (const auto& result : resource_results) {
        ok = ok && result.covered;
        std::cout << result.name << ": " << (result.covered ? "covered" : "failed") << "\n";
    }

    if (!ok) {
        std::cerr << "Level C TT-Metal layout adapter sanity failed\n";
        return 1;
    }

    std::cout << "Level C TT-Metal layout adapter sanity passed\n";
    std::cout << "layout_covered_cases=" << layout_results.size() << "\n";
    std::cout << "resource_covered_cases=" << resource_results.size() << "\n";
    std::cout << "covered_cases=" << (layout_results.size() + resource_results.size()) << "\n";
    std::cout << "AxeTensor covers element_type + AxeStorage engine + AxeLayout; AxeLayout covers D/R/O layout algebra\n";
    return 0;
}
