// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental::compiler_managed_l1::level_c {

enum class AxeElementType : uint32_t {
    invalid = 0,
    bfloat16 = 1,
    float32 = 2,
    uint32 = 3,
    int32 = 4,
    uint16 = 5,
    uint8 = 6,
};

enum class AxeMemorySpace : uint32_t {
    unknown = 0,
    l1 = 1,
    dram = 2,
    system_memory = 3,
    register_file = 4,
};

enum class AxeStorageKind : uint32_t {
    unknown = 0,
    owned_device_allocation = 1,
    owned_mesh_allocation = 2,
    borrowed_host_storage = 3,
    borrowed_device_address = 4,
    view_of_storage = 5,
    register_fragment = 6,
};

enum class AxeStorageOwnership : uint32_t {
    unknown = 0,
    owned = 1,
    borrowed = 2,
    view = 3,
    temporary = 4,
};

enum class LayoutAxisKind : uint32_t {
    none = 0,
    row = 1,
    col = 2,
    height = 3,
    width = 4,
    depth = 5,
    page = 6,
    tile_row = 7,
    tile_col = 8,
    face_row = 9,
    face_col = 10,
    bank = 11,
    core_x = 12,
    core_y = 13,
    mesh_x = 14,
    mesh_y = 15,
    storage = 16,
};

struct AxeAxis {
    std::string name{};
    LayoutAxisKind kind = LayoutAxisKind::none;
};

struct AxeIter {
    uint32_t extent = 1;
    int32_t stride = 0;
    AxeAxis axis{};
};

struct AxeCoordinateEntry {
    int32_t value = 0;
    AxeAxis axis{};
};

// Compiler IR form of Axe layout. D carries the ordered iteration domain and
// shard mapping, R carries replica dimensions, and O is one fixed coordinate
// offset represented by sparse axis-value entries. This is not a device ABI.
struct AxeLayout {
    std::vector<AxeIter> d_iters{};
    std::vector<AxeIter> r_iters{};
    std::vector<AxeCoordinateEntry> o_entries{};
};

// CuTe-style storage engine for a compiler tensor. It can describe an owning
// allocation, a borrowed pointer/address, a view, or a register fragment. A
// base_address of 0 means the storage is still symbolic at compiler IR level.
struct AxeStorage {
    std::string name{};
    AxeStorageKind kind = AxeStorageKind::unknown;
    AxeMemorySpace memory_space = AxeMemorySpace::unknown;
    AxeStorageOwnership ownership = AxeStorageOwnership::unknown;
    uint64_t size_bytes = 0;
    uint64_t base_address = 0;
    std::string root_storage_name{};
    std::string device_scope{};
};

struct AxeTensor {
    std::string name{};
    AxeElementType element_type = AxeElementType::invalid;
    AxeStorage storage{};
    AxeLayout layout{};
};

}  // namespace tt::tt_metal::experimental::compiler_managed_l1::level_c
