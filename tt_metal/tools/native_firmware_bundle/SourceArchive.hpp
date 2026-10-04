// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "BundlePublication.hpp"
#include "jit_build/types.hpp"
#include <map>
#include <nlohmann/json.hpp>

namespace tt::tt_metal::native_firmware_bundle {
// Supplier evidence only. These snapshots never substitute for image admission.
class SourceArchive {
public:
    SourceArchive(Publication&, std::filesystem::path sdk);
    std::filesystem::path capture(const std::filesystem::path&);
    void capture_tree(const std::filesystem::path&);
    void capture_recipe(const tt::jit_build::TargetRecipe&, const std::filesystem::path& image);
    void verify() const;
    nlohmann::json records = nlohmann::json::array();
private:
    void capture_dependencies(const std::filesystem::path&);
    std::vector<std::filesystem::path> dependency_snapshots_;
    Publication& publication_;
    std::filesystem::path sdk_;
    std::map<std::filesystem::path, std::filesystem::path> sources_;
};
}
