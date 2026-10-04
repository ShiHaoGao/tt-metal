// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "SourceArchive.hpp"
#include "jit_build/depend.hpp"
#include "jit_build/jit_build_utils.hpp"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace tt::tt_metal::native_firmware_bundle {
namespace {
namespace fs = std::filesystem;
fs::path within(const fs::path& path, const fs::path& root) {
    const auto relative = path.lexically_relative(root);
    if (relative.empty() || relative.is_absolute()) return {};
    for (const auto& part : relative) if (part == "..") return {};
    return relative;
}
std::string read(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read supplier source: " + path.string());
    std::string bytes(std::istreambuf_iterator<char>(input), {});
    if (input.bad()) throw std::runtime_error("cannot read supplier source: " + path.string());
    return bytes;
}
}
SourceArchive::SourceArchive(Publication& publication, fs::path sdk) :
    publication_(publication), sdk_(fs::canonical(sdk)) {}
fs::path SourceArchive::capture(const fs::path& source) {
    const auto actual = fs::canonical(source);
    if (const auto found = sources_.find(actual); found != sources_.end()) {
        if (read(actual) != read(publication_.payload() / found->second))
            throw std::runtime_error("supplier input changed during publication: " + source.string());
        return found->second;
    }
    auto relative = within(actual, sdk_);
    fs::path destination;
    if (!relative.empty()) destination = fs::path("provenance/source") / relative;
    else if (!(relative = within(actual, publication_.cache())).empty())
        destination = fs::path("provenance/generated") / relative;
    else destination = fs::path("provenance/external") / actual.relative_path();
    publication_.copy(actual, destination);
    sources_.emplace(actual, destination);
    records.push_back({{"requested_path", source.string()}, {"resolved_path", actual.string()},
        {"snapshot", destination.generic_string()}, {"bytes", fs::file_size(actual)}});
    return destination;
}
void SourceArchive::capture_tree(const fs::path& directory) {
    if (!fs::is_directory(directory))
        throw std::runtime_error("required supplier source directory is absent: " + directory.string());
    std::vector<fs::path> paths;
    for (const auto& entry : fs::recursive_directory_iterator(directory))
        if (entry.is_regular_file()) paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths) capture(path);
}
void SourceArchive::capture_dependencies(const fs::path& record) {
    if (!fs::is_regular_file(fs::symlink_status(record)))
        throw std::runtime_error("required supplier dependency record is absent: " + record.string());
    // The SDK owns both the dependency encoding and content hash. Clear its
    // metadata cache so a same-size write with restored mtime is not admitted.
    jit_build::clear_file_hash_cache();
    if (!jit_build::dependencies_up_to_date_file(record.string()))
        throw std::runtime_error("supplier dependency content differs from compiled input: " + record.string());
    const auto snapshot = capture(record);
    std::ifstream input(publication_.payload() / snapshot);
    fs::path dependency;
    uint64_t recorded_hash;
    while (input >> dependency) {
        if (!(input >> recorded_hash) || !dependency.is_absolute())
            throw std::runtime_error("invalid supplier dependency record: " + record.string());
        // The value was checked by the SDK above; it does not identify an
        // owner or replace the actual immutable source bytes.
        (void)recorded_hash;
        capture(dependency);
    }
    if (!input.eof())
        throw std::runtime_error("cannot read supplier dependency record: " + record.string());
    jit_build::clear_file_hash_cache();
    if (!jit_build::dependencies_up_to_date_file((publication_.payload() / snapshot).string()))
        throw std::runtime_error("supplier dependency changed during capture: " + record.string());
    dependency_snapshots_.push_back(snapshot);
}
void SourceArchive::capture_recipe(const tt::jit_build::TargetRecipe& recipe, const fs::path& image) {
    if (!image.is_absolute() || !fs::is_regular_file(fs::symlink_status(image)) || recipe.objs.empty() ||
        recipe.objs.size() != recipe.srcs.size())
        throw std::runtime_error("supplier recipe lacks a complete image/object inventory");
    for (const auto& object : recipe.objs) {
        const fs::path relative(object);
        if (relative.empty() || relative.has_parent_path() || relative.is_absolute())
            throw std::runtime_error("supplier recipe object must be an output basename");
        capture_dependencies(image.parent_path() / (object + ".dephash"));
    }
    capture_dependencies(image.string() + ".dephash");
    for (const auto& source : recipe.srcs) capture(source);
    if (!recipe.pch_umbrella.empty()) capture(recipe.pch_umbrella);
    if (!recipe.linker_script.empty()) capture(recipe.linker_script);
    for (const auto& object : jit_build::utils::tokenize_flags(recipe.extra_link_objs)) capture(object);
}
void SourceArchive::verify() const {
    for (const auto& [source, snapshot] : sources_)
        if (read(source) != read(publication_.payload() / snapshot))
            throw std::runtime_error("supplier input changed during publication: " + source.string());
    jit_build::clear_file_hash_cache();
    for (const auto& snapshot : dependency_snapshots_)
        if (!jit_build::dependencies_up_to_date_file((publication_.payload() / snapshot).string()))
            throw std::runtime_error("supplier dependency differs from compiled input before publication: " + snapshot.string());
}
}
