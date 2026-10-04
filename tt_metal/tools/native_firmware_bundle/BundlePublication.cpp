// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "BundlePublication.hpp"

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <linux/fs.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/syscall.h>
#include <unistd.h>

namespace tt::tt_metal::native_firmware_bundle {
namespace fs = std::filesystem;

Options parse_options(std::span<const std::string_view> arguments) {
    Options result;
    bool have_output = false;
    bool have_profile = false;
    for (const auto argument : arguments) {
        if (argument.starts_with("--output=") && !have_output) {
            const auto value = argument.substr(std::string_view("--output=").size());
            if (value.empty() || value.find('\0') != std::string_view::npos) {
                throw std::invalid_argument("--output must name a fresh directory");
            }
            result.output = value;
            have_output = true;
        } else if (argument.starts_with("--profile=") && !have_profile) {
            const auto value = argument.substr(std::string_view("--profile=").size());
            if (value == "disabled") {
                result.profile = Profile::Disabled;
            } else if (value == "program") {
                result.profile = Profile::Program;
            } else {
                throw std::invalid_argument("--profile must be disabled or program");
            }
            have_profile = true;
        } else {
            throw std::invalid_argument("unknown or duplicate argument: " + std::string(argument));
        }
    }
    if (!have_output || !have_profile) {
        throw std::invalid_argument("usage: native_firmware_bundle --output=<fresh-dir> --profile=disabled|program");
    }
    return result;
}

Publication::Publication(const fs::path& output) {
    if (output.empty()) {
        throw std::invalid_argument("output directory must not be empty");
    }
    output_ = fs::absolute(output).lexically_normal();
    if (output_.filename().empty()) {
        output_ = output_.parent_path();
    }
    if (fs::exists(fs::symlink_status(output_))) {
        throw std::runtime_error("output already exists: " + output_.string());
    }
    // Resolve the existing parent once. Staging and destination must be on the
    // same filesystem for the final atomic, no-replace rename.
    output_ = fs::canonical(output_.parent_path()) / output_.filename();
    auto pattern = (output_.parent_path() / ".native-firmware-staging_XXXXXX").string();
    const auto* stage = mkdtemp(pattern.data());
    if (stage == nullptr) {
        throw std::system_error(errno, std::generic_category(), "create private firmware staging directory");
    }
    staging_ = stage;
    payload_ = staging_ / "payload";
    try {
        fs::create_directory(payload_);
        fs::create_directory(cache());
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(staging_, ignored);
        throw;
    }
}

Publication::~Publication() {
    std::error_code ignored;
    fs::remove_all(staging_, ignored);
}

void Publication::copy(const fs::path& source, const fs::path& relative) {
    if (committed_) {
        throw std::runtime_error("firmware bundle is already published");
    }
    if (relative.empty() || relative.is_absolute()) {
        throw std::runtime_error("publication destination must be a relative file path");
    }
    for (const auto& component : relative) {
        if (component == ".." || component == "." || component.empty()) {
            throw std::runtime_error("publication destination escapes or does not name a file");
        }
    }
    if (!fs::is_regular_file(fs::symlink_status(source))) {
        throw std::runtime_error("publication source must be a regular non-symlink file: " + source.string());
    }
    const auto destination = payload_ / relative;
    fs::create_directories(destination.parent_path());
    fs::copy_file(source, destination, fs::copy_options::none);
}

void Publication::commit() {
    if (committed_ || fs::is_empty(payload_)) {
        throw std::runtime_error("firmware bundle is empty or already published");
    }
    if (syscall(SYS_renameat2, AT_FDCWD, payload_.c_str(), AT_FDCWD, output_.c_str(), RENAME_NOREPLACE) != 0) {
        throw std::system_error(errno, std::generic_category(), "publish firmware bundle without replacing output");
    }
    committed_ = true;
}
}  // namespace tt::tt_metal::native_firmware_bundle
