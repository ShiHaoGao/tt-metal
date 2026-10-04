// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <span>
#include <string_view>

namespace tt::tt_metal::native_firmware_bundle {

enum class Profile { Disabled, Program };

struct Options {
    std::filesystem::path output;
    Profile profile = Profile::Disabled;
};

Options parse_options(std::span<const std::string_view> arguments);

// Offline supplier publication only. A failed build removes its private cache
// and staged output; the public output appears only after commit().
class Publication {
public:
    explicit Publication(const std::filesystem::path& output);
    ~Publication();
    Publication(const Publication&) = delete;
    Publication& operator=(const Publication&) = delete;
    const std::filesystem::path& payload() const { return payload_; }
    std::filesystem::path cache() const { return staging_ / "cache"; }
    void copy(const std::filesystem::path& source, const std::filesystem::path& relative);
    void commit();

private:
    std::filesystem::path output_;
    std::filesystem::path staging_;
    std::filesystem::path payload_;
    bool committed_ = false;
};

}  // namespace tt::tt_metal::native_firmware_bundle
