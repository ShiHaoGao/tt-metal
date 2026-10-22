// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>

namespace tt::tt_metal {

struct BuildIdentity {
    std::string_view git_sha;
    std::string_view version;
    std::string_view build_type;
    bool git_dirty;
};

// Observe the linked SDK build without filesystem, context, or device access.
// String views remain valid while the SDK library is loaded.
BuildIdentity GetBuildIdentity() noexcept;

}  // namespace tt::tt_metal
