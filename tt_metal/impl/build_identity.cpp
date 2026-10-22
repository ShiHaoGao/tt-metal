// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/build_identity.hpp"
#include "tt_metal/impl/version.hpp"

namespace tt::tt_metal {

BuildIdentity GetBuildIdentity() noexcept {
    // Constant evaluation keeps the compiled library's values independent of
    // any consumer's inline version variables or compile definitions.
    static constexpr BuildIdentity identity{
        .git_sha = kGitSha,
        .version = kVersionFull,
        .build_type = kBuildType,
        .git_dirty = kGitDirty,
    };
    return identity;
}

}  // namespace tt::tt_metal
