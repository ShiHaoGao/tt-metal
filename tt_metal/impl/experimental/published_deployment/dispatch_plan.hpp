// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "configuration.hpp"
#include <map>
#include <string>
#include <string_view>

namespace tt::tt_metal::experimental {
enum class DispatchSourceKind { Prefetch, Dispatcher, Subordinate, SubordinateCompute };
struct DispatchSourceKernel {
    DispatchSourceKind kind;
    std::vector<HalProcessorIdentifier> processors;
    KernelBuildOptLevel opt_level;
    std::map<std::string, std::string> defines;
};
struct DispatchPlan {
    DispatchKernelConfiguration configuration;
    std::vector<DispatchSourceKernel> kernels;
};
// Pure value planning/rendering. The adapter supplies actual SDK configuration;
// this owner never queries devices or contexts, creates kernels, or writes args.
DispatchPlan plan_dispatch_kernel(tt::ARCH, const DispatchKernelConfiguration&);
void validate_published_dispatch_configuration(const DispatchKernelConfiguration&);
std::vector<HalProcessorIdentifier> dispatch_processors(const DispatchKernelConfiguration&);
std::map<std::string, std::string> render_dispatch_common(const DispatchResolvedConfiguration&);
std::string_view dispatch_source_path(DispatchSourceKind);
}  // namespace tt::tt_metal::experimental
