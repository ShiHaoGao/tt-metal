// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tt-metalium/experimental/published_deployment.hpp>
#include "jit_build/types.hpp"
#include <filesystem>
namespace tt::tt_metal::native_firmware_bundle {
struct SupplierRecipe {
    experimental::PublishedImageKind kind;
    HalProcessorIdentifier processor;
    uint32_t node;
    std::filesystem::path image;
    tt::jit_build::TargetRecipe recipe;
};
struct DeploymentBuildResult {
    experimental::PublishedDeployment deployment;
    std::string compiler;
    std::vector<SupplierRecipe> recipes;
};
// Explicit supplier build, never linked into the runtime library. Uses this
// exact configuration, builds its firmware once and links every planned
// dispatch processor against that same firmware. No catalog/default selection.
DeploymentBuildResult build_deployment(
    const JitDeviceConfig&, const llrt::RunTimeOptions&,
    const experimental::DispatchProgramPlan&);
}
