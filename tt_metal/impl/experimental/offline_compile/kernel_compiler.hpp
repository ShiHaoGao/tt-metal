// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
namespace tt::tt_metal {
class Kernel;
class BuildEnvManager;
struct DeviceBuildEnv;
class JitBuildOptions;
class Hal;
namespace experimental::offline_detail {
// Shared supplier/offline compilation leaf. The manager is local and its
// firmware must already be built for these exact inputs. No context lookup.
void generate_kernel_binaries(
    const std::shared_ptr<Kernel>&, BuildEnvManager&, const DeviceBuildEnv&, JitBuildOptions&, const Hal&);
}
}
