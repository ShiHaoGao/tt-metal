// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tt-metalium/experimental/published_deployment.hpp>

namespace tt::tt_metal::native_firmware_bundle {
// Supplier-only publication step. Preserve all load bytes, symbols and
// relocations; append the exact configuration as non-allocatable metadata.
// The complete group must still pass PublishedDeployment::admit afterwards.
std::vector<std::byte> publish_image(
    const experimental::DeploymentConfiguration&, experimental::PublishedImageKind,
    HalProcessorIdentifier, uint32_t dispatch_node, std::span<const std::byte> linked_elf);
}
