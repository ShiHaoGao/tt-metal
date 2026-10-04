// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "dispatch_program_plan.hpp"
namespace tt::tt_metal {
class Device;
class ContextDescriptor;
class DispatchMemMap;
class Hal;
struct DispatchKernelNode;
namespace experimental {
DispatchMemoryFacts capture_dispatch_memory(const DispatchMemMap&, uint8_t cq);
DispatchHalFacts capture_dispatch_hal(const Hal&);
// Actual placements are supplied by the topology after its owning core manager
// has assigned them. This adapter does not instantiate FDKernel/Program offline.
DispatchProgramInputs capture_dispatch_program(
    tt::tt_metal::Device&, const ContextDescriptor&, std::span<const DispatchKernelNode>,
    std::span<const tt_cxy_pair> placed_nodes, tt_cxy_pair completion_writer,
    uint32_t virtual_eth_cores, bool reads_dispatch_cores);
// Applies only the admitted plan's explicit writes; no compilation or layout decisions.
void initialize_dispatch_program(tt::tt_metal::Device&, const DispatchProgramPlan&);
} // namespace experimental
} // namespace tt::tt_metal
