// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>

class MatmulProfileRunner {
public:
    virtual ~MatmulProfileRunner() = default;

    virtual void run(
        const std::vector<bfloat16>& a,
        const std::vector<bfloat16>& b,
        std::vector<bfloat16>& output) = 0;

    virtual void prepare_trace(
        const std::vector<bfloat16>& a,
        const std::vector<bfloat16>& b,
        std::vector<bfloat16>& output,
        bool drain_profiler) = 0;

    virtual bool trace_ready() const = 0;
    virtual void release_trace() = 0;
};

std::unique_ptr<MatmulProfileRunner> prepare_matmul_multi_core_profile_runner(
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device);

std::unique_ptr<MatmulProfileRunner> prepare_matmul_multicore_reuse_profile_runner(
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t B,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device);

std::unique_ptr<MatmulProfileRunner> prepare_matmul_multicore_reuse_mcast_profile_runner(
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t B,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device);
