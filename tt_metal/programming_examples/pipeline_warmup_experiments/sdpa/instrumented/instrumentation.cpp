// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/programming_examples/pipeline_warmup_experiments/sdpa/instrumented/instrumentation.hpp"

#include <mutex>
#include <stdexcept>

namespace ttnn::prim::sdpa_instrumentation {

namespace {

std::mutex stats_mutex;
Snapshot stats;

}  // namespace

Snapshot& mutable_snapshot() { return stats; }

Snapshot snapshot() {
    std::lock_guard guard(stats_mutex);
    return stats;
}

void reset() {
    std::lock_guard guard(stats_mutex);
    stats = Snapshot{};
}

void add_stage(std::string_view name, double elapsed_us) {
    std::lock_guard guard(stats_mutex);
    if (name == "compute_program_hash") {
        stats.compute_program_hash.add(elapsed_us);
    } else if (name == "create_output_tensors") {
        stats.create_output_tensors.add(elapsed_us);
    } else if (name == "program_factory_create") {
        stats.program_factory_create.add(elapsed_us);
    } else if (name == "override_runtime_arguments") {
        stats.override_runtime_arguments.add(elapsed_us);
    } else {
        throw std::invalid_argument("unknown SDPA instrumentation stage");
    }
}

}  // namespace ttnn::prim::sdpa_instrumentation

