// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <string_view>

namespace ttnn::prim::sdpa_instrumentation {

struct StageStats {
    uint64_t calls = 0;
    double total_us = 0.0;
    double max_us = 0.0;

    void add(double elapsed_us) {
        ++calls;
        total_us += elapsed_us;
        if (elapsed_us > max_us) {
            max_us = elapsed_us;
        }
    }
};

struct Snapshot {
    StageStats compute_program_hash;
    StageStats create_output_tensors;
    StageStats program_factory_create;
    StageStats override_runtime_arguments;
};

Snapshot& mutable_snapshot();
Snapshot snapshot();
void reset();
void add_stage(std::string_view name, double elapsed_us);

class ScopedStageTimer {
public:
    explicit ScopedStageTimer(std::string_view name) : name_(name), start_(std::chrono::steady_clock::now()) {}

    ~ScopedStageTimer() {
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_us = std::chrono::duration<double, std::micro>(end - start_).count();
        add_stage(name_, elapsed_us);
    }

private:
    std::string_view name_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace ttnn::prim::sdpa_instrumentation

