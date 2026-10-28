// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Real-time profiler L1 layout for the block carved by DispatchMemMap
// (CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG). Not part of mailboxes_t.
//
// Consumed by tt_metal/llrt/hal/codegen/codegen.sh (same rules as fabric_telemetry_msgs.h:
// structs, enums, constants, 1-D arrays only).

#pragma once

#include <cstdint>

enum RealtimeProfilerState : uint32_t {
    REALTIME_PROFILER_STATE_IDLE = 0,       // Waiting for initialization, skip iteration
    REALTIME_PROFILER_STATE_PUSH_A = 1,     // Push real-time profiler data from buffer A
    REALTIME_PROFILER_STATE_PUSH_B = 2,     // Push real-time profiler data from buffer B
    REALTIME_PROFILER_STATE_TERMINATE = 3,  // Signal to terminate the kernel
};

struct realtime_profiler_timestamp_t {
    uint32_t time_hi;
    uint32_t time_lo;
    uint32_t id;
    uint32_t header;
};

// One qualified dispatch-tile counter interval, exactly one D2H socket page.
// State is published last by the same-tile completion monitor. Legacy A/B
// timestamp pairs never carry this protocol marker.
struct realtime_profiler_dispatch_timing_t {
    uint32_t start_hi;
    uint32_t start_lo;
    uint32_t runtime_id;
    uint32_t marker;
    uint32_t end_hi;
    uint32_t end_lo;
    uint32_t occurrence_lo;
    uint32_t occurrence_hi;
    uint32_t stream;
    uint32_t expected_count;
    uint32_t observed_count;
    uint32_t state;
    uint32_t dropped;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};

struct realtime_profiler_msg_t {
    volatile uint32_t config_buffer_addr;
    volatile uint32_t realtime_profiler_state;
    volatile uint32_t realtime_profiler_core_noc_xy;
    volatile uint32_t realtime_profiler_remote_state_addr;  // L1 addr on profiler tensix for state NOC writes
    struct realtime_profiler_timestamp_t kernel_start_a;
    struct realtime_profiler_timestamp_t kernel_end_a;
    struct realtime_profiler_timestamp_t kernel_start_b;
    struct realtime_profiler_timestamp_t kernel_end_b;
    volatile uint32_t sync_request;
    volatile uint32_t sync_host_timestamp;
    volatile uint32_t program_id_fifo[32];
    volatile uint32_t program_occurrence_lo_fifo[32];
    volatile uint32_t program_occurrence_hi_fifo[32];
    volatile uint32_t program_worker_count_fifo[32];
    volatile uint32_t program_id_fifo_start;
    volatile uint32_t program_id_fifo_end;
    volatile uint32_t dispatch_timing_published;
    volatile uint32_t dispatch_timing_completed;
    volatile uint32_t dispatch_timing_consumed;
    volatile uint32_t dispatch_timing_dropped;
    struct realtime_profiler_dispatch_timing_t dispatch_timing_records[32];
};
