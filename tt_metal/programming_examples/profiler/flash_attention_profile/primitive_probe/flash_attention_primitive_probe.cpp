// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {

enum class Primitive : uint32_t {
    DataflowSemInc = 1,
    ManualMailboxBridge = 2,
};

struct Options {
    Primitive primitive = Primitive::DataflowSemInc;
    uint32_t max_loops = 100000000;
    int device_id = 0;
};

std::string_view primitive_name(Primitive primitive) {
    switch (primitive) {
        case Primitive::DataflowSemInc: return "dataflow_sem_inc";
        case Primitive::ManualMailboxBridge: return "manual_mailbox_bridge";
    }
    return "unknown";
}

std::optional<Primitive> parse_primitive(std::string_view value) {
    if (value == "dataflow_sem_inc") {
        return Primitive::DataflowSemInc;
    }
    if (value == "manual_mailbox_bridge") {
        return Primitive::ManualMailboxBridge;
    }
    return std::nullopt;
}

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} [--primitive dataflow_sem_inc|manual_mailbox_bridge] [--max-loops N] [--device-id N]\n",
        argv0);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        auto require_value = [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                TT_THROW("{} requires a value", name);
            }
            return std::string_view(argv[++i]);
        };
        if (arg == "--primitive") {
            auto primitive = parse_primitive(require_value(arg));
            if (!primitive.has_value()) {
                print_usage(argv[0]);
                TT_THROW("unknown primitive");
            }
            options.primitive = *primitive;
        } else if (arg == "--max-loops") {
            options.max_loops = static_cast<uint32_t>(std::stoul(std::string(require_value(arg))));
        } else if (arg == "--device-id") {
            options.device_id = std::stoi(std::string(require_value(arg)));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            print_usage(argv[0]);
            TT_THROW("unknown argument {}", arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    bool pass = true;
    try {
        const Options options = parse_options(argc, argv);
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(options.device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        const CoreCoord producer_core = mesh_device->allocator()->get_logical_core_from_bank_id(0);
        const CoreCoord remote_consumer_core =
            producer_core == CoreCoord{0, 0} ? CoreCoord{0, 1} : CoreCoord{0, 0};
        const CoreCoord consumer_core = remote_consumer_core;
        const auto consumer_physical = mesh_device->worker_core_from_logical_core(consumer_core);

        const CoreRangeSet sem_cores(std::vector<CoreRange>{CoreRange(producer_core), CoreRange(remote_consumer_core)});
        const uint32_t sem_id = CreateSemaphore(program, sem_cores, 0);

        constexpr uint32_t result_bytes = 32;
        constexpr uint32_t mailbox_bytes = 32;
        constexpr uint32_t mailbox_magic = 0xfa310001u;
        distributed::DeviceLocalBufferConfig result_dram_config{
            .page_size = result_bytes, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig mailbox_l1_config{.page_size = mailbox_bytes, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig result_buffer_config{.size = result_bytes};
        distributed::ReplicatedBufferConfig mailbox_buffer_config{.size = mailbox_bytes};
        auto result_buffer = distributed::MeshBuffer::create(result_buffer_config, result_dram_config, mesh_device.get());
        auto mailbox_buffer = distributed::MeshBuffer::create(mailbox_buffer_config, mailbox_l1_config, mesh_device.get());

        std::vector<uint32_t> mailbox_init(mailbox_bytes / sizeof(uint32_t), 0);
        distributed::EnqueueWriteMeshBuffer(cq, mailbox_buffer, mailbox_init, false);

        constexpr uint32_t result_cb = CBIndex::c_0;
        CircularBufferConfig result_cb_config(result_bytes, {{result_cb, tt::DataFormat::UInt32}});
        result_cb_config = result_cb_config.set_page_size(result_cb, result_bytes);
        CreateCircularBuffer(program, CoreRange(consumer_core), result_cb_config);

        KernelHandle consumer_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/flash_attention_profile/primitive_probe/kernels/dataflow/"
            "consumer_poll.cpp",
            consumer_core,
            ReaderDataMovementConfig{{result_cb}});
        SetRuntimeArgs(
            program,
            consumer_kernel,
            consumer_core,
            {result_buffer->address(),
             0,
             sem_id,
             1,
             options.max_loops,
             static_cast<uint32_t>(options.primitive)});

        switch (options.primitive) {
            case Primitive::DataflowSemInc: {
                KernelHandle producer_kernel = CreateKernel(
                    program,
                    "tt_metal/programming_examples/profiler/flash_attention_profile/primitive_probe/kernels/dataflow/"
                    "producer_sem_inc.cpp",
                    producer_core,
                    WriterDataMovementConfig{});
                SetRuntimeArgs(
                    program, producer_kernel, producer_core, {consumer_physical.x, consumer_physical.y, sem_id, 1});
                break;
            }
            case Primitive::ManualMailboxBridge: {
                KernelHandle compute_kernel = CreateKernel(
                    program,
                    "tt_metal/programming_examples/profiler/flash_attention_profile/primitive_probe/kernels/compute/"
                    "producer_manual_mailbox_compute.cpp",
                    producer_core,
                    ComputeConfig{.compile_args = {static_cast<uint32_t>(mailbox_buffer->address()), mailbox_magic}});
                SetRuntimeArgs(program, compute_kernel, producer_core, {});

                KernelHandle bridge_kernel = CreateKernel(
                    program,
                    "tt_metal/programming_examples/profiler/flash_attention_profile/primitive_probe/kernels/dataflow/"
                    "producer_manual_mailbox_bridge.cpp",
                    producer_core,
                    WriterDataMovementConfig{});
                SetRuntimeArgs(
                    program,
                    bridge_kernel,
                    producer_core,
                    {static_cast<uint32_t>(mailbox_buffer->address()),
                     mailbox_magic,
                     options.max_loops,
                     consumer_physical.x,
                     consumer_physical.y,
                     sem_id,
                     1});
                break;
            }
        }

        auto start = std::chrono::steady_clock::now();
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        auto end = std::chrono::steady_clock::now();

        std::vector<uint32_t> result;
        distributed::EnqueueReadMeshBuffer(cq, result, result_buffer, true);
        result.resize(8, 0);
        std::vector<uint32_t> mailbox_result;
        distributed::EnqueueReadMeshBuffer(cq, mailbox_result, mailbox_buffer, true);
        mailbox_result.resize(mailbox_bytes / sizeof(uint32_t), 0);

        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
        const bool seen = result[0] == 1;
        const bool mailbox_ok = options.primitive != Primitive::ManualMailboxBridge ||
                                (mailbox_result[0] == mailbox_magic && mailbox_result[1] == 1 &&
                                 mailbox_result[4] == 0xfa310002u);
        pass = seen && result[7] == 0xface0001u && mailbox_ok;

        fmt::print(
            "FLASH_ATTN_PRIMITIVE_PROBE_RESULT primitive={} pass={} seen={} observed={} loops={} primitive_id={} "
            "sem_addr=0x{:x} expected={} max_loops={} magic=0x{:x} mailbox0=0x{:x} mailbox1={} mailbox2={} "
            "mailbox3={} mailbox4=0x{:x} elapsed_ms={:.3f}\n",
            primitive_name(options.primitive),
            pass ? "true" : "false",
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[7],
            mailbox_result[0],
            mailbox_result[1],
            mailbox_result[2],
            mailbox_result[3],
            mailbox_result[4],
            elapsed_ms);

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }

    if (!pass) {
        TT_THROW("flash_attention_primitive_probe failed");
    }
    return 0;
}
