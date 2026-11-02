// Candidate CPU build input helper. This is a catalog expectation, not a live board snapshot.
#pragma once
#include "impl/experimental/published_deployment/dispatch_program_adapter.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/rtoptions.hpp"
#include <umd/device/soc_descriptor.hpp>
#include <filesystem>
namespace tt::tt_metal::experimental::catalog {
struct Input {
    std::filesystem::path sdk_root;
    tt::ARCH architecture;
    // One explicitly selected MMIO chip. This catalog does not discover or
    // admit the N300 board's second chip or an Ethernet-serviced topology.
    ChipInfo chip;
    ChipId device_id;
    HostQueueBacking host;
    std::vector<uint32_t> active_ethernet_channels;
};
struct Output {
    JitDeviceConfig device;
    DispatchProgramInputs program;
};
Output make_input(const Input&, const Hal&, llrt::RunTimeOptions&);
}  // namespace tt::tt_metal::experimental::catalog
