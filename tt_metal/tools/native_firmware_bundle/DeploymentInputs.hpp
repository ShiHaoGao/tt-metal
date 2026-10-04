#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <umd/device/utils/semver.hpp>

namespace tt::tt_metal::native_firmware_bundle {
struct DeploymentInputFacts {
    std::string profile;
    int32_t device_id;
    tt::umd::FirmwareBundleVersion firmware_bundle;
    bool noc_translation_enabled;
    std::string board_type;
    uint64_t board_id;
    uint8_t asic_location;
    uint64_t tensix_harvesting_mask;
    uint64_t dram_harvesting_mask;
    uint64_t eth_harvesting_mask;
    uint64_t pcie_harvesting_mask;
    uint64_t l2cpu_harvesting_mask;
    uint16_t host_channel;
    uint32_t host_channel_size;
    bool host_galaxy;
    bool d2h_hugepage_fallback;
    std::optional<uint32_t> cq_size_override;
    std::vector<uint32_t> active_ethernet_channels;
};
DeploymentInputFacts parse_deployment_input(std::string_view contents);
void require_program_record(std::span<const std::byte> expected, std::span<const std::byte> actual);
}
