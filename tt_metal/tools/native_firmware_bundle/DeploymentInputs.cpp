#include "DeploymentInputs.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace tt::tt_metal::native_firmware_bundle {
namespace {
using Json = nlohmann::json;
void keys(const Json& value, std::initializer_list<std::string_view> expected) {
    if (!value.is_object() || value.size() != expected.size())
        throw std::invalid_argument("incomplete or extended deployment input object");
    for (const auto key : expected)
        if (!value.contains(std::string(key)))
            throw std::invalid_argument("missing deployment input field: " + std::string(key));
}
template<class T> T number(const Json& value) {
    if (!value.is_number_integer() || (value.is_number_integer() && !value.is_number_unsigned() && value.get<int64_t>() < 0))
        throw std::invalid_argument("deployment input requires a nonnegative integer");
    const auto v = value.get<uint64_t>();
    if (v > std::numeric_limits<T>::max()) throw std::invalid_argument("deployment input integer overflows field");
    return static_cast<T>(v);
}
bool boolean(const Json& value) {
    if (!value.is_boolean()) throw std::invalid_argument("deployment input requires a boolean");
    return value.get<bool>();
}
std::string string(const Json& value) {
    if (!value.is_string()) throw std::invalid_argument("deployment input requires a string");
    return value.get<std::string>();
}
}
DeploymentInputFacts parse_deployment_input(std::string_view contents) {
    const auto value = Json::parse(contents);
    keys(value, {"format", "architecture", "profile", "device_id", "firmware_bundle", "chip", "host_queue", "active_ethernet_channels"});
    if (string(value.at("format")) != "tt-metal-native-deployment-input-v2" ||
        string(value.at("architecture")) != "blackhole")
        throw std::invalid_argument("unsupported deployment input format or architecture");
    DeploymentInputFacts result{};
    result.profile = string(value.at("profile"));
    if (result.profile != "disabled" && result.profile != "program")
        throw std::invalid_argument("unsupported supplier observation profile");
    result.device_id = number<int32_t>(value.at("device_id"));
    const auto& firmware = value.at("firmware_bundle");
    keys(firmware, {"major", "minor", "patch", "pre_release"});
    // UMD firmware tags own four byte-wide version components. Do not accept
    // missing/ambient versions or infer capabilities from board/product names.
    result.firmware_bundle = tt::umd::FirmwareBundleVersion(
        number<uint8_t>(firmware.at("major")), number<uint8_t>(firmware.at("minor")),
        number<uint8_t>(firmware.at("patch")), number<uint8_t>(firmware.at("pre_release")));
    const auto& chip = value.at("chip");
    keys(chip, {"noc_translation_enabled", "board_type", "board_id", "asic_location", "harvesting_masks"});
    result.noc_translation_enabled = boolean(chip.at("noc_translation_enabled"));
    if (!result.noc_translation_enabled) throw std::invalid_argument("Blackhole supplier requires translated NoC facts");
    result.board_type = string(chip.at("board_type"));
    if (result.board_type != "p150" && result.board_type != "p300" &&
        result.board_type != "ubb_blackhole" && result.board_type != "ubb_blackhole_bin6")
        throw std::invalid_argument("unsupported Blackhole board type");
    result.board_id = number<uint64_t>(chip.at("board_id"));
    result.asic_location = number<uint8_t>(chip.at("asic_location"));
    const auto& masks = chip.at("harvesting_masks");
    keys(masks, {"tensix", "dram", "eth", "pcie", "l2cpu"});
    result.tensix_harvesting_mask = number<uint64_t>(masks.at("tensix"));
    result.dram_harvesting_mask = number<uint64_t>(masks.at("dram"));
    result.eth_harvesting_mask = number<uint64_t>(masks.at("eth"));
    result.pcie_harvesting_mask = number<uint64_t>(masks.at("pcie"));
    result.l2cpu_harvesting_mask = number<uint64_t>(masks.at("l2cpu"));
    const auto& host = value.at("host_queue");
    keys(host, {"channel", "channel_size", "galaxy", "d2h_hugepage_fallback", "cq_size_override"});
    result.host_channel = number<uint16_t>(host.at("channel"));
    result.host_channel_size = number<uint32_t>(host.at("channel_size"));
    result.host_galaxy = boolean(host.at("galaxy"));
    result.d2h_hugepage_fallback = boolean(host.at("d2h_hugepage_fallback"));
    if (result.host_channel_size == 0 || result.host_galaxy)
        throw std::invalid_argument("unsupported host queue backing");
    if (!host.at("cq_size_override").is_null()) {
        result.cq_size_override = number<uint32_t>(host.at("cq_size_override"));
        if (!*result.cq_size_override) throw std::invalid_argument("zero CQ size override");
    }
    const auto& channels = value.at("active_ethernet_channels");
    if (!channels.is_array() || channels.size() > 64)
        throw std::invalid_argument("invalid active Ethernet channel list");
    std::set<uint32_t> unique;
    for (const auto& channel : channels) {
        const auto id = number<uint32_t>(channel);
        if (!unique.insert(id).second) throw std::invalid_argument("duplicate active Ethernet channel");
        result.active_ethernet_channels.push_back(id);
    }
    return result;
}
void require_program_record(std::span<const std::byte> expected, std::span<const std::byte> actual) {
    if (expected.empty() || !std::ranges::equal(expected, actual))
        throw std::invalid_argument("explicit complete dispatch program record differs from SDK-derived plan");
}
}
