#include "DeploymentInputs.hpp"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace tt::tt_metal::native_firmware_bundle {
namespace {
using Json = nlohmann::json;
Json facts() {
    return {{"format", "tt-metal-native-deployment-input-v2"}, {"architecture", "blackhole"},
        {"profile", "disabled"}, {"device_id", 0},
        {"firmware_bundle", {{"major", 19}, {"minor", 5}, {"patch", 0}, {"pre_release", 0}}},
        {"chip", {{"noc_translation_enabled", true}, {"board_type", "p150"},
            {"board_id", 1}, {"asic_location", 0},
            {"harvesting_masks", {{"tensix", 0}, {"dram", 0}, {"eth", 0}, {"pcie", 0}, {"l2cpu", 0}}}}},
        {"host_queue", {{"channel", 0}, {"channel_size", 268435456}, {"galaxy", false},
            {"d2h_hugepage_fallback", false}, {"cq_size_override", 65536}}},
        {"active_ethernet_channels", Json::array({0, 1})}};
}
TEST(DeploymentInputFacts, RequiresCompleteExplicitFactSet) {
    auto value = facts();
    auto parsed = parse_deployment_input(value.dump());
    EXPECT_EQ(parsed.device_id, 0u);
    EXPECT_EQ(parsed.firmware_bundle, tt::umd::FirmwareBundleVersion(19, 5, 0, 0));
    EXPECT_EQ(parsed.host_channel_size, 268435456u);
    EXPECT_EQ(parsed.active_ethernet_channels, (std::vector<uint32_t>{0, 1}));
    value.erase("host_queue");
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
    value = facts(); value["chip"]["harvesting_masks"].erase("eth");
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
    value = facts(); value["ambient_device"] = 0;
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
}
TEST(DeploymentInputFacts, RejectsOldFormatAndUnsafeValues) {
    auto value = facts(); value["format"] = "tt-metal-native-deployment-input-v0";
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
    value = facts(); value["profile"] = "fallback";
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
    value = facts(); value["host_queue"]["channel_size"] = 0;
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
    value = facts(); value["active_ethernet_channels"] = Json::array({0, 0});
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
}
TEST(DeploymentInputFacts, RejectsAbsentUnknownAndUnboundedFirmwareVersions) {
    auto value = facts(); value["format"] = "tt-metal-native-deployment-input-v1";
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
    value = facts(); value.erase("firmware_bundle");
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
    for (const auto field : {"major", "minor", "patch", "pre_release"}) {
        SCOPED_TRACE(field);
        value = facts(); value["firmware_bundle"].erase(field);
        EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
        value = facts(); value["firmware_bundle"][field] = -1;
        EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
        value = facts(); value["firmware_bundle"][field] = 256;
        EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
        value = facts(); value["firmware_bundle"][field] = true;
        EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
    }
    value = facts(); value["firmware_bundle"] = "19.5.0";
    EXPECT_THROW(parse_deployment_input(value.dump()), std::invalid_argument);
}

TEST(DeploymentInputFacts, RequiresExactFullProgramRecord) {
    const std::vector<std::byte> actual{std::byte{1}, std::byte{2}, std::byte{3}};
    EXPECT_NO_THROW(require_program_record(actual, actual));
    const std::vector<std::byte> changed{std::byte{1}, std::byte{2}, std::byte{4}};
    EXPECT_THROW(require_program_record(changed, actual), std::invalid_argument);
    EXPECT_THROW(require_program_record({}, actual), std::invalid_argument);
}
}
}
