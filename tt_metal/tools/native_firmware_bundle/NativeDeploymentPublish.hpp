#pragma once
#include <filesystem>
#include <span>
#include <string_view>

namespace tt::tt_metal::native_firmware_bundle {
struct DeploymentPublishOptions {
    std::filesystem::path output;
    std::filesystem::path input;
    std::filesystem::path program_record;
};
DeploymentPublishOptions parse_deployment_options(std::span<const std::string_view> arguments);
}
