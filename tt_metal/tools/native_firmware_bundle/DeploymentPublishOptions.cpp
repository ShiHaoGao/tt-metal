#include "NativeDeploymentPublish.hpp"
#include <stdexcept>
#include <string>

namespace tt::tt_metal::native_firmware_bundle {
DeploymentPublishOptions parse_deployment_options(std::span<const std::string_view> arguments) {
    DeploymentPublishOptions result;
    bool output = false, input = false, program = false;
    for (const auto argument : arguments) {
        const auto parse_path = [&](std::string_view flag, std::filesystem::path& destination, bool& seen) {
            if (!argument.starts_with(flag) || seen) return false;
            const auto value = argument.substr(flag.size());
            if (value.empty() || value.find('\0') != std::string_view::npos)
                throw std::invalid_argument("empty supplier path: " + std::string(flag));
            destination = value;
            seen = true;
            return true;
        };
        if (parse_path("--output=", result.output, output) ||
            parse_path("--input=", result.input, input) ||
            parse_path("--program-record=", result.program_record, program)) continue;
        throw std::invalid_argument("unknown or duplicate supplier argument: " + std::string(argument));
    }
    if (!output || !input || !program)
        throw std::invalid_argument("usage: native_deployment_publish --output=<fresh-resource-root> --input=<facts.json> --program-record=<canonical-record>");
    return result;
}
}
