#include "NativeDeploymentPublish.hpp"
#include <gtest/gtest.h>

namespace tt::tt_metal::native_firmware_bundle {
namespace {
DeploymentPublishOptions parse(std::initializer_list<std::string_view> args) {
    return parse_deployment_options(std::span(args.begin(), args.size()));
}
TEST(NativeDeploymentPublish, RequiresAllExplicitInputs) {
    EXPECT_THROW(parse({}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=/tmp/resources", "--input=/tmp/facts.json"}), std::invalid_argument);
    EXPECT_THROW(parse({"--input=/tmp/facts.json", "--program-record=/tmp/plan"}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=/tmp/resources", "--input=/tmp/facts.json", "--program-record=/tmp/plan", "--device=0"}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=/tmp/resources", "--input=/tmp/facts.json", "--program-record=/tmp/plan", "--output=/tmp/other"}), std::invalid_argument);
}
TEST(NativeDeploymentPublish, PreservesExactInputPaths) {
    const auto value = parse({"--output=/tmp/resources", "--input=/tmp/facts.json", "--program-record=/tmp/plan"});
    EXPECT_EQ(value.output, "/tmp/resources");
    EXPECT_EQ(value.input, "/tmp/facts.json");
    EXPECT_EQ(value.program_record, "/tmp/plan");
}
}
}
