// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "BundlePublication.hpp"

#include <gtest/gtest.h>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

namespace tt::tt_metal::native_firmware_bundle {
namespace {
namespace fs = std::filesystem;

Options parse(std::initializer_list<std::string_view> args) {
    return parse_options(std::span(args.begin(), args.size()));
}

TEST(NativeFirmwareBundleArguments, RequiresExplicitOutputAndProfile) {
    EXPECT_THROW(parse({}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=/tmp/bundle"}), std::invalid_argument);
    EXPECT_THROW(parse({"--profile=disabled"}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=", "--profile=disabled"}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=/tmp/bundle", "--profile=streaming"}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=/tmp/bundle", "--profile=disabled", "--profile=program"}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=a", "--output=b", "--profile=program"}), std::invalid_argument);
    EXPECT_THROW(parse({"--output=a", "--profile=program", "--device=0"}), std::invalid_argument);
}

TEST(NativeFirmwareBundleArguments, PreservesExplicitPathAndProfile) {
    const auto disabled = parse({"--output=path with spaces", "--profile=disabled"});
    EXPECT_EQ(disabled.output, "path with spaces");
    EXPECT_EQ(disabled.profile, Profile::Disabled);
    const auto program = parse({"--profile=program", "--output=program-bundle"});
    EXPECT_EQ(program.output, "program-bundle");
    EXPECT_EQ(program.profile, Profile::Program);
}

class NativeFirmwareBundlePublication : public ::testing::Test {
protected:
    void SetUp() override {
        auto pattern = (fs::temp_directory_path() / "native_bundle_test_XXXXXX").string();
        const auto* result = mkdtemp(pattern.data());
        ASSERT_NE(result, nullptr);
        root = result;
        std::ofstream(root / "full.elf", std::ios::binary) << "full bytes";
        std::ofstream(root / "weak.elf", std::ios::binary) << "weak bytes";
    }
    void TearDown() override { if (!root.empty()) fs::remove_all(root); }
    fs::path root;
};

TEST_F(NativeFirmwareBundlePublication, RejectsExistingOutputWithoutModifyingIt) {
    const auto output = root / "bundle";
    fs::create_directory(output);
    std::ofstream(output / "keep") << "existing";
    EXPECT_THROW(Publication{output}, std::runtime_error);
    EXPECT_TRUE(fs::exists(output / "keep"));
    fs::remove(output / "keep");
    EXPECT_THROW(Publication{output}, std::runtime_error);
    fs::create_symlink(root / "absent", root / "dangling");
    EXPECT_THROW(Publication{root / "dangling"}, std::runtime_error);
    EXPECT_TRUE(fs::is_symlink(root / "dangling"));
}

TEST_F(NativeFirmwareBundlePublication, FailureLeavesNoPartialBundleOrStaging) {
    const auto output = root / "bundle";
    fs::path stage;
    {
        Publication publication(output);
        stage = publication.payload().parent_path();
        publication.copy(root / "full.elf", "tt_metal/pre-compiled/11/brisc/brisc.elf");
        EXPECT_FALSE(fs::exists(output));
        EXPECT_THROW(publication.copy(root / "missing", "missing.elf"), std::runtime_error);
    }
    EXPECT_FALSE(fs::exists(output));
    EXPECT_FALSE(fs::exists(stage));
}

TEST_F(NativeFirmwareBundlePublication, PublishesCompleteAndWeakBytesTogether) {
    const auto output = root / "bundle";
    Publication publication(output);
    publication.copy(root / "full.elf", "tt_metal/pre-compiled/11/brisc/brisc.elf");
    publication.copy(root / "weak.elf", "tt_metal/pre-compiled/11/brisc/brisc_weakened.elf");
    EXPECT_FALSE(fs::exists(output));
    publication.commit();
    std::ifstream full(output / "tt_metal/pre-compiled/11/brisc/brisc.elf", std::ios::binary);
    std::ifstream weak(output / "tt_metal/pre-compiled/11/brisc/brisc_weakened.elf", std::ios::binary);
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(full), {}), "full bytes");
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(weak), {}), "weak bytes");
}

TEST_F(NativeFirmwareBundlePublication, DoesNotReplaceConcurrentlyCreatedOutput) {
    const auto output = root / "bundle";
    Publication publication(output);
    publication.copy(root / "full.elf", "full.elf");
    fs::create_directory(output);
    EXPECT_THROW(publication.commit(), std::runtime_error);
    EXPECT_TRUE(fs::is_empty(output));
}

TEST_F(NativeFirmwareBundlePublication, RejectsSymlinkInputsAndEscapingOutputs) {
    Publication publication(root / "bundle");
    fs::create_symlink(root / "full.elf", root / "link");
    EXPECT_THROW(publication.copy(root / "link", "linked.elf"), std::runtime_error);
    EXPECT_THROW(publication.copy(root / "full.elf", "../escaped.elf"), std::runtime_error);
    EXPECT_THROW(publication.copy(root / "full.elf", root / "absolute.elf"), std::runtime_error);
    EXPECT_FALSE(fs::exists(root / "escaped.elf"));
    EXPECT_FALSE(fs::exists(root / "absolute.elf"));
}

}  // namespace
}  // namespace tt::tt_metal::native_firmware_bundle
