// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "jit_build/build.hpp"
#include "jit_build/depend.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

// Exercise the real linker and its dependency publication with small ELF inputs.
// The test only replaces the program inputs; no MetalContext or device is opened.
class LinkDependencyProbe : public JitBuildState {
public:
    LinkDependencyProbe(const JitBuildEnv& env, const Hal& hal, const std::filesystem::path& root, bool firmware) :
        JitBuildState(
            env,
            {.core_type = HalProgrammableCoreType::TENSIX,
             .processor_class = HalProcessorClassType::DM,
             .processor_id = 0,
             .is_fw = firmware},
            hal) {
        target_name_ = "probe";
        linker_script_ = (root / "probe.ld").string();
        lflags_ = "-nostdlib -Wl,-T," + linker_script_ + " ";
        weakened_firmware_name_ = (root / "firmware.elf").string();
        firmware_is_kernel_object_ = false;
        extra_link_objs_.clear();
        for (const auto* name : {"noc.o", "substitutes.o", "tmu-crt0.o"}) {
            extra_link_objs_ += (root / name).string() + " ";
        }
    }

    void run(const std::filesystem::path& root) const {
        link(root.string() + "/", nullptr, (root / "main.o").string() + " ");
    }

    bool requires_link(const std::filesystem::path& root) const { return need_link(root.string() + "/"); }
};

class JitLinkDependencyTests : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "jit_link_dependencies_XXXXXX").string();
        const auto* directory = mkdtemp(pattern.data());
        ASSERT_NE(directory, nullptr);
        root_ = directory;
        jit_build::clear_file_hash_cache();
        options_ = std::make_unique<llrt::RunTimeOptions>(llrt::RunTimeOptions::ExplicitBuildOptions{
            .root_dir = llrt::RunTimeOptions().get_root_dir(), .cache_dir = root_.string()});
        const JitDeviceConfig config{DeviceConfiguration{.arch = tt::ARCH::BLACKHOLE, .max_cbs = 32}, &hal_};
        env_.init(1, config, *options_, {});
        std::ofstream(root_ / "probe.ld") <<
            "PHDRS { text PT_LOAD FLAGS(5); data PT_LOAD FLAGS(6); } "
            "SECTIONS { . = 0x10000; .text : { *(.text*) } :text "
            ".data : { *(.data*) } :data .bss : { *(.bss*) } :data }\n";
        compile("main.o", "extern \"C\" void _start() {}\n");
        compile("firmware.o", "int firmware_symbol = 1;\n");
        run_compiler({"-nostdlib", "-Wl,-T," + (root_ / "probe.ld").string(),
                      (root_ / "firmware.o").string(), "-o", (root_ / "firmware.elf").string()});
        for (size_t i = 0; i < objects_.size(); ++i) {
            compile(objects_[i], "int dependency_" + std::to_string(i) + " = 1;\n");
        }
    }

    void TearDown() override {
        std::filesystem::remove_all(root_);
        jit_build::clear_file_hash_cache();
    }

    void run_compiler(const std::vector<std::string>& arguments) {
        auto command = jit_build::utils::tokenize_flags(env_.get_gpp());
        command.insert(command.end(), arguments.begin(), arguments.end());
        const auto log = root_ / "compile.log";
        if (!jit_build::utils::exec_command(command, root_.string(), log.string())) {
            std::ifstream stream(log);
            throw std::runtime_error(std::string(std::istreambuf_iterator<char>(stream), {}));
        }
    }

    void compile(const std::string& object, const std::string& contents) {
        const auto source = root_ / (object + ".cpp");
        std::ofstream(source) << contents;
        run_compiler({"-c", source.string(), "-o", (root_ / object).string()});
    }

    static constexpr std::array objects_{"noc.o", "substitutes.o", "tmu-crt0.o"};
    std::filesystem::path root_;
    const Hal hal_{tt::ARCH::BLACKHOLE, false, false, 0, false};
    std::unique_ptr<llrt::RunTimeOptions> options_;
    JitBuildEnv env_;
};

TEST_P(JitLinkDependencyTests, ExtraLinkObjectChangesRequireRelink) {
    const LinkDependencyProbe state(env_, hal_, root_, GetParam());
    for (size_t i = 0; i < objects_.size(); ++i) {
        SCOPED_TRACE(objects_[i]);
        state.run(root_);
        ASSERT_FALSE(state.requires_link(root_));
        compile(objects_[i], "int dependency_" + std::to_string(i) + "[2] = {2, 3};\n");
        jit_build::clear_file_hash_cache();
        EXPECT_FALSE(jit_build::dependencies_up_to_date(root_.string(), "probe.elf"));
        EXPECT_TRUE(state.requires_link(root_));
    }
}

TEST_P(JitLinkDependencyTests, MissingExtraLinkObjectRequiresRelink) {
    const LinkDependencyProbe state(env_, hal_, root_, GetParam());
    state.run(root_);
    ASSERT_FALSE(state.requires_link(root_));
    std::filesystem::remove(root_ / "noc.o");
    EXPECT_FALSE(jit_build::dependencies_up_to_date(root_.string(), "probe.elf"));
    EXPECT_TRUE(state.requires_link(root_));
}

INSTANTIATE_TEST_SUITE_P(FirmwareAndKernel, JitLinkDependencyTests, ::testing::Bool());

}  // namespace
}  // namespace tt::tt_metal
