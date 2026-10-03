// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Native bodies own numerical configuration and initialization. These tests run
// real SDK kernel preparation/JIT against a mock Blackhole, never silicon.

#include <gtest/gtest.h>

#include <exception>
#include <memory>
#include <string>
#include <type_traits>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_build_options.hpp"
#include "mock_blackhole_fixture.hpp"

namespace tt::tt_metal {
namespace {

// Keep the first RED executable on the SDK before the new typed field lands.
// Missing Native is a test failure, never a fallback to the SDK body mode.
template <typename Config>
::testing::AssertionResult select_native_body(Config& config) {
    if constexpr (requires { config.body_mode; }) {
        using Mode = std::remove_cvref_t<decltype(config.body_mode)>;
        if constexpr (requires { Mode::Sdk; Mode::Native; }) {
            config.body_mode = Mode::Native;
            return ::testing::AssertionSuccess();
        }
    }
    return ::testing::AssertionFailure() << "SDK configuration has no typed Native body mode";
}

template <typename Config>
bool has_native_body(const Config& config) {
    if constexpr (requires { config.body_mode; }) {
        using Mode = std::remove_cvref_t<decltype(config.body_mode)>;
        if constexpr (requires { Mode::Native; }) {
            return config.body_mode == Mode::Native;
        }
    }
    return false;
}

constexpr const char* native_empty_body = "void kernel_main() {}\n";

// chlkc_list.h includes the authored kernel after ckernel.h, then defines
// run_kernel(). These compile traps detect an actual subsequent SDK clear
// call. They do not replace firmware, CRT, profiler, CB, or kernel_main.
constexpr const char* native_body_rejecting_implicit_clears = R"cpp(
void kernel_main() {}
#if defined(TRISC_MATH)
#define zeroacc(...) static_assert(false, "unexpected SDK Dst initialization")
#endif
#if defined(TRISC_UNPACK)
#define zerosrc(...) static_assert(false, "unexpected SDK source initialization")
#endif
)cpp";

}  // namespace

// External linkage is required by gtest TEST_F classes in GCC unity builds.
class NativeKernelBodyMockBlackholeFixture : public MockBlackholeMeshDispatchFixture {
protected:
    std::shared_ptr<Kernel> add_compute(
        Program& program, const ComputeConfig& config, const std::string& source = native_empty_body) {
        const auto handle = CreateKernelFromString(program, source, CoreCoord{0, 0}, config);
        return program.impl().get_kernel(handle);
    }

    std::shared_ptr<Kernel> add_data_movement(Program& program, const DataMovementConfig& config) {
        const auto handle = CreateKernelFromString(program, native_empty_body, CoreCoord{0, 0}, config);
        return program.impl().get_kernel(handle);
    }

    void add_buffer(Program& program, uint8_t index, tt::DataFormat format, uint32_t page_size) {
        const auto cb = static_cast<tt::CBIndex>(index);
        auto config = CircularBufferConfig(page_size, {{cb, format}}).set_page_size(cb, page_size);
        CreateCircularBuffer(program, CoreCoord{0, 0}, config);
    }

    void compile(Program& program) { program.impl().compile(devices_.at(0).get()); }

    void expect_compile_error(Program& program, const std::string& diagnostic) {
        try {
            compile(program);
            FAIL() << "SDK preparation unexpectedly accepted the invalid LLK configuration";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find(diagnostic), std::string::npos) << error.what();
        }
    }

    const JitBuildEnv& environment(const Kernel& kernel) {
        return BuildEnvManager::get_instance(kernel.get_context_id())
            .get_device_build_env(devices_.at(0)->build_id())
            .build_env;
    }
};

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeComputeModeReachesBuildOptions) {
    ComputeConfig config;
    ASSERT_TRUE(select_native_body(config));
    Program program = CreateProgram();
    auto kernel = add_compute(program, config);
    JitBuildOptions options(environment(*kernel));
    kernel->set_build_options(options);
    EXPECT_TRUE(has_native_body(options));
    EXPECT_EQ(options.hlk_desc.get_hlk_math_fidelity(), MathFidelity::Invalid);
}

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeDataMovementModeReachesBuildOptions) {
    DataMovementConfig config;
    ASSERT_TRUE(select_native_body(config));
    Program program = CreateProgram();
    auto kernel = add_data_movement(program, config);
    JitBuildOptions options(environment(*kernel));
    kernel->set_build_options(options);
    EXPECT_TRUE(has_native_body(options));
}

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeAndSdkBodiesHaveDistinctCacheKeys) {
    ComputeConfig native;
    ASSERT_TRUE(select_native_body(native));
    Program sdk_program = CreateProgram();
    Program native_program = CreateProgram();
    auto sdk_kernel = add_compute(sdk_program, ComputeConfig{});
    auto native_kernel = add_compute(native_program, native);
    EXPECT_NE(sdk_kernel->compute_hash(), native_kernel->compute_hash());

    DataMovementConfig native_dm;
    ASSERT_TRUE(select_native_body(native_dm));
    auto sdk_dm = add_data_movement(sdk_program, DataMovementConfig{});
    auto native_dm_kernel = add_data_movement(native_program, native_dm);
    EXPECT_NE(sdk_dm->compute_hash(), native_dm_kernel->compute_hash());
}

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeCacheDoesNotDependOnUnusedLlkNumericalOptions) {
    ComputeConfig first;
    ASSERT_TRUE(select_native_body(first));
    ComputeConfig second = first;
    second.math_fidelity = MathFidelity::LoFi;
    second.fp32_dest_acc_en = true;
    second.dst_full_sync_en = true;
    second.math_approx_mode = true;
    second.unpack_to_dest_mode = {UnpackToDestMode::UnpackToDestFp32};
    second.bfp8_pack_precise = true;
    Program first_program = CreateProgram();
    Program second_program = CreateProgram();
    EXPECT_EQ(add_compute(first_program, first)->compute_hash(), add_compute(second_program, second)->compute_hash());
}

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeComputeDescriptorPreservesBodyMode) {
    ComputeConfigDescriptor config;
    ASSERT_TRUE(select_native_body(config));
    KernelDescriptor descriptor = {
        .kernel_source = native_empty_body,
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .config = config,
    };
    Program program(ProgramDescriptor{.kernels = {descriptor}});
    const auto projected = program.impl().get_kernel(0)->config();
    ASSERT_TRUE(std::holds_alternative<ComputeConfig>(projected));
    EXPECT_TRUE(has_native_body(std::get<ComputeConfig>(projected)));
}

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeDataMovementDescriptorPreservesBodyMode) {
    DataMovementConfigDescriptor config;
    ASSERT_TRUE(select_native_body(config));
    KernelDescriptor descriptor = {
        .kernel_source = native_empty_body,
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .config = config,
    };
    Program program(ProgramDescriptor{.kernels = {descriptor}});
    const auto projected = program.impl().get_kernel(0)->config();
    ASSERT_TRUE(std::holds_alternative<DataMovementConfig>(projected));
    EXPECT_TRUE(has_native_body(std::get<DataMovementConfig>(projected)));
}

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeProgramAcceptsIndependentCbExponentFormats) {
    ComputeConfig compute;
    DataMovementConfig data_movement;
    ASSERT_TRUE(select_native_body(compute));
    ASSERT_TRUE(select_native_body(data_movement));
    Program program = CreateProgram();
    add_buffer(program, 0, tt::DataFormat::Float16, 2048);
    add_buffer(program, 1, tt::DataFormat::Float16_b, 2048);
    add_compute(program, compute);
    add_data_movement(program, data_movement);
    EXPECT_NO_THROW(compile(program));
}

TEST_F(NativeKernelBodyMockBlackholeFixture, SdkProgramRetainsLlkExponentFormatValidation) {
    Program program = CreateProgram();
    add_buffer(program, 0, tt::DataFormat::Float16, 2048);
    add_buffer(program, 1, tt::DataFormat::Float16_b, 2048);
    add_compute(program, ComputeConfig{});
    expect_compile_error(program, "same exponent format");
}

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeBodyDoesNotRequireGlobalFp32ForUnusedFp8Buffer) {
    ComputeConfig config;
    ASSERT_TRUE(select_native_body(config));
    Program program = CreateProgram();
    add_buffer(program, 0, tt::DataFormat::Fp8_e4m3, 1024);
    add_compute(program, config);
    EXPECT_NO_THROW(compile(program));
}

TEST_F(NativeKernelBodyMockBlackholeFixture, SdkBodyRetainsFp8DestinationValidation) {
    Program program = CreateProgram();
    add_buffer(program, 0, tt::DataFormat::Fp8_e4m3, 1024);
    add_compute(program, ComputeConfig{});
    expect_compile_error(program, "require fp32_dest_acc_en=true");
}

TEST_F(NativeKernelBodyMockBlackholeFixture, SdkEmptyBodyStillCompiles) {
    Program program = CreateProgram();
    add_compute(program, ComputeConfig{});
    EXPECT_NO_THROW(compile(program));
}

TEST_F(NativeKernelBodyMockBlackholeFixture, NativeStartupDoesNotInsertNumericalClears) {
    ComputeConfig config;
    ASSERT_TRUE(select_native_body(config));
    Program program = CreateProgram();
    add_compute(program, config, native_body_rejecting_implicit_clears);
    EXPECT_NO_THROW(compile(program));
}

TEST_F(NativeKernelBodyMockBlackholeFixture, SdkStartupStillCallsNumericalClears) {
    Program program = CreateProgram();
    add_compute(program, ComputeConfig{}, native_body_rejecting_implicit_clears);
    EXPECT_ANY_THROW(compile(program));
}

TEST_F(NativeKernelBodyMockBlackholeFixture, UserDefinesCannotSelectNativeStartup) {
    for (const char* value : {"0", "1"}) {
        Program program = CreateProgram();
        ComputeConfig compute;
        compute.defines.emplace("TT_METAL_NATIVE_BODY", value);
        EXPECT_ANY_THROW(add_compute(program, compute));
        EXPECT_EQ(program.impl().num_kernels(), 0u);

        DataMovementConfig data_movement;
        data_movement.defines.emplace("TT_METAL_NATIVE_BODY", value);
        EXPECT_ANY_THROW(add_data_movement(program, data_movement));
        EXPECT_EQ(program.impl().num_kernels(), 0u);
    }
}

TEST_F(NativeKernelBodyMockBlackholeFixture, LateDefinesCannotOverrideStartupMode) {
    Program program = CreateProgram();
    auto kernel = add_compute(program, ComputeConfig{});
    const auto original_hash = kernel->compute_hash();
    EXPECT_ANY_THROW(kernel->add_defines({{"ANOTHER_DEFINE", "2"}, {"TT_METAL_NATIVE_BODY", "1"}}));
    EXPECT_TRUE(kernel->defines().empty());
    EXPECT_EQ(kernel->compute_hash(), original_hash);
}

TEST_F(NativeKernelBodyMockBlackholeFixture, DescriptorDefinesCannotSelectNativeStartup) {
    KernelDescriptor descriptor = {
        .kernel_source = native_empty_body,
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRange(CoreCoord{0, 0}),
        .defines = {{"TT_METAL_NATIVE_BODY", "1"}},
        .config = ComputeConfigDescriptor{},
    };
    EXPECT_ANY_THROW(Program program(ProgramDescriptor{.kernels = {descriptor}}));
    descriptor.config = DataMovementConfigDescriptor{};
    EXPECT_ANY_THROW(Program program(ProgramDescriptor{.kernels = {descriptor}}));
}

}  // namespace tt::tt_metal
