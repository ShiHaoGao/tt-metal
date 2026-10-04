#include "jit_build/jit_device_config.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include "impl/context/metal_env_accessor.hpp"
#include "impl/context/metal_env_impl.hpp"
#include <gtest/gtest.h>
namespace tt::tt_metal {
namespace {
class JitDeviceConfigProfile : public ::testing::Test {
protected:
    void SetUp() override { experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1); }
    void TearDown() override {
        MetalContext::destroy_all_instances(false);
        experimental::disable_mock_mode();
    }
    void receive(DeviceProfilerMode mode) {
        MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc().value());
        descriptor.set_device_profiler_mode(mode);
        MetalEnv env(std::move(descriptor));
        const auto id = ContextId{MetalEnvAccessor(env).impl().ensure_context_registered(env)};
        auto& context = MetalContext::instance(id);
        // Dispatch facts only; do not compile or launch mock firmware.
        context.initialize(DispatchCoreConfig{DispatchCoreType::WORKER, DispatchCoreAxis::COL},
                           1, {}, DEFAULT_WORKER_L1_SIZE, true);
        EXPECT_NE(id, DEFAULT_CONTEXT_ID);
        const auto actual = create_jit_device_config(0, 1, id);
        const auto expected = mode == DeviceProfilerMode::Program
            ? get_profiler_dram_bank_size_per_risc_bytes(context.rtoptions()) : 0;
        EXPECT_EQ(actual.profiler_dram_bank_size_per_risc_bytes, expected);
        EXPECT_EQ(actual.hal, &context.hal());
        const Hal expected_hal{tt::ARCH::BLACKHOLE, false, true,
            get_profiler_dram_bank_size_for_hal_allocation(mode), false, false, true, true};
        EXPECT_EQ(context.hal().get_dev_size(HalDramMemAddrType::PROFILER),
                  expected_hal.get_dev_size(HalDramMemAddrType::PROFILER));
        EXPECT_EQ(context.hal().get_dev_addr(HalDramMemAddrType::UNRESERVED),
                  expected_hal.get_dev_addr(HalDramMemAddrType::UNRESERVED));
        EXPECT_FALSE(MetalContext::instance_exists(DEFAULT_CONTEXT_ID));
    }
};
TEST_F(JitDeviceConfigProfile, DisabledHasNoAllocation) { receive(DeviceProfilerMode::Disabled); }
TEST_F(JitDeviceConfigProfile, ProgramRetainsActualAllocation) { receive(DeviceProfilerMode::Program); }
}
}
