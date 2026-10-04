// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "DeploymentBuild.hpp"
#include "PublishedImage.hpp"
#include "impl/experimental/offline_compile/kernel_compiler.hpp"
#include "impl/experimental/published_deployment/dispatch_plan.hpp"
#include "impl/experimental/published_deployment/dispatch_program_plan.hpp"
#include "impl/kernels/kernel.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_device_config.hpp"
#include "jit_build/jit_build_options.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace tt::tt_metal::native_firmware_bundle {
namespace {
std::vector<std::byte> read_image(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || input.tellg() > 64 * 1024 * 1024)
        throw std::runtime_error("supplier build did not produce a bounded nonempty ELF: " + path.string());
    std::vector<std::byte> bytes(static_cast<size_t>(input.tellg()));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("cannot read supplier ELF: " + path.string());
    return bytes;
}
}
DeploymentBuildResult build_deployment(
    const JitDeviceConfig& device, const llrt::RunTimeOptions& options,
    const experimental::DispatchProgramPlan& program) {
    using namespace experimental;
    // Validate complete explicit values before constructing a compiler environment.
    const auto configuration = DeploymentConfiguration::from_sdk(device, options, program);
    if (!options.get_disable_precompiled_fw() || !options.is_cache_dir_specified())
        throw std::invalid_argument("supplier build requires an explicit cache and precompiled firmware disabled");
    const std::filesystem::path root(options.get_root_dir());
    const std::filesystem::path cache(options.get_cache_dir());
    if (!root.is_absolute() || !cache.is_absolute() || !std::filesystem::is_directory(root) ||
        (std::filesystem::exists(cache) && !std::filesystem::is_empty(cache)))
        throw std::invalid_argument("supplier build requires an explicit root, fresh cache and precompiled firmware disabled");
    const auto& hal = *device.hal;
    BuildEnvManager manager(hal);
    manager.add_build_env(0, device, options);
    manager.build_firmware(0, true);
    const auto& environment = manager.get_device_build_env(0);
    if (environment.firmware_precompiled || environment.build_env.get_firmware_binary_root() !=
        environment.build_env.get_out_firmware_root_path())
        throw std::runtime_error("supplier deployment selected external firmware");
    std::vector<std::vector<std::byte>> owned;
    std::vector<PublishedImageInput> input;
    std::vector<SupplierRecipe> recipes;
    const auto add = [&](PublishedImageKind kind, HalProcessorIdentifier processor, uint32_t node,
                         const std::filesystem::path& path) {
        const auto linked = read_image(path);
        owned.push_back(publish_image(configuration, kind, processor, node, linked));
        input.push_back({kind, processor, node, owned.back()});
    };
    for (const auto processor : configuration.firmware_processors()) {
        const auto ci = hal.get_programmable_core_type_index(processor.core_type);
        const auto image = manager.get_firmware_binary_path(
            0, ci, static_cast<uint32_t>(processor.processor_class), processor.processor_type);
        recipes.push_back({PublishedImageKind::Firmware, processor, NoDispatchNode, image,
            manager.get_firmware_build_state(0, ci, static_cast<uint32_t>(processor.processor_class),
                processor.processor_type).export_target_recipe(nullptr)});
        add(PublishedImageKind::Firmware, processor, NoDispatchNode, image);
    }
    for (const auto& plan : program.kernel_plans()) {
        const auto& node = plan.configuration;
        const CoreRangeSet cores(CoreRange(node.logical_core, node.logical_core));
        for (const auto& source : plan.kernels) {
            const auto path = root / dispatch_source_path(source.kind);
            const auto kernel_source = KernelSource::from_path(DEFAULT_CONTEXT_ID, path);
            const KernelBuildContext context{hal, options, nullptr};
            std::shared_ptr<Kernel> kernel;
            if (source.kind == DispatchSourceKind::SubordinateCompute) {
                ComputeConfig config;
                config.defines = source.defines;
                config.opt_level = source.opt_level;
                kernel = std::make_shared<ComputeKernel>(context, kernel_source, cores, config);
            } else {
                DataMovementConfig config;
                config.processor = node.processor.processor_type == 0 ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1;
                config.noc = node.nocs.non_dispatch_noc;
                config.defines = source.defines;
                config.opt_level = source.opt_level;
                kernel = std::make_shared<DataMovementKernel>(context, kernel_source, cores, config);
            }
            JitBuildOptions build_options(environment.build_env);
            kernel->set_build_options(build_options);
            // A fresh supplier workspace gives each typed node/source its own
            // build directory. Names only locate outputs; they admit no image.
            const auto name = "dispatch/" + std::to_string(node.node_id) + "/" +
                std::to_string(static_cast<unsigned>(source.kind)) + "/";
            kernel->set_full_name(name);
            build_options.set_name(name);
            offline_detail::generate_kernel_binaries(kernel, manager, environment, build_options, hal);
            for (const auto processor : source.processors) {
                const auto ci = hal.get_programmable_core_type_index(processor.core_type);
                const auto image = manager.get_kernel_binary_path(0, ci,
                    static_cast<uint32_t>(processor.processor_class), processor.processor_type,
                    environment.build_env.get_out_kernel_root_path(), kernel->get_full_kernel_name());
                recipes.push_back({PublishedImageKind::Dispatch, processor, node.node_id, image,
                    manager.get_kernel_build_state(0, ci, static_cast<uint32_t>(processor.processor_class),
                        processor.processor_type).export_target_recipe(kernel.get())});
                add(PublishedImageKind::Dispatch, processor, node.node_id, image);
            }
        }
    }
    return {PublishedDeployment::admit(configuration, input), environment.build_env.get_gpp(), std::move(recipes)};
}
}
