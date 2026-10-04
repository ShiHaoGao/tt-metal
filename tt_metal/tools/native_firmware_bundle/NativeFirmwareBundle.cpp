// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "BundlePublication.hpp"
#include "SourceArchive.hpp"
#include "RecipeProvenance.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "impl/profiler/profiler_state_manager.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

#ifndef TT_NATIVE_FIRMWARE_SDK_ROOT
#error "The supplier tool must be built against an explicit SDK source root"
#endif

namespace tt::tt_metal::native_firmware_bundle {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;

fs::path within(const fs::path& path, const fs::path& root) {
    const auto relative = path.lexically_relative(root);
    if (relative.empty() || relative.is_absolute()) return {};
    for (const auto& part : relative) {
        if (part == "..") return {};
    }
    return relative;
}

void require_sdk_shell_path(const fs::path& path) {
    // The existing supplier firmware linker still constructs a shell command.
    // Reject paths that it cannot represent, before starting any compilation.
    if (path.string().find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-") !=
        std::string::npos) {
        throw std::runtime_error("SDK firmware linker cannot safely represent build path: " + path.string());
    }
}

std::vector<fs::path> files_under(const fs::path& directory) {
    if (!fs::is_directory(directory)) {
        throw std::runtime_error("required supplier source directory is absent: " + directory.string());
    }
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

Json describe_config(const JitDeviceConfig& config) {
    return {{"arch", static_cast<uint32_t>(config.arch)},
            {"num_dram_banks", config.num_dram_banks},
            {"num_l1_banks", config.num_l1_banks},
            {"pcie_core", {{"x", config.pcie_core.x}, {"y", config.pcie_core.y}}},
            {"harvesting_mask", config.harvesting_mask},
            {"dispatch_core_type", static_cast<uint32_t>(config.dispatch_core_type)},
            {"resolved_dispatch_core_type", static_cast<uint32_t>(config.resolved_dispatch_core_type)},
            {"dispatch_core_axis", static_cast<uint32_t>(config.dispatch_core_axis)},
            {"coordinate_virtualization_enabled", config.coordinate_virtualization_enabled},
            {"dispatch_message_addr", config.dispatch_message_addr},
            {"max_cbs", config.max_cbs},
            {"num_hw_cqs", config.num_hw_cqs},
            {"routing_fw_enabled", config.routing_fw_enabled},
            {"profiler_dram_bank_size_per_risc_bytes", config.profiler_dram_bank_size_per_risc_bytes}};
}

Json describe_contract(const tt_native_image_record& record) {
    return {{"magic", record.magic}, {"version", record.version}, {"kind", record.kind}, {"role", record.role},
            {"architecture", record.architecture}, {"loading", record.loading}, {"profile", record.profile},
            {"print", record.print}, {"worker_stream_abi", record.worker_stream_abi},
            {"worker_stream_owner", record.worker_stream_owner}, {"reserved", record.reserved}};
}

void capture_build_inputs(const DeviceBuildEnv& environment, SourceArchive& sources) {
    for (const auto& state : environment.firmware_build_states)
        sources.capture_recipe(state.export_target_recipe(nullptr),
            fs::path(state.get_out_path() + state.get_target_full_path()));
}

Json publish_artifacts(Publication& publication, const DeviceBuildEnv& environment, const std::string& key) {
    const auto root = fs::path(environment.build_env.get_out_firmware_root_path()).lexically_normal();
    const auto destination = fs::path("tt_metal/pre-compiled") / key;
    Json result = {{"sdk_build_key", key}, {"directory", destination.generic_string()},
                   {"targets", Json::array()}, {"artifacts", Json::array()}, {"native_roles", Json::array()}};
    std::set<fs::path> artifacts;
    for (const auto& state : environment.firmware_build_states) {
        const auto full = fs::path(state.get_out_path() + state.get_target_full_path()).lexically_normal();
        const auto weak = fs::path(state.get_weakened_firmware_name()).lexically_normal();
        for (const auto& file : {full, weak}) {
            if (!fs::is_regular_file(fs::symlink_status(file)) || fs::file_size(file) == 0 || within(file, root).empty()) {
                throw std::runtime_error("missing or misplaced complete/weak firmware artifact: " + file.string());
            }
            artifacts.insert(file);
        }
        result["targets"].push_back({{"target", state.get_target_name()},
            {"complete_image", (destination / within(full, root)).generic_string()},
            {"weakened_image", (destination / within(weak, root)).generic_string()},
            {"firmware_is_kernel_object", state.get_firmware_is_kernel_object()}});
    }
    // Include every actual firmware ELF, including non-Tensix cores. Build
    // states also retain any weakened artifact whose suffix is .o.
    for (const auto& file : files_under(root)) {
        if (file.extension() == ".elf") artifacts.insert(file.lexically_normal());
    }
    for (const auto& file : artifacts) {
        const auto published = destination / within(file, root);
        publication.copy(file, published);
        result["artifacts"].push_back({{"path", published.generic_string()}, {"bytes", fs::file_size(file)}});
    }
    // Consume the SDK's immutable, validated bundle; do not reimplement its
    // image/firmware admission rules in supplier tooling.
    if (!environment.native_firmware) throw std::runtime_error("built firmware lacks the native five-role ABI");
    for (uint8_t role = 0; role < TT_NATIVE_ROLE_COUNT; ++role) {
        result["native_roles"].push_back(describe_contract(environment.native_firmware->contract(
            static_cast<experimental::TensixKernelRole>(role))));
    }
    return result;
}

void publish(const Options& arguments) {
    // Validate the output before constructing SDK objects or probing tools.
    Publication publication(arguments.output);
    const auto sdk = fs::canonical(TT_NATIVE_FIRMWARE_SDK_ROOT);
    require_sdk_shell_path(sdk);
    require_sdk_shell_path(publication.cache());
    const auto profile = arguments.profile == Profile::Program ? DeviceProfilerMode::Program : DeviceProfilerMode::Disabled;
    llrt::RunTimeOptions options(llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = sdk.string(), .cache_dir = publication.cache().string(), .profiler_mode = profile});
    options.set_disable_precompiled_fw(true);
    options.set_enable_2_erisc_mode(true);
    const uint32_t profiler_bytes = get_profiler_dram_bank_size_for_hal_allocation(options);

    SourceArchive sources(publication, sdk);
    Json report = {{"format", "tt-metal-offline-firmware-publication-v1"}, {"sdk_source_root", sdk.string()},
                   {"profile", arguments.profile == Profile::Program ? "program" : "disabled"},
                   {"architecture", "blackhole"}, {"configuration_enumerator", "enumerate_offline_compile_device_configs"},
                   {"sdk_build_key_purpose", "precompiled publication directory only"},
                   {"hal_options", {{"enable_2_erisc_mode", true}, {"enable_dram_backed_cq", false},
                                    {"is_simulator", false}, {"enable_blackhole_dram_programmable_cores", true},
                                    {"enable_aerisc_ptp_trace", true}}},
                   {"configurations", Json::array()}, {"builds", Json::array()}, {"toolchains", Json::array()}};
    for (const auto& license : {sdk / "LICENSE", sdk / "tt_metal/tt-llk/LICENSE", sdk / "tt_metal/tt-llk/LICENSE_understanding.txt"}) {
        report["license_snapshots"].push_back(sources.capture(license).generic_string());
    }
    // Preserve the canonical catalog and its actual YAML inputs, not a second
    // hand-maintained list used to select configurations.
    sources.capture(sdk / "tt_metal/jit_build/jit_device_config.cpp");
    sources.capture(sdk / "tt_metal/jit_build/jit_device_config.hpp");
    sources.capture_tree(sdk / "tt_metal/core_descriptors");
    sources.capture_tree(sdk / "tt_metal/soc_descriptors");
    // Linker INCLUDE inputs are not all represented by compiler depfiles.
    sources.capture_tree(sdk / "tt_metal/hw/toolchain");
    sources.capture_tree(sdk / "tt_metal/tools/native_firmware_bundle");

    std::map<uint64_t, Json> recipes_by_key;
    std::map<std::string, size_t> toolchain_indices;
    enumerate_offline_compile_device_configs(options, [&](const JitDeviceConfig& authored) {
        if (authored.arch != tt::ARCH::BLACKHOLE) return;
        // The catalog intentionally supplies a profiler-disabled HAL. Rebuild
        // that HAL with the explicit allocation, preserving its other choices.
        Hal hal(authored.arch, authored.routing_fw_enabled, options.get_enable_2_erisc_mode(), profiler_bytes,
                false, false, true, true);
        auto config = authored;
        config.hal = &hal;
        config.profiler_dram_bank_size_per_risc_bytes = profiler_bytes;
        BuildEnvManager manager(hal);
        manager.add_build_env(0, config, options);
        const auto& environment = manager.get_device_build_env(0);
        if (environment.firmware_precompiled || environment.build_env.get_firmware_binary_root() !=
            environment.build_env.get_out_firmware_root_path()) {
            throw std::runtime_error("supplier compilation was redirected to pre-existing firmware");
        }
        const auto key = environment.build_env.get_build_key();
        auto tuple = describe_config(config);
        tuple["sdk_build_key"] = std::to_string(key);
        tuple["enumeration_index"] = report["configurations"].size();
        report["configurations"].push_back(std::move(tuple));
        Json recipes = Json::array();
        for (const auto& state : environment.firmware_build_states) recipes.push_back(describe_recipe(state.export_target_recipe(nullptr)));
        const auto [previous, fresh] = recipes_by_key.emplace(key, recipes);
        if (!fresh) {
            if (previous->second != recipes) throw std::runtime_error("one SDK build key selected different firmware recipes");
            return;
        }
        const auto& command = environment.build_env.get_gpp();
        const auto [toolchain, new_toolchain] = toolchain_indices.emplace(command, toolchain_indices.size());
        if (new_toolchain) {
            const auto version = jit_build::utils::compiler_version(command);
            auto argv = jit_build::utils::tokenize_flags(command);
            if (argv.empty()) throw std::runtime_error("SDK selected an empty firmware compiler command");
            const auto compiler = fs::canonical(argv.back());
            const auto transcript = publication.cache() / ("compiler-version-" + std::to_string(toolchain->second) + ".txt");
            argv.push_back("--version");
            if (!jit_build::utils::exec_command(argv, sdk.string(), transcript.string())) {
                throw std::runtime_error("cannot record actual firmware compiler version");
            }
            const auto version_path = fs::path("provenance/toolchains") / std::to_string(toolchain->second) / "version.txt";
            publication.copy(transcript, version_path);
            report["toolchains"].push_back({{"command", command}, {"executable", compiler.string()},
                {"version", version}, {"version_and_license_output", version_path.generic_string()}});
        }
        std::cout << "Building Blackhole firmware " << key << " (" << environment.firmware_build_states.size()
                  << " targets)" << std::endl;
        manager.build_firmware(0, /*ignore_precompiled=*/true);
        auto build = publish_artifacts(publication, environment, std::to_string(key));
        for (const auto& record : build["native_roles"]) {
            if (record["profile"] != (profile == DeviceProfilerMode::Program ? TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM : TT_NATIVE_PROFILE_ABSENT)) {
                throw std::runtime_error("built firmware disagrees with explicit profiler mode");
            }
        }
        build["toolchain_index"] = toolchain->second;
        build["recipes"] = std::move(recipes);
        capture_build_inputs(environment, sources);
        report["builds"].push_back(std::move(build));
    });
    if (report["configurations"].empty()) throw std::runtime_error("official offline catalog has no Blackhole configurations");
    sources.verify();
    report["source_snapshots"] = std::move(sources.records);
    const auto description = publication.payload() / "publication.json";
    std::ofstream output(description, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << report.dump(2) << '\n';
    output.close();
    publication.commit();
    std::cout << "Published " << report["builds"].size() << " SDK build keys for " << report["configurations"].size()
              << " Blackhole catalog configurations to " << arguments.output << '\n';
}
}  // namespace
}  // namespace tt::tt_metal::native_firmware_bundle

int main(int argc, char** argv) {
    try {
        std::vector<std::string_view> arguments;
        for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
        const auto options = tt::tt_metal::native_firmware_bundle::parse_options(arguments);
        tt::tt_metal::native_firmware_bundle::publish(options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native_firmware_bundle: " << error.what() << '\n';
        return 1;
    }
}
