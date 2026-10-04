#include "BundlePublication.hpp"
#include "DeploymentBuild.hpp"
#include "DeploymentCatalog.hpp"
#include "DeploymentInputs.hpp"
#include "NativeDeploymentPublish.hpp"
#include "RecipeProvenance.hpp"
#include "SourceArchive.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include "llrt/hal.hpp"
#include "llrt/firmware_capability.hpp"
#include "llrt/rtoptions.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TT_NATIVE_FIRMWARE_SDK_ROOT
#error "The supplier tool must be built against an explicit SDK source root"
#endif

namespace tt::tt_metal::native_firmware_bundle {
namespace {
namespace fs = std::filesystem;

void require_shell_path(const fs::path& path) {
    if (path.string().find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-") !=
        std::string::npos)
        throw std::invalid_argument("SDK supplier build cannot safely represent path: " + path.string());
}

std::vector<std::byte> read_bytes(const fs::path& path, size_t limit) {
    if (!fs::is_regular_file(fs::symlink_status(path)))
        throw std::invalid_argument("supplier input must be a regular non-symlink file: " + path.string());
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || static_cast<uint64_t>(file.tellg()) > limit)
        throw std::invalid_argument("supplier input is empty or exceeds its size limit: " + path.string());
    std::vector<std::byte> result(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(result.data()), result.size()))
        throw std::runtime_error("cannot read supplier input: " + path.string());
    return result;
}

void write_bytes(const fs::path& path, std::span<const std::byte> bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();
}

void publish(const DeploymentPublishOptions& paths) {
    const auto input_bytes = read_bytes(paths.input, 1024 * 1024);
    const auto expected_program = read_bytes(paths.program_record, 16 * 1024 * 1024);
    const auto facts = parse_deployment_input(std::string_view(
        reinterpret_cast<const char*>(input_bytes.data()), input_bytes.size()));
    Publication publication(paths.output);
    const auto sdk = fs::canonical(TT_NATIVE_FIRMWARE_SDK_ROOT);
    require_shell_path(sdk);
    require_shell_path(publication.cache());
    const auto mode = facts.profile == "program" ? DeviceProfilerMode::Program : DeviceProfilerMode::Disabled;
    llrt::RunTimeOptions options(llrt::RunTimeOptions::ExplicitBuildOptions{
        .root_dir = sdk.string(), .cache_dir = publication.cache().string(), .profiler_mode = mode});
    options.set_disable_precompiled_fw(true);
    options.set_enable_2_erisc_mode(true);
    const auto profiler_bytes = get_profiler_dram_bank_size_for_hal_allocation(options);
    FirmwareCapabilityRequest requested;
    requested.dram_programmable_cores = true;
    FirmwareCapabilityResult capabilities;
    check_firmware_capabilities(tt::ARCH::BLACKHOLE,
        {.firmware_bundle = facts.firmware_bundle}, requested, capabilities);
    Hal hal(tt::ARCH::BLACKHOLE, false, true, profiler_bytes, false, false,
        capabilities.dram_programmable_cores, true);
    tt::ChipInfo chip{};
    chip.noc_translation_enabled = facts.noc_translation_enabled;
    chip.board_type = tt::board_type_from_string(facts.board_type);
    chip.board_id = facts.board_id;
    chip.asic_location = facts.asic_location;
    chip.harvesting_masks = {facts.tensix_harvesting_mask, facts.dram_harvesting_mask,
        facts.eth_harvesting_mask, facts.pcie_harvesting_mask, facts.l2cpu_harvesting_mask};
    HostQueueBacking host{facts.host_channel, facts.host_channel_size, facts.host_galaxy,
        facts.d2h_hugepage_fallback, facts.cq_size_override};
    const experimental::catalog::Input input{sdk, chip, facts.device_id, host, facts.active_ethernet_channels};
    const auto actual = experimental::catalog::make_input(input, hal, options);
    const auto program = experimental::plan_dispatch_program(actual.program);
    require_program_record(expected_program, program.canonical_record());
    SourceArchive sources(publication, sdk);
    nlohmann::json provenance = {{"format", "tt-metal-native-deployment-publication-v2"},
        {"sdk_source_root", sdk.string()}, {"profile", facts.profile},
        {"license_snapshots", nlohmann::json::array()}, {"recipes", nlohmann::json::array()}};
    for (const auto& license : {sdk / "LICENSE", sdk / "tt_metal/tt-llk/LICENSE",
                                sdk / "tt_metal/tt-llk/LICENSE_understanding.txt"})
        provenance["license_snapshots"].push_back(sources.capture(license).generic_string());
    for (const auto directory : {"tt_metal/core_descriptors", "tt_metal/soc_descriptors",
        "tt_metal/hw/toolchain", "tt_metal/tools/native_firmware_bundle", "tt_metal/jit_build",
        "tt_metal/impl/experimental/offline_compile"})
        sources.capture_tree(sdk / directory);
    auto built = build_deployment(actual.device, options, program);
    const auto archive = built.deployment.to_archive();
    const auto decoded = experimental::PublishedDeployment::from_archive(archive, hal);
    if (!std::ranges::equal(decoded.to_archive(), archive))
        throw std::runtime_error("supplier v2 archive roundtrip differs from admitted deployment");
    auto compiler_argv = jit_build::utils::tokenize_flags(built.compiler);
    if (compiler_argv.empty()) throw std::runtime_error("supplier selected an empty compiler command");
    const auto compiler = fs::canonical(compiler_argv.back());
    const auto version = jit_build::utils::compiler_version(built.compiler);
    compiler_argv.push_back("--version");
    const auto transcript = publication.cache() / "compiler-version.txt";
    if (!jit_build::utils::exec_command(compiler_argv, sdk.string(), transcript.string()))
        throw std::runtime_error("cannot record supplier compiler version");
    publication.copy(transcript, "provenance/toolchain-version.txt");
    provenance["toolchain"] = {{"command", built.compiler}, {"executable", compiler.string()},
        {"version", version}, {"version_and_license_output", "provenance/toolchain-version.txt"}};
    for (const auto& recipe : built.recipes) {
        provenance["recipes"].push_back({{"kind", static_cast<uint32_t>(recipe.kind)},
            {"node", recipe.node}, {"core_type", static_cast<uint32_t>(recipe.processor.core_type)},
            {"processor_class", static_cast<uint32_t>(recipe.processor.processor_class)},
            {"processor_type", recipe.processor.processor_type}, {"image", recipe.image.string()},
            {"recipe", describe_recipe(recipe.recipe)}});
        sources.capture_recipe(recipe.recipe, recipe.image);
    }
    sources.verify();
    provenance["source_snapshots"] = std::move(sources.records);
    std::ofstream report(publication.payload() / "publication.json", std::ios::binary);
    report.exceptions(std::ios::failbit | std::ios::badbit);
    report << provenance.dump(2) << '\n';
    report.close();
    const auto staged = publication.cache() / "published-deployment.v2";
    const auto staged_input = publication.cache() / "inputs.json";
    const auto staged_program = publication.cache() / "dispatch-program.record";
    write_bytes(staged, archive);
    write_bytes(staged_input, input_bytes);
    write_bytes(staged_program, expected_program);
    publication.copy(staged, "deployment/published-deployment.v2");
    publication.copy(staged_input, "provenance/inputs.json");
    publication.copy(staged_program, "provenance/dispatch-program.record");
    publication.commit();
    std::cout << "Published complete v2 deployment (" << archive.size() << " bytes) to "
              << paths.output << '\n';
}
}
}

int main(int argc, char** argv) {
    try {
        std::vector<std::string_view> arguments;
        for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
        tt::tt_metal::native_firmware_bundle::publish(
            tt::tt_metal::native_firmware_bundle::parse_deployment_options(arguments));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native_deployment_publish: " << error.what() << '\n';
        return 1;
    }
}
