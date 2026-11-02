// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include "impl/experimental/published_deployment/storage.hpp"
#include "impl/experimental/published_deployment/record_writer.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include "tools/native_firmware_bundle/DeploymentCatalog.hpp"

#ifndef TT_NATIVE_FIRMWARE_SDK_ROOT
#error "Wormhole archive tests require an explicit SDK source root"
#endif

namespace tt::tt_metal::experimental {
namespace {
using namespace test;

// Independent wire helpers for malformed archive inputs. They intentionally
// use fixed V2 envelope offsets rather than the product decoder.
uint64_t get_word(std::span<const std::byte> bytes, size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 8) {
        throw std::logic_error("bad test offset");
    }
    uint64_t value = 0;
    for (unsigned i = 0; i != 8; ++i) {
        value |= uint64_t(std::to_integer<uint8_t>(bytes[offset + i])) << (8 * i);
    }
    return value;
}
void put_word(std::vector<std::byte>& bytes, size_t offset, uint64_t value) {
    if (offset > bytes.size() || bytes.size() - offset < 8) {
        throw std::logic_error("bad test offset");
    }
    for (unsigned i = 0; i != 8; ++i) {
        bytes[offset + i] = std::byte(value >> (8 * i));
    }
}
void append_word(std::vector<std::byte>& bytes, uint64_t value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + 8);
    put_word(bytes, offset, value);
}
std::vector<std::byte> wire_archive(std::span<const std::byte> config, std::span<const PublishedImageInput> images) {
    constexpr char magic[] = "TTDPUB\r\n";
    std::vector<std::byte> bytes;
    for (size_t i = 0; i != 8; ++i) {
        bytes.push_back(std::byte(magic[i]));
    }
    append_word(bytes, 2);
    append_word(bytes, 0);
    append_word(bytes, config.size());
    bytes.insert(bytes.end(), config.begin(), config.end());
    append_word(bytes, images.size());
    for (const auto& image : images) {
        append_word(bytes, static_cast<uint64_t>(image.kind));
        append_word(bytes, static_cast<uint64_t>(image.processor.core_type));
        append_word(bytes, static_cast<uint64_t>(image.processor.processor_class));
        append_word(bytes, image.processor.processor_type);
        append_word(bytes, image.dispatch_node);
        append_word(bytes, image.bytes.size());
        bytes.insert(bytes.end(), image.bytes.begin(), image.bytes.end());
    }
    put_word(bytes, 16, bytes.size());
    return bytes;
}
size_t program_record_offset(std::span<const std::byte> config) {
    // V2 DeviceConfiguration is 106 bytes; BuildConfiguration is 19 bytes.
    size_t offset = 125;
    const size_t processors = get_word(config, offset);
    offset += 8;
    for (size_t processor = 0; processor != processors; ++processor) {
        deployment_detail::RecordWriter processor_layout;
        processor_layout.append(HalProcessorIdentifier{});
        processor_layout.append(HalJitBuildConfig{});
        offset += processor_layout.bytes.size();
        deployment_detail::RecordWriter regions;
        regions.append(HalProcessorImageRegions{});
        offset += regions.bytes.size();
    }
    const size_t cores = get_word(config, offset);
    offset += 8;
    for (size_t core = 0; core != cores; ++core) {
        offset += 8;  // core type
        for (unsigned list = 0; list != 2; ++list) {
            const size_t count = get_word(config, offset);
            offset += 8;
            for (size_t i = 0; i != count; ++i) {
                const bool present = config[offset++] != std::byte{};
                if (present) {
                    offset += 8;
                }
            }
        }
    }
    for (unsigned list = 0; list != 2; ++list) {
        offset += 8 + get_word(config, offset) * 8;
    }
    return offset + 8;  // skip program byte count
}
template <class T>
size_t encoded_size(const T& value) {
    deployment_detail::RecordWriter writer;
    writer.append(value);
    return writer.bytes.size();
}
size_t first_dispatch(const DeploymentConfiguration& config) {
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    return program_record_offset(canonical) + 16 + encoded_size(config.dispatch_program().inputs()) + 8;
}

struct WormholeArchiveCandidate {
    llrt::RunTimeOptions options;
    Hal hal;
    catalog::Output actual;
    DeploymentConfiguration config;
    PublicationInput images;

    explicit WormholeArchiveCandidate(DeviceProfilerMode mode = DeviceProfilerMode::Disabled) :
        options(
            llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = TT_NATIVE_FIRMWARE_SDK_ROOT, .profiler_mode = mode}),
        hal(tt::ARCH::WORMHOLE_B0,
            true,
            false,
            get_profiler_dram_bank_size_for_hal_allocation(options),
            false,
            false,
            false,
            options.get_eth_ptp_trace()),
        actual([&] {
            options.set_enable_2_erisc_mode(false);
            const catalog::Input input{
                TT_NATIVE_FIRMWARE_SDK_ROOT,
                tt::ARCH::WORMHOLE_B0,
                {.noc_translation_enabled = true,
                 .harvesting_masks = {.tensix_harvesting_mask = 0x201},
                 .board_type = BoardType::N300},
                7,
                HostQueueBacking{0, 0x40000000, false, false, std::nullopt},
                {}};
            return catalog::make_input(input, hal, options);
        }()),
        config(DeploymentConfiguration::from_sdk(actual.device, options, plan_dispatch_program(actual.program))),
        images(config, hal) {
        // The shared ELF fixture defaults to Blackhole. Give every actual
        // Wormhole Tensix firmware role its own native architecture/profile.
        for (size_t i = 0; i != images.input.size(); ++i) {
            const auto& image = images.input[i];
            if (image.kind != PublishedImageKind::Firmware ||
                image.processor.core_type != HalProgrammableCoreType::TENSIX) {
                continue;
            }
            images.images[i].put(
                0x90 + offsetof(tt_native_image_record, architecture), uint8_t{TT_NATIVE_ARCH_WORMHOLE});
            images.images[i].put(
                0x90 + offsetof(tt_native_image_record, profile),
                static_cast<uint8_t>(
                    mode == DeviceProfilerMode::Program ? TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM
                                                        : TT_NATIVE_PROFILE_ABSENT));
        }
        for (size_t i = 0; i != images.input.size(); ++i) {
            auto& image = images.input[i];
            if (image.kind != PublishedImageKind::Dispatch) {
                continue;
            }
            const auto regions = hal.get_processor_image_regions(image.processor).value();
            const auto text_end = (regions.firmware_text.base + 4 + regions.firmware_kernel_pad + 15) & ~uint64_t{15};
            const auto data_end = (regions.local_data.base + 4 + 15) & ~uint64_t{15};
            const auto kernel =
                hal.get_processor_kernel_regions(image.processor, text_end, data_end, options.get_erisc_iram_enabled())
                    .value();
            images.images[i] = PublishedElf(
                kernel.text.base,
                config.image_record(PublishedImageKind::Dispatch, image.processor, image.dispatch_node),
                std::nullopt,
                true,
                kernel.data.base,
                regions.local_data.base);
            image.bytes = images.images[i].bytes;
        }
    }
};

void remove_firmware_layout_exports(PublishedElf& image) {
    for (size_t slot = 3; slot != 5; ++slot) {
        // These fixture entries name the two main.ld-only absolute exports.
        image.put(0x140 + slot * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_name), uint32_t{0});
    }
}

TEST(WormholePublishedDeploymentArchive, ProcessorImageRegionsMatchWormholeLinkerMemoryAbi) {
    Hal hal{tt::ARCH::WORMHOLE_B0, true, false, 0, false, false, false, true};
    struct Expected {
        HalProcessorIdentifier processor;
        DeviceAddr text;
        uint32_t text_size;
        DeviceAddr data;
        uint32_t data_size;
        uint32_t kernel_size;
        uint32_t pad;
        DeviceAddr initialization;
        uint32_t initialization_size;
    };
    // WH dev_mem_map.h and main.ld reserve a per-role stack from local data.
    // Active Ethernet instead uses the fixed erisc-b0-memory.ld regions.
    const std::array expected{
        Expected{
            {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0},
            14944,
            6528,
            0xffb00000,
            4096 - 256,
            1432 * 1024,
            0,
            35104,
            4096},
        Expected{
            {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 1},
            21472,
            2432,
            0xffb00000,
            4096 - 256,
            16 * 1024,
            0,
            39200,
            4096},
        Expected{
            {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0},
            23904,
            2144,
            0xffb00000,
            2048 - 192,
            1432 * 1024,
            0,
            43296,
            2048},
        Expected{
            {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 1},
            26048,
            1184,
            0xffb00000,
            2048 - 192,
            1432 * 1024,
            0,
            45344,
            2048},
        Expected{
            {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 2},
            27232,
            2080,
            0xffb00000,
            2048 - 256,
            1432 * 1024,
            0,
            47392,
            2048},
        Expected{
            {HalProgrammableCoreType::IDLE_ETH, HalProcessorClassType::DM, 0},
            13264,
            16 * 1024,
            0xffb00000,
            4096 - 128,
            24 * 1024,
            32,
            32240,
            4096},
        Expected{
            {HalProgrammableCoreType::ACTIVE_ETH, HalProcessorClassType::DM, 0},
            0x9040,
            6 * 1024,
            0x103e0,
            3104,
            17312,
            0,
            0x103e0,
            3104}};
    for (const auto& value : expected) {
        SCOPED_TRACE(static_cast<unsigned>(value.processor.core_type));
        SCOPED_TRACE(static_cast<unsigned>(value.processor.processor_class));
        SCOPED_TRACE(value.processor.processor_type);
        const auto regions = hal.get_processor_image_regions(value.processor);
        ASSERT_TRUE(regions.has_value());
        EXPECT_EQ(regions->firmware_text.base, value.text);
        EXPECT_EQ(regions->firmware_text.size, value.text_size);
        EXPECT_EQ(regions->local_data.base, value.data);
        EXPECT_EQ(regions->local_data.size, value.data_size);
        EXPECT_EQ(regions->kernel_text_size, value.kernel_size);
        EXPECT_EQ(regions->firmware_kernel_pad, value.pad);
        EXPECT_EQ(regions->local_initialization.base, value.initialization);
        EXPECT_EQ(regions->local_initialization.size, value.initialization_size);
        const auto core = hal.get_programmable_core_type_index(value.processor.core_type);
        EXPECT_EQ(
            regions->firmware_text.base,
            hal.get_jit_build_config(
                   core, static_cast<uint32_t>(value.processor.processor_class), value.processor.processor_type)
                .fw_base_addr);
    }
}

TEST(WormholePublishedDeploymentArchive, ProcessorImageRegionsRejectAbsentRolesAndUnknownEnums) {
    Hal hal{tt::ARCH::WORMHOLE_B0, true, false, 0, false, false, false, true};
    const std::array absent{
        HalProcessorIdentifier{HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, -1},
        HalProcessorIdentifier{HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 2},
        HalProcessorIdentifier{HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 3},
        HalProcessorIdentifier{HalProgrammableCoreType::ACTIVE_ETH, HalProcessorClassType::DM, 1},
        HalProcessorIdentifier{HalProgrammableCoreType::ACTIVE_ETH, HalProcessorClassType::COMPUTE, 0},
        HalProcessorIdentifier{HalProgrammableCoreType::IDLE_ETH, HalProcessorClassType::DM, 1},
        HalProcessorIdentifier{HalProgrammableCoreType::DRAM, HalProcessorClassType::DM, 0},
        HalProcessorIdentifier{static_cast<HalProgrammableCoreType>(255), HalProcessorClassType::DM, 0},
        HalProcessorIdentifier{HalProgrammableCoreType::TENSIX, static_cast<HalProcessorClassType>(255), 0}};
    for (const auto processor : absent) {
        EXPECT_FALSE(hal.get_processor_image_regions(processor).has_value());
    }
}

TEST(WormholePublishedDeploymentArchive, KernelRegionsPreserveFixedIramAndEthernetLinkerContracts) {
    Hal hal{tt::ARCH::WORMHOLE_B0, true, false, 0, false, false, false, true};
    const HalProcessorIdentifier ncrisc{HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 1};
    const auto ncrisc_regions = hal.get_processor_kernel_regions(ncrisc, 21488, 0xffb00010, false);
    ASSERT_TRUE(ncrisc_regions.has_value());
    EXPECT_EQ(ncrisc_regions->text.base, 0xffc00000u);
    EXPECT_EQ(ncrisc_regions->text.size, 16u * 1024);
    EXPECT_EQ(ncrisc_regions->data.base, 0xffb00010u);
    EXPECT_EQ(ncrisc_regions->data.size, 4096u - 256 - 16);

    const HalProcessorIdentifier active{HalProgrammableCoreType::ACTIVE_ETH, HalProcessorClassType::DM, 0};
    for (const bool iram : {false, true}) {
        SCOPED_TRACE(iram);
        const auto regions = hal.get_processor_kernel_regions(active, 0x9050, 0x103f0, iram);
        ASSERT_TRUE(regions.has_value());
        EXPECT_EQ(regions->text.base, iram ? 0xffc00000u : 0xa840u);
        EXPECT_EQ(regions->text.size, iram ? 16u * 1024 : 17312u);
        EXPECT_EQ(regions->data.base, 0xebe0u);
        EXPECT_EQ(regions->data.size, 6144u);
    }
    const HalProcessorIdentifier brisc{HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0};
    const auto brisc_regions = hal.get_processor_kernel_regions(brisc, 14960, 0xffb00010, false);
    ASSERT_TRUE(brisc_regions.has_value());
    EXPECT_EQ(brisc_regions->text.base, 14960u);
    EXPECT_EQ(brisc_regions->text.size, 1432u * 1024 - 16);
    EXPECT_EQ(brisc_regions->data.base, 0xffb00010u);
    EXPECT_EQ(brisc_regions->data.size, 4096u - 256 - 16);
    const HalProcessorIdentifier idle{HalProgrammableCoreType::IDLE_ETH, HalProcessorClassType::DM, 0};
    const auto idle_regions = hal.get_processor_kernel_regions(idle, 13312, 0xffb00010, false);
    ASSERT_TRUE(idle_regions.has_value());
    EXPECT_EQ(idle_regions->text.base, 13312u);
    EXPECT_EQ(idle_regions->text.size, 24u * 1024 - 48);
}

TEST(WormholePublishedDeploymentArchive, KernelRegionsRejectAbsentRolesAndInvalidFirmwareBounds) {
    Hal hal{tt::ARCH::WORMHOLE_B0, true, false, 0, false, false, false, true};
    const HalProcessorIdentifier brisc{HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0};
    for (const auto& [text, data] : std::array<std::pair<uint64_t, uint64_t>, 5>{
             {{14943, 0xffb00010},
              {21488, 0xffb00010},
              {14960, 0xffafffff},
              {14960, 0xffb00f01},
              {UINT64_MAX, UINT64_MAX}}}) {
        EXPECT_FALSE(hal.get_processor_kernel_regions(brisc, text, data, false).has_value());
    }
    EXPECT_FALSE(hal.get_processor_kernel_regions(
                        {HalProgrammableCoreType::ACTIVE_ETH, HalProcessorClassType::COMPUTE, 0}, 0x9050, 0x103f0, true)
                     .has_value());
    EXPECT_FALSE(hal.get_processor_kernel_regions(
                        {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 1}, 0x100, 0xffb00010, false)
                     .has_value());
}

TEST_F(PublishedDeploymentTest, KernelRegionsPreserveActualBlackholeFirmwareHandoff) {
    for (uint32_t core = 0; core != hal.get_programmable_core_type_count(); ++core) {
        const auto type = hal.get_programmable_core_type(core);
        for (uint32_t cls = 0; cls != hal.get_processor_classes_count(type); ++cls) {
            for (uint32_t role = 0; role != hal.get_processor_class_num_fw_binaries(core, cls); ++role) {
                const HalProcessorIdentifier processor{
                    type, static_cast<HalProcessorClassType>(cls), static_cast<int>(role)};
                const auto images = hal.get_processor_image_regions(processor);
                ASSERT_TRUE(images.has_value());
                const auto text_end =
                    (images->firmware_text.base + 4 + images->firmware_kernel_pad + 15) & ~uint64_t{15};
                const auto data_end = (images->local_data.base + 4 + 15) & ~uint64_t{15};
                const auto kernel = hal.get_processor_kernel_regions(processor, text_end, data_end, false);
                ASSERT_TRUE(kernel.has_value());
                const auto text_limit =
                    (images->kernel_text_limit == HalKernelTextLimit::FromFirmwareBase ? images->firmware_text.base
                                                                                       : text_end) +
                    images->kernel_text_size;
                EXPECT_EQ(kernel->text.base, text_end);
                EXPECT_EQ(kernel->text.size, text_limit - text_end);
                EXPECT_EQ(kernel->data.base, data_end);
                EXPECT_EQ(kernel->data.size, images->local_data.base + images->local_data.size - data_end);
            }
        }
    }
}

TEST(WormholePublishedDeploymentArchive, RetainsActualN300ConfigurationAndCompleteImagesWithoutContext) {
    ASSERT_FALSE(MetalContext::instance_exists());
    for (const auto mode : {DeviceProfilerMode::Disabled, DeviceProfilerMode::Program}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        std::optional<WormholeArchiveCandidate> candidate;
        ASSERT_NO_THROW(candidate.emplace(mode));
        const auto& config = candidate->config;
        EXPECT_EQ(config.device().arch, tt::ARCH::WORMHOLE_B0);
        EXPECT_EQ(config.device().dispatch_core_axis, DispatchCoreAxis::ROW);
        EXPECT_EQ(config.device().num_dram_banks, 12u);
        EXPECT_EQ(config.device().num_l1_banks, 56u);
        EXPECT_EQ(config.device().harvesting_mask, 0x201u);
        EXPECT_EQ(config.device().max_cbs, candidate->hal.get_arch_num_circular_buffers());
        EXPECT_TRUE(config.device().routing_fw_enabled);
        EXPECT_FALSE(candidate->options.get_enable_2_erisc_mode());
        EXPECT_EQ(config.dispatch_program().family(), DispatchProgramFamily::WormholeSingleMmioWorkerOneCq);
        auto expected = PublishedDeployment::admit(config, candidate->images.input);
        const auto archive = expected.to_archive();
        const auto actual = PublishedDeployment::from_archive(archive, candidate->hal);
        const auto derived = PublishedDeployment::from_archive(archive);
        for (const auto* decoded : {&actual, &derived}) {
            EXPECT_EQ(decoded->to_archive(), archive);
            EXPECT_TRUE(decoded->configuration().matches(config));
            EXPECT_EQ(decoded->configuration().profiler_mode(), mode);
            EXPECT_EQ(
                decoded->tensix_firmware().contract(TensixKernelRole::Brisc).architecture, TT_NATIVE_ARCH_WORMHOLE);
            EXPECT_EQ(
                decoded->configuration().dispatch_program().family(),
                DispatchProgramFamily::WormholeSingleMmioWorkerOneCq);
            for (const auto processor : config.firmware_processors()) {
                EXPECT_TRUE(std::ranges::equal(decoded->firmware_image(processor), expected.firmware_image(processor)));
            }
            for (const auto& node : config.dispatch_nodes()) {
                for (const auto processor : dispatch_processors(node)) {
                    EXPECT_TRUE(
                        std::ranges::equal(
                            decoded->dispatch_image(node.node_id, processor),
                            expected.dispatch_image(node.node_id, processor)));
                }
            }
        }
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(WormholePublishedDeploymentArchive, AcceptsActiveEthernetFirmwareWithoutMainLinkerLayoutExports) {
    for (const auto mode : {DeviceProfilerMode::Disabled, DeviceProfilerMode::Program}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        std::optional<WormholeArchiveCandidate> candidate;
        ASSERT_NO_THROW(candidate.emplace(mode));
        size_t changed = 0;
        for (size_t i = 0; i != candidate->images.input.size(); ++i) {
            const auto& input = candidate->images.input[i];
            if (input.kind == PublishedImageKind::Firmware &&
                input.processor.core_type == HalProgrammableCoreType::ACTIVE_ETH) {
                remove_firmware_layout_exports(candidate->images.images[i]);
                ++changed;
            }
        }
        ASSERT_EQ(changed, 1u);
        std::optional<PublishedDeployment> published;
        ASSERT_NO_THROW(published.emplace(PublishedDeployment::admit(candidate->config, candidate->images.input)));
        const auto archive = published->to_archive();
        EXPECT_NO_THROW(PublishedDeployment::from_archive(archive, candidate->hal));
        EXPECT_NO_THROW(PublishedDeployment::from_archive(archive));
    }
}

TEST(WormholePublishedDeploymentArchive, RejectsMissingMainLinkerExportsOnTensixAndIdleEthernet) {
    for (const auto core : {HalProgrammableCoreType::TENSIX, HalProgrammableCoreType::IDLE_ETH}) {
        SCOPED_TRACE(static_cast<unsigned>(core));
        std::optional<WormholeArchiveCandidate> candidate;
        ASSERT_NO_THROW(candidate.emplace());
        bool changed = false;
        for (size_t i = 0; i != candidate->images.input.size(); ++i) {
            const auto& input = candidate->images.input[i];
            if (input.kind == PublishedImageKind::Firmware && input.processor.core_type == core) {
                remove_firmware_layout_exports(candidate->images.images[i]);
                changed = true;
                break;
            }
        }
        ASSERT_TRUE(changed);
        EXPECT_THROW(PublishedDeployment::admit(candidate->config, candidate->images.input), std::invalid_argument);
        const auto archive = wire_archive(
            deployment_detail::Access::configuration(candidate->config).canonical, candidate->images.input);
        EXPECT_THROW(PublishedDeployment::from_archive(archive, candidate->hal), std::invalid_argument);
        EXPECT_THROW(PublishedDeployment::from_archive(archive), std::invalid_argument);
    }
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsMissingMainLinkerExportsOnEveryBlackholeProcessor) {
    const auto config = configuration();
    for (const auto processor : config.firmware_processors()) {
        SCOPED_TRACE(static_cast<unsigned>(processor.core_type));
        SCOPED_TRACE(static_cast<unsigned>(processor.processor_class));
        SCOPED_TRACE(processor.processor_type);
        PublicationInput input(config, hal);
        bool changed = false;
        for (size_t i = 0; i != input.input.size(); ++i) {
            if (input.input[i].kind == PublishedImageKind::Firmware && input.input[i].processor == processor) {
                remove_firmware_layout_exports(input.images[i]);
                changed = true;
                break;
            }
        }
        ASSERT_TRUE(changed);
        EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
        const auto archive = wire_archive(deployment_detail::Access::configuration(config).canonical, input.input);
        EXPECT_THROW(PublishedDeployment::from_archive(archive, hal), std::invalid_argument);
        EXPECT_THROW(PublishedDeployment::from_archive(archive), std::invalid_argument);
    }
}

TEST(WormholePublishedDeploymentArchive, RejectsForeignArchitectureAndDifferentActualProfilerHal) {
    ASSERT_FALSE(MetalContext::instance_exists());
    std::optional<WormholeArchiveCandidate> candidate;
    ASSERT_NO_THROW(candidate.emplace());
    const auto archive = PublishedDeployment::admit(candidate->config, candidate->images.input).to_archive();
    Hal blackhole{tt::ARCH::BLACKHOLE, false, true, 0, false, false, true, true};
    EXPECT_THROW(PublishedDeployment::from_archive(archive, blackhole), std::invalid_argument);
    Hal profiled{tt::ARCH::WORMHOLE_B0, true, false, 8192, false, false, false, true};
    EXPECT_THROW(PublishedDeployment::from_archive(archive, profiled), std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(WormholePublishedDeploymentArchive, RejectsConsistentlyForeignNativeFirmwareArchitecture) {
    std::optional<WormholeArchiveCandidate> candidate;
    ASSERT_NO_THROW(candidate.emplace());
    for (size_t i = 0; i != candidate->images.input.size(); ++i) {
        const auto& image = candidate->images.input[i];
        if (image.kind == PublishedImageKind::Firmware &&
            image.processor.core_type == HalProgrammableCoreType::TENSIX) {
            candidate->images.images[i].put(
                0x90 + offsetof(tt_native_image_record, architecture), uint8_t{TT_NATIVE_ARCH_BLACKHOLE});
        }
    }
    EXPECT_THROW(PublishedDeployment::admit(candidate->config, candidate->images.input), std::invalid_argument);
    const auto& canonical = deployment_detail::Access::configuration(candidate->config).canonical;
    const auto archive = wire_archive(canonical, candidate->images.input);
    EXPECT_THROW(PublishedDeployment::from_archive(archive, candidate->hal), std::invalid_argument);
    EXPECT_THROW(PublishedDeployment::from_archive(archive), std::invalid_argument);
}

TEST(WormholePublishedDeploymentArchive, ConfigurationRejectsForeignDeviceAndProgramHalFacts) {
    ASSERT_FALSE(MetalContext::instance_exists());
    std::optional<WormholeArchiveCandidate> candidate;
    ASSERT_NO_THROW(candidate.emplace());
    const auto plan = plan_dispatch_program(candidate->actual.program);
    auto device = candidate->actual.device;
    Hal blackhole{tt::ARCH::BLACKHOLE, false, true, 0, false, false, true, true};
    EXPECT_THROW(
        DeploymentConfiguration::from_sdk(JitDeviceConfig{device, &blackhole}, candidate->options, plan),
        std::invalid_argument);
    for (const auto architecture : {tt::ARCH::BLACKHOLE, tt::ARCH::QUASAR, tt::ARCH::Invalid}) {
        SCOPED_TRACE(static_cast<unsigned>(architecture));
        device.arch = architecture;
        EXPECT_THROW(
            DeploymentConfiguration::from_sdk(JitDeviceConfig{device, &candidate->hal}, candidate->options, plan),
            std::invalid_argument);
    }
    auto inputs = candidate->actual.program;
    ++inputs.hal.worker_go_message;
    const auto changed = plan_dispatch_program(inputs);
    EXPECT_THROW(
        DeploymentConfiguration::from_sdk(candidate->actual.device, candidate->options, changed),
        std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(WormholePublishedDeploymentArchive, RejectsWrongFamilyAndUnknownOrForeignArchitectureBeforeImageAdmission) {
    std::optional<WormholeArchiveCandidate> candidate;
    ASSERT_NO_THROW(candidate.emplace());
    const auto& canonical = deployment_detail::Access::configuration(candidate->config).canonical;
    const auto start = program_record_offset(canonical);
    ASSERT_EQ(get_word(canonical, 0), static_cast<uint64_t>(tt::ARCH::WORMHOLE_B0));
    ASSERT_EQ(
        get_word(canonical, start + 8), static_cast<uint64_t>(DispatchProgramFamily::WormholeSingleMmioWorkerOneCq));
    ASSERT_EQ(get_word(canonical, start + 16), static_cast<uint64_t>(tt::ARCH::WORMHOLE_B0));
    for (const auto& [offset, value] : std::vector<std::pair<size_t, uint64_t>>{
             {0, static_cast<uint64_t>(tt::ARCH::BLACKHOLE)},
             {0, static_cast<uint64_t>(tt::ARCH::QUASAR)},
             {0, UINT64_MAX},
             {start + 8, static_cast<uint64_t>(DispatchProgramFamily::BlackholeSingleMmioWorkerOneCq)},
             {start + 8, UINT64_MAX},
             {start + 16, static_cast<uint64_t>(tt::ARCH::BLACKHOLE)},
             {start + 16, UINT64_MAX}}) {
        SCOPED_TRACE(offset);
        SCOPED_TRACE(value);
        auto changed = canonical;
        put_word(changed, offset, value);
        const auto archive = wire_archive(changed, candidate->images.input);
        EXPECT_THROW(PublishedDeployment::from_archive(archive, candidate->hal), std::invalid_argument);
        EXPECT_THROW(PublishedDeployment::from_archive(archive), std::invalid_argument);
    }
}

TEST(WormholePublishedDeploymentArchive, RechecksChangedHalAndDerivedFactsWithConsistentElfPublication) {
    std::optional<WormholeArchiveCandidate> candidate;
    ASSERT_NO_THROW(candidate.emplace());
    const auto& canonical = deployment_detail::Access::configuration(candidate->config).canonical;
    const auto start = program_record_offset(canonical);
    const auto& plan = candidate->config.dispatch_program();
    const auto hal = start + 16 + encoded_size(plan.inputs()) - encoded_size(plan.inputs().hal);
    const auto workers = start + 16 + encoded_size(plan.inputs().topology);
    const auto nodes = start + 16 + encoded_size(plan.inputs());
    const auto semaphores = nodes + encoded_size(plan.dispatch_nodes());
    for (const size_t offset : {hal + 8, workers, nodes + 8, semaphores + 8 + 32}) {
        SCOPED_TRACE(offset);
        auto changed = canonical;
        changed[offset] ^= std::byte{1};
        PublicationInput images = candidate->images;
        for (size_t i = 0; i != images.images.size(); ++i) {
            auto& elf = images.images[i];
            std::copy(changed.begin(), changed.end(), elf.bytes.begin() + elf.publication_offset + 64);
            images.input[i].bytes = elf.bytes;
        }
        const auto archive = wire_archive(changed, images.input);
        EXPECT_THROW(PublishedDeployment::from_archive(archive, candidate->hal), std::invalid_argument);
        EXPECT_THROW(PublishedDeployment::from_archive(archive), std::invalid_argument);
    }
}

TEST_F(PublishedDeploymentTest, ArchiveOwnsAllFirmwareAndSubordinateCompanionBytesWithoutContext) {
    ASSERT_FALSE(MetalContext::instance_exists());
    auto config = configuration();
    PublicationInput input(config, hal);
    auto expected = PublishedDeployment::admit(config, input.input);
    auto archive = expected.to_archive();
    auto restored = PublishedDeployment::from_archive(archive, hal);
    EXPECT_EQ(restored.to_archive(), archive);
    std::fill(archive.begin(), archive.end(), std::byte{});
    input.images.clear();
    input.input.clear();
    EXPECT_TRUE(restored.configuration().matches(config));
    ASSERT_EQ(restored.configuration().firmware_processors().size(), 10u);
    for (auto processor : config.firmware_processors()) {
        EXPECT_TRUE(std::ranges::equal(restored.firmware_image(processor), expected.firmware_image(processor)));
    }
    size_t dispatch_count = 0;
    for (const auto& n : config.dispatch_nodes()) {
        for (auto processor : dispatch_processors(n)) {
            ++dispatch_count;
            EXPECT_TRUE(
                std::ranges::equal(
                    restored.dispatch_image(n.node_id, processor), expected.dispatch_image(n.node_id, processor)));
        }
    }
    EXPECT_EQ(dispatch_count, 6u);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsDifferentActualHal) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto archive = PublishedDeployment::admit(config, input.input).to_archive();
    Hal other{tt::ARCH::BLACKHOLE, false, true, 0, false, false, false, true};
    EXPECT_THROW(PublishedDeployment::from_archive(archive, other), std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedDeploymentTest, ArchiveDerivesBothDramHalRequirementsWithoutContext) {
    ASSERT_FALSE(MetalContext::instance_exists());
    for (bool dram : {false, true}) {
        SCOPED_TRACE(dram);
        Hal expected_hal{tt::ARCH::BLACKHOLE, false, true, 0, false, false, dram, true};
        auto config = DeploymentConfiguration::from_sdk(JitDeviceConfig{device, &expected_hal}, options, plan());
        PublicationInput input(config, expected_hal);
        auto archive = PublishedDeployment::admit(config, input.input).to_archive();
        const auto decoded = PublishedDeployment::from_archive(archive);
        EXPECT_EQ(decoded.to_archive(), archive);
        EXPECT_TRUE(decoded.configuration().matches(config));
        EXPECT_EQ(decoded.configuration().firmware_processors().size(), dram ? 10u : 9u);
        Hal different{tt::ARCH::BLACKHOLE, false, true, 0, false, false, !dram, true};
        EXPECT_THROW(PublishedDeployment::from_archive(archive, different), std::invalid_argument);
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedDeploymentTest, ArchiveDerivedHalStillRejectsForgedStoredLayouts) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto archive = PublishedDeployment::admit(config, input.input).to_archive();
    // Configuration begins after the 32-byte envelope. Device/build occupy
    // 125 bytes, followed by the processor-layout count and processor identity.
    const size_t first_layout = 32 + 125 + 8;
    put_word(archive, first_layout + 3 * 8, get_word(archive, first_layout + 3 * 8) + 4);
    EXPECT_THROW(PublishedDeployment::from_archive(archive), std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedDeploymentTest, ArchiveDerivedHalRetainsEnvelopeAndBoundsRejection) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto archive = PublishedDeployment::admit(config, input.input).to_archive();
    for (size_t length : {size_t{0}, size_t{8}, size_t{31}, size_t{32}, archive.size() - 1}) {
        SCOPED_TRACE(length);
        EXPECT_THROW(PublishedDeployment::from_archive(std::span(archive).first(length)), std::invalid_argument);
    }
    archive.push_back(std::byte{});
    EXPECT_THROW(PublishedDeployment::from_archive(archive), std::invalid_argument);
    put_word(archive, 16, archive.size());
    EXPECT_THROW(PublishedDeployment::from_archive(archive), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, ArchiveWireContainsCompleteConfigurationAndUnmodifiedElfBytes) {
    auto config = configuration();
    PublicationInput input(config, hal);
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    EXPECT_EQ(PublishedDeployment::admit(config, input.input).to_archive(), wire_archive(canonical, input.input));
    auto decoded = PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal);
    EXPECT_TRUE(decoded.configuration().matches(config));
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsEveryTruncationAndTrailingBytes) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto bytes = PublishedDeployment::admit(config, input.input).to_archive();
    for (size_t length = 0; length != bytes.size(); ++length) {
        SCOPED_TRACE(length);
        EXPECT_THROW(PublishedDeployment::from_archive(std::span(bytes).first(length), hal), std::invalid_argument);
    }
    bytes.push_back(std::byte{});
    EXPECT_THROW(PublishedDeployment::from_archive(bytes, hal), std::invalid_argument);
    put_word(bytes, 16, bytes.size());
    EXPECT_THROW(PublishedDeployment::from_archive(bytes, hal), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsEveryTruncatedConfigurationWithValidEnvelopeLength) {
    auto config = configuration();
    PublicationInput input(config, hal);
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    for (size_t length = 0; length != canonical.size(); ++length) {
        SCOPED_TRACE(length);
        EXPECT_THROW(
            PublishedDeployment::from_archive(wire_archive(std::span(canonical).first(length), input.input), hal),
            std::invalid_argument);
    }
}

TEST_F(PublishedDeploymentTest, ArchiveRetainsProgramProfileAndRejectsFirmwareProfileMismatch) {
    llrt::RunTimeOptions profiled{
        llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = "/tmp", .profiler_mode = DeviceProfilerMode::Program}};
    profiled.set_enable_2_erisc_mode(true);
    constexpr uint32_t profiler_bytes = 8192;
    Hal profiled_hal{tt::ARCH::BLACKHOLE, false, true, profiler_bytes, false, false, true, true};
    device.profiler_dram_bank_size_per_risc_bytes = profiler_bytes;
    program.hal = capture_dispatch_hal(profiled_hal);
    const auto config = DeploymentConfiguration::from_sdk(JitDeviceConfig{device, &profiled_hal}, profiled, plan());
    PublicationInput input(config, profiled_hal);
    for (size_t role = 0; role != 5; ++role) {
        input.images[role].put(
            0x90 + offsetof(tt_native_image_record, profile), uint8_t{TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM});
    }
    auto bytes = PublishedDeployment::admit(config, input.input).to_archive();
    auto decoded = PublishedDeployment::from_archive(bytes, profiled_hal);
    EXPECT_EQ(decoded.configuration().profiler_mode(), DeviceProfilerMode::Program);
    EXPECT_EQ(decoded.configuration().device().profiler_dram_bank_size_per_risc_bytes, profiler_bytes);
    EXPECT_EQ(decoded.to_archive(), bytes);
    const auto derived = PublishedDeployment::from_archive(bytes);
    EXPECT_EQ(derived.to_archive(), bytes);
    EXPECT_EQ(derived.configuration().profiler_mode(), DeviceProfilerMode::Program);
    EXPECT_EQ(derived.configuration().device().profiler_dram_bank_size_per_risc_bytes, profiler_bytes);
    for (size_t role = 0; role != 5; ++role) {
        input.images[role].put(0x90 + offsetof(tt_native_image_record, profile), uint8_t{TT_NATIVE_PROFILE_ABSENT});
    }
    EXPECT_THROW(
        PublishedDeployment::from_archive(
            wire_archive(deployment_detail::Access::configuration(config).canonical, input.input), profiled_hal),
        std::invalid_argument);
    EXPECT_THROW(PublishedDeployment::from_archive(bytes, hal), std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsVersionAndOversizedLengthsBeforeAllocation) {
    auto config = configuration();
    PublicationInput input(config, hal);
    const auto original = PublishedDeployment::admit(config, input.input).to_archive();
    const size_t image_count = 32 + get_word(original, 24);
    for (const auto& [offset, value] : std::vector<std::pair<size_t, uint64_t>>{
             {0, 0},
             {8, 0},
             {8, 1},
             {8, UINT64_MAX},
             {16, UINT64_MAX},
             {24, 0},
             {24, UINT64_MAX},
             {24, 16 * 1024 * 1024 + 1},
             {image_count, 4097},
             {image_count, UINT64_MAX},
             {image_count + 8 + 40, 0},
             {image_count + 8 + 40, UINT64_MAX},
             {image_count + 8 + 40, 64 * 1024 * 1024 + 1}}) {
        auto bytes = original;
        put_word(bytes, offset, value);
        SCOPED_TRACE(offset);
        EXPECT_THROW(PublishedDeployment::from_archive(bytes, hal), std::invalid_argument);
    }
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsInvalidEnumBooleanNarrowingAndSequenceCounts) {
    auto config = configuration();
    PublicationInput input(config, hal);
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    const size_t node = first_dispatch(config);
    ASSERT_EQ(get_word(canonical, node), 7u);
    // Resolved record occupies 165 bytes after its presence byte. The fixed
    // node prefix is 96 bytes (four identifiers, processor, coord and NOCs).
    const size_t resolved = node + 96;
    ASSERT_EQ(canonical[resolved], std::byte{1});
    const size_t variant = resolved + 1 + 165;
    ASSERT_EQ(get_word(canonical, variant), 1u);
    for (const size_t offset :
         {size_t{0},
          size_t{48},
          size_t{56},
          size_t{64},
          size_t{106},
          size_t{133},
          size_t{141},
          size_t{189},
          node + 72}) {
        auto changed = canonical;
        put_word(changed, offset, UINT64_MAX);
        SCOPED_TRACE(offset);
        EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(changed, input.input), hal), std::invalid_argument);
    }
    for (const size_t offset : {size_t{72}, size_t{97}, size_t{114}, resolved, variant + 8}) {
        auto changed = canonical;
        changed[offset] = std::byte{2};
        SCOPED_TRACE(offset);
        EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(changed, input.input), hal), std::invalid_argument);
    }
    for (const auto& [offset, value] : std::vector<std::pair<size_t, uint64_t>>{
             {89, 256},
             {149, UINT64_MAX},
             {125, UINT64_MAX},
             {125, 4097},
             {node - 8, UINT64_MAX},
             {variant, 0},
             {variant, 4},
             {variant, UINT64_MAX}}) {
        auto changed = canonical;
        put_word(changed, offset, value);
        SCOPED_TRACE(offset);
        EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(changed, input.input), hal), std::invalid_argument);
    }
    auto extra = canonical;
    extra.push_back(std::byte{});
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(extra, input.input), hal), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, ArchiveRechecksRequiredDispatchFieldsAndDuplicateNodes) {
    auto config = configuration();
    PublicationInput input(config, hal);
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    const size_t node = first_dispatch(config);
    const size_t resolved = node + 96;
    auto absent = canonical;
    absent.erase(absent.begin() + resolved + 1, absent.begin() + resolved + 1 + 165);
    absent[resolved] = std::byte{};
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(absent, input.input), hal), std::invalid_argument);
    // First prefetch static field is mandatory even when its value is zero.
    const size_t field = resolved + 1 + 165 + 8;
    absent = canonical;
    absent.erase(absent.begin() + field + 1, absent.begin() + field + 9);
    absent[field] = std::byte{};
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(absent, input.input), hal), std::invalid_argument);
    auto duplicate = canonical;
    duplicate.insert(duplicate.end(), canonical.begin() + node, canonical.end());
    put_word(duplicate, node - 8, 2);
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(duplicate, input.input), hal), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsMissingDuplicateUnknownImagesAndProcessorRoles) {
    auto config = configuration();
    PublicationInput input(config, hal);
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    const auto original = input.input;
    for (size_t index = 0; index != original.size(); ++index) {
        input.input = original;
        input.input.erase(input.input.begin() + index);
        SCOPED_TRACE(index);
        EXPECT_THROW(
            PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::invalid_argument);
        input.input = original;
        input.input[index] = original[(index + 1) % original.size()];
        EXPECT_THROW(
            PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::invalid_argument);
    }
    input.input = original;
    input.input.front().kind = static_cast<PublishedImageKind>(2);
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::invalid_argument);
    input.input = original;
    input.input.front().processor.processor_type = -1;
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::invalid_argument);
    input.input = original;
    input.input.front().dispatch_node = 7;
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, ArchiveRechecksElfMetadataEntryAndFirmwareImports) {
    auto config = configuration();
    PublicationInput input(config, hal);
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    auto& elf = input.images.back();
    const auto original = elf.bytes;
    elf.put(elf.publication_offset + 8, uint64_t{1});
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::exception);
    std::copy(original.begin(), original.end(), elf.bytes.begin());
    elf.put(offsetof(Elf32_Ehdr, e_entry), uint32_t{0});
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::exception);
    std::copy(original.begin(), original.end(), elf.bytes.begin());
    elf.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value), uint32_t{0x2004});
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::exception);
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsForgedHalEvenWithConsistentElfPublicationRecords) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto canonical = deployment_detail::Access::configuration(config).canonical;
    // Change the first firmware's local-init layout, not an ELF payload. Every
    // ELF is updated to make its self-reported publication match the archive.
    put_word(canonical, 165, get_word(canonical, 165) + 4);
    for (auto& elf : input.images) {
        std::copy(canonical.begin(), canonical.end(), elf.bytes.begin() + elf.publication_offset + 64);
    }
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(canonical, input.input), hal), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, ArchiveRetainsActualDeviceBuildAndDispatchConfigurationDifferences) {
    auto config = configuration();
    PublicationInput original(config, hal);
    const auto original_bytes = PublishedDeployment::admit(config, original.input).to_archive();
    auto check = [&] {
        const auto changed = configuration();
        PublicationInput input(changed, hal);
        const auto bytes = PublishedDeployment::admit(changed, input.input).to_archive();
        auto decoded = PublishedDeployment::from_archive(bytes, hal);
        EXPECT_TRUE(decoded.configuration().matches(changed));
        EXPECT_FALSE(decoded.configuration().matches(config));
        EXPECT_NE(bytes, original_bytes);
    };
    device.harvesting_mask = 1;
    check();
    device.harvesting_mask = 0;
    device.num_l1_banks = 112;
    program.workers.count = 112;
    check();
    device.num_l1_banks = 120;
    program.workers.count = 120;
    device.pcie_core.x = 18;
    check();
    device.pcie_core.x = 19;
    options.set_enable_2_erisc_mode(false);
    check();
    options.set_enable_2_erisc_mode(true);
    program.workers.multicast[0] += 1;
    check();
    program.workers.multicast[0] -= 1;
    queue_size(0x40000);
    check();
}

TEST_F(PublishedDeploymentTest, ArchiveRetainsCompletePlanIncludingNonNodeActions) {
    auto config = configuration();
    PublicationInput input(config, hal);
    auto archive = PublishedDeployment::admit(config, input.input).to_archive();
    auto restored = PublishedDeployment::from_archive(archive, hal);
    const auto& expected = config.dispatch_program();
    const auto& actual = restored.configuration().dispatch_program();
    EXPECT_TRUE(std::ranges::equal(expected.canonical_record(), actual.canonical_record()));
    EXPECT_EQ(actual.semaphores().size(), 6u);
    EXPECT_EQ(actual.runtime_arguments().size(), 2u);
    EXPECT_EQ(actual.initialization().size(), expected.initialization().size());
    EXPECT_EQ(actual.core_registrations().size(), 2u);
    EXPECT_EQ(restored.configuration().dispatch_nodes().data(), actual.dispatch_nodes().data());
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsProgramVersionFamilyAndMalformedTypedInputs) {
    auto config = configuration();
    PublicationInput images(config, hal);
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    const size_t start = program_record_offset(canonical);
    // Independent offsets for the topology prefix and first placed node:
    // arch/device, two singleton device lists, four booleans, core/CQ,
    // two booleans, then the placed-node sequence.
    const size_t node_count = start + 16 + 16 + 16 + 16 + 4 + 8 + 8 + 2;
    const size_t node = node_count + 8;
    ASSERT_EQ(get_word(canonical, node_count), 3u);
    ASSERT_EQ(get_word(canonical, node), 7u);
    const size_t placement = node + 40 + 8 + 24 + 24;
    const size_t noc_array = placement + 32;
    const size_t tunnel = noc_array + 8 + 32;
    ASSERT_EQ(get_word(canonical, noc_array), 2u);
    ASSERT_EQ(get_word(canonical, tunnel), UINT64_MAX);
    const size_t profiler_array = start + 16 + encoded_size(program) - 80;
    ASSERT_EQ(get_word(canonical, profiler_array), 9u);
    const size_t topology_start = start + 16;
    const size_t topology_boolean = topology_start + 48;
    for (const auto& [offset, value] : std::vector<std::pair<size_t, uint64_t>>{
             {start - 8, UINT64_MAX},
             {start, 0},
             {start, 2},
             {start + 8, 99},
             {node_count, 0},
             {node_count, 2},
             {node_count, 4},
             {node_count, 4097},
             {node_count, UINT64_MAX},
             {node + 24, 256},
             {node + 32, UINT64_MAX},
             {node + 40, UINT64_MAX},
             {noc_array, 1},
             {noc_array, 3},
             {noc_array, UINT64_MAX},
             {tunnel, 0},
             {tunnel, UINT64_MAX - 1},
             {profiler_array, 8},
             {profiler_array, 10},
             {profiler_array, UINT64_MAX}}) {
        SCOPED_TRACE(offset);
        SCOPED_TRACE(value);
        auto changed = canonical;
        put_word(changed, offset, value);
        EXPECT_THROW(
            PublishedDeployment::from_archive(wire_archive(changed, images.input), hal), std::invalid_argument);
    }
    auto changed = canonical;
    changed[topology_boolean] = std::byte{2};
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(changed, images.input), hal), std::invalid_argument);
    // A nonnegative node identifier does not inherit the tunnel's -1 exception.
    changed = canonical;
    put_word(changed, node, UINT64_MAX);
    EXPECT_THROW(PublishedDeployment::from_archive(wire_archive(changed, images.input), hal), std::invalid_argument);
}

TEST_F(PublishedDeploymentTest, ArchiveRejectsChangedInputsAndEveryDerivedOutputEvenWithMatchingElfs) {
    auto config = configuration();
    PublicationInput images(config, hal);
    const auto& canonical = deployment_detail::Access::configuration(config).canonical;
    const size_t start = program_record_offset(canonical);
    const auto& p = config.dispatch_program();
    const size_t nodes = start + 16 + encoded_size(p.inputs());
    const size_t semaphores = nodes + encoded_size(p.dispatch_nodes());
    const size_t arguments = semaphores + encoded_size(p.semaphores());
    const size_t initialization = arguments + encoded_size(p.runtime_arguments());
    const size_t registrations = initialization + encoded_size(p.initialization());
    ASSERT_EQ(get_word(canonical, semaphores), 6u);
    ASSERT_EQ(get_word(canonical, arguments), 2u);
    for (const size_t offset :
         {start + 16 + encoded_size(p.inputs().topology),  // changed worker count
          nodes + 8,
          semaphores + 8 + 32,      // node ID / semaphore ID
          arguments + 8 + 64,       // first runtime word
          initialization + 8 + 40,  // first initialization byte
          registrations + 8}) {
        SCOPED_TRACE(offset);
        auto changed = canonical;
        changed[offset] ^= std::byte{1};
        for (auto& elf : images.images) {
            std::copy(changed.begin(), changed.end(), elf.bytes.begin() + elf.publication_offset + 64);
        }
        EXPECT_THROW(
            PublishedDeployment::from_archive(wire_archive(changed, images.input), hal), std::invalid_argument);
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental
