// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include "impl/experimental/published_deployment/storage.hpp"
#include "tools/native_firmware_bundle/PublishedImage.hpp"
#include <array>
#include <fstream>
namespace tt::tt_metal::experimental {
namespace {
using namespace test;
TEST_F(PublishedDeploymentTest, RegionsKeepTensixFirmwareContiguousBeforeDynamicCounters) {
    constexpr std::array<HalProcessorIdentifier, 5> processors{{
        {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0},
        {HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 1},
        {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0},
        {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 1},
        {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 2},
    }};
    const auto first = hal.get_processor_image_regions(processors.front()).value().firmware_text.base;
    auto next = first;
    const auto core = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    for (const auto processor : processors) {
        SCOPED_TRACE(processor.processor_type);
        const auto region = hal.get_processor_image_regions(processor).value().firmware_text;
        const auto& launch = hal.get_jit_build_config(
            core, static_cast<uint32_t>(processor.processor_class), processor.processor_type);
        EXPECT_EQ(region.base, next);
        EXPECT_EQ(launch.fw_base_addr, region.base);
        if (processor != processors.front()) EXPECT_EQ(launch.fw_launch_addr_value, region.base);
        EXPECT_GT(region.size, 0u);
        EXPECT_EQ(region.base % 16, 0u);
        EXPECT_EQ(region.size % 16, 0u);
        next += region.size;
    }
    // Freeze the shared envelope, not the individual role budgets: moving its
    // end would silently move the dynamic-counter ABI consumed by firmware.
    EXPECT_EQ(next - first, 18944u);
    EXPECT_EQ(next, hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::NOC_DYNAMIC_COUNTERS));
    EXPECT_EQ(next, 0x8520u);
}
TEST_F(PublishedDeploymentTest, RegionsAdmitFullFirmwareTextAndRejectOneInstructionBeyondItsRole) {
    auto config = configuration();
    PublicationInput input(config, hal);
    // Expand TRISC0's actual text and move every matching dispatch consumer
    // to the new firmware handoff; other roles keep their original layout.
    const size_t index = 2;
    ASSERT_EQ(input.input[index].processor,
        (HalProcessorIdentifier{HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0}));
    const auto processor = input.input[index].processor;
    const auto regions = hal.get_processor_image_regions(processor).value();
    const auto region = regions.firmware_text;
    auto& image = input.images[index];
    const uint32_t offset = (image.bytes.size() + 3) & ~uint32_t{3};
    image.bytes.resize(offset + region.size + 4);
    image.put(offset, uint32_t{0x00008067});
    image.put(sizeof(Elf32_Ehdr) + offsetof(Elf32_Phdr, p_offset), offset);
    image.put(image.sections_offset + sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_offset), offset);
    for (const uint32_t bytes : {region.size, region.size + 4}) {
        SCOPED_TRACE(bytes);
        image.put(sizeof(Elf32_Ehdr) + offsetof(Elf32_Phdr, p_filesz), bytes);
        image.put(sizeof(Elf32_Ehdr) + offsetof(Elf32_Phdr, p_memsz), bytes);
        image.put(image.sections_offset + sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_size), bytes);
        const auto text_end = static_cast<uint32_t>((region.base + bytes + 15) & ~uint64_t{15});
        image.put(0x140 + 3 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value), text_end);
        input.input[index].bytes = image.bytes;
        for (size_t consumer = 0; consumer != input.input.size(); ++consumer) {
            auto& published = input.input[consumer];
            if (published.kind != PublishedImageKind::Dispatch || published.processor != processor) continue;
            const auto data_end = (regions.local_data.base + 4 + 15) & ~uint64_t{15};
            input.images[consumer] = PublishedElf(text_end,
                config.image_record(PublishedImageKind::Dispatch, processor, published.dispatch_node),
                std::nullopt, true, data_end, regions.local_data.base);
            published.bytes = input.images[consumer].bytes;
        }
        if (bytes == region.size) {
            EXPECT_NO_THROW(PublishedDeployment::admit(config, input.input));
        } else {
            EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
        }
    }
}
TEST_F(PublishedDeploymentTest, RegionsRejectFirmwareEntryInsideRegionButDifferentFromLaunchAddress) {
    auto config = configuration();
    PublicationInput input(config, hal);
    ASSERT_NO_THROW(PublishedDeployment::admit(config, input.input));
    const size_t index = 2;
    const auto processor = input.input[index].processor;
    const auto regions = hal.get_processor_image_regions(processor).value();
    const auto entry = static_cast<uint32_t>(regions.firmware_text.base + 16);
    // All ELF entry, segment, section, startup and export addresses agree.
    // The only invalid relation is to the actual HAL launch address.
    input.images[index] = PublishedElf(entry, config.image_record(PublishedImageKind::Firmware, processor),
        TT_NATIVE_ROLE_TRISC0, false, regions.local_data.base, regions.local_data.base,
        entry + 16, (regions.local_data.base + 19) & ~uint64_t{15});
    input.input[index].bytes = input.images[index].bytes;
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, RegionsRejectRetiredSubordinateFirmwareBasesDespiteCurrentPublicationRecord) {
    auto config = configuration();
    constexpr std::array<uint32_t, 4> retired_bases{0x5d20, 0x6720, 0x7120, 0x7b20};
    for (size_t role = 1; role != 5; ++role) {
        SCOPED_TRACE(role);
        PublicationInput input(config, hal);
        ASSERT_NO_THROW(PublishedDeployment::admit(config, input.input));
        const auto processor = input.input[role].processor;
        const auto regions = hal.get_processor_image_regions(processor).value();
        const auto entry = retired_bases[role - 1];
        ASSERT_NE(entry, regions.firmware_text.base);
        input.images[role] = PublishedElf(entry, config.image_record(PublishedImageKind::Firmware, processor),
            static_cast<uint8_t>(role), false, regions.local_data.base, regions.local_data.base,
            entry + 16, (regions.local_data.base + 19) & ~uint64_t{15});
        input.input[role].bytes = input.images[role].bytes;
        EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    }
}
TEST_F(PublishedDeploymentTest, RegionsUseActualProcessorLimitsIncludingMainEthernetAndDramStaging) {
    auto brisc = hal.get_processor_image_regions({HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 0}).value();
    auto trisc = hal.get_processor_image_regions({HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0}).value();
    auto ethernet = hal.get_processor_image_regions({HalProgrammableCoreType::ACTIVE_ETH, HalProcessorClassType::DM, 0}).value();
    auto dram = hal.get_processor_image_regions({HalProgrammableCoreType::DRAM, HalProcessorClassType::DM, 0}).value();
    EXPECT_EQ(brisc.local_data.base, 0xffb00000u); EXPECT_EQ(brisc.local_data.size, 8192u - 256u);
    EXPECT_EQ(trisc.local_data.size, 4096u - 192u);
    EXPECT_EQ(ethernet.local_data.base, 0xffb00700u); EXPECT_EQ(ethernet.local_data.size, 8192u - 0x700u);
    EXPECT_EQ(dram.local_initialization.size, 1024u); EXPECT_EQ(dram.local_data.size, 8192u - 256u);
    EXPECT_FALSE(hal.get_processor_image_regions({HalProgrammableCoreType::TENSIX, HalProcessorClassType::DM, 2}));
}
TEST_F(PublishedDeploymentTest, RegionsRejectFirmwarePayloadOutsideActualLocalMemory) {
    auto config = configuration(); PublicationInput input(config, hal); auto& image = input.images.front();
    constexpr uint32_t outside = 0x12340000;
    image.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_vaddr), outside);
    image.put(image.sections_offset + 2 * sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_addr), outside);
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, RegionsRejectFirmwareClaimsNotDerivedFromActualSegments) {
    auto config = configuration(); PublicationInput input(config, hal); auto& image = input.images.front();
    image.put(0x140 + 3 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value), uint32_t{0x1000});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, RegionsRejectDispatchOutsideActualFirmwareTextHandoff) {
    auto config = configuration(); PublicationInput input(config, hal); auto& image = input.images.back();
    Elf32_Ehdr h{}; std::memcpy(&h, image.bytes.data(), sizeof(h)); const auto entry = h.e_entry + 16;
    image.put(offsetof(Elf32_Ehdr, e_entry), entry);
    image.put(sizeof(Elf32_Ehdr) + offsetof(Elf32_Phdr, p_vaddr), entry);
    image.put(sizeof(Elf32_Ehdr) + offsetof(Elf32_Phdr, p_paddr), entry);
    image.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_paddr), entry + 4);
    image.put(image.sections_offset + sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_addr), entry);
    image.put(0x140 + sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value), entry);
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, RegionsRejectTriscStaticDataOverflowAndMainEthernetBaseFirmwareOverwrite) {
    auto config = configuration(); PublicationInput input(config, hal);
    for (const size_t index : {size_t{2}, size_t{5}}) {
        SCOPED_TRACE(index);
        auto original = input.images[index].bytes; auto& image = input.images[index];
        if (index == 2) {
            const auto regions = hal.get_processor_image_regions(input.input[index].processor).value();
            image.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_memsz), regions.local_data.size + 4);
        } else {
            image.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_vaddr), uint32_t{0xffb00000});
            image.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_paddr), uint32_t{0xffb00000});
            image.put(image.sections_offset + 2 * sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_addr), uint32_t{0xffb00000});
        }
        EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
        std::copy(original.begin(), original.end(), image.bytes.begin());
    }
}
TEST_F(PublishedDeploymentTest, RegionsRejectDramInitializedDataOverflowWithLegalPrivateDataFootprint) {
    auto config = configuration(); PublicationInput input(config, hal);
    const size_t index = 9;
    auto& image = input.images[index];
    const auto region = hal.get_processor_image_regions(input.input[index].processor).value();
    const uint32_t offset = (image.bytes.size() + 3) & ~uint32_t{3};
    const uint32_t bytes = region.local_initialization.size + 4;
    ASSERT_LT(bytes, region.local_data.size);
    image.bytes.resize(offset + bytes);
    image.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_offset), offset);
    image.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_filesz), bytes);
    image.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_memsz), bytes);
    image.put(image.sections_offset + 2 * sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_offset), offset);
    image.put(image.sections_offset + 2 * sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_size), bytes);
    image.put(0x140 + 4 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value),
        static_cast<uint32_t>((region.local_data.base + bytes + 15) & ~uint64_t{15}));
    input.input[index].bytes = image.bytes;
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, RegionsRejectLocalLayoutExportAndDispatchStackOverlap) {
    auto config = configuration(); PublicationInput input(config, hal);
    auto& firmware = input.images.front(); const auto original = firmware.bytes;
    firmware.put(0x140 + 4 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value), uint32_t{0xffb01000});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    std::copy(original.begin(), original.end(), firmware.bytes.begin());
    const auto region = hal.get_processor_image_regions(input.input.back().processor).value();
    auto& dispatch = input.images.back();
    dispatch.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_memsz), region.local_data.size);
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
} // namespace
} // namespace tt::tt_metal::experimental
