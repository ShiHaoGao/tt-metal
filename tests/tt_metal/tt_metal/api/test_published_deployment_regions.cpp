// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include "impl/experimental/published_deployment/storage.hpp"
#include "tools/native_firmware_bundle/PublishedImage.hpp"
#include <fstream>
namespace tt::tt_metal::experimental {
namespace {
using namespace test;
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
