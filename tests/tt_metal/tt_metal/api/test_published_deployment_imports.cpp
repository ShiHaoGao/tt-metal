// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
namespace tt::tt_metal::experimental {
namespace {
using namespace test;
void change_absolute_symbol(PublishedElf& image, std::string_view name) {
    Elf32_Shdr strings{};
    std::memcpy(&strings, image.bytes.data() + image.sections_offset + 6 * sizeof(Elf32_Shdr), sizeof(strings));
    const size_t string_offset = strings.sh_size;
    if (strings.sh_offset + strings.sh_size + name.size() + 1 > 0x200)
        throw std::logic_error("fixture symbol exceeds reserved string bytes");
    std::memcpy(image.bytes.data() + strings.sh_offset + string_offset, name.data(), name.size());
    image.bytes[strings.sh_offset + string_offset + name.size()] = std::byte{};
    image.put(image.sections_offset + 6 * sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_size),
        static_cast<uint32_t>(strings.sh_size + name.size() + 1));
    Elf32_Ehdr header{}; std::memcpy(&header, image.bytes.data(), sizeof(header));
    image.put(0x140 + 2 * sizeof(Elf32_Sym), Elf32_Sym{static_cast<uint32_t>(string_offset), header.e_entry, 0,
        ELF32_ST_INFO(STB_GLOBAL, STT_NOTYPE), 0, SHN_ABS});
}
TEST_F(PublishedDeploymentTest, ImportsRejectMissingAbsoluteFirmwareProvider) {
    auto config = configuration(); PublicationInput input(config, hal);
    input.images[0].put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_name), Elf32_Word{2});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, ImportsRejectUndefinedWeakFirmwareProvider) {
    auto config = configuration(); PublicationInput input(config, hal); auto& firmware = input.images[0];
    firmware.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_shndx), Elf32_Half{SHN_UNDEF});
    firmware.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_info),
        static_cast<unsigned char>(ELF32_ST_INFO(STB_WEAK, STT_OBJECT)));
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, ImportsRejectFirmwareProviderSizeAndTypeMismatch) {
    auto config = configuration(); PublicationInput input(config, hal); auto& firmware = input.images[0];
    const auto original = firmware.bytes;
    firmware.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_size), Elf32_Word{1});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    std::copy(original.begin(), original.end(), firmware.bytes.begin());
    firmware.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_info),
        static_cast<unsigned char>(ELF32_ST_INFO(STB_GLOBAL, STT_FUNC)));
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, ImportsValidateExactNcriscLinkerSymbolOwnerAndValue) {
    auto config = configuration(); PublicationInput input(config, hal);
    const size_t index = config.firmware_processors().size() + 2;
    ASSERT_EQ(input.input[index].processor.processor_type, 1);
    auto& image = input.images[index]; change_absolute_symbol(image, "__kernel_text_start");
    ASSERT_NO_THROW(PublishedDeployment::admit(config, input.input));
    const auto original = image.bytes;
    image.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value), uint32_t{0});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    std::copy(original.begin(), original.end(), image.bytes.begin());
    image.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_size), uint32_t{4});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    std::copy(original.begin(), original.end(), image.bytes.begin());
    image.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_info),
        static_cast<unsigned char>(ELF32_ST_INFO(STB_WEAK, STT_NOTYPE)));
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, ImportsRejectLinkerSymbolOnWrongProcessorAndPrefixImpersonation) {
    auto config = configuration(); PublicationInput input(config, hal); auto& image = input.images.back();
    const auto original = image.bytes;
    change_absolute_symbol(image, "__kernel_text_start");
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    std::copy(original.begin(), original.end(), image.bytes.begin());
    change_absolute_symbol(image, "__kernel_text_start_extra");
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, ImportsRejectProviderOutsideItsRealAllocation) {
    auto config = configuration(); PublicationInput input(config, hal); auto& firmware = input.images[0];
    const auto original = firmware.bytes;
    firmware.put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_shndx), Elf32_Half{5});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
    std::copy(original.begin(), original.end(), firmware.bytes.begin());
    // Keep import and export identical but point both beyond the real .data allocation.
    for (auto index : {size_t{0}, input.images.size() - 1})
        input.images[index].put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_value), uint32_t{0xffb01234});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, ImportsRejectObjectsDisguisedAsFirmwareAbsoluteConstants) {
    auto config = configuration(); PublicationInput input(config, hal);
    input.images.front().put(0x140 + 2 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_shndx), Elf32_Half{SHN_ABS});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
} // namespace
} // namespace tt::tt_metal::experimental
