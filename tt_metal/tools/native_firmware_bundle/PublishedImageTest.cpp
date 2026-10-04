// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "PublishedImage.hpp"
#include "DeploymentBuild.hpp"
#include "tests/tt_metal/tt_metal/api/published_deployment_test_support.hpp"
#include "llrt/tt_elffile.hpp"

namespace tt::tt_metal::native_firmware_bundle {
namespace {
using namespace experimental;
class SupplierImageTest : public experimental::test::PublishedDeploymentTest {};
template<class T> T read_at(std::span<const std::byte> bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}
void remove_publication(experimental::test::PublishedElf& elf) {
    auto header = read_at<Elf32_Ehdr>(elf.bytes, 0);
    // The fixture's final section is publication metadata. Nothing else changes.
    header.e_shnum = 8;
    elf.put(0, header);
}
TEST_F(SupplierImageTest, PreservesEveryLoadByteSymbolAndRelocation) {
    const auto config = configuration();
    experimental::test::PublicationInput input(config, hal);
    auto elf = input.images.front();
    remove_publication(elf);
    const auto original = elf.bytes;
    auto result = publish_image(config, PublishedImageKind::Firmware,
        config.firmware_processors().front(), NoDispatchNode, original);
    EXPECT_EQ(elf.bytes, original);
    const auto before = read_at<Elf32_Ehdr>(original, 0);
    const auto after = read_at<Elf32_Ehdr>(result, 0);
    EXPECT_EQ(after.e_entry, before.e_entry);
    EXPECT_EQ(after.e_phoff, before.e_phoff);
    EXPECT_EQ(after.e_phnum, before.e_phnum);
    for (unsigned i = 0; i != before.e_phnum; ++i) {
        const auto ph = read_at<Elf32_Phdr>(original, before.e_phoff + i * sizeof(Elf32_Phdr));
        EXPECT_TRUE(std::equal(original.begin() + ph.p_offset,
            original.begin() + ph.p_offset + ph.p_filesz, result.begin() + ph.p_offset));
    }
    for (unsigned i = 0; i != before.e_shnum; ++i) {
        if (i == before.e_shstrndx) continue;
        const auto old_section = read_at<Elf32_Shdr>(original, before.e_shoff + i * sizeof(Elf32_Shdr));
        const auto new_section = read_at<Elf32_Shdr>(result, after.e_shoff + i * sizeof(Elf32_Shdr));
        EXPECT_EQ(std::memcmp(&old_section, &new_section, sizeof(old_section)), 0);
        if (old_section.sh_type != SHT_NOBITS)
            EXPECT_TRUE(std::equal(original.begin() + old_section.sh_offset,
                original.begin() + old_section.sh_offset + old_section.sh_size,
                result.begin() + new_section.sh_offset));
    }
    const auto section = read_at<Elf32_Shdr>(result, after.e_shoff + before.e_shnum * sizeof(Elf32_Shdr));
    EXPECT_EQ(section.sh_flags, 0u);
    EXPECT_EQ(section.sh_addr, 0u);
    ll_api::ElfFile parsed;
    parsed.ReadImage(result, "supplier result");
    EXPECT_TRUE(std::ranges::equal(parsed.GetMetadataSection(PublishedImageSection),
        config.image_record(PublishedImageKind::Firmware, config.firmware_processors().front())));
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST_F(SupplierImageTest, AdmitsTheCompletePublishedGroupWithoutContext) {
    const auto config = configuration();
    experimental::test::PublicationInput source(config, hal);
    std::vector<std::vector<std::byte>> owned;
    std::vector<PublishedImageInput> inputs;
    owned.reserve(source.images.size());
    for (size_t i = 0; i != source.images.size(); ++i) {
        remove_publication(source.images[i]);
        const auto& old = source.input[i];
        owned.push_back(publish_image(config, old.kind, old.processor, old.dispatch_node, source.images[i].bytes));
        inputs.push_back({old.kind, old.processor, old.dispatch_node, owned.back()});
    }
    EXPECT_NO_THROW(PublishedDeployment::admit(config, inputs));
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST_F(SupplierImageTest, RejectsExistingPublicationRatherThanRelabelingIt) {
    const auto config = configuration();
    experimental::test::PublicationInput source(config, hal);
    const auto& old = source.input.front();
    EXPECT_THROW(publish_image(config, old.kind, old.processor, old.dispatch_node, old.bytes), std::invalid_argument);
}
TEST_F(SupplierImageTest, RejectsTruncatedOrWrongElfBeforePublication) {
    const auto config = configuration();
    experimental::test::PublicationInput source(config, hal);
    auto elf = source.images.front();
    remove_publication(elf);
    const auto& old = source.input.front();
    EXPECT_ANY_THROW(publish_image(config, old.kind, old.processor, old.dispatch_node,
        std::span<const std::byte>(elf.bytes).first(20)));
    auto header = read_at<Elf32_Ehdr>(elf.bytes, 0);
    header.e_type = ET_REL;
    elf.put(0, header);
    EXPECT_THROW(publish_image(config, old.kind, old.processor, old.dispatch_node, elf.bytes), std::invalid_argument);
}
TEST_F(SupplierImageTest, RejectsImplicitSupplierEnvironmentBeforeCompilerOrContext) {
    // These fixture options deliberately have no explicit cache and permit
    // precompiled lookup. A supplier must not discover or reuse such resources.
    EXPECT_THROW(build_deployment(JitDeviceConfig{device, &hal}, options, plan()), std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST_F(SupplierImageTest, RejectsIncompleteSupplierConfigurationBeforeToolAccess) {
    // A complete typed dispatch plan cannot represent an unresolved node.
    // A mismatched supplier device must still fail before tool access.
    device.num_l1_banks += 1;
    EXPECT_THROW(build_deployment(JitDeviceConfig{device, &hal}, options, plan()), std::invalid_argument);
    EXPECT_FALSE(MetalContext::instance_exists());
}
TEST_F(SupplierImageTest, RejectsPublicationThatWouldModifyLoadedHeaderBytes) {
    const auto config = configuration();
    experimental::test::PublicationInput source(config, hal);
    auto elf = source.images.front();
    remove_publication(elf);
    auto segment = read_at<Elf32_Phdr>(elf.bytes, sizeof(Elf32_Ehdr));
    segment.p_offset = 0;
    segment.p_filesz = segment.p_memsz = 0x84;
    elf.put(sizeof(Elf32_Ehdr), segment);
    const auto& old = source.input.front();
    EXPECT_THROW(publish_image(config, old.kind, old.processor, old.dispatch_node, elf.bytes), std::invalid_argument);
}
} // namespace
} // namespace tt::tt_metal::native_firmware_bundle
