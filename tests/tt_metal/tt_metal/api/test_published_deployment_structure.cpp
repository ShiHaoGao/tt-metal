// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "published_deployment_test_support.hpp"
#include <gtest/gtest.h>
#include <elf.h>
#include <cstring>
namespace tt::tt_metal::experimental {
namespace {
using namespace test;
TEST_F(PublishedDeploymentTest, RejectsNonFunctionStartupSymbol) {
    auto config = configuration();
    PublicationInput input(config, hal);
    // _start remains at the entry address but is declared as an object.
    input.images[0].put(0x140 + sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_info),
                        static_cast<unsigned char>(ELF32_ST_INFO(STB_GLOBAL, STT_OBJECT)));
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, RejectsStartupSymbolInNonLoadSection) {
    auto config = configuration();
    PublicationInput input(config, hal);
    // _start remains defined at e_entry but points at .symtab, not executable .text.
    input.images[0].put(0x140 + sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_shndx), Elf32_Half{5});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, RejectsNonExecutableEntrySegment) {
    auto config = configuration(); PublicationInput input(config, hal);
    input.images[0].put(sizeof(Elf32_Ehdr) + offsetof(Elf32_Phdr, p_flags), Elf32_Word{PF_R | PF_W});
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
TEST_F(PublishedDeploymentTest, RejectsOverlappingFirmwareLoadSegments) {
    auto config = configuration(); PublicationInput input(config, hal);
    auto& firmware = input.images[0];
    Elf32_Ehdr header{}; std::memcpy(&header, firmware.bytes.data(), sizeof(header));
    firmware.put(sizeof(Elf32_Ehdr) + sizeof(Elf32_Phdr) + offsetof(Elf32_Phdr, p_vaddr), header.e_entry);
    firmware.put(firmware.sections_offset + 2 * sizeof(Elf32_Shdr) + offsetof(Elf32_Shdr, sh_addr), header.e_entry);
    EXPECT_THROW(PublishedDeployment::admit(config, input.input), std::invalid_argument);
}
} // namespace
} // namespace tt::tt_metal::experimental
