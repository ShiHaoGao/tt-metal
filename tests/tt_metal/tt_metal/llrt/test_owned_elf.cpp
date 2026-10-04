// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <elf.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_elffile.hpp"
#include "llrt/tt_memory.h"

namespace ll_api {
namespace {

// A complete ELF32 executable with two real RISC-V instructions referring to
// its text segment. Fixed offsets make each malformed-input case independent
// of the parser. No toolchain, fixture archive or device context is required.
struct LinkedElf {
    static constexpr size_t text_offset = 0x80;
    static constexpr size_t reloc_offset = 0x90;
    static constexpr size_t symbol_offset = 0xa8;
    static constexpr size_t strings_offset = 0xc8;
    static constexpr size_t names_offset = 0xd0;
    static constexpr size_t segments_offset = 0x130;
    static constexpr size_t sections_offset = 0x180;
    static constexpr size_t text_section = 1;
    static constexpr size_t relocation_section = 3;
    static constexpr size_t symbol_section = 4;
    static constexpr size_t names_section = 6;
    std::vector<std::byte> bytes = std::vector<std::byte>(sections_offset + 8 * sizeof(Elf32_Shdr));

    template <typename T>
    void put(size_t offset, const T& value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    template <typename T>
    T get(size_t offset) const {
        T value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    }

    Elf32_Shdr section(size_t index) const { return get<Elf32_Shdr>(sections_offset + index * sizeof(Elf32_Shdr)); }
    void section(size_t index, const Elf32_Shdr& value) { put(sections_offset + index * sizeof(Elf32_Shdr), value); }

    explicit LinkedElf(uint32_t text_address = 0x1000, uint32_t data_address = 0x2000) {
        Elf32_Ehdr header{};
        std::memcpy(header.e_ident, ELFMAG, SELFMAG);
        header.e_ident[EI_CLASS] = ELFCLASS32;
        header.e_ident[EI_DATA] = ELFDATA2LSB;
        header.e_ident[EI_VERSION] = EV_CURRENT;
        header.e_type = ET_EXEC;
        header.e_machine = EM_RISCV;
        header.e_version = EV_CURRENT;
        header.e_entry = text_address;
        header.e_phoff = sizeof(header);
        header.e_shoff = sections_offset;
        header.e_ehsize = sizeof(header);
        header.e_phentsize = sizeof(Elf32_Phdr);
        header.e_phnum = 2;
        header.e_shentsize = sizeof(Elf32_Shdr);
        header.e_shnum = 8;
        header.e_shstrndx = names_section;
        put(0, header);
        put(sizeof(header), Elf32_Phdr{PT_LOAD, text_offset, text_address, text_address, 12, 12, PF_R | PF_X, 4});
        put(sizeof(header) + sizeof(Elf32_Phdr),
            Elf32_Phdr{PT_LOAD, text_offset + 12, data_address, text_address + 12, 4, 4, PF_R | PF_W, 4});
        // lui ra, hi(target); addi ra, ra, lo(target); ret
        put(text_offset, std::array<uint32_t, 4>{text_address | 0xb7u, 0x00808093u, 0x00008067u, 0x12345678u});
        put(reloc_offset, Elf32_Rela{text_address, ELF32_R_INFO(1, R_RISCV_HI20), 0});
        put(reloc_offset + sizeof(Elf32_Rela), Elf32_Rela{text_address + 4, ELF32_R_INFO(1, R_RISCV_LO12_I), 0});
        Elf32_Sym symbol{};
        symbol.st_name = 1;
        symbol.st_value = text_address + 8;
        symbol.st_info = ELF32_ST_INFO(STB_GLOBAL, STT_FUNC);
        symbol.st_shndx = text_section;
        put(symbol_offset + sizeof(Elf32_Sym), symbol);
        constexpr char strings[] = "\0target\0";
        std::memcpy(bytes.data() + strings_offset, strings, sizeof(strings));
        constexpr char names[] = "\0.text\0.data\0.rela.text\0.symtab\0.strtab\0.shstrtab\0.segments\0";
        std::memcpy(bytes.data() + names_offset, names, sizeof(names));
        const auto name = [&](const char* value) {
            size_t offset = 1;
            while (std::strcmp(names + offset, value)) offset += std::strlen(names + offset) + 1;
            return static_cast<uint32_t>(offset);
        };
        section(1, Elf32_Shdr{name(".text"), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_address,
                             text_offset, 12, 0, 0, 4, 0});
        section(2, Elf32_Shdr{name(".data"), SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, data_address,
                             text_offset + 12, 4, 0, 0, 4, 0});
        section(3, Elf32_Shdr{name(".rela.text"), SHT_RELA, 0, 0, reloc_offset, 2 * sizeof(Elf32_Rela),
                             symbol_section, text_section, 4, sizeof(Elf32_Rela)});
        section(4, Elf32_Shdr{name(".symtab"), SHT_SYMTAB, 0, 0, symbol_offset, 2 * sizeof(Elf32_Sym),
                             5, 1, 4, sizeof(Elf32_Sym)});
        section(5, Elf32_Shdr{name(".strtab"), SHT_STRTAB, 0, 0, strings_offset, sizeof(strings), 0, 0, 1, 0});
        section(6, Elf32_Shdr{name(".shstrtab"), SHT_STRTAB, 0, 0, names_offset, sizeof(names), 0, 0, 1, 0});
        section(7, Elf32_Shdr{name(".segments"), SHT_PROGBITS, 0, 0, segments_offset, 24, 0, 0, 4, 0});
        put(segments_offset, std::array<uint32_t, 6>{text_address, text_address, 12, data_address, data_address, 4});
    }
};

std::vector<uint32_t> text_words(const ElfFile& elf) {
    const auto text = elf.GetSegments().front().contents;
    return {text.begin(), text.end()};
}

TEST(OwnedElf, CopiesCallerStorageAndMovesParserOwnership) {
    ElfFile retained;
    {
        LinkedElf input;
        ElfFile initial;
        initial.ReadImage(input.bytes, std::string("temporary label"));
        std::fill(input.bytes.begin(), input.bytes.end(), std::byte{0});
        ElfFile intermediate(std::move(initial));
        retained = std::move(intermediate);
    }
    EXPECT_EQ(text_words(retained), (std::vector<uint32_t>{0x000010b7u, 0x00808093u, 0x00008067u}));
    retained.MakeExecuteInPlace();
    EXPECT_EQ(text_words(retained), (std::vector<uint32_t>{0x00000097u, 0x00808093u, 0x00008067u}));
    EXPECT_EQ(retained.GetSegments().front().address, 0u);
}

TEST(OwnedElf, XipPackingTransformsInstructionsWithoutContextOrInputMutation) {
    ASSERT_FALSE(tt::tt_metal::MetalContext::instance_exists());
    LinkedElf input;
    const auto original = input.bytes;
    const memory image(input.bytes, memory::Loading::CONTIGUOUS_XIP, "owned ELF");
    EXPECT_EQ(image.data(), (std::vector<uint32_t>{0x00000097u, 0x00808093u, 0x00008067u, 0x12345678u}));
    EXPECT_EQ(image.get_text_size(), 12u);
    EXPECT_EQ(image.get_text_addr(), 0u);
    EXPECT_EQ(input.bytes, original);
    EXPECT_FALSE(tt::tt_metal::MetalContext::instance_exists());
}

TEST(OwnedElf, DiscretePackingPreservesAddressOrderingAndCallerLifetime) {
    memory image;
    {
        LinkedElf input(0x2000, 0x1000);
        image = memory(input.bytes, memory::Loading::DISCRETE, "owned ELF");
    }
    EXPECT_EQ(image.data(), (std::vector<uint32_t>{0x12345678u, 0x000020b7u, 0x00808093u, 0x00008067u}));
    std::vector<uint64_t> addresses;
    std::as_const(image).process_spans([&](auto, uint64_t address, auto) { addresses.push_back(address); });
    EXPECT_EQ(addresses, (std::vector<uint64_t>{0x1000, 0x2000}));
    EXPECT_EQ(image.get_text_addr(), 0x2000u);
}

TEST(OwnedElf, AcceptsEmptyAllocatableSectionWithoutInventingLoadPayload) {
    LinkedElf input;
    auto header = input.get<Elf32_Ehdr>(0);
    header.e_phnum = 1;
    input.put(0, header);
    auto data = input.section(2);
    data.sh_size = 0;
    input.section(2, data);
    auto segments = input.section(7);
    segments.sh_size = 12;
    input.section(7, segments);

    ElfFile elf;
    ASSERT_NO_THROW(elf.ReadImage(input.bytes));
    ASSERT_EQ(elf.GetSegments().size(), 1u);
    EXPECT_EQ(elf.GetSegments().front().contents.size_bytes(), 12u);
    const memory image(input.bytes, memory::Loading::CONTIGUOUS_XIP);
    EXPECT_EQ(image.data(), (std::vector<uint32_t>{0x00000097u, 0x00808093u, 0x00008067u}));

    // A nonempty section at the same unmapped address still requires storage.
    data.sh_size = 4;
    input.section(2, data);
    EXPECT_THROW(elf.ReadImage(input.bytes), std::exception);
}

TEST(OwnedElf, AcceptsEmptyAllocationMetadataWithoutInventingLoadPayload) {
    LinkedElf input;
    auto header = input.get<Elf32_Ehdr>(0);
    header.e_phnum = 1;
    input.put(0, header);
    auto data = input.section(2);
    data.sh_size = 0;
    input.section(2, data);
    // The real SDK linker script retains its data-region metadata even when
    // LLD omits the empty PT_LOAD. The limit is capacity, not payload size.
    input.put(LinkedElf::segments_offset + 20, uint32_t{4096});
    ElfFile elf;
    ASSERT_NO_THROW(elf.ReadImage(input.bytes));
    ASSERT_EQ(elf.GetSegments().size(), 1u);
    EXPECT_EQ(elf.GetSegments().front().contents.size_bytes(), 12u);
    const memory image(input.bytes, memory::Loading::CONTIGUOUS_XIP);
    EXPECT_EQ(image.data(), (std::vector<uint32_t>{0x00000097u, 0x00808093u, 0x00008067u}));
}

TEST(OwnedElf, RejectsUnbackedMetadataWithoutAnEmptyAllocationSection) {
    LinkedElf input;
    auto header = input.get<Elf32_Ehdr>(0);
    header.e_phnum = 1;
    input.put(0, header);
    auto data = input.section(2);
    data.sh_size = 0;
    data.sh_flags = 0;
    input.section(2, data);
    ElfFile elf;
    EXPECT_ANY_THROW(elf.ReadImage(input.bytes));
}

TEST(OwnedElf, RejectsNonzeroTrimForAnEmptyAllocationSection) {
    LinkedElf input;
    auto header = input.get<Elf32_Ehdr>(0);
    header.e_phnum = 1;
    input.put(0, header);
    auto data = input.section(2);
    data.sh_size = 0;
    input.section(2, data);
    input.put(LinkedElf::segments_offset + 16, uint32_t{0x2004});
    ElfFile elf;
    EXPECT_ANY_THROW(elf.ReadImage(input.bytes));
}

TEST(OwnedElf, RejectedReplacementRetainsAcceptedImage) {
    LinkedElf input;
    ElfFile image;
    image.ReadImage(input.bytes);
    auto header = input.get<Elf32_Ehdr>(0);
    header.e_machine = EM_X86_64;
    input.put(0, header);
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
    EXPECT_EQ(text_words(image), (std::vector<uint32_t>{0x000010b7u, 0x00808093u, 0x00008067u}));
    EXPECT_NO_THROW(image.MakeExecuteInPlace());
}

TEST(OwnedElf, RejectsTruncatedHeadersAndTablesWithoutPublishingSegments) {
    LinkedElf input;
    ElfFile image;
    for (size_t size = 0; size < sizeof(Elf32_Ehdr); ++size) {
        SCOPED_TRACE(size);
        EXPECT_THROW(image.ReadImage(std::span<const std::byte>(input.bytes).first(size)), std::exception);
        EXPECT_TRUE(image.GetSegments().empty());
    }
    auto header = input.get<Elf32_Ehdr>(0);
    header.e_shoff = 0xfffffffcu;
    input.put(0, header);
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
    header.e_shoff = LinkedElf::sections_offset;
    header.e_phoff = 0xfffffffcu;
    input.put(0, header);
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
    input.bytes[EI_CLASS] = std::byte{ELFCLASS64};
    EXPECT_THROW(image.ReadImage(std::span<const std::byte>(input.bytes).first(sizeof(Elf64_Ehdr) - 1)), std::exception);
}

TEST(OwnedElf, RejectsUnsupportedWideRelocation) {
    LinkedElf input;
    auto relocation = input.get<Elf32_Rela>(LinkedElf::reloc_offset);
    relocation.r_info = ELF32_R_INFO(1, R_RISCV_64);
    input.put(LinkedElf::reloc_offset, relocation);
    // Remove the unmatched LO12 so the only issue is the unsupported relocation.
    auto section = input.section(LinkedElf::relocation_section);
    section.sh_size = sizeof(Elf32_Rela);
    input.section(LinkedElf::relocation_section, section);
    EXPECT_THROW((memory(input.bytes, memory::Loading::CONTIGUOUS_XIP)), std::exception);
}

TEST(OwnedElf, RejectsOutOfFileSectionPayload) {
    LinkedElf input;
    auto section = input.section(5);
    section.sh_offset = input.bytes.size();
    input.section(5, section);
    ElfFile image;
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
    EXPECT_TRUE(image.GetSegments().empty());
}

TEST(OwnedElf, RejectsMalformedRelocationEntrySize) {
    LinkedElf input;
    auto section = input.section(LinkedElf::relocation_section);
    section.sh_entsize = 1;
    input.section(LinkedElf::relocation_section, section);
    ElfFile image;
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
    EXPECT_TRUE(image.GetSegments().empty());
}

TEST(OwnedElf, RejectsMisalignedRelocationRecordsBeforeReadingThem) {
    LinkedElf input;
    auto section = input.section(LinkedElf::relocation_section);
    ++section.sh_offset;
    input.section(LinkedElf::relocation_section, section);
    ElfFile image;
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
}

TEST(OwnedElf, RejectsOutOfFileLoadSegment) {
    LinkedElf input;
    auto segment = input.get<Elf32_Phdr>(sizeof(Elf32_Ehdr));
    segment.p_filesz = 0xfffffffcu;
    segment.p_memsz = segment.p_filesz;
    input.put(sizeof(Elf32_Ehdr), segment);
    input.put(LinkedElf::segments_offset + 8, uint32_t{0xffffffff});
    ElfFile image;
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
}

TEST(OwnedElf, RejectsUnterminatedSectionName) {
    LinkedElf input;
    auto names = input.section(LinkedElf::names_section);
    std::fill(input.bytes.begin() + names.sh_offset, input.bytes.begin() + names.sh_offset + names.sh_size, std::byte{'x'});
    ElfFile image;
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
}

TEST(OwnedElf, RejectsRelocationWithForeignSymbolIndexBeforeTransformation) {
    LinkedElf input;
    auto relocation = input.get<Elf32_Rela>(LinkedElf::reloc_offset);
    relocation.r_info = ELF32_R_INFO(2, R_RISCV_HI20);
    input.put(LinkedElf::reloc_offset, relocation);
    ElfFile image;
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
}

TEST(OwnedElf, RejectsRelocationReferencesOutsideSectionTable) {
    for (const bool target : {false, true}) {
        SCOPED_TRACE(target);
        LinkedElf input;
        auto section = input.section(LinkedElf::relocation_section);
        if (target) section.sh_info = 8;
        else section.sh_link = 8;
        input.section(LinkedElf::relocation_section, section);
        ElfFile image;
        EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
    }
}

TEST(OwnedElf, RejectsSegmentTrimOutsideItsContents) {
    LinkedElf input;
    input.put(LinkedElf::segments_offset + 4, uint32_t{0x1010});
    input.put(LinkedElf::segments_offset + 8, uint32_t{0xffffffff});
    ElfFile image;
    EXPECT_THROW(image.ReadImage(input.bytes), std::exception);
}

}  // namespace
}  // namespace ll_api
