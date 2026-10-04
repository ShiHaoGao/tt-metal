// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "PublishedImage.hpp"
#include "llrt/tt_elffile.hpp"
#include <algorithm>
#include <cstring>
#include <elf.h>
#include <limits>
#include <stdexcept>
#include <string_view>
namespace tt::tt_metal::native_firmware_bundle {
namespace {
template<class T> T read_at(std::span<const std::byte> bytes, size_t offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
        throw std::invalid_argument("supplier ELF has a truncated table");
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}
template<class T> void write_at(std::vector<std::byte>& bytes, size_t offset, const T& value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}
}
std::vector<std::byte> publish_image(
    const experimental::DeploymentConfiguration& config, experimental::PublishedImageKind kind,
    HalProcessorIdentifier processor, uint32_t node, std::span<const std::byte> linked_elf) {
    // The image and configuration must come from this supplier build. Adding
    // metadata does not itself establish that the complete deployment matches.
    const auto record = config.image_record(kind, processor, node);
    std::vector<std::byte> result(linked_elf.begin(), linked_elf.end());
    auto header = read_at<Elf32_Ehdr>(result, 0);
    if (header.e_ident[EI_CLASS] != ELFCLASS32 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
        header.e_type != ET_EXEC || header.e_machine != EM_RISCV ||
        header.e_shentsize != sizeof(Elf32_Shdr) || !header.e_shnum ||
        header.e_shnum >= SHN_LORESERVE - 1 || !header.e_shstrndx ||
        header.e_shstrndx >= header.e_shnum)
        throw std::invalid_argument("supplier publication requires an ELF32 RISC-V executable with ordinary section tables");
    ll_api::ElfFile parsed;
    parsed.ReadImage(result, "supplier linked image");
    std::vector<Elf32_Shdr> sections;
    for (unsigned i = 0; i != header.e_shnum; ++i)
        sections.push_back(read_at<Elf32_Shdr>(result, uint64_t(header.e_shoff) + i * sizeof(Elf32_Shdr)));
    auto& names_section = sections[header.e_shstrndx];
    if (names_section.sh_type != SHT_STRTAB || (names_section.sh_flags & SHF_ALLOC) ||
        names_section.sh_offset > result.size() || names_section.sh_size > result.size() - names_section.sh_offset)
        throw std::invalid_argument("supplier ELF requires a non-allocatable section-name table");
    const std::string_view names(reinterpret_cast<const char*>(result.data() + names_section.sh_offset),
                                 names_section.sh_size);
    for (const auto& section : sections) {
        if (section.sh_name >= names.size())
            throw std::invalid_argument("supplier ELF section name exceeds its table");
        const auto end = names.find('\0', section.sh_name);
        if (end == std::string_view::npos)
            throw std::invalid_argument("supplier ELF section name is unterminated");
        if (names.substr(section.sh_name, end - section.sh_name) == experimental::PublishedImageSection)
            throw std::invalid_argument("supplier ELF already carries a publication record");
    }
    const uint64_t record_offset = result.size();
    const uint64_t names_offset = record_offset + record.size();
    const uint64_t names_size = names.size() + sizeof(experimental::PublishedImageSection);
    const uint64_t sections_offset = (names_offset + names_size + 3) & ~uint64_t{3};
    const uint64_t total_size = sections_offset + (sections.size() + 1) * sizeof(Elf32_Shdr);
    if (total_size > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("supplier publication exceeds ELF32 file offsets");
    const auto old_names_offset = names_section.sh_offset;
    const auto old_names_size = names_section.sh_size;
    result.resize(total_size);
    std::memcpy(result.data() + record_offset, record.data(), record.size());
    // resize invalidates the earlier string_view; copy only from owned offsets.
    std::memcpy(result.data() + names_offset, result.data() + old_names_offset, old_names_size);
    std::memcpy(result.data() + names_offset + old_names_size, experimental::PublishedImageSection,
                sizeof(experimental::PublishedImageSection));
    names_section.sh_offset = names_offset;
    names_section.sh_size = names_size;
    sections.push_back(Elf32_Shdr{old_names_size, SHT_PROGBITS, 0, 0,
        static_cast<uint32_t>(record_offset), static_cast<uint32_t>(record.size()), 0, 0, 1, 0});
    std::memcpy(result.data() + sections_offset, sections.data(), sections.size() * sizeof(Elf32_Shdr));
    header.e_shoff = sections_offset;
    header.e_shnum = sections.size();
    write_at(result, 0, header);
    ll_api::ElfFile published;
    published.ReadImage(result, "supplier published image");
    const auto& before = parsed.GetSegments();
    const auto& after = published.GetSegments();
    if (before.size() != after.size())
        throw std::invalid_argument("supplier publication changed the load segment count");
    for (size_t i = 0; i != before.size(); ++i) {
        if (before[i].address != after[i].address || before[i].lma != after[i].lma ||
            before[i].membytes != after[i].membytes || before[i].relocs != after[i].relocs ||
            !std::ranges::equal(before[i].contents, after[i].contents))
            throw std::invalid_argument("supplier publication would change loaded bytes or relocations");
    }
    if (!std::ranges::equal(published.GetMetadataSection(experimental::PublishedImageSection), record))
        throw std::logic_error("supplier publication did not retain its exact image record");
    return result;
}
}
