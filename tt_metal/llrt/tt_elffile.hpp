// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// C++
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// An ELF executable loader
// This is a replacement for tt_hexfile stuff.

namespace ll_api {

class ElfFile {
public:
    // On ELF64 systems, we're statically restricted to the first 4GB
    using address_t = std::uint32_t;  // Address in memory
    using offset_t = std::uint32_t;   // Offset within region
    using word_t = std::uint32_t;     // Contents

    struct Segment {
        std::vector<offset_t> relocs;      // 32-bit relocs to apply
        std::span<const word_t> contents;  // Non-owning span
        address_t address = 0;             // Byte execution address (0 for
                                           // XIP)
        address_t lma = 0;                 // Byte load address
        offset_t membytes = 0;             // Byte size of memory image.

    public:
        Segment(std::span<const word_t> contents, address_t addr, address_t lma, offset_t membytes) :
            contents(contents), address(addr), lma(lma), membytes(membytes) {}
    };

    ElfFile() = default;
    ~ElfFile();

    // Uncopyable -- because of the owning buffer & pimpl object.
    ElfFile(ElfFile const&) = delete;
    ElfFile operator=(ElfFile const&) = delete;

    // Move constructable & assignable -- take ownership
    ElfFile(ElfFile&& s) noexcept;
    ElfFile& operator=(ElfFile&& s) noexcept;

    std::vector<Segment> const& GetSegments() const { return segments_; }

    // Get the contents of a named section. Returns empty span if not found.
    std::span<std::byte> GetSectionContents(std::string_view section_name, uint64_t& virtual_address) const;

    // Unique host metadata only; reject duplicate names and loaded sections.
    std::span<const std::byte> GetMetadataSection(std::string_view name) const;
    std::span<const std::byte> GetImageContents() const { return contents_; }

    // Release the implementation data, leaving the segments and
    // contents. Use this, after processing, if the elf object is long-lived.
    void ReleaseImpl();

    // Read an elf file, populate segments vector.
    void ReadImage(const std::string& path);

    // Copy an ELF image into private mutable storage before parsing. The
    // caller may release or change its bytes after this call. The label is
    // copied for diagnostics only; it is never used for file I/O or identity.
    void ReadImage(std::span<const std::byte> image, std::string_view label = "<memory>");

    // Write the (now-processed) elf file.
    void WriteImage(const std::string& path);

    // Weaken data symbols, remove all others. Keep STRONG_NAMES
    // strong (can be non-data symbols).  Names can be exact or simple
    // globs ending in '*'.
    void WeakenDataSymbols(std::span<std::string_view const> strong_names);

    // Convert the executable to a relinkable object file. Relocations
    // are removed, allocatable sections are placed at zero, their
    // symbols are adjusted to remain section-relative. The elf type
    // becomes 'ET_REL'. Remember, objectifying people is bad, but
    // objectifying an executable is perfectly fine (if a little strange).
    void ObjectifyExecutable();

    // XIPify
    void MakeExecuteInPlace();

private:
    class Impl;

    void Reset() noexcept;

    // Reset releases the opaque parser before its backing mapping.
    Impl* pimpl_ = nullptr;

    std::span<std::byte> contents_;  // Owning buffer

    // The first segment is the text segment, regardless of VMA ordering.
    std::vector<Segment> segments_;
};

}  // namespace ll_api
