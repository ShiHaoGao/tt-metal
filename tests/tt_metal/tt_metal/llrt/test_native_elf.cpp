// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <elf.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>
#include "tt-metalium/experimental/native_kernel.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/allocator/allocator_types.hpp"
#include "impl/experimental/native_kernel/native_image.hpp"
#include "impl/experimental/native_kernel/native_kernel.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal::experimental {
namespace {
// Complete linked RISC-V ELF, with distinct loaded text/data and host-only
// support metadata. The instruction is ret; no external toolchain is needed.
struct NativeElf {
    static constexpr size_t note_offset = 0x90;
    static constexpr size_t sections_offset = 0x200;
    std::vector<std::byte> bytes = std::vector<std::byte>(sections_offset + 8 * sizeof(Elf32_Shdr));
    template <class T>
    void put(size_t offset, const T& v) {
        std::memcpy(bytes.data() + offset, &v, sizeof(v));
    }
    NativeElf(uint8_t kind, uint8_t role) {
        Elf32_Ehdr h{};
        std::memcpy(h.e_ident, ELFMAG, SELFMAG);
        h.e_ident[EI_CLASS] = ELFCLASS32;
        h.e_ident[EI_DATA] = ELFDATA2LSB;
        h.e_ident[EI_VERSION] = EV_CURRENT;
        h.e_type = ET_EXEC;
        h.e_machine = EM_RISCV;
        h.e_version = EV_CURRENT;
        h.e_entry = 0x1000;
        h.e_phoff = sizeof(h);
        h.e_phnum = 2;
        h.e_phentsize = sizeof(Elf32_Phdr);
        h.e_ehsize = sizeof(h);
        h.e_shoff = sections_offset;
        h.e_shnum = 8;
        h.e_shentsize = sizeof(Elf32_Shdr);
        h.e_shstrndx = 4;
        put(0, h);
        put(sizeof(h), Elf32_Phdr{PT_LOAD, 0x80, 0x1000, 0x1000, 4, 4, PF_R | PF_X, 4});
        put(sizeof(h) + sizeof(Elf32_Phdr), Elf32_Phdr{PT_LOAD, 0x84, 0x2000, 0x1004, 4, 4, PF_R | PF_W, 4});
        put(0x80, uint32_t{0x00008067});
        put(0x84, uint32_t{0x12345678});
        put(note_offset,
            tt_native_image_record{
                TT_NATIVE_IMAGE_MAGIC,
                TT_NATIVE_IMAGE_ABI_VERSION,
                kind,
                role,
                TT_NATIVE_ARCH_BLACKHOLE,
                TT_NATIVE_LOADING_DISCRETE,
                TT_NATIVE_PROFILE_ABSENT,
                TT_NATIVE_PRINT_ABSENT,
                1,
                TT_NATIVE_WORKER_STREAM_SDK,
                0});
        constexpr char names[] = "\0.text\0.data\0.tt_native_image\0.shstrtab\0.symtab\0.strtab\0.rela.text\0";
        std::memcpy(bytes.data() + 0x100, names, sizeof(names));
        put(sections_offset + sizeof(Elf32_Shdr),
            Elf32_Shdr{1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x1000, 0x80, 4, 0, 0, 4, 0});
        put(sections_offset + 2 * sizeof(Elf32_Shdr),
            Elf32_Shdr{7, SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 0x2000, 0x84, 4, 0, 0, 4, 0});
        put(sections_offset + 3 * sizeof(Elf32_Shdr), Elf32_Shdr{13, SHT_PROGBITS, 0, 0, note_offset, 16, 0, 0, 4, 0});
        put(sections_offset + 4 * sizeof(Elf32_Shdr),
            Elf32_Shdr{30, SHT_STRTAB, 0, 0, 0x100, sizeof(names), 0, 0, 1, 0});
        put(sections_offset + 5 * sizeof(Elf32_Shdr),
            Elf32_Shdr{40, SHT_SYMTAB, 0, 0, 0x150, sizeof(Elf32_Sym), 6, 1, 4, sizeof(Elf32_Sym)});
        put(sections_offset + 6 * sizeof(Elf32_Shdr), Elf32_Shdr{48, SHT_STRTAB, 0, 0, 0x170, 1, 0, 0, 1, 0});
        put(sections_offset + 7 * sizeof(Elf32_Shdr),
            Elf32_Shdr{56, SHT_RELA, 0, 0, 0x180, 0, 5, 1, 4, sizeof(Elf32_Rela)});
    }
    void niu_metadata(
        std::span<const std::byte> metadata, uint32_t flags = 0, std::string_view section_name = ".tt_niu_handoff") {
        Elf32_Ehdr header{};
        std::memcpy(&header, bytes.data(), sizeof(header));
        std::array<Elf32_Shdr, 8> original{};
        std::memcpy(original.data(), bytes.data() + header.e_shoff, sizeof(original));
        const auto& names_section = original[header.e_shstrndx];
        std::string names(reinterpret_cast<const char*>(bytes.data() + names_section.sh_offset), names_section.sh_size);
        const auto name_offset = names.size();
        names.append(section_name);
        names.push_back('\0');
        const size_t names_offset = std::max<size_t>(0x500, (0x400 + metadata.size() + 0xff) & ~size_t{0xff});
        const size_t table_offset = std::max<size_t>(0x600, (names_offset + names.size() + 0xff) & ~size_t{0xff});
        bytes.resize(table_offset + 9 * sizeof(Elf32_Shdr));
        std::memcpy(bytes.data() + 0x400, metadata.data(), metadata.size());
        std::memcpy(bytes.data() + names_offset, names.data(), names.size());
        original[header.e_shstrndx].sh_offset = names_offset;
        original[header.e_shstrndx].sh_size = names.size();
        std::memcpy(bytes.data() + table_offset, original.data(), sizeof(original));
        put(table_offset + 8 * sizeof(Elf32_Shdr),
            Elf32_Shdr{
                static_cast<uint32_t>(name_offset),
                SHT_PROGBITS,
                flags,
                0,
                0x400,
                static_cast<uint32_t>(metadata.size()),
                0,
                0,
                4,
                0});
        header.e_shoff = table_offset;
        header.e_shnum = 9;
        put(0, header);
    }
};

struct FirmwareInput {
    std::array<NativeElf, 5> images{{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}}};
    auto spans() const {
        return std::array<std::span<const std::byte>, 5>{
            images[0].bytes, images[1].bytes, images[2].bytes, images[3].bytes, images[4].bytes};
    }
    void niu_metadata() {
        for (uint8_t role = 0; role < images.size(); ++role) {
            const tt_niu_handoff_record record{
                TT_NIU_HANDOFF_MAGIC,
                TT_NIU_HANDOFF_VERSION,
                TT_NIU_HANDOFF_RECORD_BYTES,
                role,
                TT_NATIVE_ARCH_BLACKHOLE,
                2,
                16,
                TT_NIU_HANDOFF_LAYOUT_VERSION,
                TT_NIU_HANDOFF_PROTOCOL,
                0xa040,
                TT_NIU_HANDOFF_L1_BYTES,
                0x8520,
                TT_NIU_HANDOFF_COUNTER_BYTES,
                TT_NIU_HANDOFF_COUNTER_CONTRACT,
                TT_NIU_HANDOFF_PROCESSOR_MASK,
                TT_NIU_HANDOFF_REQUEST_OFFSET,
                TT_NIU_HANDOFF_REQUEST_BYTES,
                TT_NIU_HANDOFF_RECEIPT_OFFSET,
                TT_NIU_HANDOFF_RECEIPT_BYTES,
                TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1};
            images[role].niu_metadata(std::as_bytes(std::span(&record, 1)));
        }
    }
};

using NocEndpointBytes = std::array<uint8_t, 4>;

struct NocWindowWire {
    uint32_t formal;
    uint32_t abi_word_offset;
    uint32_t relative_begin;
    uint32_t relative_end;
    uint32_t base_alignment;
    uint8_t access;
    uint8_t addressing;
    uint8_t noc_mask;
    uint8_t reserved = 0;
};

std::vector<std::byte> noc_window_metadata(
    std::span<const NocWindowWire> windows = {}, std::span<const NocEndpointBytes> endpoints = {}) {
    // The independent v3 fixture catches accidental dependence on host padding,
    // old header framing, or a window table that overwrites the endpoint table.
    std::vector<std::byte> bytes;
    const auto append = [&](uint32_t value, unsigned count) {
        for (unsigned index = 0; index != count; ++index) {
            bytes.push_back(static_cast<std::byte>((value >> (8 * index)) & 0xff));
        }
    };
    append(TT_NATIVE_NOC_MAGIC, 4);
    append(3, 2);
    append(32 + 4 * endpoints.size() + 24 * windows.size(), 2);
    append(TT_NATIVE_ROLE_BRISC, 1);
    append(TT_NATIVE_ARCH_BLACKHOLE, 1);
    append(0, 2);
    append(TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1, 4);
    append(0, 4);
    append(0, 4);
    append(endpoints.size(), 2);
    append(1, 2);
    append(windows.size(), 2);
    append(24, 2);
    for (const auto& endpoint : endpoints) {
        for (auto field : endpoint) {
            append(field, 1);
        }
    }
    for (const auto& window : windows) {
        append(window.formal, 4);
        append(window.abi_word_offset, 4);
        append(window.relative_begin, 4);
        append(window.relative_end, 4);
        append(window.base_alignment, 4);
        append(window.access, 1);
        append(window.addressing, 1);
        append(window.noc_mask, 1);
        append(window.reserved, 1);
    }
    return bytes;
}

void expect_noc_windows(const KernelElfImage& image, std::span<const NocWindowWire> expected) {
    const auto windows = image.noc_windows();
    static_assert(std::is_same_v<decltype(image.noc_windows()), std::span<const tt_native_noc_window>>);
    ASSERT_EQ(windows.size(), expected.size());
    for (size_t index = 0; index != expected.size(); ++index) {
        SCOPED_TRACE(index);
        const auto& actual = windows[index];
        const auto& original = expected[index];
        EXPECT_EQ(actual.formal, original.formal);
        EXPECT_EQ(actual.abi_word_offset, original.abi_word_offset);
        EXPECT_EQ(actual.relative_begin, original.relative_begin);
        EXPECT_EQ(actual.relative_end, original.relative_end);
        EXPECT_EQ(actual.base_alignment, original.base_alignment);
        EXPECT_EQ(actual.access, original.access);
        EXPECT_EQ(actual.addressing, original.addressing);
        EXPECT_EQ(actual.noc_mask, original.noc_mask);
        EXPECT_EQ(actual.reserved, original.reserved);
    }
}

std::vector<std::byte> noc_endpoint_metadata(
    std::span<const NocEndpointBytes> endpoints = {}, uint8_t role = TT_NATIVE_ROLE_BRISC) {
    // Encode the wire independently of the host's struct representation.
    std::vector<std::byte> bytes;
    const auto append = [&](uint32_t value, unsigned count) {
        for (unsigned index = 0; index != count; ++index) {
            bytes.push_back(static_cast<std::byte>((value >> (8 * index)) & 0xff));
        }
    };
    append(TT_NATIVE_NOC_MAGIC, 4);
    append(TT_NATIVE_NOC_ABI_VERSION, 2);
    append(32 + 4 * endpoints.size(), 2);
    append(role, 1);
    append(TT_NATIVE_ARCH_BLACKHOLE, 1);
    append(0, 2);
    append(TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1, 4);
    append(0x20000, 4);
    append(0x20004, 4);
    append(endpoints.size(), 2);
    append(0, 2);
    append(0, 2);
    append(24, 2);
    for (const auto& endpoint : endpoints) {
        for (auto field : endpoint) {
            append(field, 1);
        }
    }
    return bytes;
}

TEST(NativeElf, CopiesFirmwareAndKernelWithoutContext) {
    ASSERT_FALSE(MetalContext::instance_exists());
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    NativeElf kernel(1, 0);
    const auto image = KernelElfImage::from_bytes(kernel.bytes, firmware, "brisc");
    const auto expected = kernel.bytes;
    std::fill(kernel.bytes.begin(), kernel.bytes.end(), std::byte{});
    std::fill(input.images[0].bytes.begin(), input.images[0].bytes.end(), std::byte{});
    EXPECT_TRUE(std::equal(image.image_bytes().begin(), image.image_bytes().end(), expected.begin(), expected.end()));
    EXPECT_EQ(image.role(), TensixKernelRole::Brisc);
    EXPECT_EQ(image.label(), "brisc");
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(NativeElf, WormholeContractsRetainGenerationWithoutGrantingNiuSupport) {
    ASSERT_FALSE(MetalContext::instance_exists());
    FirmwareInput input;
    // Wire value 2 is the Wormhole architecture, independent of producer enums.
    for (auto& image : input.images) {
        image.put(NativeElf::note_offset + 8, uint8_t{2});
    }
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (uint8_t role = 0; role != TT_NATIVE_ROLE_COUNT; ++role) {
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, role);
        kernel.put(NativeElf::note_offset + 8, uint8_t{2});
        const auto image = KernelElfImage::from_bytes(kernel.bytes, firmware);
        EXPECT_EQ(image.contract().architecture, 2);
        EXPECT_EQ(firmware.contract(static_cast<TensixKernelRole>(role)).architecture, 2);
        EXPECT_FALSE(image.noc_contract());
    }
    EXPECT_FALSE(firmware.niu_handoff());
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(NativeElf, FirmwareRequiresOneKnownArchitectureAcrossAllRoles) {
    for (uint8_t architecture : {uint8_t{0}, uint8_t{3}, uint8_t{255}}) {
        FirmwareInput input;
        for (auto& image : input.images) {
            image.put(NativeElf::note_offset + 8, architecture);
        }
        EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
    }
    for (uint8_t differing_role = 0; differing_role != TT_NATIVE_ROLE_COUNT; ++differing_role) {
        FirmwareInput input;
        input.images[differing_role].put(NativeElf::note_offset + 8, uint8_t{2});
        EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
    }
}

TEST(NativeElf, KernelArchitectureMustMatchItsActualFirmware) {
    for (uint8_t firmware_architecture : {uint8_t{1}, uint8_t{2}}) {
        FirmwareInput input;
        for (auto& image : input.images) {
            image.put(NativeElf::note_offset + 8, firmware_architecture);
        }
        const auto firmware = FirmwareBundle::from_images(input.spans());
        for (uint8_t role = 0; role != TT_NATIVE_ROLE_COUNT; ++role) {
            NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, role);
            kernel.put(NativeElf::note_offset + 8, uint8_t{3 - firmware_architecture});
            EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
        }
    }
}

TEST(NativeElf, BlackholeNiuCapabilityCannotBeAttachedToWormholeFirmware) {
    FirmwareInput input;
    input.niu_metadata();
    for (auto& image : input.images) {
        image.put(NativeElf::note_offset + 8, uint8_t{2});
    }
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
}

TEST(NativeElf, WormholeKernelCannotInheritBlackholeAlgorithmNocQualification) {
    FirmwareInput input;
    for (auto& image : input.images) {
        image.put(NativeElf::note_offset + 8, uint8_t{2});
    }
    const auto firmware = FirmwareBundle::from_images(input.spans());
    NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    kernel.put(NativeElf::note_offset + 8, uint8_t{2});
    kernel.niu_metadata(noc_endpoint_metadata(), 0, TT_NATIVE_NOC_SECTION);
    EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
}

TEST(NativeElf, RejectsWrongRoleBundleAndLoadedMetadata) {
    FirmwareInput input;
    input.images[4].put(NativeElf::note_offset + 7, uint8_t{3});
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
    input.images[4] = NativeElf(0, 4);
    input.images[4].put(NativeElf::sections_offset + 3 * sizeof(Elf32_Shdr) + 8, uint32_t{SHF_ALLOC});
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
}

TEST(NativeElf, ByteEqualityDoesNotUseLabelsOrClaimedKeys) {
    FirmwareInput a, b;
    auto first = FirmwareBundle::from_images(a.spans(), "one");
    auto same = FirmwareBundle::from_images(b.spans(), "two");
    EXPECT_TRUE(first.matches(same));
    b.images[2].put(0x84, uint32_t{0x98765432});
    auto different = FirmwareBundle::from_images(b.spans(), "one");
    EXPECT_FALSE(first.matches(different));
}

TEST(NativeElf, RejectsUnsupportedOrUnavailableContracts) {
    FirmwareInput input;
    auto firmware = FirmwareBundle::from_images(input.spans());
    NativeElf kernel(1, 0);
    kernel.put(NativeElf::note_offset + 10, uint8_t{TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM});
    EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    kernel = NativeElf(1, 0);
    kernel.put(NativeElf::note_offset + 4, uint16_t{2});
    EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    kernel = NativeElf(0, 0);
    EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
}

TEST(NativeElf, RejectsMalformedDeclaredNiuHandoffCapability) {
    FirmwareInput input;
    // A declared firmware capability must be parsed, never silently treated as
    // absent. These bytes cannot describe a supported NIU handoff protocol.
    const std::array<std::byte, 48> malformed{};
    input.images[0].niu_metadata(malformed);
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
    input.images[0] = NativeElf(0, 0);
    input.images[0].niu_metadata(std::span(malformed).first(47));
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
    input.images[0] = NativeElf(0, 0);
    input.images[0].niu_metadata({});
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
}

TEST(NativeElf, RejectsMalformedDeclaredAlgorithmNocContract) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (const auto size : {size_t{0}, size_t{15}, size_t{16}, size_t{17}}) {
        SCOPED_TRACE(size);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        std::vector<std::byte> metadata(size);
        kernel.niu_metadata(metadata, 0, ".tt_native_noc");
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocContractRetainsOriginalRoleBytesWithoutGrantingBoot) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (uint8_t role = 0; role != TT_NATIVE_ROLE_COUNT; ++role) {
        SCOPED_TRACE(role);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, role);
        EXPECT_FALSE(KernelElfImage::from_bytes(kernel.bytes, firmware).noc_contract());
        const tt_native_noc_record record{
            TT_NATIVE_NOC_MAGIC,
            TT_NATIVE_NOC_ABI_VERSION,
            sizeof(tt_native_noc_record),
            role,
            TT_NATIVE_ARCH_BLACKHOLE,
            0,
            TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1,
            0,
            0,
            0,
            0,
            0,
            TT_NATIVE_NOC_WINDOW_BYTES};
        kernel.niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
        const auto image = KernelElfImage::from_bytes(kernel.bytes, firmware);
        std::fill(kernel.bytes.begin(), kernel.bytes.end(), std::byte{});
        ASSERT_TRUE(image.noc_contract());
        auto copy = *image.noc_contract();
        EXPECT_EQ(std::memcmp(&copy, &record, sizeof(record)), 0);
        copy.traffic_contract = 0;
        EXPECT_EQ(image.noc_contract()->traffic_contract, TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1);
        EXPECT_FALSE(image.firmware().niu_handoff());
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(NativeElf, AlgorithmNocContractRejectsWrongRoleVersionAndLoadedDeclarations) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const tt_native_noc_record record{
        TT_NATIVE_NOC_MAGIC,
        TT_NATIVE_NOC_ABI_VERSION,
        sizeof(tt_native_noc_record),
        TT_NATIVE_ROLE_BRISC,
        TT_NATIVE_ARCH_BLACKHOLE,
        0,
        TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1,
        0,
        0,
        0,
        0,
        0,
        TT_NATIVE_NOC_WINDOW_BYTES};
    for (const auto offset :
         {offsetof(tt_native_noc_record, magic),
          offsetof(tt_native_noc_record, version),
          offsetof(tt_native_noc_record, record_bytes),
          offsetof(tt_native_noc_record, role),
          offsetof(tt_native_noc_record, architecture),
          offsetof(tt_native_noc_record, reserved),
          offsetof(tt_native_noc_record, traffic_contract)}) {
        SCOPED_TRACE(offset);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
        kernel.bytes[0x400 + offset] ^= std::byte{0x80};
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
    NativeElf allocated(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    allocated.niu_metadata(std::as_bytes(std::span(&record, 1)), SHF_ALLOC, TT_NATIVE_NOC_SECTION);
    EXPECT_THROW(KernelElfImage::from_bytes(allocated.bytes, firmware), std::exception);
    NativeElf overlapped(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    overlapped.niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
    overlapped.put(sizeof(Elf32_Ehdr) + offsetof(Elf32_Phdr, p_filesz), uint32_t{0x400});
    overlapped.put(sizeof(Elf32_Ehdr) + offsetof(Elf32_Phdr, p_memsz), uint32_t{0x400});
    EXPECT_THROW(KernelElfImage::from_bytes(overlapped.bytes, firmware), std::exception);
    NativeElf duplicate(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    duplicate.niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
    Elf32_Shdr section{};
    std::memcpy(&section, duplicate.bytes.data() + 0x600 + 8 * sizeof(section), sizeof(section));
    duplicate.bytes.resize(0x600 + 10 * sizeof(section));
    duplicate.put(0x600 + 9 * sizeof(section), section);
    duplicate.put(offsetof(Elf32_Ehdr, e_shnum), uint16_t{10});
    EXPECT_THROW(KernelElfImage::from_bytes(duplicate.bytes, firmware), std::exception);
}

TEST(NativeElf, AlgorithmNocPayloadClassificationRetainsBothKindsWithOriginalImage) {
    ASSERT_FALSE(MetalContext::instance_exists());
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (uint8_t role = 0; role != TT_NATIVE_ROLE_COUNT; ++role) {
        for (uint8_t payload : {uint8_t{0}, uint8_t{1}}) {
            SCOPED_TRACE(role);
            SCOPED_TRACE(payload);
            auto metadata = noc_endpoint_metadata({}, role);
            // Independent v3 wire encoding retains the classification before
            // the relative-window table header.
            metadata[4] = std::byte{3};
            metadata[5] = std::byte{0};
            metadata[26] = static_cast<std::byte>(payload);
            NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, role);
            kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
            std::optional<KernelElfImage> retained;
            ASSERT_NO_THROW(retained.emplace(KernelElfImage::from_bytes(kernel.bytes, firmware)));
            std::fill(kernel.bytes.begin(), kernel.bytes.end(), std::byte{});
            std::fill(metadata.begin(), metadata.end(), std::byte{});
            const auto record = retained->noc_contract();
            ASSERT_TRUE(record);
            const auto bytes = std::as_bytes(std::span(&*record, 1));
            ASSERT_EQ(bytes.size(), 32u);
            EXPECT_EQ(std::to_integer<uint8_t>(bytes[26]), payload);
            EXPECT_EQ(std::to_integer<uint8_t>(bytes[27]), 0u);
            EXPECT_EQ(record->role, role);
            EXPECT_EQ(record->traffic_contract, TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1);
            // A static image classification cannot manufacture firmware or
            // a live invocation's completion/visibility reception.
            EXPECT_FALSE(retained->firmware().niu_handoff());
        }
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(NativeElf, AlgorithmNocPayloadClassificationRejectsRetiredVersionAndUnknownKinds) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (uint16_t payload : {uint16_t{0}, uint16_t{1}}) {
        auto metadata = noc_endpoint_metadata();
        metadata[4] = std::byte{1};
        metadata[5] = std::byte{0};
        metadata[26] = static_cast<std::byte>(payload);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
    for (uint16_t payload : {uint16_t{2}, uint16_t{255}, uint16_t{256}, uint16_t{65535}}) {
        SCOPED_TRACE(payload);
        auto metadata = noc_endpoint_metadata();
        metadata[4] = std::byte{3};
        metadata[5] = std::byte{0};
        metadata[26] = static_cast<std::byte>(payload & 0xff);
        metadata[27] = static_cast<std::byte>(payload >> 8);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocContractRejectsMalformedMemoryEnvelope) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (const auto [begin, end] :
         {std::pair{0u, 4u}, std::pair{0x20000u, 0u}, std::pair{0x20000u, 0x20000u}, std::pair{0x30000u, 0x20000u}}) {
        NativeElf elf(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        const tt_native_noc_record record{
            TT_NATIVE_NOC_MAGIC,
            TT_NATIVE_NOC_ABI_VERSION,
            sizeof(tt_native_noc_record),
            TT_NATIVE_ROLE_BRISC,
            TT_NATIVE_ARCH_BLACKHOLE,
            0,
            TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1,
            begin,
            end,
            0,
            0,
            0,
            TT_NATIVE_NOC_WINDOW_BYTES};
        elf.niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(elf.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocEndpointsRetainCanonicalBytesWithTheirImageOwner) {
    ASSERT_FALSE(MetalContext::instance_exists());
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const std::array<NocEndpointBytes, 4> expected{{{0, 0, 0, 0}, {0, 0, 63, 0}, {0, 63, 0, 0}, {1, 63, 63, 0}}};
    auto metadata = noc_endpoint_metadata(expected);
    const auto retained = [&] {
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
        const auto image = KernelElfImage::from_bytes(kernel.bytes, firmware);
        static_assert(std::is_same_v<decltype(image.noc_endpoints()), std::span<const tt_native_noc_endpoint>>);
        auto copy = image;
        EXPECT_EQ(copy.noc_endpoints().data(), image.noc_endpoints().data());
        std::fill(kernel.bytes.begin(), kernel.bytes.end(), std::byte{});
        std::fill(metadata.begin(), metadata.end(), std::byte{});
        return copy;
    }();
    ASSERT_TRUE(retained.noc_contract());
    EXPECT_EQ(retained.noc_contract()->record_bytes, 32 + 4 * expected.size());
    EXPECT_EQ(retained.noc_contract()->endpoint_count, expected.size());
    EXPECT_EQ(retained.noc_contract()->payload_contract, TT_NATIVE_NOC_PAYLOAD_UNCLASSIFIED);
    ASSERT_EQ(retained.noc_endpoints().size(), expected.size());
    for (size_t index = 0; index != expected.size(); ++index) {
        const auto& endpoint = retained.noc_endpoints()[index];
        EXPECT_EQ(endpoint.noc, expected[index][0]);
        EXPECT_EQ(endpoint.x, expected[index][1]);
        EXPECT_EQ(endpoint.y, expected[index][2]);
        EXPECT_EQ(endpoint.reserved, 0);
    }
    EXPECT_FALSE(retained.firmware().niu_handoff());
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(NativeElf, AlgorithmNocEndpointsAreEmptyForAbsentAndEmptyDeclarations) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    const auto absent = KernelElfImage::from_bytes(kernel.bytes, firmware);
    EXPECT_FALSE(absent.noc_contract());
    EXPECT_TRUE(absent.noc_endpoints().empty());
    kernel.niu_metadata(noc_endpoint_metadata(), 0, TT_NATIVE_NOC_SECTION);
    const auto empty = KernelElfImage::from_bytes(kernel.bytes, firmware);
    ASSERT_TRUE(empty.noc_contract());
    EXPECT_EQ(empty.noc_contract()->record_bytes, 32);
    EXPECT_EQ(empty.noc_contract()->endpoint_count, 0);
    EXPECT_TRUE(empty.noc_endpoints().empty());
}

TEST(NativeElf, AlgorithmNocEndpointsRejectIncompleteHeaderAndUnknownPayloadContract) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const auto metadata = noc_endpoint_metadata();
    for (size_t length = 0; length != 32; ++length) {
        SCOPED_TRACE(length);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(std::span(metadata).first(length), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
    for (size_t offset : {size_t{26}, size_t{27}}) {
        SCOPED_TRACE(offset);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        auto changed = metadata;
        changed[offset] = std::byte{2};
        kernel.niu_metadata(changed, 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocEndpointsRejectCountLengthAndTrailingByteDisagreement) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const std::array<NocEndpointBytes, 2> endpoints{{{0, 1, 2, 0}, {1, 3, 4, 0}}};
    for (unsigned mutation = 0; mutation != 9; ++mutation) {
        SCOPED_TRACE(mutation);
        auto metadata = noc_endpoint_metadata(endpoints);
        switch (mutation) {
            case 0: metadata[24] = std::byte{0}; break;
            case 1: metadata[24] = std::byte{1}; break;
            case 2: metadata[24] = std::byte{3}; break;
            case 3: metadata[6] = std::byte{32}; break;
            case 4: metadata[6] = std::byte{39}; break;
            case 5: metadata[6] = std::byte{41}; break;
            case 6: metadata.pop_back(); break;
            case 7: metadata.push_back(std::byte{}); break;
            case 8:
                metadata.push_back(std::byte{});
                metadata[6] = static_cast<std::byte>(metadata.size());
                break;
        }
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocEndpointsRejectDuplicatesAndEveryUnsortedCoordinate) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const std::array<std::array<NocEndpointBytes, 2>, 4> invalid{{
        {{{0, 1, 2, 0}, {0, 1, 2, 0}}},
        {{{1, 0, 0, 0}, {0, 63, 63, 0}}},
        {{{0, 2, 0, 0}, {0, 1, 63, 0}}},
        {{{0, 1, 2, 0}, {0, 1, 1, 0}}},
    }};
    for (size_t index = 0; index != invalid.size(); ++index) {
        SCOPED_TRACE(index);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(noc_endpoint_metadata(invalid[index]), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocEndpointsRejectInvalidNocCoordinatesAndReservedBytes) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const std::array<NocEndpointBytes, 8> invalid{
        {{2, 0, 0, 0},
         {255, 0, 0, 0},
         {0, 64, 0, 0},
         {0, 255, 0, 0},
         {0, 0, 64, 0},
         {0, 0, 255, 0},
         {0, 0, 0, 1},
         {0, 0, 0, 255}}};
    for (size_t index = 0; index != invalid.size(); ++index) {
        SCOPED_TRACE(index);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(noc_endpoint_metadata(std::span(&invalid[index], 1)), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocEndpointsAcceptTheCompleteBoundedCoordinateDomain) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    std::vector<NocEndpointBytes> endpoints;
    for (uint8_t noc = 0; noc != 2; ++noc) {
        for (uint8_t x = 0; x != 64; ++x) {
            for (uint8_t y = 0; y != 64; ++y) {
                endpoints.push_back({noc, x, y, 0});
            }
        }
    }
    ASSERT_EQ(endpoints.size(), 8192u);
    NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    kernel.niu_metadata(noc_endpoint_metadata(endpoints), 0, TT_NATIVE_NOC_SECTION);
    const auto image = KernelElfImage::from_bytes(kernel.bytes, firmware);
    ASSERT_TRUE(image.noc_contract());
    EXPECT_EQ(image.noc_contract()->record_bytes, 32800);
    EXPECT_EQ(image.noc_contract()->endpoint_count, 8192);
    ASSERT_EQ(image.noc_endpoints().size(), endpoints.size());
    for (size_t index = 0; index != endpoints.size(); ++index) {
        const auto& endpoint = image.noc_endpoints()[index];
        EXPECT_EQ(endpoint.noc, endpoints[index][0]);
        EXPECT_EQ(endpoint.x, endpoints[index][1]);
        EXPECT_EQ(endpoint.y, endpoints[index][2]);
        EXPECT_EQ(endpoint.reserved, 0);
    }
}

TEST(NativeElf, AlgorithmNocEndpointsRejectAnOverlargeDeclaredInventory) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    std::vector<NocEndpointBytes> endpoints(8193);
    NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    kernel.niu_metadata(noc_endpoint_metadata(endpoints), 0, TT_NATIVE_NOC_SECTION);
    EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
}

TEST(NativeElf, AlgorithmNocWindowsRetainOriginalBytesAndEndpointTableWithTheirImage) {
    ASSERT_FALSE(MetalContext::instance_exists());
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const std::array<NocEndpointBytes, 2> endpoints{{{0, 1, 2, 0}, {1, 3, 4, 0}}};
    const std::array<NocWindowWire, 3> expected{{
        {0, 0, 0, 16, 16, 1, 1, 0},
        {0, 0, 16, 32, 16, 2, 2, 1},
        {3, 7, 64, 4096, 64, 3, 2, 3},
    }};
    auto metadata = noc_window_metadata(expected, endpoints);
    NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
    std::optional<KernelElfImage> retained;
    ASSERT_NO_THROW(retained.emplace(KernelElfImage::from_bytes(kernel.bytes, firmware)));
    auto copy = *retained;
    retained.reset();
    std::fill(kernel.bytes.begin(), kernel.bytes.end(), std::byte{});
    std::fill(metadata.begin(), metadata.end(), std::byte{});
    ASSERT_TRUE(copy.noc_contract());
    EXPECT_EQ(copy.noc_contract()->version, 3);
    EXPECT_EQ(copy.noc_contract()->record_bytes, 112);
    EXPECT_EQ(copy.noc_contract()->l1_begin, 0);
    EXPECT_EQ(copy.noc_contract()->l1_end, 0);
    ASSERT_EQ(copy.noc_endpoints().size(), 2u);
    EXPECT_EQ(copy.noc_endpoints()[0].noc, 0);
    EXPECT_EQ(copy.noc_endpoints()[0].x, 1);
    EXPECT_EQ(copy.noc_endpoints()[1].noc, 1);
    EXPECT_EQ(copy.noc_endpoints()[1].y, 4);
    expect_noc_windows(copy, expected);
    EXPECT_FALSE(copy.firmware().niu_handoff());
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(NativeElf, AlgorithmNocWindowsAreEmptyForAbsentAndEmptyDeclarations) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    expect_noc_windows(KernelElfImage::from_bytes(kernel.bytes, firmware), {});
    kernel.niu_metadata(noc_window_metadata(), 0, TT_NATIVE_NOC_SECTION);
    std::optional<KernelElfImage> image;
    ASSERT_NO_THROW(image.emplace(KernelElfImage::from_bytes(kernel.bytes, firmware)));
    expect_noc_windows(*image, {});
    ASSERT_TRUE(image->noc_contract());
    EXPECT_EQ(image->noc_contract()->record_bytes, 32);
}

TEST(NativeElf, AlgorithmNocWindowsAcceptFullUnsignedFormalsAndLastNonwrappingThreeWordOffset) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const std::array<NocWindowWire, 1> expected{
        {{UINT32_MAX, UINT32_MAX - 2, 0, UINT32_MAX, uint32_t{1} << 31, 3, 2, 3}}};
    NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    kernel.niu_metadata(noc_window_metadata(expected), 0, TT_NATIVE_NOC_SECTION);
    std::optional<KernelElfImage> image;
    ASSERT_NO_THROW(image.emplace(KernelElfImage::from_bytes(kernel.bytes, firmware)));
    expect_noc_windows(*image, expected);
}

TEST(NativeElf, AlgorithmNocWindowsRejectRetiredVersionsIncludingTheExactV2Frame) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (uint16_t version : {uint16_t{0}, uint16_t{1}, uint16_t{2}, uint16_t{4}, uint16_t{65535}}) {
        SCOPED_TRACE(version);
        auto metadata = noc_window_metadata();
        metadata[4] = static_cast<std::byte>(version & 0xff);
        metadata[5] = static_cast<std::byte>(version >> 8);
        if (version == 2) {
            metadata.resize(28);
            metadata[6] = std::byte{28};
        }
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocWindowsRejectInvalidGeometryOffsetAndTags) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (unsigned mutation = 0; mutation != 19; ++mutation) {
        SCOPED_TRACE(mutation);
        std::array<NocWindowWire, 1> windows{{{1, 3, 0, 64, 16, 1, 2, 1}}};
        auto& window = windows.front();
        switch (mutation) {
            case 0: window.abi_word_offset = UINT32_MAX; break;
            case 1: window.abi_word_offset = UINT32_MAX - 1; break;
            case 2: window.relative_begin = window.relative_end; break;
            case 3: window.relative_begin = window.relative_end + 1; break;
            case 4: window.base_alignment = 0; break;
            case 5: window.base_alignment = 3; break;
            case 6: window.base_alignment = UINT32_MAX; break;
            case 7: window.access = 0; break;
            case 8: window.access = 4; break;
            case 9: window.access = 255; break;
            case 10: window.addressing = 0; break;
            case 11: window.addressing = 3; break;
            case 12: window.addressing = 255; break;
            case 13: window.noc_mask = 4; break;
            case 14: window.noc_mask = 255; break;
            case 15: window.reserved = 1; break;
            case 16: window.reserved = 255; break;
            case 17: window.relative_end = 0; break;
            case 18: window.noc_mask = 0; break;
        }
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(noc_window_metadata(windows), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocWindowsRequireTheExactEntryWidthForAnEmptyTable) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (uint16_t width : {uint16_t{0}, uint16_t{23}, uint16_t{25}, uint16_t{65535}}) {
        SCOPED_TRACE(width);
        auto metadata = noc_window_metadata();
        metadata[30] = static_cast<std::byte>(width & 0xff);
        metadata[31] = static_cast<std::byte>(width >> 8);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocWindowsRequireCanonicalOrderAcrossEveryFieldAndUniqueRecords) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (unsigned field = 0; field != 8; ++field) {
        SCOPED_TRACE(field);
        std::array<NocWindowWire, 2> windows{{{1, 3, 0, 64, 16, 1, 1, 1}, {1, 3, 0, 64, 16, 1, 1, 1}}};
        auto& last = windows.back();
        switch (field) {
            case 0: ++last.formal; break;
            case 1: ++last.abi_word_offset; break;
            case 2: ++last.relative_begin; break;
            case 3: ++last.relative_end; break;
            case 4: last.base_alignment = 32; break;
            case 5: last.access = 2; break;
            case 6: last.addressing = 2; break;
            case 7: last.noc_mask = 2; break;
        }
        NativeElf sorted(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        sorted.niu_metadata(noc_window_metadata(windows), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_NO_THROW(KernelElfImage::from_bytes(sorted.bytes, firmware));
        std::swap(windows.front(), windows.back());
        NativeElf unsorted(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        unsorted.niu_metadata(noc_window_metadata(windows), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(unsorted.bytes, firmware), std::exception);
        windows.back() = windows.front();
        NativeElf duplicate(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        duplicate.niu_metadata(noc_window_metadata(windows), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(duplicate.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocWindowsRejectTruncationCountsEntrySizesAndTrailingBytes) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    const std::array<NocWindowWire, 1> windows{{{1, 3, 0, 64, 16, 1, 2, 1}}};
    const auto complete = noc_window_metadata(windows);
    ASSERT_EQ(complete.size(), 56u);
    for (size_t size = 0; size != complete.size(); ++size) {
        SCOPED_TRACE(size);
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(std::span(complete).first(size), 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
    for (unsigned mutation = 0; mutation != 9; ++mutation) {
        SCOPED_TRACE(mutation);
        auto metadata = complete;
        switch (mutation) {
            case 0: metadata[28] = std::byte{0}; break;
            case 1: metadata[28] = std::byte{2}; break;
            case 2: metadata[30] = std::byte{0}; break;
            case 3: metadata[30] = std::byte{23}; break;
            case 4: metadata[30] = std::byte{25}; break;
            case 5: metadata[31] = std::byte{1}; break;
            case 6: metadata[6] = std::byte{32}; break;
            case 7: metadata.push_back(std::byte{}); break;
            case 8:
                metadata.push_back(std::byte{});
                metadata[6] = std::byte{57};
                break;
        }
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        kernel.niu_metadata(metadata, 0, TT_NATIVE_NOC_SECTION);
        EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes, firmware), std::exception);
    }
}

TEST(NativeElf, AlgorithmNocWindowsBoundTheCompleteDeclaredInventory) {
    FirmwareInput input;
    const auto firmware = FirmwareBundle::from_images(input.spans());
    std::vector<NocWindowWire> windows;
    for (uint32_t formal = 0; formal != 1024; ++formal) {
        windows.push_back({formal, formal * 3, 0, 64, 16, 1, 2, 1});
    }
    NativeElf accepted(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    accepted.niu_metadata(noc_window_metadata(windows), 0, TT_NATIVE_NOC_SECTION);
    std::optional<KernelElfImage> image;
    ASSERT_NO_THROW(image.emplace(KernelElfImage::from_bytes(accepted.bytes, firmware)));
    expect_noc_windows(*image, windows);
    windows.push_back({1024, 3072, 0, 64, 16, 1, 2, 1});
    NativeElf overlarge(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    overlarge.niu_metadata(noc_window_metadata(windows), 0, TT_NATIVE_NOC_SECTION);
    EXPECT_THROW(KernelElfImage::from_bytes(overlarge.bytes, firmware), std::exception);
}

TEST(NativeElf, RetainsNiuHandoffDeclarationFromAllFiveOriginalImages) {
    FirmwareInput input;
    input.niu_metadata();
    const auto firmware = FirmwareBundle::from_images(input.spans());
    for (auto& image : input.images) {
        std::fill(image.bytes.begin(), image.bytes.end(), std::byte{});
    }
    ASSERT_TRUE(firmware.niu_handoff());
    EXPECT_EQ(firmware.niu_handoff()->role, TT_NATIVE_ROLE_BRISC);
    EXPECT_EQ(firmware.niu_handoff()->l1_base, 0xa040u);
    EXPECT_EQ(firmware.niu_handoff()->dynamic_counter_base, 0x8520u);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(NativeElf, AbsentNiuHandoffDeclarationDoesNotCreateACapability) {
    FirmwareInput input;
    EXPECT_FALSE(FirmwareBundle::from_images(input.spans()).niu_handoff());
}

TEST(NativeElf, NiuHandoffUsesActualHalReservedLayout) {
    const Hal hal(tt::ARCH::BLACKHOLE, false, false, 0, false);
    const auto core = HalProgrammableCoreType::TENSIX;
    const auto region = hal.get_dev_addr(core, HalL1MemAddrType::NIU_HANDOFF);
    EXPECT_NE(region, 0u);
    EXPECT_EQ(region % TT_NIU_HANDOFF_ALIGNMENT, 0u);
    EXPECT_EQ(hal.get_dev_size(core, HalL1MemAddrType::NIU_HANDOFF), sizeof(tt_niu_handoff_region));
    EXPECT_EQ(region + sizeof(tt_niu_handoff_region), hal.get_dev_addr(core, HalL1MemAddrType::KERNEL_CONFIG));
    const auto counters = hal.get_dev_addr(core, HalL1MemAddrType::NOC_DYNAMIC_COUNTERS);
    EXPECT_NE(counters, 0u);
    EXPECT_EQ(hal.get_dev_size(core, HalL1MemAddrType::NOC_DYNAMIC_COUNTERS), TT_NIU_HANDOFF_COUNTER_BYTES);
    EXPECT_LE(counters + TT_NIU_HANDOFF_COUNTER_BYTES, region);
    EXPECT_EQ(hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::NIU_HANDOFF), 0u);
}

TEST(NativeElf, NiuHandoffAdmissionRejectsForeignHalAndDeclaredAddresses) {
    const Hal blackhole(tt::ARCH::BLACKHOLE, false, false, 0, false);
    const Hal wormhole(tt::ARCH::WORMHOLE_B0, false, false, 0, false);
    FirmwareInput input;
    input.niu_metadata();
    const auto firmware = FirmwareBundle::from_images(input.spans());
    EXPECT_NO_THROW(native_detail::NativeImageAccess::validate_niu_handoff(firmware, blackhole));
    EXPECT_THROW(native_detail::NativeImageAccess::validate_niu_handoff(firmware, wormhole), std::exception);
    for (const auto offset :
         {offsetof(tt_niu_handoff_record, l1_base), offsetof(tt_niu_handoff_record, dynamic_counter_base)}) {
        SCOPED_TRACE(offset);
        FirmwareInput changed;
        changed.niu_metadata();
        for (auto& image : changed.images) {
            image.put(0x400 + offset, uint32_t{0xb000});
        }
        const auto foreign = FirmwareBundle::from_images(changed.spans());
        EXPECT_THROW(native_detail::NativeImageAccess::validate_niu_handoff(foreign, blackhole), std::exception);
    }
    FirmwareInput ordinary;
    EXPECT_NO_THROW(
        native_detail::NativeImageAccess::validate_niu_handoff(
            FirmwareBundle::from_images(ordinary.spans()), blackhole));
}

TEST(NativeElf, RejectsPartialOrDisagreeingNiuHandoffDeclaration) {
    for (uint8_t role = 0; role < 5; ++role) {
        SCOPED_TRACE(role);
        FirmwareInput missing;
        missing.niu_metadata();
        missing.images[role] = NativeElf(0, role);
        EXPECT_THROW(FirmwareBundle::from_images(missing.spans()), std::exception);
        FirmwareInput changed;
        changed.niu_metadata();
        changed.images[role].put(0x400 + offsetof(tt_niu_handoff_record, l1_base), uint32_t{0xa060});
        EXPECT_THROW(FirmwareBundle::from_images(changed.spans()), std::exception);
    }
}

TEST(NativeElf, RejectsUnsupportedNiuHandoffFieldsAndUnsafeAddressRanges) {
    const auto reject = [](size_t offset, auto value) {
        SCOPED_TRACE(offset);
        FirmwareInput input;
        input.niu_metadata();
        for (auto& image : input.images) {
            image.put(0x400 + offset, value);
        }
        EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
    };
    reject(offsetof(tt_niu_handoff_record, magic), uint32_t{0});
    reject(offsetof(tt_niu_handoff_record, version), uint16_t{2});
    reject(offsetof(tt_niu_handoff_record, record_bytes), uint16_t{47});
    reject(offsetof(tt_niu_handoff_record, role), uint8_t{5});
    reject(offsetof(tt_niu_handoff_record, architecture), uint8_t{0});
    reject(offsetof(tt_niu_handoff_record, noc_count), uint8_t{1});
    reject(offsetof(tt_niu_handoff_record, selectors_per_noc), uint8_t{32});
    reject(offsetof(tt_niu_handoff_record, layout_version), uint16_t{2});
    reject(offsetof(tt_niu_handoff_record, protocol), uint16_t{2});
    reject(offsetof(tt_niu_handoff_record, l1_base), uint32_t{0});
    reject(offsetof(tt_niu_handoff_record, l1_base), uint32_t{0xa041});
    reject(offsetof(tt_niu_handoff_record, l1_base), uint32_t{0xffffffe0});
    reject(offsetof(tt_niu_handoff_record, l1_base), uint32_t{0x8520});
    reject(offsetof(tt_niu_handoff_record, l1_bytes), uint32_t{576});
    reject(offsetof(tt_niu_handoff_record, dynamic_counter_base), uint32_t{0});
    reject(offsetof(tt_niu_handoff_record, dynamic_counter_base), uint32_t{0x8521});
    reject(offsetof(tt_niu_handoff_record, dynamic_counter_base), uint32_t{0xfffffffc});
    reject(offsetof(tt_niu_handoff_record, counter_bytes), uint16_t{40});
    reject(offsetof(tt_niu_handoff_record, counter_contract), uint16_t{0});
    reject(offsetof(tt_niu_handoff_record, processor_mask), uint32_t{0x03});
    reject(offsetof(tt_niu_handoff_record, request_offset), uint16_t{4});
    reject(offsetof(tt_niu_handoff_record, request_bytes), uint16_t{32});
    reject(offsetof(tt_niu_handoff_record, receipt_offset), uint16_t{0});
    reject(offsetof(tt_niu_handoff_record, receipt_bytes), uint16_t{512});
    reject(offsetof(tt_niu_handoff_record, traffic_contract), uint32_t{0});
}

TEST(NativeElf, RejectsNiuHandoffDuplicateTypeAndLoadedPayloadOverlap) {
    for (unsigned mutation = 0; mutation < 3; ++mutation) {
        SCOPED_TRACE(mutation);
        FirmwareInput input;
        input.niu_metadata();
        auto& elf = input.images[0];
        constexpr auto section = 0x600 + 8 * sizeof(Elf32_Shdr);
        Elf32_Shdr metadata{};
        std::memcpy(&metadata, elf.bytes.data() + section, sizeof(metadata));
        if (mutation == 0) {
            elf.bytes.resize(0x600 + 10 * sizeof(Elf32_Shdr));
            elf.put(0x600 + 9 * sizeof(Elf32_Shdr), metadata);
            elf.put(offsetof(Elf32_Ehdr, e_shnum), uint16_t{10});
        } else if (mutation == 1) {
            metadata.sh_type = SHT_NOTE;
            elf.put(section, metadata);
        } else {
            metadata.sh_offset = 0x80;
            elf.put(section, metadata);
        }
        EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
    }
}

TEST(NativeElf, RejectsLoadedNiuHandoffMetadata) {
    FirmwareInput input;
    const std::array<std::byte, 48> record{};
    input.images[0].niu_metadata(record, SHF_ALLOC);
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()), std::exception);
}

TEST(NativeElf, ComputeGroupRejectsRolePermutationAndDifferentFirmware) {
    FirmwareInput input;
    auto firmware = FirmwareBundle::from_images(input.spans());
    const auto make = [&](uint8_t role, const FirmwareBundle& bundle) {
        return KernelElfImage::from_bytes(NativeElf(1, role).bytes, bundle);
    };
    NativeComputeConfig group{make(2, firmware), make(3, firmware), make(4, firmware)};
    EXPECT_NO_THROW(native_detail::NativeImageAccess::validate_compute(group));
    std::swap(group.trisc1, group.trisc2);
    EXPECT_THROW(native_detail::NativeImageAccess::validate_compute(group), std::exception);
    std::swap(group.trisc1, group.trisc2);
    input.images[0].put(0x84, uint32_t{0x87654321});
    group.trisc2 = make(4, FirmwareBundle::from_images(input.spans()));
    EXPECT_THROW(native_detail::NativeImageAccess::validate_compute(group), std::exception);
}

TEST(NativeElf, RetainsOriginalImageAlongsideXipPackedBytes) {
    FirmwareInput input;
    auto firmware = FirmwareBundle::from_images(input.spans());
    NativeElf kernel(1, 0);
    kernel.put(NativeElf::note_offset + 9, uint8_t{TT_NATIVE_LOADING_CONTIGUOUS_XIP});
    auto image = KernelElfImage::from_bytes(kernel.bytes, firmware);
    const auto& packed = native_detail::NativeImageAccess::memory(image);
    EXPECT_EQ(packed.get_loading(), ll_api::memory::Loading::CONTIGUOUS_XIP);
    EXPECT_EQ(packed.get_text_addr(), 0u);
    EXPECT_EQ(packed.data(), (std::vector<uint32_t>{0x00008067u, 0x12345678u}));
    EXPECT_TRUE(
        std::equal(image.image_bytes().begin(), image.image_bytes().end(), kernel.bytes.begin(), kernel.bytes.end()));
}

class NativeKernel : public ::testing::Test {
protected:
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = "/tmp"}};
    Hal hal{tt::ARCH::BLACKHOLE, false, false, 0, false, false, true, true};
    KernelBuildContext context{hal, options, nullptr};
    CoreRangeSet cores{CoreRange({0, 0}, {0, 0})};
    FirmwareInput input;
    FirmwareBundle firmware = FirmwareBundle::from_images(input.spans());
    KernelElfImage image(uint8_t role) {
        return KernelElfImage::from_bytes(NativeElf(TT_NATIVE_IMAGE_KERNEL, role).bytes, firmware, "native");
    }
    KernelElfImage image(uint8_t role, std::span<const NocEndpointBytes> endpoints) {
        NativeElf kernel(TT_NATIVE_IMAGE_KERNEL, role);
        kernel.niu_metadata(noc_endpoint_metadata(endpoints, role), 0, TT_NATIVE_NOC_SECTION);
        return KernelElfImage::from_bytes(kernel.bytes, firmware, "native");
    }
};

TEST_F(NativeKernel, RetainsImageAndRejectsSourceCompilationWithoutContext) {
    ASSERT_FALSE(MetalContext::instance_exists());
    auto bytes = image(TT_NATIVE_ROLE_BRISC);
    NativeElfKernel kernel(context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{bytes});
    EXPECT_TRUE(kernel.is_native());
    EXPECT_EQ(kernel.expected_num_binaries(), 1u);
    EXPECT_EQ(kernel.get_kernel_processor_type(0), 0u);
    EXPECT_EQ(kernel.name(), "native");
    EXPECT_THROW(kernel.kernel_source(), std::bad_variant_access);
    EXPECT_THROW(kernel.compute_hash(), std::exception);
    EXPECT_THROW(kernel.get_compiler_opt_level(), std::logic_error);
    EXPECT_THROW(kernel.get_linker_opt_level(), std::logic_error);
    EXPECT_THROW(kernel.read_binaries(nullptr, "/absent"), std::logic_error);
    const auto& binaries = kernel.owned_binaries();
    ASSERT_EQ(binaries.size(), 1u);
    EXPECT_EQ(binaries.front(), &native_detail::NativeImageAccess::memory(kernel.native_images().images.front()));
    EXPECT_THROW(kernel.binaries(uint64_t{17}), std::exception);
    EXPECT_THROW(kernel.set_binaries(uint64_t{17}, {binaries.front()}), std::exception);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ComputeRetainsExactRoleImagesAndPackedStorage) {
    const NativeComputeConfig config{
        image(TT_NATIVE_ROLE_TRISC0), image(TT_NATIVE_ROLE_TRISC1), image(TT_NATIVE_ROLE_TRISC2)};
    NativeElfKernel kernel(context, DEFAULT_CONTEXT_ID, cores, config);
    EXPECT_TRUE(kernel.is_native());
    EXPECT_EQ(kernel.expected_num_binaries(), 3u);
    const auto& binaries = kernel.owned_binaries();
    ASSERT_EQ(binaries.size(), 3u);
    for (size_t index = 0; index != 3; ++index) {
        EXPECT_EQ(kernel.get_kernel_processor_type(index), index);
        EXPECT_EQ(binaries[index], &native_detail::NativeImageAccess::memory(kernel.native_images().images[index]));
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedTrafficRequiresAlgorithmAndMatchingFirmwareDeclarations) {
    const auto qualified = [](uint8_t role, const FirmwareBundle& firmware) {
        NativeElf elf(TT_NATIVE_IMAGE_KERNEL, role);
        const tt_native_noc_record record{
            TT_NATIVE_NOC_MAGIC,
            TT_NATIVE_NOC_ABI_VERSION,
            sizeof(tt_native_noc_record),
            role,
            TT_NATIVE_ARCH_BLACKHOLE,
            0,
            TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1,
            0,
            0,
            0,
            0,
            0,
            TT_NATIVE_NOC_WINDOW_BYTES};
        elf.niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
        return KernelElfImage::from_bytes(elf.bytes, firmware);
    };
    FirmwareInput capable;
    capable.niu_metadata();
    const auto capable_firmware = FirmwareBundle::from_images(capable.spans());
    const auto unqualified = KernelElfImage::from_bytes(NativeElf(TT_NATIVE_IMAGE_KERNEL, 0).bytes, capable_firmware);
    const NativeDataMovementConfig missing_algorithm{unqualified, NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC};
    NativeElfKernel no_algorithm(context, DEFAULT_CONTEXT_ID, cores, missing_algorithm);
    EXPECT_THROW(no_algorithm.validate_managed_traffic(), std::exception);
    NativeElfKernel no_firmware(
        context,
        DEFAULT_CONTEXT_ID,
        cores,
        NativeDataMovementConfig{qualified(0, firmware), NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC});
    EXPECT_THROW(no_firmware.validate_managed_traffic(), std::exception);
    NativeElfKernel static_noc(
        context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{qualified(0, capable_firmware)});
    EXPECT_THROW(static_noc.validate_managed_traffic(), std::exception);
    NativeElfKernel eligible(
        context,
        DEFAULT_CONTEXT_ID,
        cores,
        NativeDataMovementConfig{qualified(0, capable_firmware), NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC});
    EXPECT_NO_THROW(eligible.validate_managed_traffic());
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedTrafficRejectsUnboundRelativeWindowsEvenWithMatchingFirmwareAndSafeFixedEnvelope) {
    FirmwareInput capable;
    capable.niu_metadata();
    const auto capable_firmware = FirmwareBundle::from_images(capable.spans());
    const std::array<NocWindowWire, 1> windows{{{1, 3, 0, 64, 16, 1, 2, 1}}};
    NativeElf elf(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    elf.niu_metadata(noc_window_metadata(windows), 0, TT_NATIVE_NOC_SECTION);
    std::optional<KernelElfImage> image;
    ASSERT_NO_THROW(image.emplace(KernelElfImage::from_bytes(elf.bytes, capable_firmware)));
    NativeElfKernel unbound(
        context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{*image, NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC});
    EXPECT_THROW(unbound.validate_managed_traffic(), std::exception);
    AllocatorConfig layout;
    layout.l1_unreserved_base = 0x20000;
    layout.worker_l1_size = 0x180000;
    EXPECT_THROW(unbound.validate_managed_traffic(layout), std::exception);
    const std::array<const NativeElfKernel*, 1> group{&unbound};
    EXPECT_THROW(NativeElfKernel::validate_managed_group(group), std::exception);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedComputeTrafficRequiresEveryActualRoleDeclaration) {
    FirmwareInput capable;
    capable.niu_metadata();
    const auto capable_firmware = FirmwareBundle::from_images(capable.spans());
    std::array<NativeElf, 3> elves{{{1, 2}, {1, 3}, {1, 4}}};
    for (uint8_t index = 0; index != elves.size(); ++index) {
        const tt_native_noc_record record{
            TT_NATIVE_NOC_MAGIC,
            TT_NATIVE_NOC_ABI_VERSION,
            sizeof(tt_native_noc_record),
            static_cast<uint8_t>(index + 2),
            TT_NATIVE_ARCH_BLACKHOLE,
            0,
            TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1,
            0,
            0,
            0,
            0,
            0,
            TT_NATIVE_NOC_WINDOW_BYTES};
        elves[index].niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
    }
    const auto unpack = KernelElfImage::from_bytes(elves[0].bytes, capable_firmware);
    const auto math = KernelElfImage::from_bytes(elves[1].bytes, capable_firmware);
    const auto pack = KernelElfImage::from_bytes(elves[2].bytes, capable_firmware);
    const auto missing = KernelElfImage::from_bytes(NativeElf(1, 3).bytes, capable_firmware);
    NativeElfKernel incomplete(context, DEFAULT_CONTEXT_ID, cores, NativeComputeConfig{unpack, missing, pack});
    EXPECT_THROW(incomplete.validate_managed_traffic(), std::exception);
    NativeElfKernel complete(context, DEFAULT_CONTEXT_ID, cores, NativeComputeConfig{unpack, math, pack});
    EXPECT_NO_THROW(complete.validate_managed_traffic());
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedTrafficExcludesTheActualConfiguredKernelLoadRegion) {
    FirmwareInput capable;
    capable.niu_metadata();
    const auto capable_firmware = FirmwareBundle::from_images(capable.spans());
    const auto with_range = [&](uint32_t begin, uint32_t end) {
        NativeElf elf(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
        const tt_native_noc_record record{
            TT_NATIVE_NOC_MAGIC,
            TT_NATIVE_NOC_ABI_VERSION,
            sizeof(tt_native_noc_record),
            TT_NATIVE_ROLE_BRISC,
            TT_NATIVE_ARCH_BLACKHOLE,
            0,
            TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1,
            begin,
            end,
            0,
            0,
            0,
            TT_NATIVE_NOC_WINDOW_BYTES};
        elf.niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
        return NativeDataMovementConfig{
            KernelElfImage::from_bytes(elf.bytes, capable_firmware), NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC};
    };
    AllocatorConfig layout;
    layout.l1_unreserved_base = 0x20000;
    layout.worker_l1_size = 0x180000;
    NativeElfKernel code_write(context, DEFAULT_CONTEXT_ID, cores, with_range(0x10000, 0x10004));
    EXPECT_THROW(code_write.validate_managed_traffic(layout), std::exception);
    NativeElfKernel crossing(context, DEFAULT_CONTEXT_ID, cores, with_range(0x1fffc, 0x20004));
    EXPECT_THROW(crossing.validate_managed_traffic(layout), std::exception);
    NativeElfKernel outside(context, DEFAULT_CONTEXT_ID, cores, with_range(0x17fffc, 0x180004));
    EXPECT_THROW(outside.validate_managed_traffic(layout), std::exception);
    NativeElfKernel data(context, DEFAULT_CONTEXT_ID, cores, with_range(0x20000, 0x20004));
    EXPECT_NO_THROW(data.validate_managed_traffic(layout));
    // Same algorithm becomes ineligible when the actual config ring grows.
    layout.l1_unreserved_base = 0x30000;
    EXPECT_THROW(data.validate_managed_traffic(layout), std::exception);
    NativeElfKernel no_access(context, DEFAULT_CONTEXT_ID, cores, with_range(0, 0));
    EXPECT_NO_THROW(no_access.validate_managed_traffic(layout));
    layout.worker_l1_size = 0;
    EXPECT_THROW(no_access.validate_managed_traffic(layout), std::exception);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedGroupRequiresDynamicDmOnEveryActualComputeCore) {
    FirmwareInput capable;
    capable.niu_metadata();
    const auto capable_firmware = FirmwareBundle::from_images(capable.spans());
    const auto qualified = [&](uint8_t role) {
        NativeElf elf(TT_NATIVE_IMAGE_KERNEL, role);
        const tt_native_noc_record record{
            TT_NATIVE_NOC_MAGIC,
            TT_NATIVE_NOC_ABI_VERSION,
            sizeof(tt_native_noc_record),
            role,
            TT_NATIVE_ARCH_BLACKHOLE,
            0,
            TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1,
            0,
            0,
            0,
            0,
            0,
            TT_NATIVE_NOC_WINDOW_BYTES};
        elf.niu_metadata(std::as_bytes(std::span(&record, 1)), 0, TT_NATIVE_NOC_SECTION);
        return KernelElfImage::from_bytes(elf.bytes, capable_firmware);
    };
    const CoreRangeSet both({CoreRange({0, 0}, {1, 0})});
    const CoreRangeSet first({CoreRange({0, 0}, {0, 0})});
    const CoreRangeSet second({CoreRange({1, 0}, {1, 0})});
    NativeElfKernel compute(
        context, DEFAULT_CONTEXT_ID, both, NativeComputeConfig{qualified(2), qualified(3), qualified(4)});
    NativeElfKernel first_dm(
        context,
        DEFAULT_CONTEXT_ID,
        first,
        NativeDataMovementConfig{qualified(0), NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC});
    NativeElfKernel second_dm(
        context,
        DEFAULT_CONTEXT_ID,
        second,
        NativeDataMovementConfig{qualified(1), NOC::NOC_1, NOC_MODE::DM_DYNAMIC_NOC});
    const std::array<const NativeElfKernel*, 1> compute_only{&compute};
    EXPECT_THROW(NativeElfKernel::validate_managed_group(compute_only), std::exception);
    const std::array<const NativeElfKernel*, 2> uncovered{&compute, &first_dm};
    EXPECT_THROW(NativeElfKernel::validate_managed_group(uncovered), std::exception);
    const std::array<const NativeElfKernel*, 3> covered{&compute, &first_dm, &second_dm};
    EXPECT_NO_THROW(NativeElfKernel::validate_managed_group(covered));
    const std::array<const NativeElfKernel*, 1> dm_only{&first_dm};
    EXPECT_NO_THROW(NativeElfKernel::validate_managed_group(dm_only));
    EXPECT_THROW(NativeElfKernel::validate_managed_group({}), std::exception);
    const std::array<const NativeElfKernel*, 1> absent{nullptr};
    EXPECT_THROW(NativeElfKernel::validate_managed_group(absent), std::exception);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedEndpointsAcceptAnUnorderedSupersetOnBothNocs) {
    const std::array<NocEndpointBytes, 2> endpoints{{{0, 1, 2, 0}, {1, 3, 4, 0}}};
    NativeElfKernel kernel(
        context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{image(TT_NATIVE_ROLE_BRISC, endpoints)});
    const std::array<tt_native_noc_endpoint, 3> allowed{{{1, 3, 4, 0}, {0, 63, 63, 0}, {0, 1, 2, 0}}};
    EXPECT_NO_THROW(kernel.validate_managed_endpoints(allowed));
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedEndpointsRejectMissingNocCoordinateOrRemoteCore) {
    const std::array<NocEndpointBytes, 2> endpoints{{{0, 1, 2, 0}, {1, 3, 4, 0}}};
    NativeElfKernel kernel(
        context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{image(TT_NATIVE_ROLE_BRISC, endpoints)});
    const std::array<std::array<tt_native_noc_endpoint, 2>, 4> incomplete{{
        {{{0, 1, 2, 0}, {0, 3, 4, 0}}},
        {{{0, 1, 2, 0}, {1, 4, 4, 0}}},
        {{{0, 1, 2, 0}, {1, 3, 5, 0}}},
        {{{1, 1, 2, 0}, {1, 3, 4, 0}}},
    }};
    for (size_t index = 0; index != incomplete.size(); ++index) {
        SCOPED_TRACE(index);
        EXPECT_THROW(kernel.validate_managed_endpoints(incomplete[index]), std::exception);
    }
    const std::array<tt_native_noc_endpoint, 1> first_only{{{0, 1, 2, 0}}};
    EXPECT_THROW(kernel.validate_managed_endpoints(first_only), std::exception);
    EXPECT_THROW(kernel.validate_managed_endpoints({}), std::exception);
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedEndpointsAllowAnEmptyDeclarationAndEmptyAllowedInventory) {
    NativeElfKernel kernel(
        context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{image(TT_NATIVE_ROLE_NCRISC, {})});
    EXPECT_NO_THROW(kernel.validate_managed_endpoints({}));
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, ManagedEndpointsRequireEveryActualComputeRole) {
    const std::array<NocEndpointBytes, 3> endpoints{{{0, 1, 2, 0}, {0, 3, 4, 0}, {1, 5, 6, 0}}};
    NativeElfKernel kernel(
        context,
        DEFAULT_CONTEXT_ID,
        cores,
        NativeComputeConfig{
            image(TT_NATIVE_ROLE_TRISC0, std::span(&endpoints[0], 1)),
            image(TT_NATIVE_ROLE_TRISC1, std::span(&endpoints[1], 1)),
            image(TT_NATIVE_ROLE_TRISC2, std::span(&endpoints[2], 1))});
    const std::array<tt_native_noc_endpoint, 3> allowed{{{1, 5, 6, 0}, {0, 1, 2, 0}, {0, 3, 4, 0}}};
    EXPECT_NO_THROW(kernel.validate_managed_endpoints(allowed));
    for (size_t omitted = 0; omitted != allowed.size(); ++omitted) {
        SCOPED_TRACE(omitted);
        std::vector<tt_native_noc_endpoint> incomplete;
        for (size_t index = 0; index != allowed.size(); ++index) {
            if (index != omitted) {
                incomplete.push_back(allowed[index]);
            }
        }
        EXPECT_THROW(kernel.validate_managed_endpoints(incomplete), std::exception);
    }
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST_F(NativeKernel, PreparationWithoutContextRejectsBeforeSourceEnvironmentAccess) {
    ASSERT_FALSE(MetalContext::instance_exists());
    NativeElfKernel kernel(context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{image(0)});
    try {
        kernel.prepare(nullptr, tt::worker_stream_state::Owner::SdkCircularBuffers);
        FAIL() << "Native preparation requires a real device and boot";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("existing device context"), std::string::npos);
    }
    EXPECT_FALSE(MetalContext::instance_exists());
    EXPECT_EQ(kernel.get_binary_packed_size(nullptr, 0), kernel.owned_binaries().front()->get_packed_size());
    EXPECT_EQ(kernel.get_binary_text_size(nullptr, 0), kernel.owned_binaries().front()->get_text_size());
}

TEST_F(NativeKernel, RejectsUnsupportedWatcherEvenWhenAssertionsAreDisabled) {
    options.set_watcher_enabled(true);
    options.disable_watcher_assert();
    EXPECT_THROW(
        (NativeElfKernel(context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{image(0)})), std::exception);
}

TEST_F(NativeKernel, MapsWorkerStreamOwnershipAcrossDistinctWireEnums) {
    NativeElf sdk_bytes(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    auto sdk_image = KernelElfImage::from_bytes(sdk_bytes.bytes, firmware);
    NativeElfKernel sdk_kernel(context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{sdk_image});
    EXPECT_NO_THROW(sdk_kernel.validate_stream_owner(tt::worker_stream_state::Owner::SdkCircularBuffers));
    EXPECT_THROW(sdk_kernel.validate_stream_owner(tt::worker_stream_state::Owner::Program), std::exception);
    sdk_bytes.put(NativeElf::note_offset + 14, uint8_t{TT_NATIVE_WORKER_STREAM_PROGRAM});
    auto program_image = KernelElfImage::from_bytes(sdk_bytes.bytes, firmware);
    NativeElfKernel program_kernel(context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{program_image});
    EXPECT_NO_THROW(program_kernel.validate_stream_owner(tt::worker_stream_state::Owner::Program));
    EXPECT_THROW(
        program_kernel.validate_stream_owner(tt::worker_stream_state::Owner::SdkCircularBuffers), std::exception);
    EXPECT_THROW(program_kernel.validate_stream_owner(static_cast<tt::worker_stream_state::Owner>(0)), std::exception);
}
}  // namespace
}  // namespace tt::tt_metal::experimental
