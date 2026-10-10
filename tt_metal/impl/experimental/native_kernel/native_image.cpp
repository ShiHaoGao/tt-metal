// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "native_image.hpp"

#include <elf.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "hostdev/worker_stream_state_contract.h"
#include "llrt/tt_elffile.hpp"

namespace tt::tt_metal::experimental {
namespace {
size_t index(TensixKernelRole role) {
    auto result = static_cast<size_t>(role);
    if (result >= TT_NATIVE_ROLE_COUNT) throw std::invalid_argument("invalid native processor role");
    return result;
}

tt_native_image_record read_contract(ll_api::ElfFile& elf, uint8_t kind) {
    const auto bytes = elf.GetMetadataSection(TT_NATIVE_IMAGE_SECTION);
    if (bytes.size() != sizeof(tt_native_image_record))
        throw std::invalid_argument("native ELF requires one exact support ABI record");
    tt_native_image_record result{};
    std::memcpy(&result, bytes.data(), sizeof(result));
    if (result.magic != TT_NATIVE_IMAGE_MAGIC || result.version != TT_NATIVE_IMAGE_ABI_VERSION ||
        result.kind != kind || result.role >= TT_NATIVE_ROLE_COUNT ||
        result.architecture != TT_NATIVE_ARCH_BLACKHOLE ||
        result.loading > TT_NATIVE_LOADING_CONTIGUOUS_XIP ||
        result.profile > TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM ||
        result.print > TT_NATIVE_PRINT_TENSIX_SHARED_BUFFER ||
        result.worker_stream_abi != tt::worker_stream_state::kVersion ||
        result.worker_stream_owner > TT_NATIVE_WORKER_STREAM_PROGRAM || result.reserved)
        throw std::invalid_argument("unsupported native ELF support contract");
    if (kind == TT_NATIVE_IMAGE_FIRMWARE &&
        (result.loading != TT_NATIVE_LOADING_DISCRETE || result.worker_stream_owner != TT_NATIVE_WORKER_STREAM_SDK))
        throw std::invalid_argument("firmware requires discrete loading and SDK default worker ownership");
    return result;
}

void validate_executable(std::span<const std::byte> bytes, const ll_api::ElfFile& elf) {
    Elf32_Ehdr header{};
    if (bytes.size() < sizeof(header)) throw std::invalid_argument("truncated native ELF header");
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (header.e_ident[EI_CLASS] != ELFCLASS32 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
        header.e_machine != EM_RISCV || header.e_type != ET_EXEC ||
        (header.e_flags & ~uint32_t(EF_RISCV_RVC)))
        throw std::invalid_argument("native image must be an ELF32 little-endian RISC-V soft-float executable");
    if (elf.GetSegments().empty() || header.e_entry != elf.GetSegments().front().address)
        throw std::invalid_argument("native entry must start the executable text segment");
}

using NativeImageSymbols = std::unordered_map<std::string, Elf32_Sym>;
NativeImageSymbols symbols(const ll_api::ElfFile& elf) {
    uint64_t address = 0;
    auto table = elf.GetSectionContents(".symtab", address);
    auto strings = elf.GetSectionContents(".strtab", address);
    NativeImageSymbols result;
    for (size_t offset = 0; offset + sizeof(Elf32_Sym) <= table.size(); offset += sizeof(Elf32_Sym)) {
        Elf32_Sym symbol{};
        std::memcpy(&symbol, table.data() + offset, sizeof(symbol));
        if (symbol.st_name == 0 || ELF32_ST_BIND(symbol.st_info) == STB_LOCAL) continue;
        if (symbol.st_name >= strings.size() ||
            !std::memchr(strings.data() + symbol.st_name, 0, strings.size() - symbol.st_name))
            throw std::invalid_argument("native ELF symbol string is out of bounds");
        result.emplace(reinterpret_cast<const char*>(strings.data() + symbol.st_name), symbol);
    }
    return result;
}

struct Image {
    std::vector<std::byte> bytes;
    std::vector<std::byte> link_bytes;
    tt_native_image_record contract{};
    ll_api::memory memory;
    NativeImageSymbols exports;
};
}  // namespace

struct FirmwareBundle::Impl {
    std::array<Image, TT_NATIVE_ROLE_COUNT> images;
};

struct KernelElfImage::Impl {
    Image image;
    FirmwareBundle firmware;
    std::string label;
    Impl(FirmwareBundle fw, std::string_view text) : firmware(std::move(fw)), label(text) {}
};

FirmwareBundle::FirmwareBundle(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
KernelElfImage::KernelElfImage(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

FirmwareBundle FirmwareBundle::from_images(
    const std::array<std::span<const std::byte>, TT_NATIVE_ROLE_COUNT>& inputs, std::string_view label) {
    auto owner = std::make_shared<Impl>();
    for (size_t role = 0; role != inputs.size(); ++role) {
        auto& image = owner->images[role];
        image.bytes.assign(inputs[role].begin(), inputs[role].end());
        ll_api::ElfFile elf;
        elf.ReadImage(image.bytes, label);
        validate_executable(image.bytes, elf);
        image.contract = read_contract(elf, TT_NATIVE_IMAGE_FIRMWARE);
        if (image.contract.role != role) throw std::invalid_argument("firmware bundle role mismatch");
        if (role && (image.contract.profile != owner->images[0].contract.profile ||
                     image.contract.print != owner->images[0].contract.print))
            throw std::invalid_argument("firmware bundle observation transports differ");
        image.exports = symbols(elf);
        image.memory = ll_api::memory(image.bytes, ll_api::memory::Loading::DISCRETE, label);
        constexpr std::array<std::string_view, 2> strong{"__fw_export_*", "__global_pointer$"};
        elf.WeakenDataSymbols(strong);
        auto linked = elf.GetImageContents();
        image.link_bytes.assign(linked.begin(), linked.end());
    }
    return FirmwareBundle(std::move(owner));
}

std::span<const std::byte> FirmwareBundle::image_bytes(TensixKernelRole role) const {
    return impl_->images[index(role)].bytes;
}
std::span<const std::byte> FirmwareBundle::link_image_bytes(TensixKernelRole role) const {
    return impl_->images[index(role)].link_bytes;
}
const tt_native_image_record& FirmwareBundle::contract(TensixKernelRole role) const {
    return impl_->images[index(role)].contract;
}
bool FirmwareBundle::matches(const FirmwareBundle& other) const {
    if (impl_ == other.impl_) return true;
    for (size_t role = 0; role != TT_NATIVE_ROLE_COUNT; ++role)
        if (impl_->images[role].bytes != other.impl_->images[role].bytes) return false;
    return true;
}

KernelElfImage KernelElfImage::from_bytes(
    std::span<const std::byte> input, const FirmwareBundle& firmware, std::string_view label) {
    auto owner = std::make_shared<Impl>(firmware, label);
    auto& image = owner->image;
    image.bytes.assign(input.begin(), input.end());
    ll_api::ElfFile elf;
    elf.ReadImage(image.bytes, label);
    validate_executable(image.bytes, elf);
    image.contract = read_contract(elf, TT_NATIVE_IMAGE_KERNEL);
    const auto role = static_cast<TensixKernelRole>(image.contract.role);
    const auto& available = firmware.contract(role);
    if ((image.contract.profile && image.contract.profile != available.profile) ||
        (image.contract.print && image.contract.print != available.print))
        throw std::invalid_argument("native observation transport is unavailable in linked firmware");
    ll_api::ElfFile fw;
    fw.ReadImage(firmware.image_bytes(role), "linked firmware");
    const auto exports = symbols(fw);
    const auto imports = symbols(elf);
    for (const auto& [name, imported] : imports) {
        if (imported.st_shndx == SHN_UNDEF && ELF32_ST_BIND(imported.st_info) != STB_WEAK)
            throw std::invalid_argument("native executable has an unresolved symbol: " + name);
        if (imported.st_shndx != SHN_ABS) continue;
        auto exported = exports.find(name);
        if (exported != exports.end() && exported->second.st_value != imported.st_value)
            throw std::invalid_argument("native firmware import address mismatch: " + name);
    }
    if (image.contract.profile) {
        for (const auto* name : {"_ZN15kernel_profiler20profiler_data_bufferE", "_ZN15kernel_profiler6wIndexE",
                                "_ZN15kernel_profiler9stackSizeE"}) {
            auto symbol = exports.find(name);
            if (symbol == exports.end() || symbol->second.st_shndx == SHN_UNDEF ||
                ELF32_ST_TYPE(symbol->second.st_info) != STT_OBJECT || symbol->second.st_size != sizeof(uint32_t))
                throw std::invalid_argument("profile firmware lacks resident object: " + std::string(name));
        }
    }
    auto loading = image.contract.loading == TT_NATIVE_LOADING_DISCRETE
        ? ll_api::memory::Loading::DISCRETE : ll_api::memory::Loading::CONTIGUOUS_XIP;
    image.memory = ll_api::memory(image.bytes, loading, label);
    return KernelElfImage(std::move(owner));
}

TensixKernelRole KernelElfImage::role() const { return static_cast<TensixKernelRole>(impl_->image.contract.role); }
std::span<const std::byte> KernelElfImage::image_bytes() const { return impl_->image.bytes; }
const tt_native_image_record& KernelElfImage::contract() const { return impl_->image.contract; }
const FirmwareBundle& KernelElfImage::firmware() const { return impl_->firmware; }
std::string_view KernelElfImage::label() const { return impl_->label; }

const ll_api::memory& native_detail::NativeImageAccess::memory(const KernelElfImage& image) { return image.impl_->image.memory; }
const ll_api::memory& native_detail::NativeImageAccess::memory(const FirmwareBundle& image, TensixKernelRole role) {
    return image.impl_->images[index(role)].memory;
}
void native_detail::NativeImageAccess::validate_compute(const NativeComputeConfig& images) {
    if (images.trisc0.role() != TensixKernelRole::Trisc0 || images.trisc1.role() != TensixKernelRole::Trisc1 ||
        images.trisc2.role() != TensixKernelRole::Trisc2)
        throw std::invalid_argument("native compute requires TRISC0, TRISC1, TRISC2 in exact role order");
    for (const auto* image : {&images.trisc1, &images.trisc2}) {
        const auto& a = images.trisc0.contract();
        const auto& b = image->contract();
        if (!images.trisc0.firmware().matches(image->firmware()) || a.profile != b.profile ||
            a.print != b.print || a.worker_stream_owner != b.worker_stream_owner)
            throw std::invalid_argument("native compute images require one firmware and observation contract");
    }
}
}  // namespace tt::tt_metal::experimental
