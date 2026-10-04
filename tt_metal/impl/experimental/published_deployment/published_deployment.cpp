// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "storage.hpp"
#include "dispatch_plan.hpp"
#include "llrt/tt_elffile.hpp"
#include <elf.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tt::tt_metal::experimental {
namespace {
using Symbols = std::map<std::string, Elf32_Sym>;
Symbols read_symbols(const ll_api::ElfFile& elf) {
    uint64_t address = 0;
    auto table = elf.GetSectionContents(".symtab", address);
    auto strings = elf.GetSectionContents(".strtab", address);
    if (table.empty() || table.size() % sizeof(Elf32_Sym) || strings.empty())
        throw std::invalid_argument("published image requires a complete symbol table");
    Symbols result;
    for (size_t offset = 0; offset != table.size(); offset += sizeof(Elf32_Sym)) {
        Elf32_Sym symbol{};
        std::memcpy(&symbol, table.data() + offset, sizeof(symbol));
        if (!symbol.st_name || ELF32_ST_BIND(symbol.st_info) == STB_LOCAL) continue;
        if (symbol.st_name >= strings.size() ||
            !std::memchr(strings.data() + symbol.st_name, 0, strings.size() - symbol.st_name))
            throw std::invalid_argument("published image symbol string exceeds its table");
        if (symbol.st_shndx == SHN_UNDEF && ELF32_ST_BIND(symbol.st_info) != STB_WEAK)
            throw std::invalid_argument("published image contains an unresolved symbol");
        std::string name(reinterpret_cast<const char*>(strings.data() + symbol.st_name));
        if (!result.emplace(name, symbol).second)
            throw std::invalid_argument("published image has ambiguous global symbols");
    }
    return result;
}
template<class T> T elf_record(std::span<const std::byte> bytes, size_t offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
        throw std::invalid_argument("published ELF record is outside the image");
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}
std::vector<Elf32_Phdr> validate_startup_and_segments(std::span<const std::byte> bytes, const Elf32_Ehdr& header, const Elf32_Sym& start) {
    if (ELF32_ST_TYPE(start.st_info) != STT_FUNC || start.st_shndx >= header.e_shnum ||
        start.st_shndx >= SHN_LORESERVE)
        throw std::invalid_argument("published startup is not a defined executable function");
    const auto section = elf_record<Elf32_Shdr>(bytes, size_t(header.e_shoff) + start.st_shndx * sizeof(Elf32_Shdr));
    const uint64_t function_size = std::max(uint64_t{1}, uint64_t{start.st_size});
    if (section.sh_type != SHT_PROGBITS ||
        (section.sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) != (SHF_ALLOC | SHF_EXECINSTR) ||
        start.st_value < section.sh_addr ||
        uint64_t(start.st_value) + function_size > uint64_t(section.sh_addr) + section.sh_size)
        throw std::invalid_argument("published startup does not belong to executable section bytes");
    std::vector<Elf32_Phdr> loads;
    bool executable_start = false;
    for (unsigned index = 0; index != header.e_phnum; ++index) {
        const auto segment = elf_record<Elf32_Phdr>(bytes, size_t(header.e_phoff) + index * sizeof(Elf32_Phdr));
        if (segment.p_type != PT_LOAD) continue;
        for (const auto& previous : loads) {
            if (segment.p_memsz && previous.p_memsz &&
                uint64_t(segment.p_vaddr) < uint64_t(previous.p_vaddr) + previous.p_memsz &&
                uint64_t(previous.p_vaddr) < uint64_t(segment.p_vaddr) + segment.p_memsz)
                throw std::invalid_argument("published load segments overlap in execution memory");
        }
        loads.push_back(segment);
        if ((segment.p_flags & PF_X) && start.st_value >= segment.p_vaddr &&
            uint64_t(start.st_value) + function_size <= uint64_t(segment.p_vaddr) + segment.p_filesz &&
            uint64_t(section.sh_offset) + start.st_value - section.sh_addr ==
                uint64_t(segment.p_offset) + start.st_value - segment.p_vaddr)
            executable_start = true;
    }
    if (!executable_start)
        throw std::invalid_argument("published startup has no executable load bytes");
    return loads;
}
struct FirmwareLayout { uint64_t text_end; uint64_t data_end; };
struct OwnedImage {
    PublishedImageKind kind;
    HalProcessorIdentifier processor;
    uint32_t node;
    std::vector<std::byte> bytes;
    ll_api::memory memory;
    Symbols symbols;
    std::vector<Elf32_Phdr> loads;
    std::optional<FirmwareLayout> firmware_layout;
};
bool contains(HalImageRegion region, uint64_t address, uint64_t size) {
    return address >= region.base && address - region.base <= region.size && size <= region.size - (address - region.base);
}
uint64_t aligned_end(uint64_t end) { return (end + 15) & ~uint64_t{15}; }
void require_export(const Symbols& symbols, const char* name, uint64_t value) {
    const auto found = symbols.find(name);
    if (found == symbols.end() || found->second.st_shndx != SHN_ABS ||
        ELF32_ST_BIND(found->second.st_info) != STB_GLOBAL || ELF32_ST_TYPE(found->second.st_info) != STT_NOTYPE ||
        found->second.st_value != value)
        throw std::invalid_argument("firmware layout export differs from its actual segments");
}
FirmwareLayout validate_firmware_regions(const OwnedImage& image, const HalProcessorImageRegions& regions) {
    uint64_t text_end = regions.firmware_text.base;
    uint64_t data_end = regions.local_data.base;
    for (const auto& segment : image.loads) {
        const bool executable = segment.p_flags & PF_X;
        const auto region = executable ? regions.firmware_text : regions.local_data;
        if (!contains(region, segment.p_vaddr, segment.p_memsz) || segment.p_paddr != segment.p_vaddr ||
            (executable && (segment.p_flags & PF_W)))
            throw std::invalid_argument("firmware load exceeds its actual processor image region");
        if (executable) text_end = std::max(text_end, uint64_t(segment.p_vaddr) + segment.p_memsz);
        else {
            data_end = std::max(data_end, uint64_t(segment.p_vaddr) + segment.p_memsz);
            const uint64_t offset = segment.p_vaddr - regions.local_data.base;
            if (segment.p_filesz &&
                (offset > regions.local_initialization.size || segment.p_filesz > regions.local_initialization.size - offset))
                throw std::invalid_argument("firmware initialized local data exceeds actual staging region");
        }
    }
    FirmwareLayout result{aligned_end(text_end + regions.firmware_kernel_pad), aligned_end(data_end)};
    // main.ld owns these two mandatory ABI exports. Their values are derived
    // independently from real segments and actual HAL limits before use.
    require_export(image.symbols, "__fw_export_text_end", result.text_end);
    require_export(image.symbols, "__fw_export_ldm_end", result.data_end);
    return result;
}
void validate_dispatch_regions(const OwnedImage& image, const OwnedImage& firmware,
                               const HalProcessorImageRegions& regions) {
    if (!firmware.firmware_layout || image.loads.empty() || image.loads.front().p_vaddr != firmware.firmware_layout->text_end)
        throw std::invalid_argument("dispatch entry differs from actual firmware text handoff");
    const auto layout = *firmware.firmware_layout;
    const uint64_t text_limit = (regions.kernel_text_limit == HalKernelTextLimit::FromFirmwareBase
        ? regions.firmware_text.base : layout.text_end) + regions.kernel_text_size;
    if (layout.text_end > text_limit || layout.data_end > regions.local_data.base + regions.local_data.size)
        throw std::invalid_argument("firmware leaves no valid dispatch image region");
    for (const auto& segment : image.loads) {
        const bool executable = segment.p_flags & PF_X;
        const uint64_t begin = executable ? layout.text_end : layout.data_end;
        const uint64_t end = executable ? text_limit : regions.local_data.base + regions.local_data.size;
        if (segment.p_vaddr < begin || uint64_t(segment.p_vaddr) + segment.p_memsz > end ||
            (executable && (segment.p_flags & PF_W)))
            throw std::invalid_argument("dispatch load exceeds actual firmware/processor image region");
    }
}
enum class AbsoluteSymbolOwner { FirmwareExport, KernelTextStart };
AbsoluteSymbolOwner absolute_symbol_owner(std::string_view name) {
    // Exact SDK linker ABI from hw/toolchain/main.ld: TYPE_KERNEL plus
    // COMPILE_FOR_NCRISC defines this symbol at the kernel text entry.
    // Every other absolute symbol must resolve to the selected firmware.
    return name == "__kernel_text_start" ? AbsoluteSymbolOwner::KernelTextStart : AbsoluteSymbolOwner::FirmwareExport;
}
void validate_firmware_symbol_storage(const OwnedImage& firmware, const Elf32_Sym& symbol) {
    if (symbol.st_shndx == SHN_ABS) {
        if (ELF32_ST_BIND(symbol.st_info) != STB_GLOBAL ||
            ELF32_ST_TYPE(symbol.st_info) != STT_NOTYPE || symbol.st_size != 0)
            throw std::invalid_argument("firmware absolute provider is not a linker constant");
        return;
    }
    const auto header = elf_record<Elf32_Ehdr>(firmware.bytes, 0);
    if (symbol.st_shndx == SHN_UNDEF || symbol.st_shndx >= SHN_LORESERVE || symbol.st_shndx >= header.e_shnum)
        throw std::invalid_argument("firmware provider symbol has no real definition");
    const auto section = elf_record<Elf32_Shdr>(firmware.bytes,
        size_t(header.e_shoff) + symbol.st_shndx * sizeof(Elf32_Shdr));
    const auto type = ELF32_ST_TYPE(symbol.st_info);
    if (!(section.sh_flags & SHF_ALLOC) ||
        (section.sh_type != SHT_PROGBITS && section.sh_type != SHT_NOBITS) ||
        symbol.st_value < section.sh_addr ||
        uint64_t(symbol.st_value) + symbol.st_size > uint64_t(section.sh_addr) + section.sh_size ||
        (type == STT_FUNC && (!(section.sh_flags & SHF_EXECINSTR) || section.sh_type != SHT_PROGBITS)))
        throw std::invalid_argument("firmware provider symbol differs from its allocated section");
    const auto allocated = std::any_of(firmware.loads.begin(), firmware.loads.end(), [&](const auto& segment) {
        return symbol.st_value >= segment.p_vaddr &&
            uint64_t(symbol.st_value) + symbol.st_size <= uint64_t(segment.p_vaddr) + segment.p_memsz &&
            (type != STT_FUNC || ((segment.p_flags & PF_X) &&
                uint64_t(symbol.st_value) + std::max(uint64_t{1}, uint64_t{symbol.st_size}) <= uint64_t(segment.p_vaddr) + segment.p_filesz &&
                uint64_t(section.sh_offset) + symbol.st_value - section.sh_addr ==
                    uint64_t(segment.p_offset) + symbol.st_value - segment.p_vaddr));
    });
    if (!allocated) throw std::invalid_argument("firmware provider symbol is not resident in an actual load segment");
}
void validate_absolute_symbols(const OwnedImage& image, const OwnedImage& firmware) {
    for (const auto& [name, imported] : image.symbols) {
        if (imported.st_shndx != SHN_ABS) continue;
        switch (absolute_symbol_owner(name)) {
        case AbsoluteSymbolOwner::KernelTextStart:
            if (image.processor.core_type != HalProgrammableCoreType::TENSIX ||
                image.processor.processor_class != HalProcessorClassType::DM || image.processor.processor_type != 1 ||
                ELF32_ST_BIND(imported.st_info) != STB_GLOBAL || ELF32_ST_TYPE(imported.st_info) != STT_NOTYPE ||
                imported.st_size != 0 || image.loads.empty() || imported.st_value != image.loads.front().p_vaddr)
                throw std::invalid_argument("kernel text linker symbol differs from processor startup contract");
            break;
        case AbsoluteSymbolOwner::FirmwareExport: {
            const auto provider = firmware.symbols.find(name);
            if (provider == firmware.symbols.end() || provider->second.st_shndx == SHN_UNDEF ||
                provider->second.st_value != imported.st_value || provider->second.st_size != imported.st_size ||
                ELF32_ST_TYPE(provider->second.st_info) != ELF32_ST_TYPE(imported.st_info))
                throw std::invalid_argument("dispatch absolute import lacks a matching defined firmware provider");
            validate_firmware_symbol_storage(firmware, provider->second);
            break;
        }
        }
    }
}
size_t firmware_index(const DeploymentConfiguration& config, HalProcessorIdentifier processor) {
    auto processors = config.firmware_processors();
    auto found = std::find(processors.begin(), processors.end(), processor);
    if (found == processors.end()) throw std::invalid_argument("unknown published firmware processor");
    return std::distance(processors.begin(), found);
}
size_t dispatch_index(const DeploymentConfiguration& config, uint32_t node, HalProcessorIdentifier processor) {
    size_t offset = 0;
    for (const auto& value : config.dispatch_nodes()) {
        const auto processors = dispatch_processors(value);
        if (value.node_id == node) {
            const auto found = std::find(processors.begin(), processors.end(), processor);
            if (found == processors.end()) throw std::invalid_argument("unknown published dispatch processor");
            return offset + std::distance(processors.begin(), found);
        }
        offset += processors.size();
    }
    throw std::invalid_argument("unknown published dispatch node");
}
OwnedImage admit_image(const DeploymentConfiguration& config, const PublishedImageInput& input) {
    // Copy before parsing. ElfFile and memory perform the actual existing ELF
    // structural admission/packing without opening paths or creating a context.
    OwnedImage image{input.kind, input.processor, input.dispatch_node,
        std::vector<std::byte>(input.bytes.begin(), input.bytes.end()), {}, {}, {}, {}};
    const auto expected = config.image_record(input.kind, input.processor, input.dispatch_node);
    ll_api::ElfFile elf;
    elf.ReadImage(image.bytes, "published deployment image");
    const auto actual = elf.GetMetadataSection(PublishedImageSection);
    if (!std::ranges::equal(actual, expected))
        throw std::invalid_argument("image publication record differs from complete deployment configuration");
    Elf32_Ehdr header{};
    if (image.bytes.size() < sizeof(header)) throw std::invalid_argument("truncated published image");
    std::memcpy(&header, image.bytes.data(), sizeof(header));
    if (header.e_ident[EI_CLASS] != ELFCLASS32 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
        header.e_machine != EM_RISCV || header.e_type != ET_EXEC || (header.e_flags & ~uint32_t(EF_RISCV_RVC)) ||
        elf.GetSegments().empty() || header.e_entry != elf.GetSegments().front().address)
        throw std::invalid_argument("published image requires a complete ELF32 RISC-V soft-float executable entry");
    image.symbols = read_symbols(elf);
    auto start = image.symbols.find("_start");
    if (start == image.symbols.end() || start->second.st_value != header.e_entry ||
        start->second.st_shndx == SHN_UNDEF || start->second.st_shndx == SHN_ABS)
        throw std::invalid_argument("published image lacks its executable startup symbol");
    image.loads = validate_startup_and_segments(image.bytes, header, start->second);
    const auto& storage = deployment_detail::Access::configuration(config);
    const auto& processor_layout = storage.layouts[firmware_index(config, input.processor)];
    const auto& layout = processor_layout.layout;
    auto loading = ll_api::memory::Loading::DISCRETE;
    if (input.kind == PublishedImageKind::Firmware) {
        image.firmware_layout = validate_firmware_regions(image, processor_layout.regions);
        if (header.e_entry != layout.fw_base_addr)
            throw std::invalid_argument("firmware entry differs from selected HAL processor");
        if (input.processor.core_type != HalProgrammableCoreType::TENSIX &&
            !elf.GetMetadataSection(TT_NATIVE_IMAGE_SECTION).empty())
            throw std::invalid_argument("non-Tensix firmware cannot carry a Tensix native support record");
    } else {
        // SDK dispatch has its own publication/startup contract. It is never a
        // TReX numerical native kernel and cannot borrow its support record.
        if (!elf.GetMetadataSection(TT_NATIVE_IMAGE_SECTION).empty())
            throw std::invalid_argument("vendor dispatch cannot carry a numerical native support record");
        loading = layout.memory_load;
    }
    image.memory = ll_api::memory(image.bytes, loading, "published deployment image");
    return image;
}
} // namespace
struct PublishedDeployment::Impl {
    DeploymentConfiguration config;
    FirmwareBundle tensix;
    std::vector<OwnedImage> firmware;
    std::vector<OwnedImage> dispatch;
    Impl(DeploymentConfiguration c, FirmwareBundle fw) : config(std::move(c)), tensix(std::move(fw)) {}
};
PublishedDeployment::PublishedDeployment(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
PublishedDeployment PublishedDeployment::admit(const DeploymentConfiguration& config, std::span<const PublishedImageInput> input) {
    const auto firmware_count = config.firmware_processors().size();
    size_t dispatch_count = 0;
    for (const auto& node : config.dispatch_nodes()) dispatch_count += dispatch_processors(node).size();
    if (input.size() != firmware_count + dispatch_count)
        throw std::invalid_argument("published deployment requires all firmware and dispatch images");
    std::vector<std::optional<OwnedImage>> firmware(firmware_count), dispatch(dispatch_count);
    for (const auto& value : input) {
        const bool is_firmware = value.kind == PublishedImageKind::Firmware;
        // image_record validates kind, processor, node association before any
        // publication slot is touched; arbitrary enum values never mean dispatch.
        (void)config.image_record(value.kind, value.processor, value.dispatch_node);
        auto index = is_firmware ? firmware_index(config, value.processor) : dispatch_index(config, value.dispatch_node, value.processor);
        auto& slot = is_firmware ? firmware[index] : dispatch[index];
        if (slot) throw std::invalid_argument("duplicate published image role or node");
        slot = admit_image(config, value);
    }
    std::array<std::span<const std::byte>, TT_NATIVE_ROLE_COUNT> native;
    for (const auto& image : firmware) {
        if (!image) throw std::invalid_argument("missing published firmware processor");
        if (image->processor.core_type == HalProgrammableCoreType::TENSIX) {
            const auto role = image->processor.processor_class == HalProcessorClassType::DM
                ? image->processor.processor_type : 2 + image->processor.processor_type;
            if (role < 0 || role >= TT_NATIVE_ROLE_COUNT) throw std::invalid_argument("unsupported Tensix processor");
            native[role] = image->bytes;
        }
    }
    auto tensix = FirmwareBundle::from_images(native, "published Tensix firmware");
    const auto expected_profile = config.profiler_mode() == DeviceProfilerMode::Program
        ? TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM : TT_NATIVE_PROFILE_ABSENT;
    if (tensix.contract(TensixKernelRole::Brisc).profile != expected_profile)
        throw std::invalid_argument("firmware observation contract differs from deployment profile");
    for (const auto& image : dispatch) {
        if (!image) throw std::invalid_argument("missing published dispatch image");
        const auto processor_index = firmware_index(config, image->processor);
        const auto& provider = *firmware[processor_index];
        validate_dispatch_regions(*image, provider, deployment_detail::Access::configuration(config).layouts[processor_index].regions);
        validate_absolute_symbols(*image, provider);
    }
    auto owner = std::make_shared<Impl>(config, std::move(tensix));
    for (auto& image : firmware) owner->firmware.push_back(std::move(*image));
    for (auto& image : dispatch) owner->dispatch.push_back(std::move(*image));
    return PublishedDeployment(std::move(owner));
}
const DeploymentConfiguration& PublishedDeployment::configuration() const { return impl_->config; }
const FirmwareBundle& PublishedDeployment::tensix_firmware() const { return impl_->tensix; }
std::span<const std::byte> PublishedDeployment::firmware_image(HalProcessorIdentifier processor) const {
    return impl_->firmware[firmware_index(impl_->config, processor)].bytes;
}
std::span<const std::byte> PublishedDeployment::dispatch_image(uint32_t node, HalProcessorIdentifier processor) const {
    return impl_->dispatch[dispatch_index(impl_->config, node, processor)].bytes;
}
void PublishedDeployment::validate_configuration(const DeploymentConfiguration& other) const {
    if (!impl_->config.matches(other)) throw std::invalid_argument("actual device deployment configuration differs from publication");
}
const ll_api::memory& deployment_detail::Access::firmware(const PublishedDeployment& deployment, HalProcessorIdentifier processor) {
    return deployment.impl_->firmware[firmware_index(deployment.impl_->config, processor)].memory;
}
const ll_api::memory& deployment_detail::Access::dispatch(
    const PublishedDeployment& deployment, uint32_t node, HalProcessorIdentifier processor) {
    return deployment.impl_->dispatch[dispatch_index(deployment.impl_->config, node, processor)].memory;
}
} // namespace tt::tt_metal::experimental
