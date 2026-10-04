// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <gtest/gtest.h>
#include <reflect>
#include <elf.h>
#include <cstring>
#include <algorithm>
#include <limits>
#include "tt-metalium/experimental/published_deployment.hpp"
#include "tt-metalium/experimental/context/metal_env.hpp"
#include "impl/experimental/published_deployment/configuration.hpp"
#include "impl/experimental/published_deployment/dispatch_plan.hpp"
#include "impl/experimental/published_deployment/dispatch_program_adapter.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/context/metal_context.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::experimental::test {
template<class T> void initialize_config(T& object) {
    // Explicit zero values for unused dispatch slots, including inactive fabric
    // slots, are part of the normalized input. Expectations never use this helper.
    if constexpr (requires { typename T::value_type; object.has_value(); }) {
        object = typename T::value_type{};
    } else {
        reflect::for_each([&](auto i) { initialize_config(reflect::get<i>(object)); }, object);
    }
}

class PublishedDeploymentTest : public ::testing::Test {
protected:
    llrt::RunTimeOptions options{llrt::RunTimeOptions::ExplicitBuildOptions{.root_dir = "/tmp"}};
    Hal hal{tt::ARCH::BLACKHOLE, false, true, 0, false, false, true, true};
    DeviceConfiguration device{
        .arch = tt::ARCH::BLACKHOLE, .num_dram_banks = 8, .num_l1_banks = 120,
        .pcie_core = {19, 24}, .harvesting_mask = 0,
        .dispatch_core_type = DispatchCoreType::WORKER, .resolved_dispatch_core_type = tt::CoreType::WORKER,
        .dispatch_core_axis = DispatchCoreAxis::ROW, .coordinate_virtualization_enabled = true,
        .dispatch_message_addr = 0xffb30c04, .max_cbs = 64, .num_hw_cqs = 1,
        .routing_fw_enabled = false, .profiler_dram_bank_size_per_risc_bytes = 0};
    DispatchProgramInputs program{};

    PublishedDeploymentTest() {
        options.set_enable_2_erisc_mode(true);
        const auto placement = [](uint32_t x, uint32_t y) -> DispatchCorePlacement {
            return {{x, y}, {x + 16, y + 16}, {{{x + 16, y + 16}, {x + 16, y + 16}}}};
        };
        auto& t = program.topology;
        t.arch = device.arch; t.device = 0; t.mmio = true;
        t.deployment_devices = {0}; t.serviced_devices = {0};
        t.core_type = CoreType::WORKER; t.num_hw_cqs = 1; t.subordinate_enabled = true;
        t.nodes = {
            {7, 0, 0, 0, PREFETCH_HD, {}, {8, 9}, {NOC_0, NOC_0, NOC_0}, placement(0, 0)},
            {8, 0, 0, 0, DISPATCH_HD, {7}, {9}, {NOC_0, NOC_1, NOC_0}, placement(1, 0)},
            {9, 0, 0, 0, DISPATCH_S, {7}, {8}, {NOC_1, NOC_1, NOC_1}, placement(1, 0)}};
        t.completion_writer = placement(1, 0); t.unused_core = placement(0, 0);
        program.workers = {120, {0x10111213, 0x20212223}, 0, 0};
        program.observation = {false, false, options.watcher_dispatch_disabled(), false,
            options.get_dispatch_telemetry_disabled(), false};
        DispatchMemMap memory{CoreType::WORKER, 1, hal, false, {false, 1}, options};
        program.hal = capture_dispatch_hal(hal);
        program.memory = capture_dispatch_memory(memory, 0);
        program.queue.inputs = {HostQueueBacking{0, 0x40000000, false, false, std::nullopt}, 1,
            memory.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED),
            hal.get_alignment(HalMemType::HOST), 0x1000};
        program.queue.plan = plan_system_memory_queues(program.queue.inputs);
    }
    DispatchProgramPlan plan() const { return plan_dispatch_program(program); }
    DeploymentConfiguration configuration() const {
        return DeploymentConfiguration::from_sdk(JitDeviceConfig{device, &hal}, options, plan());
    }
    void queue_size(uint32_t bytes) {
        std::get<HostQueueBacking>(program.queue.inputs.backing).cq_size_override = bytes;
        program.queue.plan = plan_system_memory_queues(program.queue.inputs);
    }
};

struct PublishedElf {
    std::vector<std::byte> bytes;
    size_t publication_offset = 0x300;
    size_t sections_offset;
    template<class T> void put(size_t offset, const T& value) { std::memcpy(bytes.data() + offset, &value, sizeof(value)); }
    PublishedElf(uint32_t entry, std::span<const std::byte> publication, std::optional<uint8_t> native_role,
                 bool dispatch = false, uint32_t data_address = 0x2000, uint32_t shared_address = 0x2000,
                 uint32_t firmware_text_end = 0, uint32_t firmware_data_end = 0) {
        sections_offset = (publication_offset + publication.size() + 15) & ~size_t{15};
        bytes.resize(sections_offset + 9 * sizeof(Elf32_Shdr));
        Elf32_Ehdr h{};
        std::memcpy(h.e_ident, ELFMAG, SELFMAG);
        h.e_ident[EI_CLASS] = ELFCLASS32; h.e_ident[EI_DATA] = ELFDATA2LSB; h.e_ident[EI_VERSION] = EV_CURRENT;
        h.e_type = ET_EXEC; h.e_machine = EM_RISCV; h.e_version = EV_CURRENT; h.e_entry = entry;
        h.e_ehsize = sizeof(h); h.e_phoff = sizeof(h); h.e_phnum = 2; h.e_phentsize = sizeof(Elf32_Phdr);
        h.e_shoff = sections_offset; h.e_shnum = 9; h.e_shentsize = sizeof(Elf32_Shdr); h.e_shstrndx = 4;
        put(0, h);
        put(sizeof(h), Elf32_Phdr{PT_LOAD, 0x80, entry, entry, 4, 4, PF_R | PF_X, 4});
        put(sizeof(h) + sizeof(Elf32_Phdr), Elf32_Phdr{PT_LOAD, 0x84, data_address, dispatch ? entry + 4 : data_address, 4, 4, PF_R | PF_W, 4});
        put(0x80, uint32_t{0x00008067}); put(0x84, uint32_t{0x12345678});
        if (native_role) put(0x90, tt_native_image_record{TT_NATIVE_IMAGE_MAGIC, TT_NATIVE_IMAGE_ABI_VERSION,
            TT_NATIVE_IMAGE_FIRMWARE, *native_role, TT_NATIVE_ARCH_BLACKHOLE, TT_NATIVE_LOADING_DISCRETE,
            TT_NATIVE_PROFILE_ABSENT, TT_NATIVE_PRINT_TENSIX_SHARED_BUFFER, 1, TT_NATIVE_WORKER_STREAM_SDK, 0});
        constexpr char names[] = "\0.text\0.data\0.tt_native_image\0.shstrtab\0.symtab\0.strtab\0.rela.text\0.tt_published_image\0";
        std::memcpy(bytes.data() + 0x200, names, sizeof(names));
        std::memcpy(bytes.data() + publication_offset, publication.data(), publication.size());
        constexpr char symbols[] = "\0_start\0shared_state\0__fw_export_text_end\0__fw_export_ldm_end\0";
        std::memcpy(bytes.data() + 0x190, symbols, sizeof(symbols));
        put(0x140 + sizeof(Elf32_Sym), Elf32_Sym{1, entry, 4, ELF32_ST_INFO(STB_GLOBAL, STT_FUNC), 0, 1});
        put(0x140 + 2 * sizeof(Elf32_Sym), Elf32_Sym{8, shared_address, 4, ELF32_ST_INFO(STB_GLOBAL, STT_OBJECT), 0,
            static_cast<Elf32_Half>(dispatch ? SHN_ABS : 2)});
        if (!dispatch) {
            const auto symbol_names = std::string_view(symbols, sizeof(symbols));
            put(0x140 + 3 * sizeof(Elf32_Sym), Elf32_Sym{
                static_cast<uint32_t>(symbol_names.find("__fw_export_text_end")), firmware_text_end, 0,
                ELF32_ST_INFO(STB_GLOBAL, STT_NOTYPE), 0, SHN_ABS});
            put(0x140 + 4 * sizeof(Elf32_Sym), Elf32_Sym{
                static_cast<uint32_t>(symbol_names.find("__fw_export_ldm_end")), firmware_data_end, 0,
                ELF32_ST_INFO(STB_GLOBAL, STT_NOTYPE), 0, SHN_ABS});
        }
        auto section = [&](unsigned index, std::string_view name, uint32_t type, uint32_t flags,
                           uint32_t addr, uint32_t offset, uint32_t size, uint32_t link = 0,
                           uint32_t info = 0, uint32_t align = 4, uint32_t entsize = 0) {
            auto pos = std::string_view(names, sizeof(names)).find(name);
            put(sections_offset + index * sizeof(Elf32_Shdr),
                Elf32_Shdr{static_cast<uint32_t>(pos), type, flags, addr, offset, size, link, info, align, entsize});
        };
        section(1, ".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, entry, 0x80, 4);
        section(2, ".data", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, data_address, 0x84, 4);
        section(3, ".tt_native_image", SHT_PROGBITS, 0, 0, 0x90, native_role ? 16 : 0);
        section(4, ".shstrtab", SHT_STRTAB, 0, 0, 0x200, sizeof(names), 0, 0, 1);
        section(5, ".symtab", SHT_SYMTAB, 0, 0, 0x140, (dispatch ? 3 : 5) * sizeof(Elf32_Sym), 6, 1, 4, sizeof(Elf32_Sym));
        section(6, ".strtab", SHT_STRTAB, 0, 0, 0x190, sizeof(symbols), 0, 0, 1);
        section(7, ".rela.text", SHT_RELA, 0, 0, 0x1c0, 0, 5, 1, 4, sizeof(Elf32_Rela));
        section(8, ".tt_published_image", SHT_PROGBITS, 0, 0, publication_offset, publication.size());
    }
};
struct PublicationInput {
    std::vector<PublishedElf> images;
    std::vector<PublishedImageInput> input;
    PublicationInput(const DeploymentConfiguration& config, const Hal& hal) {
        size_t dispatch_count = 0;
        for (const auto& node : config.dispatch_nodes()) dispatch_count += dispatch_processors(node).size();
        images.reserve(config.firmware_processors().size() + dispatch_count);
        for (const auto processor : config.firmware_processors()) {
            auto ci = hal.get_programmable_core_type_index(processor.core_type);
            auto entry = hal.get_jit_build_config(ci, static_cast<uint32_t>(processor.processor_class), processor.processor_type).fw_base_addr;
            std::optional<uint8_t> role;
            if (processor.core_type == HalProgrammableCoreType::TENSIX)
                role = processor.processor_class == HalProcessorClassType::DM ? processor.processor_type : 2 + processor.processor_type;
            const auto regions = hal.get_processor_image_regions(processor).value();
            const auto text_end = (entry + 4 + regions.firmware_kernel_pad + 15) & ~uint32_t{15};
            const auto data_end = (regions.local_data.base + 4 + 15) & ~uint64_t{15};
            images.emplace_back(entry, config.image_record(PublishedImageKind::Firmware, processor), role,
                false, regions.local_data.base, regions.local_data.base, text_end, data_end);
            input.push_back({PublishedImageKind::Firmware, processor, NoDispatchNode, images.back().bytes});
        }
        for (const auto& node : config.dispatch_nodes()) {
            for (auto processor : dispatch_processors(node)) {
                const auto regions = hal.get_processor_image_regions(processor).value();
                const auto text_end = (regions.firmware_text.base + 4 + regions.firmware_kernel_pad + 15) & ~uint64_t{15};
                const auto data_end = (regions.local_data.base + 4 + 15) & ~uint64_t{15};
                images.emplace_back(text_end, config.image_record(PublishedImageKind::Dispatch, processor, node.node_id),
                    std::nullopt, true, data_end, regions.local_data.base);
                input.push_back({PublishedImageKind::Dispatch, processor, node.node_id, images.back().bytes});
            }
        }
    }
};

} // namespace tt::tt_metal::experimental::test
