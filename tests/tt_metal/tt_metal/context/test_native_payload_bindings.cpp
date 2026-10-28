// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <elf.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/experimental/native_kernel.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include "../api/metal2_host_api/test_helpers.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/experimental/native_kernel/native_kernel.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_execution.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {
namespace {
// A complete immutable ELF fixture, parsed by the production image reader.
// Declared firmware bytes are never registered as an actual successful boot.
struct PayloadElf {
    static constexpr size_t sections_offset = 0x200;
    std::vector<std::byte> bytes = std::vector<std::byte>(sections_offset + 8 * sizeof(Elf32_Shdr));

    template <class T>
    void put(size_t offset, const T& value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    PayloadElf(uint8_t kind, uint8_t role) {
        Elf32_Ehdr header{};
        std::memcpy(header.e_ident, ELFMAG, SELFMAG);
        header.e_ident[EI_CLASS] = ELFCLASS32;
        header.e_ident[EI_DATA] = ELFDATA2LSB;
        header.e_ident[EI_VERSION] = EV_CURRENT;
        header.e_type = ET_EXEC;
        header.e_machine = EM_RISCV;
        header.e_version = EV_CURRENT;
        header.e_entry = 0x1000;
        header.e_phoff = sizeof(header);
        header.e_phnum = 2;
        header.e_phentsize = sizeof(Elf32_Phdr);
        header.e_ehsize = sizeof(header);
        header.e_shoff = sections_offset;
        header.e_shnum = 8;
        header.e_shentsize = sizeof(Elf32_Shdr);
        header.e_shstrndx = 4;
        put(0, header);
        put(sizeof(header), Elf32_Phdr{PT_LOAD, 0x80, 0x1000, 0x1000, 4, 4, PF_R | PF_X, 4});
        put(sizeof(header) + sizeof(Elf32_Phdr),
            Elf32_Phdr{PT_LOAD, 0x84, 0x2000, 0x1004, 4, 4, PF_R | PF_W, 4});
        put(0x80, uint32_t{0x00008067});
        put(0x84, uint32_t{0x12345678});
        put(0x90, tt_native_image_record{
                      TT_NATIVE_IMAGE_MAGIC, TT_NATIVE_IMAGE_ABI_VERSION, kind, role, TT_NATIVE_ARCH_BLACKHOLE,
                      TT_NATIVE_LOADING_DISCRETE, TT_NATIVE_PROFILE_ABSENT, TT_NATIVE_PRINT_ABSENT, 1,
                      TT_NATIVE_WORKER_STREAM_SDK, 0});
        constexpr char names[] = "\0.text\0.data\0.tt_native_image\0.shstrtab\0.symtab\0.strtab\0.rela.text\0";
        std::memcpy(bytes.data() + 0x100, names, sizeof(names));
        put(sections_offset + sizeof(Elf32_Shdr),
            Elf32_Shdr{1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x1000, 0x80, 4, 0, 0, 4, 0});
        put(sections_offset + 2 * sizeof(Elf32_Shdr),
            Elf32_Shdr{7, SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 0x2000, 0x84, 4, 0, 0, 4, 0});
        put(sections_offset + 3 * sizeof(Elf32_Shdr),
            Elf32_Shdr{13, SHT_PROGBITS, 0, 0, 0x90, 16, 0, 0, 4, 0});
        put(sections_offset + 4 * sizeof(Elf32_Shdr),
            Elf32_Shdr{30, SHT_STRTAB, 0, 0, 0x100, sizeof(names), 0, 0, 1, 0});
        put(sections_offset + 5 * sizeof(Elf32_Shdr),
            Elf32_Shdr{40, SHT_SYMTAB, 0, 0, 0x150, sizeof(Elf32_Sym), 6, 1, 4, sizeof(Elf32_Sym)});
        put(sections_offset + 6 * sizeof(Elf32_Shdr), Elf32_Shdr{48, SHT_STRTAB, 0, 0, 0x170, 1, 0, 0, 1, 0});
        put(sections_offset + 7 * sizeof(Elf32_Shdr),
            Elf32_Shdr{56, SHT_RELA, 0, 0, 0x180, 0, 5, 1, 4, sizeof(Elf32_Rela)});
    }

    void metadata(std::span<const std::byte> metadata, std::string_view section_name) {
        Elf32_Ehdr header{};
        std::memcpy(&header, bytes.data(), sizeof(header));
        std::array<Elf32_Shdr, 8> sections{};
        std::memcpy(sections.data(), bytes.data() + header.e_shoff, sizeof(sections));
        const auto names_section = sections[header.e_shstrndx];
        std::string names(reinterpret_cast<const char*>(bytes.data() + names_section.sh_offset), names_section.sh_size);
        const auto name_offset = names.size();
        names.append(section_name);
        names.push_back('\0');
        const size_t names_offset = std::max<size_t>(0x500, (0x400 + metadata.size() + 0xff) & ~size_t{0xff});
        const size_t table_offset = std::max<size_t>(0x600, (names_offset + names.size() + 0xff) & ~size_t{0xff});
        bytes.resize(table_offset + 9 * sizeof(Elf32_Shdr));
        std::memcpy(bytes.data() + 0x400, metadata.data(), metadata.size());
        std::memcpy(bytes.data() + names_offset, names.data(), names.size());
        sections[header.e_shstrndx].sh_offset = names_offset;
        sections[header.e_shstrndx].sh_size = names.size();
        std::memcpy(bytes.data() + table_offset, sections.data(), sizeof(sections));
        put(table_offset + 8 * sizeof(Elf32_Shdr),
            Elf32_Shdr{static_cast<uint32_t>(name_offset), SHT_PROGBITS, 0, 0, 0x400,
                       static_cast<uint32_t>(metadata.size()), 0, 0, 4, 0});
        header.e_shoff = table_offset;
        header.e_shnum = 9;
        put(0, header);
    }
};

experimental::KernelElfImage payload_image(uint8_t addressing) {
    std::array<PayloadElf, TT_NATIVE_ROLE_COUNT> firmware{{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}}};
    for (uint8_t role = 0; role != firmware.size(); ++role) {
        const tt_niu_handoff_record record{
            TT_NIU_HANDOFF_MAGIC, TT_NIU_HANDOFF_VERSION, TT_NIU_HANDOFF_RECORD_BYTES, role,
            TT_NATIVE_ARCH_BLACKHOLE, 2, 16, TT_NIU_HANDOFF_LAYOUT_VERSION, TT_NIU_HANDOFF_PROTOCOL,
            0xa040, TT_NIU_HANDOFF_L1_BYTES, 0x8520, TT_NIU_HANDOFF_COUNTER_BYTES,
            TT_NIU_HANDOFF_COUNTER_CONTRACT, TT_NIU_HANDOFF_PROCESSOR_MASK, TT_NIU_HANDOFF_REQUEST_OFFSET,
            TT_NIU_HANDOFF_REQUEST_BYTES, TT_NIU_HANDOFF_RECEIPT_OFFSET, TT_NIU_HANDOFF_RECEIPT_BYTES,
            TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1};
        firmware[role].metadata(std::as_bytes(std::span(&record, 1)), ".tt_niu_handoff");
    }
    const auto bundle = experimental::FirmwareBundle::from_images(
        {firmware[0].bytes, firmware[1].bytes, firmware[2].bytes, firmware[3].bytes, firmware[4].bytes});
    const tt_native_noc_record header{
        TT_NATIVE_NOC_MAGIC, TT_NATIVE_NOC_ABI_VERSION,
        sizeof(tt_native_noc_record) + sizeof(tt_native_noc_window), TT_NATIVE_ROLE_BRISC,
        TT_NATIVE_ARCH_BLACKHOLE, 0, TT_NIU_TRAFFIC_TRACKED_ISSUING_NIU_V1, 0, 0, 0,
        TT_NATIVE_NOC_PAYLOAD_WORKER_L1_NONPOSTED, 1, TT_NATIVE_NOC_WINDOW_BYTES};
    const tt_native_noc_window window{1, 3, 0, 64, 64, TT_NATIVE_NOC_WINDOW_READ, addressing, 1, 0};
    std::array<std::byte, sizeof(header) + sizeof(window)> metadata;
    std::memcpy(metadata.data(), &header, sizeof(header));
    std::memcpy(metadata.data() + sizeof(header), &window, sizeof(window));
    PayloadElf kernel(TT_NATIVE_IMAGE_KERNEL, TT_NATIVE_ROLE_BRISC);
    kernel.metadata(metadata, TT_NATIVE_NOC_SECTION);
    return experimental::KernelElfImage::from_bytes(kernel.bytes, bundle);
}

class NativePayloadBindingsTest : public ::testing::Test {
protected:
    void SetUp() override {
        slow.emplace();
        std::optional<std::string> previous;
        if (const auto* value = std::getenv("TT_METAL_ALLOCATOR_MODE_HYBRID")) previous = value;
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", 1);
        MetalEnvDescriptor descriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value());
        descriptor.set_device_profiler_mode(DeviceProfilerMode::Disabled);
        environment = std::make_unique<MetalEnv>(std::move(descriptor));
        context_id = MetalContext::create_instance(*environment);
        context = &MetalContext::instance(*context_id);
        const DispatchCoreConfig dispatch{DispatchCoreType::WORKER, DispatchCoreAxis::COL};
        context->initialize(dispatch, 1, {}, DEFAULT_WORKER_L1_SIZE, true);
        mesh = distributed::MeshDeviceImpl::create(
            *context_id, distributed::MeshDeviceConfig(distributed::MeshShape{1, 2}), DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE, 1, dispatch, {}, DEFAULT_WORKER_L1_SIZE);
        if (previous) setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", previous->c_str(), 1);
        else unsetenv("TT_METAL_ALLOCATOR_MODE_HYBRID");
        ASSERT_EQ(context->get_cluster().get_target_device_type(), tt::TargetDevice::Mock);
        ASSERT_EQ(mesh->get_devices().size(), 2u);
        for (auto* device : mesh->get_devices()) ASSERT_FALSE(context->native_firmware(device->id()));

        BufferShardingArgs sharding(
            ShardSpecBuffer(CoreRangeSet(CoreRange({0, 0}, {1, 0})), {1, 1}, ShardOrientation::ROW_MAJOR,
                            {1, 1}, {2, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED);
        experimental::per_core_allocation::set_per_core_allocation(sharding, true);
        distributed::DeviceLocalBufferConfig local;
        local.page_size = 1024;
        local.buffer_type = BufferType::L1;
        local.sharding_args = sharding;
        allocation = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{2048}, local, mesh.get());
        make_workload(TT_NATIVE_NOC_WINDOW_RUNTIME_ENDPOINT);
    }

    void TearDown() override {
        kernel.reset();
        work.reset();
        allocation.reset();
        if (mesh) {
            mesh->close();
            mesh.reset();
        }
        if (context_id) MetalContext::destroy_instance(false, *context_id);
        environment.reset();
        slow.reset();
    }

    void make_workload(uint8_t addressing) {
        Program program(std::make_shared<detail::ProgramImpl>(*context_id));
        const auto handle = experimental::CreateKernelFromElf(
            program, CoreRangeSet(CoreRange({0, 0}, {1, 0})),
            experimental::NativeDataMovementConfig{payload_image(addressing), NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC});
        kernel = program.impl().get_kernel(handle);
        work = std::make_unique<distributed::MeshWorkload>();
        work->add_program(
            distributed::MeshCoordinateRange(distributed::MeshCoordinate{0, 0}, distributed::MeshCoordinate{0, 1}),
            std::move(program));
    }

    Program& program() { return work->get_programs().begin()->second; }

    std::array<uint32_t, 3> arguments(const distributed::MeshCoordinate& coordinate, CoreCoord home) {
        auto* device = mesh->impl().get_device(coordinate);
        const auto endpoint = device->virtual_noc0_coordinate(0, device->worker_core_from_logical_core(home));
        const auto base = experimental::per_core_allocation::get_per_core_address(
            *allocation->get_device_buffer(coordinate), home);
        return {endpoint.x, endpoint.y, static_cast<uint32_t>(base)};
    }

    template <class Execution>
    auto requests() {
        std::vector<typename Execution::PayloadBindingRequest> result;
        for (uint32_t chip = 0; chip != 2; ++chip) {
            const distributed::MeshCoordinate coordinate{0, chip};
            auto* device = mesh->impl().get_device(coordinate);
            for (uint32_t x = 0; x != 2; ++x) {
                const CoreCoord issuer{x, 0};
                result.push_back({
                    .program = &program(), .kernel = kernel, .role = experimental::TensixKernelRole::Brisc,
                    .window = 0, .coordinate = coordinate, .issuer = issuer,
                    .arguments = arguments(coordinate, issuer),
                    .allocation = device->allocator_impl()->pin_l1(allocation, coordinate, issuer, BufferRegion{0, 64})});
            }
        }
        return result;
    }

    enum class Case {
        HostPreparation, BatchContract, MissingCoverage, DuplicateCoverage, ForeignProgram, ForeignKernel,
        WrongRole, WrongWindow, WrongPlacement, WrongIssuer, WrongDevice, WrongEndpoint,
        ChildBounds, BaseAlignment, IssuerLocal, IssuerLocalTuple, MissingBootConsumesBatch, ProgramChanged,
        WorkloadChanged
    };

    template <class Execution>
    void exercise(Case test) {
        if constexpr (requires { typename Execution::PayloadBindingRequest; typename Execution::PayloadBindings; }) {
            if (test == Case::IssuerLocal || test == Case::IssuerLocalTuple)
                make_workload(TT_NATIVE_NOC_WINDOW_ISSUER_LOCAL);
            auto rows = requests<Execution>();
            const std::array workloads{work.get()};
            if (test == Case::HostPreparation || test == Case::BatchContract || test == Case::IssuerLocalTuple) {
                EXPECT_TRUE(kernel->cores_with_runtime_args().empty());
                if (test == Case::IssuerLocalTuple) {
                    const auto remote = arguments(distributed::MeshCoordinate{0, 0}, {1, 0});
                    rows.front().arguments[0] = remote[0];
                    rows.front().arguments[1] = remote[1];
                }
                {
                    auto bindings = Execution::prepare_payload_bindings(*mesh, workloads, std::move(rows));
                    EXPECT_FALSE(std::is_default_constructible_v<typename Execution::PayloadBindings>);
                    EXPECT_FALSE(std::is_aggregate_v<typename Execution::PayloadBindings>);
                    EXPECT_FALSE(std::is_copy_constructible_v<typename Execution::PayloadBindings>);
                    EXPECT_TRUE(std::is_move_constructible_v<typename Execution::PayloadBindings>);
                    EXPECT_THROW(allocation->deallocate(), std::runtime_error);
                    for (auto* device : mesh->get_devices()) EXPECT_FALSE(context->native_firmware(device->id()));
                }
                EXPECT_NO_THROW(allocation->deallocate());
                return;
            }
            if (test == Case::MissingBootConsumesBatch || test == Case::ProgramChanged || test == Case::WorkloadChanged) {
                auto bindings = Execution::prepare_payload_bindings(*mesh, workloads, std::move(rows));
                if (test == Case::ProgramChanged) {
                    experimental::CreateKernelFromElf(
                        program(), CoreRangeSet(CoreRange({2, 0}, {2, 0})),
                        experimental::NativeDataMovementConfig{
                            payload_image(TT_NATIVE_NOC_WINDOW_RUNTIME_ENDPOINT), NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC});
                } else if (test == Case::WorkloadChanged) {
                    work->get_programs().clear();
                }
                try {
                    auto invocation = Execution::acquire_managed(*mesh, workloads, std::move(bindings));
                    FAIL() << "Host preparation must not manufacture a successful actual boot";
                } catch (const std::exception& error) {
                    if (test == Case::MissingBootConsumesBatch)
                        EXPECT_NE(std::string(error.what()).find("successful actual SDK boot"), std::string::npos);
                    else
                        EXPECT_EQ(std::string(error.what()).find("successful actual SDK boot"), std::string::npos)
                            << "Changed original objects must be rejected before consulting boot: " << error.what();
                }
                try {
                    auto invocation = Execution::acquire_managed(*mesh, workloads, std::move(bindings));
                    FAIL() << "A payload binding batch must be consumed even when acquisition fails";
                } catch (const std::exception& error) {
                    EXPECT_NE(std::string(error.what()).find("consum"), std::string::npos) << error.what();
                }
                EXPECT_NO_THROW(allocation->deallocate());
                return;
            }
            Program foreign(std::make_shared<detail::ProgramImpl>(*context_id));
            switch (test) {
                case Case::MissingCoverage: rows.pop_back(); break;
                case Case::DuplicateCoverage: {
                    auto extra = requests<Execution>();
                    rows.push_back(std::move(extra.front()));
                    break;
                }
                case Case::ForeignProgram: rows.front().program = &foreign; break;
                case Case::ForeignKernel: {
                    const auto handle = experimental::CreateKernelFromElf(
                        foreign, CoreRangeSet(CoreRange({0, 0}, {1, 0})),
                        experimental::NativeDataMovementConfig{
                            payload_image(TT_NATIVE_NOC_WINDOW_RUNTIME_ENDPOINT), NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC});
                    rows.front().kernel = foreign.impl().get_kernel(handle);
                    break;
                }
                case Case::WrongRole: rows.front().role = experimental::TensixKernelRole::Ncrisc; break;
                case Case::WrongWindow: rows.front().window = 1; break;
                case Case::WrongPlacement: rows.front().coordinate = distributed::MeshCoordinate{0, 2}; break;
                case Case::WrongIssuer: rows.front().issuer = CoreCoord{2, 0}; break;
                case Case::WrongDevice: std::swap(rows.front().allocation, rows.back().allocation); break;
                case Case::WrongEndpoint: rows.front().arguments = arguments(distributed::MeshCoordinate{0, 0}, {1, 0}); break;
                case Case::ChildBounds: rows.front().arguments[2] += 64; break;
                case Case::BaseAlignment: {
                    const distributed::MeshCoordinate coordinate{0, 0};
                    rows.front().allocation = mesh->impl().get_device(coordinate)->allocator_impl()->pin_l1(
                        allocation, coordinate, {0, 0}, BufferRegion{0, 128});
                    rows.front().arguments[2] += 1;
                    break;
                }
                case Case::IssuerLocal: {
                    // The remote tuple must not widen an issuer-local window's home.
                    const distributed::MeshCoordinate coordinate{0, 0};
                    rows.front().arguments = arguments(coordinate, {1, 0});
                    rows.front().allocation = mesh->impl().get_device(coordinate)->allocator_impl()->pin_l1(
                        allocation, coordinate, {1, 0}, BufferRegion{0, 64});
                    break;
                }
                default: FAIL() << "Unexpected payload binding test case"; return;
            }
            EXPECT_THROW(Execution::prepare_payload_bindings(*mesh, workloads, std::move(rows)), std::exception);
        } else {
            ADD_FAILURE() << "SDK lacks original-allocation payload preparation and its single-use managed binding batch";
        }
    }

    std::optional<experimental::test_helpers::ScopedSlowDispatchOverride> slow;
    std::unique_ptr<MetalEnv> environment;
    std::optional<ContextId> context_id;
    MetalContext* context = nullptr;
    std::shared_ptr<distributed::MeshDevice> mesh;
    std::shared_ptr<distributed::MeshBuffer> allocation;
    std::unique_ptr<distributed::MeshWorkload> work;
    std::shared_ptr<Kernel> kernel;
};

TEST_F(NativePayloadBindingsTest, OriginalImageAndAllocationDoNotManufactureFirmwareOrUnboundTrafficEligibility) {
    auto* native = dynamic_cast<NativeElfKernel*>(kernel.get());
    ASSERT_NE(native, nullptr);
    ASSERT_EQ(native->native_images().images.front().noc_windows().size(), 1u);
    EXPECT_TRUE(kernel->cores_with_runtime_args().empty());
    EXPECT_THROW(native->validate_managed_traffic(), std::exception);
    const distributed::MeshCoordinate coordinate{0, 0};
    auto pin = mesh->impl().get_device(coordinate)->allocator_impl()->pin_l1(
        allocation, coordinate, {0, 0}, BufferRegion{0, 64});
    EXPECT_TRUE(pin.live());
    for (auto* device : mesh->get_devices()) EXPECT_FALSE(context->native_firmware(device->id()));
}

#define PAYLOAD_BINDING_TEST(name, scenario) \
    TEST_F(NativePayloadBindingsTest, name) { exercise<detail::ProgramExecution>(Case::scenario); }
PAYLOAD_BINDING_TEST(HostPreparationNeedsNeitherBootNorFinalRuntimeArgumentsAndRetainsOriginalPins, HostPreparation)
PAYLOAD_BINDING_TEST(BatchIsOpaqueMoveOnlyAndCannotBeCallerMinted, BatchContract)
PAYLOAD_BINDING_TEST(RejectsMissingActualChipIssuerWindowCoverage, MissingCoverage)
PAYLOAD_BINDING_TEST(RejectsDuplicateActualChipIssuerWindowCoverage, DuplicateCoverage)
PAYLOAD_BINDING_TEST(RejectsForeignProgramOwner, ForeignProgram)
PAYLOAD_BINDING_TEST(RejectsCopiedImageInForeignKernelOwner, ForeignKernel)
PAYLOAD_BINDING_TEST(RejectsRoleAbsentFromOriginalKernelImage, WrongRole)
PAYLOAD_BINDING_TEST(RejectsWindowAbsentFromOriginalKernelImage, WrongWindow)
PAYLOAD_BINDING_TEST(RejectsCoordinateOutsideOriginalPlacement, WrongPlacement)
PAYLOAD_BINDING_TEST(RejectsIssuerOutsideOriginalKernel, WrongIssuer)
PAYLOAD_BINDING_TEST(RejectsPinFromAnotherActualDevice, WrongDevice)
PAYLOAD_BINDING_TEST(RejectsTupleEndpointDifferentFromOriginalPinHome, WrongEndpoint)
PAYLOAD_BINDING_TEST(RejectsRelativeWindowOutsideExactLogicalChild, ChildBounds)
PAYLOAD_BINDING_TEST(RejectsBaseThatViolatesOriginalWindowAlignment, BaseAlignment)
PAYLOAD_BINDING_TEST(IssuerLocalWindowUsesOriginalIssuerAndCannotAdoptTupleHome, IssuerLocal)
PAYLOAD_BINDING_TEST(IssuerLocalWindowAcceptsAnotherTupleEndpointWithoutChangingItsOriginalHome, IssuerLocalTuple)
PAYLOAD_BINDING_TEST(AcquireRequiresActualBootAndConsumesBatchOnFailure, MissingBootConsumesBatch)
PAYLOAD_BINDING_TEST(AcquireRevalidatesOriginalProgramBeforeBoot, ProgramChanged)
PAYLOAD_BINDING_TEST(AcquireRevalidatesCompleteWorkloadBeforeBoot, WorkloadChanged)
#undef PAYLOAD_BINDING_TEST

}  // namespace
}  // namespace tt::tt_metal
