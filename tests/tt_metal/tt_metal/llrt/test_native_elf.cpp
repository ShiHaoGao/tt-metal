// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <elf.h>
#include <array>
#include <cstring>
#include <vector>
#include "tt-metalium/experimental/native_kernel.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/experimental/native_kernel/native_image.hpp"
#include "impl/experimental/native_kernel/native_kernel.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::experimental {
namespace {
// Complete linked RISC-V ELF, with distinct loaded text/data and host-only
// support metadata. The instruction is ret; no external toolchain is needed.
struct NativeElf {
    static constexpr size_t note_offset = 0x90;
    static constexpr size_t sections_offset = 0x200;
    std::vector<std::byte> bytes = std::vector<std::byte>(sections_offset + 8 * sizeof(Elf32_Shdr));
    template<class T> void put(size_t offset, const T& v) { std::memcpy(bytes.data()+offset, &v, sizeof(v)); }
    NativeElf(uint8_t kind, uint8_t role) {
        Elf32_Ehdr h{};
        std::memcpy(h.e_ident, ELFMAG, SELFMAG);
        h.e_ident[EI_CLASS]=ELFCLASS32; h.e_ident[EI_DATA]=ELFDATA2LSB; h.e_ident[EI_VERSION]=EV_CURRENT;
        h.e_type=ET_EXEC; h.e_machine=EM_RISCV; h.e_version=EV_CURRENT; h.e_entry=0x1000;
        h.e_phoff=sizeof(h); h.e_phnum=2; h.e_phentsize=sizeof(Elf32_Phdr); h.e_ehsize=sizeof(h);
        h.e_shoff=sections_offset; h.e_shnum=8; h.e_shentsize=sizeof(Elf32_Shdr); h.e_shstrndx=4;
        put(0,h);
        put(sizeof(h),Elf32_Phdr{PT_LOAD,0x80,0x1000,0x1000,4,4,PF_R|PF_X,4});
        put(sizeof(h)+sizeof(Elf32_Phdr),Elf32_Phdr{PT_LOAD,0x84,0x2000,0x1004,4,4,PF_R|PF_W,4});
        put(0x80,uint32_t{0x00008067}); put(0x84,uint32_t{0x12345678});
        put(note_offset,tt_native_image_record{TT_NATIVE_IMAGE_MAGIC,TT_NATIVE_IMAGE_ABI_VERSION,kind,role,
            TT_NATIVE_ARCH_BLACKHOLE,TT_NATIVE_LOADING_DISCRETE,TT_NATIVE_PROFILE_ABSENT,
            TT_NATIVE_PRINT_ABSENT,1,TT_NATIVE_WORKER_STREAM_SDK,0});
        constexpr char names[]="\0.text\0.data\0.tt_native_image\0.shstrtab\0.symtab\0.strtab\0.rela.text\0";
        std::memcpy(bytes.data()+0x100,names,sizeof(names));
        put(sections_offset+sizeof(Elf32_Shdr),Elf32_Shdr{1,SHT_PROGBITS,SHF_ALLOC|SHF_EXECINSTR,0x1000,0x80,4,0,0,4,0});
        put(sections_offset+2*sizeof(Elf32_Shdr),Elf32_Shdr{7,SHT_PROGBITS,SHF_ALLOC|SHF_WRITE,0x2000,0x84,4,0,0,4,0});
        put(sections_offset+3*sizeof(Elf32_Shdr),Elf32_Shdr{13,SHT_PROGBITS,0,0,note_offset,16,0,0,4,0});
        put(sections_offset+4*sizeof(Elf32_Shdr),Elf32_Shdr{30,SHT_STRTAB,0,0,0x100,sizeof(names),0,0,1,0});
        put(sections_offset+5*sizeof(Elf32_Shdr),Elf32_Shdr{40,SHT_SYMTAB,0,0,0x150,sizeof(Elf32_Sym),6,1,4,sizeof(Elf32_Sym)});
        put(sections_offset+6*sizeof(Elf32_Shdr),Elf32_Shdr{48,SHT_STRTAB,0,0,0x170,1,0,0,1,0});
        put(sections_offset+7*sizeof(Elf32_Shdr),Elf32_Shdr{56,SHT_RELA,0,0,0x180,0,5,1,4,sizeof(Elf32_Rela)});
    }
};

struct FirmwareInput {
    std::array<NativeElf,5> images{{{0,0},{0,1},{0,2},{0,3},{0,4}}};
    auto spans() const {
        return std::array<std::span<const std::byte>,5>{images[0].bytes,images[1].bytes,images[2].bytes,
            images[3].bytes,images[4].bytes};
    }
};

TEST(NativeElf, CopiesFirmwareAndKernelWithoutContext) {
    ASSERT_FALSE(MetalContext::instance_exists());
    FirmwareInput input;
    const auto firmware=FirmwareBundle::from_images(input.spans());
    NativeElf kernel(1,0);
    const auto image=KernelElfImage::from_bytes(kernel.bytes,firmware,"brisc");
    const auto expected=kernel.bytes;
    std::fill(kernel.bytes.begin(),kernel.bytes.end(),std::byte{});
    std::fill(input.images[0].bytes.begin(),input.images[0].bytes.end(),std::byte{});
    EXPECT_TRUE(std::equal(image.image_bytes().begin(),image.image_bytes().end(),expected.begin(),expected.end()));
    EXPECT_EQ(image.role(),TensixKernelRole::Brisc);
    EXPECT_EQ(image.label(),"brisc");
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(NativeElf, RejectsWrongRoleBundleAndLoadedMetadata) {
    FirmwareInput input;
    input.images[4].put(NativeElf::note_offset+7,uint8_t{3});
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()),std::exception);
    input.images[4]=NativeElf(0,4);
    input.images[4].put(NativeElf::sections_offset+3*sizeof(Elf32_Shdr)+8,uint32_t{SHF_ALLOC});
    EXPECT_THROW(FirmwareBundle::from_images(input.spans()),std::exception);
}

TEST(NativeElf, ByteEqualityDoesNotUseLabelsOrClaimedKeys) {
    FirmwareInput a,b;
    auto first=FirmwareBundle::from_images(a.spans(),"one");
    auto same=FirmwareBundle::from_images(b.spans(),"two");
    EXPECT_TRUE(first.matches(same));
    b.images[2].put(0x84,uint32_t{0x98765432});
    auto different=FirmwareBundle::from_images(b.spans(),"one");
    EXPECT_FALSE(first.matches(different));
}

TEST(NativeElf, RejectsUnsupportedOrUnavailableContracts) {
    FirmwareInput input;
    auto firmware=FirmwareBundle::from_images(input.spans());
    NativeElf kernel(1,0);
    kernel.put(NativeElf::note_offset+10,uint8_t{TT_NATIVE_PROFILE_CLASSIC_DRAM_PROGRAM});
    EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes,firmware),std::exception);
    kernel=NativeElf(1,0);
    kernel.put(NativeElf::note_offset+4,uint16_t{2});
    EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes,firmware),std::exception);
    kernel=NativeElf(0,0);
    EXPECT_THROW(KernelElfImage::from_bytes(kernel.bytes,firmware),std::exception);
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
    EXPECT_TRUE(std::equal(image.image_bytes().begin(), image.image_bytes().end(), kernel.bytes.begin(), kernel.bytes.end()));
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
    const NativeComputeConfig config{image(TT_NATIVE_ROLE_TRISC0), image(TT_NATIVE_ROLE_TRISC1),
                                     image(TT_NATIVE_ROLE_TRISC2)};
    NativeElfKernel kernel(context, DEFAULT_CONTEXT_ID, cores, config);
    EXPECT_TRUE(kernel.is_native());
    EXPECT_EQ(kernel.expected_num_binaries(), 3u);
    const auto& binaries = kernel.owned_binaries();
    ASSERT_EQ(binaries.size(), 3u);
    for (size_t index = 0; index != 3; ++index) {
        EXPECT_EQ(kernel.get_kernel_processor_type(index), index);
        EXPECT_EQ(binaries[index],
                  &native_detail::NativeImageAccess::memory(kernel.native_images().images[index]));
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
    EXPECT_THROW((NativeElfKernel(context, DEFAULT_CONTEXT_ID, cores, NativeDataMovementConfig{image(0)})),
                 std::exception);
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
    EXPECT_THROW(program_kernel.validate_stream_owner(tt::worker_stream_state::Owner::SdkCircularBuffers), std::exception);
    EXPECT_THROW(program_kernel.validate_stream_owner(static_cast<tt::worker_stream_state::Owner>(0)), std::exception);
}
}  // namespace
}  // namespace tt::tt_metal::experimental
