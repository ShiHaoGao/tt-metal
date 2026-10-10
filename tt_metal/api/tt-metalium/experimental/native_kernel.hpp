// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "hostdev/native_kernel_contract.h"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"

namespace tt::tt_metal {
class IDevice;
class Program;
class RiscFirmwareInitializer;
namespace experimental {
namespace native_detail { class NativeImageAccess; }

enum class TensixKernelRole : uint8_t {
    Brisc = TT_NATIVE_ROLE_BRISC,
    Ncrisc = TT_NATIVE_ROLE_NCRISC,
    Trisc0 = TT_NATIVE_ROLE_TRISC0,
    Trisc1 = TT_NATIVE_ROLE_TRISC1,
    Trisc2 = TT_NATIVE_ROLE_TRISC2,
};

// CPU-only immutable owner. Every input is copied. Linking resources are
// derived from these complete deployment ELFs, never supplied separately.
class FirmwareBundle {
public:
    static FirmwareBundle from_images(
        const std::array<std::span<const std::byte>, TT_NATIVE_ROLE_COUNT>& images,
        std::string_view label = "firmware");
    std::span<const std::byte> image_bytes(TensixKernelRole role) const;
    std::span<const std::byte> link_image_bytes(TensixKernelRole role) const;
    const tt_native_image_record& contract(TensixKernelRole role) const;
    bool matches(const FirmwareBundle& other) const;
private:
    struct Impl;
    explicit FirmwareBundle(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class native_detail::NativeImageAccess;
};

// Retains original bytes, their linked firmware owner, and the loader's packed
// image. The linked support record selects the exact role and loading mode.
class KernelElfImage {
public:
    static KernelElfImage from_bytes(
        std::span<const std::byte> image, const FirmwareBundle& firmware,
        std::string_view label = "native kernel");
    TensixKernelRole role() const;
    std::span<const std::byte> image_bytes() const;
    const tt_native_image_record& contract() const;
    const FirmwareBundle& firmware() const;
    std::string_view label() const;
private:
    struct Impl;
    explicit KernelElfImage(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class native_detail::NativeImageAccess;
};

struct NativeDataMovementConfig {
    KernelElfImage image;
    NOC noc = NOC::NOC_0;
    NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;
};

struct NativeComputeConfig {
    KernelElfImage trisc0;
    KernelElfImage trisc1;
    KernelElfImage trisc2;
};

KernelHandle CreateKernelFromElf(Program&, const CoreRangeSet&, const NativeDataMovementConfig&);
KernelHandle CreateKernelFromElf(Program&, const CoreRangeSet&, const NativeComputeConfig&);

// Validate and prepare a program composed exclusively of compiler-published
// native ELF kernels for the supplied live device.  The implementation keeps
// the SDK-owned worker-stream, firmware-generation, placement and native-image
// checks in ProgramImpl; this public entry point avoids exposing that private
// implementation header to installed consumers.
void PrepareNativeProgram(Program&, IDevice&);

// Returns only the bundle actually used by successful firmware boot. This
// does not create a context, build firmware, reopen files, or mint boot state.
FirmwareBundle GetLoadedFirmwareBundle(IDevice& device);

}  // namespace experimental
}  // namespace tt::tt_metal
