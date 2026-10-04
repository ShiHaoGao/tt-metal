// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "storage.hpp"
#include "dispatch_plan.hpp"
#include "record_writer.hpp"
#include "impl/experimental/published_deployment/dispatch_program_adapter.hpp"
#include <reflect>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tt::tt_metal::experimental {
namespace {
// V2 envelope: magic[8], version:u64, total_bytes:u64, configuration_bytes:u64,
// complete structural configuration, image_count:u64, then images in HAL/node
// order. Each image is kind:u64, processor:(u64,u64,u64), node:u64,
// bytes:u64, ELF[bytes]. Every integer is little-endian. Configuration uses the
// same full structural record as .tt_published_image (no digest or path lookup).
constexpr std::array Magic{std::byte{'T'}, std::byte{'T'}, std::byte{'D'}, std::byte{'P'},
                           std::byte{'U'}, std::byte{'B'}, std::byte{'\r'}, std::byte{'\n'}};
constexpr uint64_t Version = 2;
constexpr size_t MaxArchiveBytes = 512 * 1024 * 1024;
constexpr size_t MaxConfigurationBytes = 16 * 1024 * 1024;
constexpr size_t MaxImageBytes = 64 * 1024 * 1024;
constexpr size_t MaxImages = 4096;
constexpr size_t MaxSequenceElements = 4096;

[[noreturn]] void malformed() { throw std::invalid_argument("invalid published deployment archive"); }

template<class T> bool valid_enum(uint64_t value) {
    const auto is = [value](auto... members) { return ((value == static_cast<uint64_t>(members)) || ...); };
    if constexpr (std::is_same_v<T, tt::ARCH>) return is(tt::ARCH::BLACKHOLE);
    else if constexpr (std::is_same_v<T, DispatchCoreType>) return is(DispatchCoreType::WORKER, DispatchCoreType::ETH);
    else if constexpr (std::is_same_v<T, DispatchCoreAxis>) return is(DispatchCoreAxis::ROW, DispatchCoreAxis::COL);
    else if constexpr (std::is_same_v<T, tt::CoreType>) return is(tt::CoreType::WORKER, tt::CoreType::ETH);
    else if constexpr (std::is_same_v<T, DeviceProfilerMode>) return is(DeviceProfilerMode::Disabled, DeviceProfilerMode::Program);
    else if constexpr (std::is_same_v<T, HalProgrammableCoreType>)
        return is(HalProgrammableCoreType::TENSIX, HalProgrammableCoreType::ACTIVE_ETH,
                  HalProgrammableCoreType::IDLE_ETH, HalProgrammableCoreType::DRAM, HalProgrammableCoreType::DISPATCH);
    else if constexpr (std::is_same_v<T, HalProcessorClassType>) return is(HalProcessorClassType::DM, HalProcessorClassType::COMPUTE);
    else if constexpr (std::is_same_v<T, ll_api::memory::Loading>)
        return is(ll_api::memory::Loading::DISCRETE, ll_api::memory::Loading::CONTIGUOUS, ll_api::memory::Loading::CONTIGUOUS_XIP);
    else if constexpr (std::is_same_v<T, HalKernelTextLimit>)
        return is(HalKernelTextLimit::FromFirmwareBase, HalKernelTextLimit::AfterFirmware);
    else if constexpr (std::is_same_v<T, NOC>) return is(NOC_0, NOC_1);
    else if constexpr (std::is_same_v<T, PublishedImageKind>) return is(PublishedImageKind::Firmware, PublishedImageKind::Dispatch);
    else if constexpr (std::is_same_v<T, DispatchProgramFamily>) return is(DispatchProgramFamily::BlackholeSingleMmioWorkerOneCq);
    else if constexpr (std::is_same_v<T, DispatchWorkerType>) return is(PREFETCH_HD, DISPATCH_HD, DISPATCH_S);
    else static_assert(!std::is_same_v<T, T>, "archive enum requires an explicit legal value set");
}

template<class T> struct IsArray : std::false_type {};
template<class T, size_t N> struct IsArray<std::array<T, N>> : std::true_type {};

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : remaining_(bytes) {}
    size_t remaining() const { return remaining_.size(); }
    std::span<const std::byte> take(size_t count) {
        if (count > remaining()) malformed();
        auto result = remaining_.first(count);
        remaining_ = remaining_.subspan(count);
        return result;
    }
    uint64_t integer() {
        uint64_t result = 0;
        const auto bytes = take(8);
        for (unsigned i = 0; i != 8; ++i) result |= uint64_t(std::to_integer<uint8_t>(bytes[i])) << (8 * i);
        return result;
    }
    size_t count(size_t limit, size_t minimum_bytes = 1) {
        const auto result = integer();
        // Bounds precede narrowing, subspan construction and every allocation.
        if (result > limit || result > remaining() / minimum_bytes) malformed();
        return static_cast<size_t>(result);
    }
    template<class T> void read(T& value) {
        if constexpr (std::is_same_v<T, bool>) {
            const auto byte = std::to_integer<uint8_t>(take(1)[0]);
            if (byte > 1) malformed();
            value = byte != 0;
        } else if constexpr (std::is_enum_v<T>) {
            const auto number = integer();
            if (!valid_enum<T>(number)) malformed();
            value = static_cast<T>(number);
        } else if constexpr (std::is_integral_v<T>) {
            const auto number = integer();
            // Signed identifiers are nonnegative. The topology tunnel sentinel
            // has its own typed branch below, never a global signed exception.
            if (number > static_cast<uint64_t>(std::numeric_limits<T>::max())) malformed();
            value = static_cast<T>(number);
        } else if constexpr (std::is_same_v<T, CoreCoord>) {
            read(value.x); read(value.y);
        } else if constexpr (std::is_same_v<T, tt_cxy_pair>) {
            read(value.chip); read(value.x); read(value.y);
        } else if constexpr (std::is_same_v<T, DispatchPlacedNode>) {
            read(value.id); read(value.device); read(value.servicing_device); read(value.cq); read(value.role);
            read(value.upstream); read(value.downstream); read(value.nocs); read(value.core);
            if (integer() != UINT64_MAX) malformed();
            value.tunnel_index = -1;
        } else if constexpr (requires { value.has_value(); }) {
            bool present; read(present);
            if (present) { value.emplace(); read(*value); } else value.reset();
        } else if constexpr (requires { std::variant_size<T>::value; }) {
            variant(value, integer());
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            malformed();
        } else if constexpr (IsArray<T>::value) {
            if (count(MaxSequenceElements) != value.size()) malformed();
            for (auto& member : value) read(member);
        } else if constexpr (requires { value.size(); value.begin(); }) {
            const size_t length = count(MaxSequenceElements);
            value.clear();
            // Decode before growing: a tiny truncated record cannot reserve a
            // large vector of heavyweight dispatch configurations.
            for (size_t i = 0; i != length; ++i) {
                typename T::value_type member{};
                read(member);
                value.push_back(std::move(member));
            }
        } else {
            static_assert(std::is_aggregate_v<T>, "archive requires a structural value owner");
            reflect::for_each([&](auto i) { read(reflect::get<i>(value)); }, value);
        }
    }
private:
    template<size_t I = 0, class T> void variant(T& value, uint64_t index) {
        if constexpr (I == std::variant_size_v<T>) malformed();
        else if (index == I) { value.template emplace<I>(); read(std::get<I>(value)); }
        else variant<I + 1>(value, index);
    }
    std::span<const std::byte> remaining_;
};

void append(std::vector<std::byte>& bytes, uint64_t value) {
    for (unsigned i = 0; i != 8; ++i) bytes.push_back(std::byte(value >> (8 * i)));
}
std::vector<PublishedImageInput> images(const PublishedDeployment& deployment) {
    std::vector<PublishedImageInput> result;
    auto add = [&](PublishedImageInput input) {
        if (result.size() == MaxImages || input.bytes.empty() || input.bytes.size() > MaxImageBytes) malformed();
        result.push_back(input);
    };
    for (auto processor : deployment.configuration().firmware_processors())
        add({PublishedImageKind::Firmware, processor, NoDispatchNode, deployment.firmware_image(processor)});
    for (const auto& node : deployment.configuration().dispatch_nodes())
        for (auto processor : dispatch_processors(node))
            add({PublishedImageKind::Dispatch, processor, node.node_id, deployment.dispatch_image(node.node_id, processor)});
    return result;
}
} // namespace

std::vector<std::byte> PublishedDeployment::to_archive() const {
    const auto& config = deployment_detail::Access::configuration(configuration()).canonical;
    if (config.empty() || config.size() > MaxConfigurationBytes) malformed();
    const auto input = images(*this);
    size_t total = 40 + config.size();
    for (const auto& image : input) {
        if (total > MaxArchiveBytes - 48 || image.bytes.size() > MaxArchiveBytes - total - 48) malformed();
        total += 48 + image.bytes.size();
    }
    std::vector<std::byte> result;
    result.reserve(total);
    result.insert(result.end(), Magic.begin(), Magic.end());
    append(result, Version); append(result, total); append(result, config.size());
    result.insert(result.end(), config.begin(), config.end());
    append(result, input.size());
    for (const auto& image : input) {
        append(result, static_cast<uint64_t>(image.kind));
        append(result, static_cast<uint64_t>(image.processor.core_type));
        append(result, static_cast<uint64_t>(image.processor.processor_class));
        append(result, image.processor.processor_type); append(result, image.dispatch_node);
        append(result, image.bytes.size());
        result.insert(result.end(), image.bytes.begin(), image.bytes.end());
    }
    return result;
}

namespace {
PublishedDeployment decode_archive(std::span<const std::byte> bytes, const Hal* actual_hal) {
    if (bytes.size() > MaxArchiveBytes) malformed();
    Reader archive(bytes);
    if (!std::ranges::equal(archive.take(Magic.size()), Magic) || archive.integer() != Version ||
        archive.integer() != bytes.size()) malformed();
    const size_t config_size = archive.count(MaxConfigurationBytes);
    const auto config_bytes = archive.take(config_size);
    Reader record(config_bytes);
    DeviceConfiguration device{};
    deployment_detail::BuildConfiguration build{};
    std::vector<deployment_detail::ProcessorLayout> layouts;
    std::vector<deployment_detail::CoreLayout> cores;
    std::vector<DeviceAddr> dram_bases;
    std::vector<uint32_t> dram_sizes;
    record.read(device); record.read(build); record.read(layouts);
    record.read(cores); record.read(dram_bases); record.read(dram_sizes);
    const auto program_bytes = record.take(record.count(MaxConfigurationBytes));
    if (record.remaining()) malformed();
    // Canonical device/build values and the owned core inventory are the sole
    // expected-HAL requirements. Never probe candidates or consult live state.
    // Full canonical reconstruction below still validates every stored layout.
    std::optional<Hal> expected_hal;
    if (!actual_hal) {
        const bool dram_cores = std::ranges::any_of(cores, [](const auto& core) {
            return core.type == HalProgrammableCoreType::DRAM;
        });
        expected_hal.emplace(device.arch, device.routing_fw_enabled, build.two_erisc,
            device.profiler_dram_bank_size_per_risc_bytes, build.dram_backed_cq,
            false, dram_cores, build.eth_ptp_trace);
        actual_hal = &*expected_hal;
    }
    Reader program_record(program_bytes);
    if (program_record.integer() != 1) malformed();
    DispatchProgramFamily family{}; program_record.read(family);
    DispatchProgramInputs program_inputs{}; program_record.read(program_inputs);
    // Bound planner allocations by the actual HAL before deriving any writes.
    // A forged L1 size must not admit a huge prefetch initialization buffer.
    deployment_detail::RecordWriter actual_dispatch_hal, stored_dispatch_hal;
    actual_dispatch_hal.append(capture_dispatch_hal(*actual_hal));
    stored_dispatch_hal.append(program_inputs.hal);
    if (actual_dispatch_hal.bytes != stored_dispatch_hal.bytes) malformed();
    const auto program = plan_dispatch_program(program_inputs);
    // Outputs are reconstructed from typed inputs. Supplied nodes, semaphore
    // IDs, runtime words or initialization bytes never become a second owner.
    if (program.family() != family || !std::ranges::equal(program_bytes, program.canonical_record()))
        throw std::invalid_argument("published dispatch program differs from its derived canonical plan");
    auto config = deployment_detail::Access::admit_configuration(device, build, program, *actual_hal);
    // Reconstruct from the actual HAL and shared admission. A self-consistent
    // archive plus ELF records cannot substitute a forged HAL layout.
    if (!std::ranges::equal(config_bytes, deployment_detail::Access::configuration(config).canonical))
        throw std::invalid_argument("published archive configuration differs from actual HAL or canonical configuration");
    const size_t image_count = archive.count(MaxImages, 48);
    size_t required = config.firmware_processors().size();
    for (const auto& node : config.dispatch_nodes()) required += dispatch_processors(node).size();
    if (image_count != required) malformed();
    std::vector<PublishedImageInput> input;
    input.reserve(image_count);
    for (size_t i = 0; i != image_count; ++i) {
        PublishedImageInput image{};
        archive.read(image.kind); archive.read(image.processor); archive.read(image.dispatch_node);
        const size_t length = archive.count(MaxImageBytes);
        if (!length) malformed();
        image.bytes = archive.take(length);
        input.push_back(image);
    }
    if (archive.remaining()) malformed();
    // Admission owns copies and validates every ELF, profile, metadata record,
    // firmware export/import and HAL/companion role before returning a handle.
    return PublishedDeployment::admit(config, input);
}
} // namespace

PublishedDeployment PublishedDeployment::from_archive(std::span<const std::byte> bytes) {
    return decode_archive(bytes, nullptr);
}
PublishedDeployment PublishedDeployment::from_archive(std::span<const std::byte> bytes, const Hal& actual_hal) {
    return decode_archive(bytes, &actual_hal);
}
} // namespace tt::tt_metal::experimental
