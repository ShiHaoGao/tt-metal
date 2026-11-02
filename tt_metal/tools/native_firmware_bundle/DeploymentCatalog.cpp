#include "DeploymentCatalog.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/kernels/cq_prefetch.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include "llrt/core_descriptor.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include <yaml-cpp/yaml.h>
#include <umd/device/arch/wormhole_implementation.hpp>
#include <bit>
#include <set>
namespace tt::tt_metal::experimental::catalog {
Output make_input(const Input& input, const Hal& hal, llrt::RunTimeOptions& options) {
    const auto architecture = input.architecture;
    if ((architecture != tt::ARCH::BLACKHOLE && architecture != tt::ARCH::WORMHOLE_B0) ||
        hal.get_arch() != architecture || input.host.galaxy || options.get_dram_backed_cq()) {
        throw std::invalid_argument("catalog helper requires a matching supported architecture and host CQ");
    }
    const bool wormhole = architecture == tt::ARCH::WORMHOLE_B0;
    const bool routing_fw_enabled = wormhole && Cluster::is_base_routing_fw_enabled(ClusterType::N300);
    if (options.get_enable_2_erisc_mode() != !wormhole) {
        throw std::invalid_argument("catalog ERISC mode differs from official architecture policy");
    }
    if (!wormhole && !input.chip.noc_translation_enabled) {
        throw std::invalid_argument("Blackhole catalog requires translated NoC facts");
    }
    if (wormhole) {
        constexpr uint64_t row_mask = (uint64_t{1} << tt::umd::wormhole::TENSIX_GRID_SIZE.y) - 1;
        const auto& masks = input.chip.harvesting_masks;
        if (input.chip.board_type != BoardType::N300 || (masks.tensix_harvesting_mask & ~row_mask) != 0 ||
            masks.dram_harvesting_mask != 0 || masks.eth_harvesting_mask != 0 || masks.pcie_harvesting_mask != 0 ||
            masks.l2cpu_harvesting_mask != 0) {
            throw std::invalid_argument("catalog requires official Wormhole N300 chip facts");
        }
    }
    const auto harvested = std::popcount(input.chip.harvesting_masks.tensix_harvesting_mask);
    if (harvested > 2) {
        throw std::invalid_argument("unsupported official core catalog harvesting count");
    }
    const auto descriptor = wormhole ? "wormhole_b0_80_arch.yaml" : "blackhole_140_arch.yaml";
    auto soc_path = input.sdk_root / "tt_metal/soc_descriptors" / descriptor;
    auto cores_path = input.sdk_root / "tt_metal/core_descriptors" / descriptor;
    auto arch = std::make_shared<tt::umd::SocArchDescriptor>(soc_path.string());
    const tt::umd::SocDescriptor umd_soc(arch, input.chip);
    const metal_SocDescriptor soc(umd_soc, input.chip.board_type);
    const auto axis = resolve_dispatch_core_axis(
        DispatchCoreConfig{DispatchCoreType::WORKER}, architecture, tt_fabric::FabricTensixConfig::DISABLED);
    const auto& sku = tt::get_product_name(architecture, harvested);
    auto config = YAML::LoadFile(cores_path.string())[sku][axis == DispatchCoreAxis::ROW ? "row" : "col"][1];
    if (!config || config["dispatch_core_type"].as<std::string>() != "tensix") {
        throw std::invalid_argument("missing canonical Worker catalog entry");
    }
    auto range = config["compute_with_storage_grid_range"];
    CoreCoord start{range["start"][0].as<uint32_t>(), range["start"][1].as<uint32_t>()};
    CoreCoord end{range["end"][0].as<uint32_t>(), range["end"][1].as<uint32_t>()};
    auto grid = soc.get_grid_size(CoreType::TENSIX);
    if (start != CoreCoord{0, 0} || end.x >= grid.x || end.y >= grid.y) {
        throw std::invalid_argument("core catalog and UMD worker grid disagree");
    }
    auto project = [&](CoreCoord logical) {
        auto translated = soc.translate_coord_to(
            tt::umd::CoreCoord(logical.x, logical.y, CoreType::TENSIX, CoordSystem::LOGICAL), CoordSystem::TRANSLATED);
        CoreCoord v{translated.x, translated.y};
        // Match Device::virtual_noc0_coordinate: translated coordinates outside
        // the physical grid and all Blackhole coordinates are NoC-independent.
        auto noc1 = v;
        if (wormhole && v.x < soc.grid_size.x && v.y < soc.grid_size.y) {
            noc1 = {hal.noc_coordinate(NOC_1, soc.grid_size.x, v.x), hal.noc_coordinate(NOC_1, soc.grid_size.y, v.y)};
        }
        return DispatchCorePlacement{logical, v, {v, noc1}};
    };
    auto dispatch_core = [&](size_t index) {
        auto coord = config["dispatch_cores"][index];
        int x = coord[0].as<int>(), y = coord[1].as<int>();
        if (x < 0) {
            x += grid.x;
        }
        if (y < 0) {
            y += grid.y;
        }
        if (x < 0 || y < 0 || static_cast<size_t>(x) >= grid.x || static_cast<size_t>(y) >= grid.y) {
            throw std::invalid_argument("invalid relative core catalog coordinate");
        }
        return project({static_cast<uint32_t>(x), static_cast<uint32_t>(y)});
    };
    const auto prefetch = dispatch_core(0), dispatcher = dispatch_core(1);
    std::set<uint32_t> active(input.active_ethernet_channels.begin(), input.active_ethernet_channels.end());
    if (active.size() != input.active_ethernet_channels.size()) {
        throw std::invalid_argument("duplicate active ethernet channel");
    }
    for (auto channel : active) {
        (void)soc.get_eth_core_for_channel(channel, CoordSystem::TRANSLATED);
    }
    DispatchMemMap memory{CoreType::WORKER, 1, hal, false, {false, 1}, options};
    Output result{};
    auto& p = result.program;
    auto& t = p.topology;
    t.arch = architecture;
    t.device = input.device_id;
    t.deployment_devices = {t.device};
    t.serviced_devices = {t.device};
    t.mmio = true;
    t.core_type = CoreType::WORKER;
    t.num_hw_cqs = 1;
    t.subordinate_enabled = true;
    t.nodes = {
        {0, t.device, t.device, 0, PREFETCH_HD, {}, {1, 2}, {NOC_0, NOC_0, NOC_0}, prefetch},
        {1, t.device, t.device, 0, DISPATCH_HD, {0}, {2}, {NOC_0, NOC_1, NOC_0}, dispatcher},
        {2, t.device, t.device, 0, DISPATCH_S, {0}, {1}, {NOC_1, NOC_1, NOC_1}, dispatcher}};
    t.completion_writer = dispatcher;
    t.unused_core = project({0, 0});
    const auto lo = project(start), hi = project(end);
    p.workers = {
        static_cast<uint32_t>((end.x + 1) * (end.y + 1)),
        {hal.noc_multicast_encoding(lo.noc[0].x, lo.noc[0].y, hi.noc[0].x, hi.noc[0].y),
         hal.noc_multicast_encoding(hi.noc[1].x, hi.noc[1].y, lo.noc[1].x, lo.noc[1].y)},
        static_cast<uint32_t>(active.size()),
        static_cast<uint32_t>(active.size())};
    p.observation = {
        options.get_watcher_enabled(),
        options.get_watcher_noinline(),
        options.watcher_dispatch_disabled(),
        false,
        options.get_dispatch_telemetry_disabled(),
        false};
    p.queue.inputs = {
        input.host,
        1,
        memory.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED),
        hal.get_alignment(HalMemType::HOST),
        uint64_t{memory.max_prefetch_command_size()} *
            (memory.prefetch_q_entries() + 1 + PrefetchConstants::PREFETCH_MAX_OUTSTANDING_PCIE_READS)};
    p.queue.plan = plan_system_memory_queues(p.queue.inputs);
    p.memory = capture_dispatch_memory(memory, 0);
    p.hal = capture_dispatch_hal(hal);
    bool found = false;
    enumerate_jit_device_configs(architecture, cores_path.string(), soc_path.string(), [&](const JitDeviceConfig& c) {
        if (!found && c.num_hw_cqs == 1 && c.num_l1_banks == p.workers.count &&
            c.num_dram_banks == soc.get_num_dram_views() && c.dispatch_core_axis == axis &&
            c.dispatch_core_type == DispatchCoreType::WORKER && c.routing_fw_enabled == routing_fw_enabled) {
            result.device = c;
            result.device.hal = &hal;
            found = true;
        }
    });
    if (!found) {
        throw std::invalid_argument("no matching official offline JitDeviceConfig");
    }
    result.device.harvesting_mask = soc.harvesting_masks.tensix_harvesting_mask;
    const auto pcie = soc.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    result.device.pcie_core = pcie.empty() ? soc.grid_size : CoreCoord{pcie.front().x, pcie.front().y};
    // The official catalog describes profiler-disabled firmware. This supplier
    // input instead follows its explicit observation options and matching HAL.
    result.device.profiler_dram_bank_size_per_risc_bytes =
        options.get_profiler_enabled() ? get_profiler_dram_bank_size_per_risc_bytes(options) : 0;
    (void)plan_dispatch_program(p);
    return result;
}
}  // namespace tt::tt_metal::experimental::catalog
