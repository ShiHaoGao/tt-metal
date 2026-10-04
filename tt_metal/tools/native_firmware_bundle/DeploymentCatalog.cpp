#include "DeploymentCatalog.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/kernels/cq_prefetch.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include <yaml-cpp/yaml.h>
#include <bit>
#include <set>
namespace tt::tt_metal::experimental::catalog {
Output make_input(const Input& input, const Hal& hal, llrt::RunTimeOptions& options) {
    if (hal.get_arch() != tt::ARCH::BLACKHOLE || !input.chip.noc_translation_enabled ||
        input.host.galaxy || options.get_dram_backed_cq())
        throw std::invalid_argument("catalog helper requires translated Blackhole host-CQ input");
    const auto harvested = std::popcount(input.chip.harvesting_masks.tensix_harvesting_mask);
    if (harvested > 2) throw std::invalid_argument("unsupported Blackhole core catalog harvesting count");
    auto soc_path = input.sdk_root / "tt_metal/soc_descriptors/blackhole_140_arch.yaml";
    auto cores_path = input.sdk_root / "tt_metal/core_descriptors/blackhole_140_arch.yaml";
    auto arch = std::make_shared<tt::umd::SocArchDescriptor>(soc_path.string());
    tt::umd::SocDescriptor soc(arch, input.chip);
    const auto sku = harvested == 0 ? "unharvested" : harvested == 1 ? "1xharvested" : "2xharvested";
    auto config = YAML::LoadFile(cores_path.string())[sku]["col"][1];
    if (!config || config["dispatch_core_type"].as<std::string>() != "tensix")
        throw std::invalid_argument("missing canonical Blackhole Worker catalog entry");
    auto range = config["compute_with_storage_grid_range"];
    CoreCoord start{range["start"][0].as<uint32_t>(), range["start"][1].as<uint32_t>()};
    CoreCoord end{range["end"][0].as<uint32_t>(), range["end"][1].as<uint32_t>()};
    auto grid = soc.get_grid_size(CoreType::TENSIX);
    if (start != CoreCoord{0, 0} || end.x >= grid.x || end.y >= grid.y)
        throw std::invalid_argument("core catalog and UMD worker grid disagree");
    auto project = [&](CoreCoord logical) {
        auto translated = soc.translate_coord_to(
            tt::umd::CoreCoord(logical.x, logical.y, CoreType::TENSIX, CoordSystem::LOGICAL), CoordSystem::TRANSLATED);
        CoreCoord v{translated.x, translated.y};
        // Device::virtual_noc0_coordinate returns the same translated coords on Blackhole.
        return DispatchCorePlacement{logical, v, {v, v}};
    };
    auto dispatch_core = [&](size_t index) {
        auto coord = config["dispatch_cores"][index];
        int x = coord[0].as<int>(), y = coord[1].as<int>();
        if (x < 0) x += grid.x;
        if (y < 0) y += grid.y;
        if (x < 0 || y < 0 || static_cast<size_t>(x) >= grid.x || static_cast<size_t>(y) >= grid.y)
            throw std::invalid_argument("invalid relative core catalog coordinate");
        return project({static_cast<uint32_t>(x), static_cast<uint32_t>(y)});
    };
    const auto prefetch = dispatch_core(0), dispatcher = dispatch_core(1);
    std::set<uint32_t> active(input.active_ethernet_channels.begin(), input.active_ethernet_channels.end());
    if (active.size() != input.active_ethernet_channels.size()) throw std::invalid_argument("duplicate active ethernet channel");
    for (auto channel : active) (void)soc.get_eth_core_for_channel(channel, CoordSystem::TRANSLATED);
    DispatchMemMap memory{CoreType::WORKER, 1, hal, false, {false, 1}, options};
    Output result{};
    auto& p = result.program;
    auto& t = p.topology;
    t.arch = tt::ARCH::BLACKHOLE; t.device = input.device_id;
    t.deployment_devices = {t.device}; t.serviced_devices = {t.device}; t.mmio = true;
    t.core_type = CoreType::WORKER; t.num_hw_cqs = 1; t.subordinate_enabled = true;
    t.nodes = {
        {0, t.device, t.device, 0, PREFETCH_HD, {}, {1, 2}, {NOC_0, NOC_0, NOC_0}, prefetch},
        {1, t.device, t.device, 0, DISPATCH_HD, {0}, {2}, {NOC_0, NOC_1, NOC_0}, dispatcher},
        {2, t.device, t.device, 0, DISPATCH_S, {0}, {1}, {NOC_1, NOC_1, NOC_1}, dispatcher}};
    t.completion_writer = dispatcher; t.unused_core = project({0, 0});
    auto lo = project(start).virtual_core, hi = project(end).virtual_core;
    p.workers = {static_cast<uint32_t>((end.x + 1) * (end.y + 1)),
        {hal.noc_multicast_encoding(lo.x, lo.y, hi.x, hi.y), hal.noc_multicast_encoding(hi.x, hi.y, lo.x, lo.y)},
        static_cast<uint32_t>(active.size()), static_cast<uint32_t>(active.size())};
    p.observation = {options.get_watcher_enabled(), options.get_watcher_noinline(), options.watcher_dispatch_disabled(),
        false, options.get_dispatch_telemetry_disabled(), false};
    p.queue.inputs = {input.host, 1, memory.get_host_command_queue_addr(CommandQueueHostAddrType::UNRESERVED),
        hal.get_alignment(HalMemType::HOST), uint64_t{memory.max_prefetch_command_size()} *
            (memory.prefetch_q_entries() + 1 + PrefetchConstants::PREFETCH_MAX_OUTSTANDING_PCIE_READS)};
    p.queue.plan = plan_system_memory_queues(p.queue.inputs);
    p.memory = capture_dispatch_memory(memory, 0); p.hal = capture_dispatch_hal(hal);
    bool found = false;
    enumerate_jit_device_configs(tt::ARCH::BLACKHOLE, cores_path.string(), soc_path.string(), [&](const JitDeviceConfig& c) {
        if (!found && c.num_hw_cqs == 1 && c.num_l1_banks == p.workers.count &&
            c.num_dram_banks == static_cast<size_t>(soc.get_num_dram_channels()) &&
            c.dispatch_core_axis == DispatchCoreAxis::COL && !c.routing_fw_enabled) {
            result.device = c; result.device.hal = &hal; found = true;
        }
    });
    if (!found) throw std::invalid_argument("no matching official offline JitDeviceConfig");
    result.device.harvesting_mask = soc.harvesting_masks.tensix_harvesting_mask;
    const auto pcie = soc.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    result.device.pcie_core = pcie.empty() ? soc.grid_size : CoreCoord{pcie.front().x, pcie.front().y};
    // The official catalog describes profiler-disabled firmware. This supplier
    // input instead follows its explicit observation options and matching HAL.
    result.device.profiler_dram_bank_size_per_risc_bytes = options.get_profiler_enabled()
        ? get_profiler_dram_bank_size_per_risc_bytes(options) : 0;
    (void)plan_dispatch_program(p);
    return result;
}
} // namespace tt::tt_metal::experimental::catalog
