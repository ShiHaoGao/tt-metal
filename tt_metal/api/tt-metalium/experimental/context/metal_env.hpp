// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <tt-metalium/experimental/published_deployment.hpp>
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/context/device_profiler_config.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/system_mesh.hpp>

namespace tt::tt_metal {
class WorkerStreamStateClient;

// Describes the fabric topology and routing configuration for the devices in the environment.
// These parameters determine how devices are interconnected and how data is routed between them.
struct FabricConfigDescriptor {
    tt_fabric::FabricConfig fabric_config = tt_fabric::FabricConfig::DISABLED;
    tt_fabric::FabricReliabilityMode reliability_mode =
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    std::optional<uint8_t> num_routing_planes = std::nullopt;
    tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED;
    tt_fabric::FabricUDMMode fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED;
    tt_fabric::FabricManagerMode fabric_manager = tt_fabric::FabricManagerMode::DEFAULT;
    tt_fabric::FabricRouterConfig router_config = {};
};

// Configuration for a MetalEnv.
//
// The default descriptor discovers and connects to the physical cluster present in the system.
// A custom MetalEnvDescriptor can be supplied to target a mock/simulated cluster instead.
//
// Only one MetalEnv for the physical cluster may exist at a time  due to UMD limitations.
class MetalEnvDescriptor {
public:
    MetalEnvDescriptor() = default;

    explicit MetalEnvDescriptor(const std::string& mock_cluster_desc_path);

    explicit MetalEnvDescriptor(std::optional<std::string> mock_cluster_desc_path);

    MetalEnvDescriptor(std::optional<std::string> mock_cluster_desc_path, FabricConfigDescriptor fabric_config_desc);

    bool is_mock_device() const { return mock_cluster_desc_path_.has_value(); }
    const std::string& mock_cluster_desc_path() const { return *mock_cluster_desc_path_; }
    const FabricConfigDescriptor& fabric_config_descriptor() const { return fabric_config_desc_; }
    std::optional<DeviceProfilerMode> device_profiler_mode() const { return device_profiler_mode_; }
    void set_device_profiler_mode(DeviceProfilerMode mode) {
        if (published_deployment_ && published_deployment_->configuration().profiler_mode() != mode)
            throw std::invalid_argument("profiler mode differs from published deployment");
        device_profiler_mode_ = mode;
    }
    const std::shared_ptr<const experimental::PublishedDeployment>& published_deployment() const { return published_deployment_; }
    // Only an admitted immutable value can select publication mode. There is
    // no null setter that could silently restore source/JIT fallback.
    void set_published_deployment(experimental::PublishedDeployment deployment) {
        const auto mode = deployment.configuration().profiler_mode();
        if (device_profiler_mode_ && *device_profiler_mode_ != mode)
            throw std::invalid_argument("published deployment differs from selected profiler mode");
        auto owned = std::make_shared<const experimental::PublishedDeployment>(std::move(deployment));
        device_profiler_mode_ = mode;
        published_deployment_ = std::move(owned);
    }

protected:
    std::optional<std::string> mock_cluster_desc_path_ = std::nullopt;
    FabricConfigDescriptor fabric_config_desc_;
    // Absence preserves the existing environment-driven SDK behavior.
    std::optional<DeviceProfilerMode> device_profiler_mode_;
    std::shared_ptr<const experimental::PublishedDeployment> published_deployment_;
};

class MetalEnvImpl;

// A MetalEnv provides an interface for the runtime environment to access a homogeneous cluster of Tenstorrent devices.
// It exposes several query functions for the hardware capabilities and cluster configuration.
//
// The FabricConfigDescriptor in the MetalEnvDescriptor describes the topology of the devices — how they are
// interconnected and how traffic is routed between them. From this topology the MetalEnv constructs the
// system mesh, which virtualizes and partitions the physical hardware for placement queries.
//
// Note, MetalEnv is a RAII object. As such, it must outlive every object that uses it (e.g. MeshDevice).
// The MetalEnv should be destroyed before forking to avoid undefined behavior.
class MetalEnv {
public:
    // Construct and initialize a MetalEnv using the provided descriptor.
    explicit MetalEnv(MetalEnvDescriptor descriptor = {});
    // Acquire exclusive worker state in this environment before creating its
    // meshes. The descriptor must select an explicit profiler mode. Existing
    // contexts and repeated acquisition are rejected. The environment retains
    // the client until its context is retired.
    std::shared_ptr<const WorkerStreamStateClient> acquire_worker_stream_state_client();
    ~MetalEnv();

    MetalEnv(const MetalEnv&) = delete;
    MetalEnv& operator=(const MetalEnv&) = delete;
    MetalEnv(MetalEnv&&) = delete;
    MetalEnv& operator=(MetalEnv&&) = delete;

    /// @return The descriptor used to construct this MetalEnv.
    const MetalEnvDescriptor& get_descriptor() const;

    /// @return Architecture of this environment.
    tt::ARCH get_arch() const;

    /// @return Human-readable name of the architecture of this environment.
    std::string get_arch_name() const;

    /// @return Total number of PCIe devices in this environment.
    uint32_t get_num_pcie_devices() const;

    /// @return Number of available devices in this environment.
    uint32_t get_num_available_devices() const;

    /// @return Size in bytes of each Tensix core's L1 SRAM of this environment.
    uint32_t get_l1_size() const;

    /// @return Required address alignment in bytes for DRAM allocations of this environment.
    uint32_t get_dram_alignment() const;

    /// @return Required address alignment in bytes for L1 allocations of this environment.
    uint32_t get_l1_alignment() const;

    /// @return Maximum number of circular buffers per core of this environment.
    uint32_t get_arch_num_circular_buffers() const;

    /// @return Maximum usable L1 size in bytes when the ring-buffer size is 0 of this environment.
    uint32_t get_max_worker_l1_unreserved_size() const;

    /// @return Representable SFPU epsilon value of this environment.
    float get_eps() const;

    /// @return Representable SFPU NaN value of this environment.
    float get_nan() const;

    /// @return Representable SFPU Infinity value of this environment.
    float get_inf() const;

    /// @return The system mesh, lazily initialized.
    /// The system mesh provides a virtualized coordinate system over the physical devices, allowing
    /// MeshDevice instances to map logical coordinates to physical device IDs.
    distributed::SystemMesh& get_system_mesh();

    // Create a MeshDevice which will use this MetalEnv
    std::shared_ptr<distributed::MeshDevice> create_mesh_device(
        const distributed::MeshDeviceConfig& config,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        ttsl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

    // Create a unit mesh for the physical device ID which will use this MetalEnv
    std::shared_ptr<distributed::MeshDevice> create_unit_mesh_device(
        int device_id,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        ttsl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

    // Create a unit mesh for each physical device ID in the list which will use this MetalEnv
    std::map<int, std::shared_ptr<distributed::MeshDevice>> create_unit_meshes(
        const std::vector<int>& device_ids,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        ttsl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

    // Create a SubDevice that uses this MetalEnv
    SubDevice create_sub_device(ttsl::Span<const CoreRangeSet> cores);

private:
    friend class MetalEnvAccessor;
    std::unique_ptr<MetalEnvImpl> impl_;

    MetalEnvImpl& impl() { return *impl_; }
    MetalEnvDescriptor descriptor_;
};

}  // namespace tt::tt_metal
