if(TT_METAL_BUILD_TOOLS)
    add_subdirectory(watcher_dump)
    add_subdirectory(dispatch_telemetry_dump)
    add_subdirectory(lightmetal_runner)
    add_subdirectory(mem_bench)
    add_subdirectory(precompile_fw)
    add_subdirectory(jit_compile_server)
endif()

# Supplier publication remains explicitly buildable without entering the SDK's
# default build. Tests of the supplier implementation remain test-owned.
if(TT_METAL_BUILD_TOOLS OR TT_METAL_BUILD_TESTS)
    add_subdirectory(native_firmware_bundle)
else()
    add_subdirectory(native_firmware_bundle EXCLUDE_FROM_ALL)
endif()

include(sources.cmake)

add_library(tools OBJECT ${TOOLS_SRC})
tt_reuse_precompile_headers(tools metal_common_pch)
target_link_libraries(
    tools
    PUBLIC
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
        Metalium::Metal
    PRIVATE
        TT::Metalium::HostDevCommon
)
