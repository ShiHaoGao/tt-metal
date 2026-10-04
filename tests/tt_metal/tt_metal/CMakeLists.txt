add_library(test_metal_common_libs INTERFACE)
target_link_libraries(
    test_metal_common_libs
    INTERFACE
        Metalium::Metal
        test_common_libs
)

# Consolidated legacy gtest executable
# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(unit_tests_legacy)
target_sources(unit_tests_legacy PRIVATE ${UNIT_TESTS_LEGACY_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_legacy)
tt_reuse_precompile_headers(unit_tests_legacy metal_test_pch)
target_link_libraries(unit_tests_legacy PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_legacy
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    unit_tests_legacy
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
list(APPEND METAL_TEST_TARGETS unit_tests_legacy)

# Owned ELF admission must remain usable before any device/context exists.
# Keep this CPU test in its own process, separate from device fixtures.
add_executable(unit_tests_owned_elf llrt/test_owned_elf.cpp)
target_link_libraries(unit_tests_owned_elf PRIVATE test_metal_common_libs)
target_include_directories(unit_tests_owned_elf PRIVATE
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>")
set_target_properties(unit_tests_owned_elf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
gtest_discover_tests(unit_tests_owned_elf
    TEST_PREFIX "unit_tests_owned_elf." DISCOVERY_TIMEOUT 60)
list(APPEND METAL_TEST_TARGETS unit_tests_owned_elf)

add_executable(unit_tests_native_elf llrt/test_native_elf.cpp)
target_link_libraries(unit_tests_native_elf PRIVATE test_metal_common_libs)
target_include_directories(unit_tests_native_elf PRIVATE
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>")
set_target_properties(unit_tests_native_elf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
gtest_discover_tests(unit_tests_native_elf
    TEST_PREFIX "unit_tests_native_elf." DISCOVERY_TIMEOUT 60)
list(APPEND METAL_TEST_TARGETS unit_tests_native_elf)

add_executable(unit_tests_published_deployment
    api/test_published_deployment.cpp
    api/test_published_deployment_archive.cpp
    api/test_published_deployment_imports.cpp
    api/test_published_deployment_structure.cpp
    api/test_published_deployment_regions.cpp
    api/test_published_dispatch_kernel.cpp
    api/test_published_dispatch_consumer.cpp
    api/test_jit_device_config_profile.cpp
    api/test_profiler_allocation.cpp
    ${PROJECT_SOURCE_DIR}/tt_metal/tools/native_firmware_bundle/DeploymentCatalogTest.cpp
    api/test_semaphore_offset.cpp
    api/test_dispatch_plan.cpp
    api/test_dispatch_program_plan.cpp
    api/test_published_dispatch_program.cpp
    api/test_system_memory_queue_plan.cpp
    api/test_published_firmware.cpp)
target_link_libraries(unit_tests_published_deployment PRIVATE test_metal_common_libs native_deployment_build)
target_compile_definitions(unit_tests_published_deployment PRIVATE
    TT_NATIVE_FIRMWARE_SDK_ROOT="${PROJECT_SOURCE_DIR}")
target_include_directories(unit_tests_published_deployment PRIVATE
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>")
set_target_properties(unit_tests_published_deployment PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
gtest_discover_tests(unit_tests_published_deployment
    TEST_PREFIX "unit_tests_published_deployment." DISCOVERY_TIMEOUT 60
    PROPERTIES ENVIRONMENT "TT_METAL_RUNTIME_ROOT=${PROJECT_SOURCE_DIR}")
list(APPEND METAL_TEST_TARGETS unit_tests_published_deployment)

# Standalone test_clean_init executable (not a gtest - requires custom main with skip-teardown handling)
add_executable(test_clean_init test_clean_init.cpp)
tt_reuse_precompile_headers(test_clean_init metal_test_pch)
target_link_libraries(test_clean_init PUBLIC Metalium::Metal)
target_include_directories(
    test_clean_init
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    test_clean_init
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/api)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/data_movement)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/debug_tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deployment)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/context)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dispatch)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/eth)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hal_codegen)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/integration)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jit_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llk)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/perf_microbenchmark)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/noc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/noc_debugging)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lightmetal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sfpi)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/misc)

add_custom_target(
    metal_tests
    DEPENDS
        ${METAL_TEST_TARGETS}
        metal_perf_microbenchmark_tests
        unit_tests_api
        unit_tests_tensor
        unit_tests_hybrid_allocator
        unit_tests_data_movement
        unit_tests_debug_tools
        unit_tests_deployment
        unit_tests_inspector
        unit_tests_device
        unit_tests_context
        unit_tests_dispatch
        unit_tests_eth
        unit_tests_integration
        unit_tests_llk
        unit_tests_noc
        unit_tests_noc_debugging
        unit_tests_lightmetal
        unit_tests_sfpi
        unit_tests_misc
)

add_library(metalium_test_files INTERFACE)
# These headers are for the device, not host; will require cross compiling to verify.
set_target_properties(
    metalium_test_files
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    test_kernels/*.cpp
    test_kernels/*.hpp
    test_kernels/*.h
    test_kernels/*.inc
    perf_microbenchmark/routing/kernels/*.cpp
    perf_microbenchmark/routing/kernels/*.hpp
)
target_sources(
    metalium_test_files
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../.. # FIXME
        FILES
            ${kernels}
            perf_microbenchmark/common/kernel_utils.hpp
            perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp
            perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp
)

install(
    TARGETS
        metalium_test_files
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium # FIXME: fix the include paths for jit_build
        COMPONENT metalium-validation
)
# Integrate in tests/tt_metal/tt_metal/CMakeLists.txt (controller-owned seam).
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(runtime_root_probe llrt/runtime_root_probe.cpp)
target_link_libraries(runtime_root_probe PRIVATE Metalium::Metal)
target_compile_features(runtime_root_probe PRIVATE cxx_std_20)
target_include_directories(runtime_root_probe PRIVATE
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>")
set_target_properties(runtime_root_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
add_test(NAME unit_tests_runtime_root.relocation
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/llrt/test_runtime_root.py
        --probe $<TARGET_FILE:runtime_root_probe>
        --library $<TARGET_FILE:Metalium::Metal>
        --resources ${PROJECT_SOURCE_DIR}/tt_metal
        --libdir ${CMAKE_INSTALL_LIBDIR}
        --libexecdir ${CMAKE_INSTALL_LIBEXECDIR}
        --distro-root /usr/${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium
        -v)
set_tests_properties(unit_tests_runtime_root.relocation PROPERTIES TIMEOUT 120)
list(APPEND METAL_TEST_TARGETS runtime_root_probe)
