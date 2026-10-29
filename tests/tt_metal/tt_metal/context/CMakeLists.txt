# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# CPU allocation/lifetime coverage uses the production Metalium algorithm.
# It does not construct a firmware boot or activate device admission.
add_executable(unit_tests_niu_admission
    test_niu_admission.cpp test_niu_infrastructure.cpp
    test_native_payload_bindings.cpp)
target_compile_features(unit_tests_niu_admission PRIVATE cxx_std_20)
target_include_directories(unit_tests_niu_admission BEFORE PRIVATE
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    ${PROJECT_SOURCE_DIR}/tt_metal/impl
    ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium)
target_link_libraries(unit_tests_niu_admission PRIVATE test_metal_common_libs)
set_target_properties(unit_tests_niu_admission PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
gtest_discover_tests(unit_tests_niu_admission
    TEST_PREFIX "unit_tests_niu_admission."
    PROPERTIES ENVIRONMENT "TT_METAL_RUNTIME_ROOT=${PROJECT_SOURCE_DIR}"
    DISCOVERY_TIMEOUT 60)

# Worker-state admission belongs to Metalium and needs no TTNN algorithm build.
add_executable(unit_tests_worker_stream_state ${UNIT_TESTS_WORKER_STREAM_STATE_SOURCES})
target_include_directories(unit_tests_worker_stream_state BEFORE PRIVATE
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unit_tests_worker_stream_state PRIVATE test_metal_common_libs)
tt_reuse_precompile_headers(unit_tests_worker_stream_state metal_test_pch)
set_target_properties(unit_tests_worker_stream_state PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)

# Smoke tests
add_library(unit_tests_context_smoke OBJECT)
add_library(TT::Metalium::Test::Context::Smoke ALIAS unit_tests_context_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_context_smoke)
target_sources(unit_tests_context_smoke PRIVATE ${UNIT_TESTS_CONTEXT_SMOKE_SOURCES})
target_include_directories(
    unit_tests_context_smoke
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    unit_tests_context_smoke
    PRIVATE
        test_metal_common_libs
        TTNN::CPP
)

# Non-smoke tests
add_executable(unit_tests_context)
target_include_directories(
    unit_tests_context
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    unit_tests_context
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(
    unit_tests_context
    PRIVATE
        TT::Metalium::Test::Context::Smoke
        test_metal_common_libs
        TTNN::CPP
)
