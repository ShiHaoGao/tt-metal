set(PROFILER_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count/test_custom_cycle_count.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count_slow_dispatch/test_custom_cycle_count_slow_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer/test_full_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_op/test_multi_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_op_buffer_overflow/test_multi_op_buffer_overflow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_realtime_profiler_csv/test_realtime_profiler_csv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dispatch_cores/test_dispatch_cores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_timestamped_events/test_timestamped_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_noc_event_profiler/test_noc_event_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_device_api_debugger/test_device_api_debugger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_variants_profile/matmul_variants_profile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/flash_attention_profile.cpp
)

CREATE_PGM_EXAMPLES_EXE("${PROFILER_EXAMPLES_SRCS}" "profiler")

target_sources(
    matmul_variants_profile
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_variants_profile/host/matmul_single_core_profile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_variants_profile/host/matmul_multi_core_profile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_variants_profile/host/matmul_multicore_reuse_profile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_variants_profile/host/matmul_multicore_reuse_mcast_profile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_variants_profile/host/matmul_profile_runners.cpp
)

target_include_directories(
    matmul_variants_profile
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_variants_profile
)

target_sources(
    flash_attention_profile
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/flash_attention_profile_runners.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/copied_sdpa/device/sdpa_device_operation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/copied_sdpa/device/sdpa_program_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/split_compute_sdpa/device/sdpa_device_operation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/split_compute_sdpa/device/sdpa_program_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/llk_microflow_sdpa/device/sdpa_device_operation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/llk_microflow_sdpa/device/sdpa_program_factory.cpp
)

target_link_libraries(
    flash_attention_profile
    PUBLIC
        TTNN::TTNN
        tt_metal
        m
        pthread
)

target_include_directories(
    flash_attention_profile
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/ttnn/cpp
        ${PROJECT_SOURCE_DIR}/ttnn/api
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/copied_sdpa/device
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/split_compute_sdpa/device
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_profile/host/llk_microflow_sdpa/device
)

add_custom_target(profiler_examples DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
