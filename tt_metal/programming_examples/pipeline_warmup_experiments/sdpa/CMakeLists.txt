add_executable(pipeline_warmup_sdpa)

target_sources(
    pipeline_warmup_sdpa
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_warmup_sdpa.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/instrumented/instrumentation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/instrumented/device/instrumented_sdpa_device_operation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/instrumented/device/instrumented_sdpa_program_factory.cpp
)

target_link_libraries(
    pipeline_warmup_sdpa
    PUBLIC
        TTNN::TTNN
        tt_metal
        m
        pthread
)

target_include_directories(
    pipeline_warmup_sdpa
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/ttnn/cpp
        ${PROJECT_SOURCE_DIR}/ttnn/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(TARGET TT::CommonPCH)
    target_precompile_headers(pipeline_warmup_sdpa REUSE_FROM TT::CommonPCH)
endif()

set_target_properties(
    pipeline_warmup_sdpa
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/pipeline_warmup_experiments
)
