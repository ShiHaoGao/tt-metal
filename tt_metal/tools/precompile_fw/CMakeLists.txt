include(sources.cmake)

add_executable(precompile_fw ${PRECOMPILE_FW_SOURCES})
target_link_libraries(
    precompile_fw
    PRIVATE
        tt_metal
        Metalium::Metal::Hardware
)

target_include_directories(precompile_fw PRIVATE ${PROJECT_SOURCE_DIR}/tt_metal)

set_target_properties(
    precompile_fw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

tt_reuse_precompile_headers(precompile_fw TT::CommonPCH)

if(NOT TT_METAL_SKIP_LINKING)
    add_custom_command(
        OUTPUT
            ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp
        COMMAND
            ${CMAKE_COMMAND} -E env ASAN_OPTIONS=detect_odr_violation=0 $<TARGET_FILE:precompile_fw>
        COMMAND
            ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS
            precompile_fw
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-1xx/brisc.cc
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-1xx/trisc.cc
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/hostdev/worker_stream_state_contract.h
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/main.ld
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/blackhole/dst_initialization.h
            ${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/vptr_uint.h
        COMMENT "Pre-compiling firmware binaries"
    )

    add_custom_target(precompile-fw ALL DEPENDS ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp)
else()
    add_custom_target(precompile-fw ALL)
endif()

if(TT_INSTALL)
    install(
        DIRECTORY
            ${PROJECT_SOURCE_DIR}/tt_metal/pre-compiled/
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/pre-compiled
        COMPONENT metalium-runtime
    )
endif()
