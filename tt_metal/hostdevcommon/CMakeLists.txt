add_library(ttmetalium_hostdevcommon INTERFACE)
add_library(TT::Metalium::HostDevCommon ALIAS ttmetalium_hostdevcommon)

include(sources.cmake)

target_sources(
    ttmetalium_hostdevcommon
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS api
        FILES ${HOSTDEVCOMMON_JIT_API_HEADERS}
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS api
        FILES ${HOSTDEVCOMMON_JIT_API_HEADERS}
        FILE_SET hostdev_headers
        TYPE HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}
        FILES ${HOSTDEVCOMMON_HOSTDEV_HEADERS}
        FILE_SET worker_stream_state_api
        TYPE HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        FILES ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/hostdev/worker_stream_state_contract.h
)

target_include_directories(ttmetalium_hostdevcommon INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>")

target_link_libraries(
    ttmetalium_hostdevcommon
    INTERFACE
        fmt::fmt-header-only
        umd::tt-umd
)

install(
    TARGETS
        ttmetalium_hostdevcommon
    EXPORT Metalium
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hostdevcommon/api # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
    FILE_SET
    api
        COMPONENT metalium-dev
    FILE_SET
    worker_stream_state_api
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT metalium-dev
    FILE_SET
    hostdev_headers
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium
        COMPONENT metalium-runtime
)
