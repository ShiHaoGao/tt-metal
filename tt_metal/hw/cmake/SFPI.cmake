# SPDX-License-Identifier: Apache-2.0
# The vendor compiler is needed for SDK firmware/support, not TReX algorithm
# codegen. Query release metadata during configuration; fetch and validate the
# pinned compiler only when its explicit build dependency is requested.
set(SFPI_BASE "${TT_METAL_RUNTIME_ROOT}")
set(_sfpi_info "${CMAKE_CURRENT_SOURCE_DIR}/../sfpi-info.sh")
set(_sfpi_version "${CMAKE_CURRENT_SOURCE_DIR}/../sfpi-version")
set(_sfpi_metadata "${CMAKE_CURRENT_BINARY_DIR}/sfpi-version.cmake")
execute_process(
    COMMAND "${_sfpi_info}" CMAKE txz
    OUTPUT_FILE "${_sfpi_metadata}"
    COMMAND_ERROR_IS_FATAL ANY
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_sfpi_info}" "${_sfpi_version}")
include("${_sfpi_metadata}")
set(_sfpi_verify "${CMAKE_CURRENT_LIST_DIR}/VerifySFPI.cmake")

if(TT_USE_SYSTEM_SFPI)
    set(SFPI_BASE "/opt/tenstorrent")
endif()
set(GPP_CMD "${SFPI_BASE}/sfpi/compiler/bin/riscv-tt-elf-g++")

if(TT_USE_SYSTEM_SFPI)
    # System SDK reuse is explicit. Version/loader checks still precede every
    # resource consumer, and never run the compiler during CMake configuration.
    set(_sfpi_checked "${CMAKE_CURRENT_BINARY_DIR}/sfpi-system-verified.stamp")
    add_custom_command(
        OUTPUT "${_sfpi_checked}"
        COMMAND "${CMAKE_COMMAND}" "-DSFPI_COMPILER=${GPP_CMD}"
                "-DSFPI_VERSION=${SFPI_version}" -P "${_sfpi_verify}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_sfpi_checked}"
        DEPENDS "${GPP_CMD}" "${_sfpi_verify}" "${_sfpi_version}" "${_sfpi_metadata}"
        VERBATIM
    )
    add_custom_target(sfpi_toolchain DEPENDS "${_sfpi_checked}")
else()
    if(SFPI_hash STREQUAL "")
        message(FATAL_ERROR
            "No pinned SFPI release for ${SFPI_arch} ${SFPI_dist}. "
            "Use a supported host or explicitly provide the matching system SFPI "
            "with TT_USE_SYSTEM_SFPI=ON; configure does not build SFPI from source."
        )
    endif()
    include(ExternalProject)
    ExternalProject_Add(
        sfpi_toolchain
        PREFIX "${CMAKE_CURRENT_BINARY_DIR}/sfpi"
        SOURCE_DIR "${SFPI_BASE}/sfpi"
        URL "${SFPI_url}/${SFPI_filename}"
        URL_HASH "${SFPI_HASHTYPE}=${SFPI_hash}"
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
        TLS_VERIFY ON
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND "${CMAKE_COMMAND}" "-DSFPI_COMPILER=${GPP_CMD}"
                      "-DSFPI_VERSION=${SFPI_version}" -P "${_sfpi_verify}"
        BUILD_BYPRODUCTS "${GPP_CMD}"
        INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
    )
    ExternalProject_Add_StepDependencies(sfpi_toolchain build "${_sfpi_verify}" "${_sfpi_version}")
endif()
