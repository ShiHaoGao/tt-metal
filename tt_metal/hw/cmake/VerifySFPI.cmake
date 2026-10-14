# SPDX-License-Identifier: Apache-2.0
# Invoked as a build step after extraction (or for explicit system SDK reuse).
if(NOT EXISTS "${SFPI_COMPILER}")
    message(FATAL_ERROR "SFPI compiler is missing: ${SFPI_COMPILER}")
endif()
execute_process(
    COMMAND "${SFPI_COMPILER}" --version
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _version
    ERROR_VARIABLE _error
)
if(NOT _status EQUAL 0)
    message(FATAL_ERROR "SFPI compiler cannot run: ${SFPI_COMPILER}: ${_error}")
endif()
string(REGEX MATCH "\\([^:\n]+:([^][)\n]+)" _match "${_version}")
if(NOT _match OR NOT CMAKE_MATCH_1 STREQUAL SFPI_VERSION)
    message(FATAL_ERROR "SFPI version does not match required ${SFPI_VERSION}: ${_version}")
endif()

# Loading the driver alone does not check cc1/cc1plus and other internal tools.
# Inspect only ELF files; shell wrappers and static executables need no ldd.
get_filename_component(_bin_dir "${SFPI_COMPILER}" DIRECTORY)
get_filename_component(_compiler_dir "${_bin_dir}" DIRECTORY)
file(GLOB _binaries "${_bin_dir}/*" "${_compiler_dir}/libexec/gcc/riscv-tt-elf/*/*")
foreach(_binary IN LISTS _binaries)
    if(IS_DIRECTORY "${_binary}")
        continue()
    endif()
    file(READ "${_binary}" _magic LIMIT 4 HEX)
    if(NOT _magic STREQUAL "7f454c46")
        continue()
    endif()
    execute_process(COMMAND ldd -d "${_binary}" RESULT_VARIABLE _status
                    OUTPUT_VARIABLE _dependencies ERROR_VARIABLE _error)
    if("${_dependencies}${_error}" MATCHES "=> not found|undefined symbol")
        message(FATAL_ERROR "SFPI has unsatisfied dependencies: ${_binary}\n${_dependencies}${_error}")
    elseif(NOT _status EQUAL 0 AND NOT "${_dependencies}${_error}" MATCHES "not a dynamic executable|statically linked")
        message(FATAL_ERROR "Cannot check SFPI dependencies: ${_binary}\n${_dependencies}${_error}")
    endif()
endforeach()
