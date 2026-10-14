# Temporary workaround for Issue #8767
set(ARCHS
    wormhole
    blackhole
)
set(PROCS
    brisc
    ncrisc
    ierisc
    subordinate_ierisc
    trisc=0
    trisc=1
    trisc=2
    aerisc
    main_aerisc
    subordinate_aerisc
)
set(BH_ONLY_PROCS drisc)
set(TYPES
    firmware
    kernel
)

set(TLS_ARCH_CPUS quasar:tt-qsr32:tt-qsr64)
# name:cpu
set(TLS_PROCS
    trisc:tt-qsr32
    dm:tt-qsr64
)

# for wormhole, we need to generate two different linker scripts
set(WH_LD_SCRIPTS
    kernel # base name
    app # base name
)
set(IRAM_OPTIONS
    "" # No IRAM
    ENABLE_IRAM # With IRAM
)

# Generated resources belong to the vendor binary tree in both standalone and
# embedded builds. The embedding project may select the same runtime root.
if(NOT DEFINED TT_METAL_RUNTIME_ROOT)
    set(TT_METAL_RUNTIME_ROOT "${PROJECT_BINARY_DIR}/runtime")
endif()
include(cmake/SFPI.cmake)

# old-style
foreach(ARCH IN LISTS ARCHS)
    set(HW_INCLUDE "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}")
    set(HW_OUTPUT_DIR "${TT_METAL_RUNTIME_ROOT}/hw/toolchain/${ARCH}")
    string(TOUPPER ${ARCH} ARCH_DEFINE)

    set(ARCH_PROCS ${PROCS})

    if(ARCH STREQUAL "wormhole")
        foreach(SCRIPT_TYPE IN LISTS WH_LD_SCRIPTS)
            foreach(IRAM_OPT IN LISTS IRAM_OPTIONS)
                if(IRAM_OPT)
                    set(HW_OUTPUT_FILE "erisc-b0-${SCRIPT_TYPE}_iram.ld")
                    set(IRAM_FLAG "-D${IRAM_OPT}")
                    set(IRAM_COMMENT " with IRAM")
                else()
                    set(HW_OUTPUT_FILE "erisc-b0-${SCRIPT_TYPE}.ld")
                    set(IRAM_FLAG "")
                    set(IRAM_COMMENT "")
                endif()

                add_custom_command(
                    OUTPUT
                        ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                    COMMAND
                        ${CMAKE_COMMAND} -E make_directory ${HW_OUTPUT_DIR}
                    COMMAND
                        ${CMAKE_CXX_COMPILER} ${IRAM_FLAG} -I${HW_INCLUDE} -E -P -x c
                        -MMD -MF ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}.d -MT ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                        -o ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/erisc-b0-${SCRIPT_TYPE}.ld
                    DEPENDS
                        ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
                        ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/erisc-b0-${SCRIPT_TYPE}.ld
                        ${HW_INCLUDE}/dev_mem_map.h
                    DEPFILE ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}.d
                    COMMENT "Preprocessing ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}${IRAM_COMMENT}"
                    VERBATIM
                )

                list(APPEND LD_FILES ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE})
            endforeach()
        endforeach()
    elseif(ARCH STREQUAL "blackhole")
        list(APPEND ARCH_PROCS ${BH_ONLY_PROCS})
    endif()

    foreach(PROC IN LISTS ARCH_PROCS)
        string(TOUPPER ${PROC} PROC_DEFINE)
        string(REPLACE "=" "" PROC_FILE "${PROC}")
        foreach(TYPE IN LISTS TYPES)
            set(HW_OUTPUT_FILE "${TYPE}_${PROC_FILE}.ld")
            string(TOUPPER ${TYPE} TYPE_DEFINE)

            # custom command to preprocess/generate the output file
            add_custom_command(
                OUTPUT
                    ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                COMMAND
                    ${CMAKE_COMMAND} -E make_directory ${HW_OUTPUT_DIR}
                COMMAND
                    ${CMAKE_CXX_COMPILER} -DTYPE_${TYPE_DEFINE} -DCOMPILE_FOR_${PROC_DEFINE} -DARCH_${ARCH_DEFINE}
                    -I${HW_INCLUDE} -E -P -x c
                    -MMD -MF ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}.d -MT ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                    -o ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
                DEPENDS
                    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
                    ${HW_INCLUDE}/dev_mem_map.h
                DEPFILE ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}.d
                COMMENT "Preprocessing ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}"
                VERBATIM
            )

            # add output file to the custom target
            list(APPEND LD_FILES ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE})
        endforeach()
    endforeach()
endforeach()

# TLS-style
foreach(ARCH_CPU IN LISTS TLS_ARCH_CPUS)
    string(REPLACE ":" ";" ARCH_CPU_LIST ${ARCH_CPU})
    list(GET ARCH_CPU_LIST 0 ARCH)
    set(HW_INCLUDE "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-2xx/${ARCH}")
    set(HW_OUTPUT_DIR "${TT_METAL_RUNTIME_ROOT}/hw/toolchain/${ARCH}")
    string(TOUPPER ${ARCH} ARCH_DEFINE)

    foreach(PROC IN LISTS TLS_PROCS)
        # PROC is NAME:CPU
        string(REPLACE ":" ";" PROC_INFO ${PROC})
        list(GET PROC_INFO 0 NAME)
        list(GET PROC_INFO 1 CPU)

        foreach(TYPE IN LISTS TYPES)
            if(TYPE STREQUAL "firmware")
                set(BOUND 0)
            else()
                set(BOUND 1)
            endif()
            # bounds are inclusive [0,NUM]
            foreach(IX RANGE ${BOUND})
                set(LGC "")
                if(IX)
                    set(LGC "_lgc")
                endif()
                set(HW_OUTPUT_FILE "${TYPE}_${NAME}${LGC}.ld")
                string(TOUPPER "${TYPE}=${IX}" TYPE_DEFINE)
                string(TOUPPER "${NAME}" PROC_DEFINE)

                # custom command to preprocess/generate the output file
                add_custom_command(
                    OUTPUT
                        ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                    COMMAND
                        ${CMAKE_COMMAND} -E make_directory ${HW_OUTPUT_DIR}
                    COMMAND
                        ${CMAKE_CXX_COMPILER} -DTYPE_${TYPE_DEFINE} -DCOMPILE_FOR_${PROC_DEFINE} -DARCH_${ARCH_DEFINE}
                        -I${HW_INCLUDE} -E -P -x c
                        -MMD -MF ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}.d -MT ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                        -o ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/script_tng.ld
                    DEPENDS
                        ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
                        ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/script_tng.ld
                    DEPFILE ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}.d
                    COMMENT "Preprocessing ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}"
                    VERBATIM
                )

                # add output file to the custom target
                list(APPEND LD_FILES ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE})
            endforeach()
        endforeach()
    endforeach()
endforeach()

set(GPP_DEFINES -DTENSIX_FIRMWARE)

# Define flags for each architecture
set(GPP_FLAGS_wormhole -mcpu=tt-wh)
set(GPP_FLAGS_blackhole -mcpu=tt-bh)

# Define common flags for all architectures
set(GPP_FLAGS_common
    -std=c++17
    -flto=auto
    -ffast-math
    -fno-use-cxa-atexit
    -fno-exceptions
    -Wall
    -Werror
    -Wno-deprecated-declarations
    -Wno-unknown-pragmas
    -Wno-error=multistatement-macros
    -Wno-error=parentheses
    -Wno-error=unused-but-set-variable
    -Wno-unused-variable
    -Wno-unused-function
    -Os
    -fno-tree-loop-distribute-patterns
)

# We are going to build 5 or 6 object files foreach ARCH
# old-style
foreach(ARCH IN LISTS ARCHS)
    # These are the set of object files we are to build foreach ARCH
    set(HW_OBJ_SRC_PAIRS
        "substitutes.o:tt_metal/hw/toolchain/substitutes.cpp"
        "tdma_xmov.o:tt_metal/hw/firmware/src/tt-1xx/tdma_xmov.c"
        "noc.o:tt_metal/hw/firmware/src/tt-1xx/${ARCH}/noc.c"
        "tmu-crt0.o:tt_metal/hw/toolchain/tmu-crt0.S"
    )
    set(ARCH_ALIAS ${ARCH})

    if(ARCH STREQUAL "wormhole")
        set(ARCH_ALIAS wormhole_b0)
        list(
            APPEND
            HW_OBJ_SRC_PAIRS
            "wh-iram-start.o:tt_metal/hw/toolchain/wh-iram-start.S"
            "wh-iram-trampoline.o:tt_metal/hw/toolchain/wh-iram-trampoline.S"
        )
    endif()

    # Set GPP_FLAGS based on ARCH
    set(GPP_FLAGS
        ${GPP_FLAGS_${ARCH}}
        ${GPP_FLAGS_common}
    )

    # Dump object files to this directory
    set(HW_OBJ_DIR ${TT_METAL_RUNTIME_ROOT}/hw/lib/${ARCH})

    # Includes independent from ARCH
    set(GPP_INCLUDES
        -I.
        -I..
        -I${PROJECT_SOURCE_DIR}
        -I${PROJECT_SOURCE_DIR}/tt_metal
        -I${PROJECT_SOURCE_DIR}/tt_metal/api
        -I${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/debug
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-1xx
    )

    # Architecture specific include paths
    list(
        APPEND
        GPP_INCLUDES
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}/${ARCH_ALIAS}_defines
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}/noc
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}/overlay
        -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_ALIAS}/metal/common
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_ALIAS}/metal/llk_io
        -I${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_${ARCH_ALIAS}/common/inc
        -I${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_${ARCH_ALIAS}/llk_lib
        -I${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_${ARCH_ALIAS}/llk_lib/hal
    )

    foreach(HW_PAIR IN LISTS HW_OBJ_SRC_PAIRS)
        string(REPLACE ":" ";" HW_PAIR_LIST ${HW_PAIR})
        list(GET HW_PAIR_LIST 0 HWOBJ)
        list(GET HW_PAIR_LIST 1 HWSRC)
        set(HW_OUTPUT_FILE "${HW_OBJ_DIR}/${HWOBJ}")
        add_custom_command(
            OUTPUT
                ${HW_OUTPUT_FILE}
            COMMAND
                ${CMAKE_COMMAND} -E make_directory ${HW_OBJ_DIR}
            COMMAND
                ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES}
                -MMD -MF ${HW_OUTPUT_FILE}.d -MT ${HW_OUTPUT_FILE} -c -o ${HW_OUTPUT_FILE}
                ${PROJECT_SOURCE_DIR}/${HWSRC}
            DEPENDS
                ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
                ${PROJECT_SOURCE_DIR}/${HWSRC}
                ${GPP_CMD}
                sfpi_toolchain
            DEPFILE ${HW_OUTPUT_FILE}.d
            COMMENT "Building ${ARCH} hw lib ${HWOBJ}"
            VERBATIM
        )

        list(APPEND OBJ_FILES ${HW_OUTPUT_FILE})
    endforeach()
endforeach()

# TLS-style
foreach(ARCH_CPU IN LISTS TLS_ARCH_CPUS)
    string(REPLACE ":" ";" CPUS ${ARCH_CPU})
    list(GET CPUS 0 ARCH)
    list(POP_FRONT CPUS)

    # TODO:Review this list
    set(GPP_INCLUDES
        -I.
        -I..
        -I${PROJECT_SOURCE_DIR}
        -I${PROJECT_SOURCE_DIR}/tt_metal
        -I${PROJECT_SOURCE_DIR}/tt_metal/api
        -I${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/api/debug
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-2xx
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-2xx/${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-2xx/${ARCH}/noc
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH}/metal/common
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH}/metal/llk_io
    )
    set(HW_OBJ_DIR ${TT_METAL_RUNTIME_ROOT}/hw/lib/${ARCH})
    set(HW_OBJ_PATTERNS
        "substitutes.o:tt_metal/hw/toolchain/substitutes.cpp"
        "crt0-tls.o:tt_metal/hw/toolchain/crt0-tls.S"
        "crt0.o:tt_metal/hw/toolchain/crt0.S"
        "noc.o:tt_metal/hw/firmware/src/tt-2xx/quasar/noc.c:tt-qsr64"
    )
    set(GPP_DEFINES -D${ARCH})

    foreach(HW_PATTERN IN LISTS HW_OBJ_PATTERNS)
        string(REPLACE ":" ";" HW_LIST ${HW_PATTERN})
        list(LENGTH HW_LIST HW_LENGTH)
        list(GET HW_LIST 0 HWOBJ)
        list(GET HW_LIST 1 HWSRC)
        if(HW_LENGTH GREATER 2)
            list(GET HW_LIST 2 HWCPUS)
            string(REPLACE " " ";" HWCPUS ${HWCPUS})
        else()
            set(HWCPUS ${CPUS})
        endif()
        foreach(CPU IN LISTS CPUS)
            list(FIND HWCPUS ${CPU} HWENABLE)
            if(HWENABLE LESS 0)
                continue()
            endif()
            set(HW_OBJ_FILE "${CPU}-${HWOBJ}")

            set(GPP_FLAGS
                -mcpu=${CPU}
                ${GPP_FLAGS_common}
            )
            if(CPU STREQUAL "tt-qsr32")
                list(APPEND GPP_FLAGS -DCOMPILE_FOR_TRISC)
            endif()
            add_custom_command(
                OUTPUT
                    ${HW_OBJ_DIR}/${HW_OBJ_FILE}
                COMMAND
                    ${CMAKE_COMMAND} -E make_directory ${HW_OBJ_DIR}
                COMMAND
                    ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES}
                    -MMD -MF ${HW_OBJ_DIR}/${HW_OBJ_FILE}.d -MT ${HW_OBJ_DIR}/${HW_OBJ_FILE}
                    -c -o ${HW_OBJ_DIR}/${HW_OBJ_FILE}
                    ${PROJECT_SOURCE_DIR}/${HWSRC}
                DEPENDS
                    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
                    ${PROJECT_SOURCE_DIR}/${HWSRC}
                    ${GPP_CMD}
                    sfpi_toolchain
                DEPFILE ${HW_OBJ_DIR}/${HW_OBJ_FILE}.d
                COMMENT "Building ${ARCH}:${CPU} hw lib ${HW_OBJ_FILE} from ${HWSRC}"
                VERBATIM
            )
            list(APPEND OBJ_FILES ${HW_OBJ_DIR}/${HW_OBJ_FILE})
        endforeach()
    endforeach()
endforeach()

# custom target that depends on all the output files
add_custom_target(
    hw_toolchain
    ALL
    DEPENDS
        ${LD_FILES}
        ${OBJ_FILES}
)

add_library(hw INTERFACE)
add_library(Metalium::Metal::Hardware ALIAS hw)

target_include_directories(hw INTERFACE inc)

# These headers are for the device, not host; will require cross compiling to verify.
set_target_properties(
    hw
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# It's acceptable to use GLOB here because these files are not part of the build and developers aren't generating packages (for now).
# We may need to revisit later.
file(GLOB_RECURSE blackhole_kernels ckernels/blackhole/*.h)
file(GLOB_RECURSE wormhole_kernels ckernels/wormhole_b0/*.h)

include(sources.cmake)

target_sources(
    hw
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        FILES ${blackhole_kernels} ${wormhole_kernels} ${HW_JIT_API_HEADERS}
        FILE_SET toolchain
        TYPE HEADERS
        BASE_DIRS ${TT_METAL_RUNTIME_ROOT}
        FILES ${LD_FILES} ${OBJ_FILES}
)

target_link_libraries(hw INTERFACE TT::Metalium::HostDevCommon)

add_subdirectory(firmware)

install(
    TARGETS
        hw
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hw # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
    FILE_SET
    toolchain
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/runtime
        COMPONENT metalium-runtime
)
