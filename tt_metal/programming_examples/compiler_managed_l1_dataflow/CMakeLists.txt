add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/profiler)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/level_c)

add_custom_target(
    compiler_managed_l1_dataflow_examples
    DEPENDS
        compiler_managed_l1_dataflow_profiler_examples
        compiler_managed_l1_dataflow_level_c_examples
        ${COMPILER_MANAGED_L1_DATAFLOW_PROFILER_TARGETS}
        ${COMPILER_MANAGED_L1_DATAFLOW_LEVEL_C_TARGETS}
)
