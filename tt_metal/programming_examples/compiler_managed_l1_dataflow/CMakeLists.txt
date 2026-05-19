add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/profiler)

add_custom_target(
    compiler_managed_l1_dataflow_examples
    DEPENDS
        compiler_managed_l1_dataflow_profiler_examples
        ${COMPILER_MANAGED_L1_DATAFLOW_PROFILER_TARGETS}
)
