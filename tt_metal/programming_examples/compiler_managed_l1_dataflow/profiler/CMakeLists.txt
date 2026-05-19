set(COMPILER_MANAGED_L1_DATAFLOW_PROFILER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/microbench/cb_protocol_overhead/cb_protocol_overhead.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/real_copy_protocol/real_copy_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_pipeline/real_tile_add_protocol/real_tile_add_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/real_matmul_protocol/real_matmul_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_pipeline/static_protocol_modeling/static_protocol_modeling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_binary_ng_no_bcast_protocol/ttnn_binary_ng_no_bcast_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_binary_ng_sfpu_no_bcast_protocol/ttnn_binary_ng_sfpu_no_bcast_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_binary_ng_row_bcast_protocol/ttnn_binary_ng_row_bcast_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_bcast_to_protocol/ttnn_bcast_to_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_paged_update_cache_protocol/ttnn_paged_update_cache_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_transpose_wh_protocol/ttnn_transpose_wh_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_embedding_lookup_protocol/ttnn_embedding_lookup_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_slice_tile_protocol/ttnn_slice_tile_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ttnn_kernel_forks/ttnn_kv_cache_load_slice_protocol/ttnn_kv_cache_load_slice_protocol.cpp
)

CREATE_PGM_EXAMPLES_EXE(
    "${COMPILER_MANAGED_L1_DATAFLOW_PROFILER_SRCS}"
    "compiler_managed_l1_dataflow/profiler"
)

add_custom_target(compiler_managed_l1_dataflow_profiler_examples DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
set(COMPILER_MANAGED_L1_DATAFLOW_PROFILER_TARGETS "${PROGRAMMING_EXAMPLES_TEST_TARGETS}" PARENT_SCOPE)
