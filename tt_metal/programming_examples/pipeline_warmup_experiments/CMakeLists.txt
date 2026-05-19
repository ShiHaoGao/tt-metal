add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/add)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/matmul)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sdpa)

add_custom_target(
    pipeline_warmup_experiments
    DEPENDS
        pipeline_warmup_add
        pipeline_warmup_matmul
        pipeline_warmup_sdpa
)
