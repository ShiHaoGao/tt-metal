set(PIPELINE_WARMUP_MATMUL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_warmup_matmul.cpp
)

CREATE_PGM_EXAMPLES_EXE("${PIPELINE_WARMUP_MATMUL_SRCS}" "pipeline_warmup_experiments")
