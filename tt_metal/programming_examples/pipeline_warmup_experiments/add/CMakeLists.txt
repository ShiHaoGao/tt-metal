set(PIPELINE_WARMUP_ADD_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_warmup_add.cpp
)

CREATE_PGM_EXAMPLES_EXE("${PIPELINE_WARMUP_ADD_SRCS}" "pipeline_warmup_experiments")
