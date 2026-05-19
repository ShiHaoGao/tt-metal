cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_add_2_integers_in_compute_two_dm_read)

add_executable(metal_example_add_2_integers_in_compute_two_dm_read)
target_sources(metal_example_add_2_integers_in_compute_two_dm_read PRIVATE add_2_integers_in_compute_two_dm_read.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_add_2_integers_in_compute_two_dm_read PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_add_2_integers_in_compute_two_dm_read TT::CommonPCH)
endif()
set_target_properties(
    metal_example_add_2_integers_in_compute_two_dm_read
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/programming_examples/add_2_integers_in_compute_two_dm_read
)
