# Explicit supplier publication tool. Building/installing the SDK must never
# run this tool or add supplier firmware compilation to the product cold path.
add_library(native_deployment_build STATIC DeploymentBuild.cpp DeploymentCatalog.cpp PublishedImage.cpp)
target_compile_features(native_deployment_build PRIVATE cxx_std_20)
target_include_directories(native_deployment_build PRIVATE
    ${PROJECT_SOURCE_DIR}/tt_metal ${PROJECT_SOURCE_DIR}/tt_metal/impl
    "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>")
target_link_libraries(native_deployment_build PUBLIC tt_metal PRIVATE yaml-cpp::yaml-cpp)
add_executable(native_deployment_publish
    NativeDeploymentPublish.cpp DeploymentPublishOptions.cpp DeploymentInputs.cpp
    BundlePublication.cpp SourceArchive.cpp)
target_compile_features(native_deployment_publish PRIVATE cxx_std_20)
target_compile_definitions(native_deployment_publish PRIVATE
    TT_NATIVE_FIRMWARE_SDK_ROOT="${PROJECT_SOURCE_DIR}")
target_include_directories(native_deployment_publish PRIVATE
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/impl)
target_link_libraries(native_deployment_publish PRIVATE native_deployment_build
    Metalium::Metal::Hardware nlohmann_json::nlohmann_json)
set_target_properties(native_deployment_publish PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tools)
add_executable(native_firmware_bundle NativeFirmwareBundle.cpp BundlePublication.cpp SourceArchive.cpp)
target_compile_features(native_firmware_bundle PRIVATE cxx_std_20)
target_compile_definitions(native_firmware_bundle PRIVATE
    TT_NATIVE_FIRMWARE_SDK_ROOT="${PROJECT_SOURCE_DIR}")
target_include_directories(native_firmware_bundle PRIVATE
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/impl)
target_link_libraries(native_firmware_bundle PRIVATE
    tt_metal Metalium::Metal::Hardware nlohmann_json::nlohmann_json)
set_target_properties(native_firmware_bundle PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tools)

if(TT_METAL_BUILD_TESTS)
    add_executable(unit_tests_native_deployment_publish
        NativeDeploymentPublishTest.cpp DeploymentInputsTest.cpp
        DeploymentPublishOptions.cpp DeploymentInputs.cpp)
    target_compile_features(unit_tests_native_deployment_publish PRIVATE cxx_std_20)
    target_link_libraries(unit_tests_native_deployment_publish PRIVATE
        GTest::gtest_main nlohmann_json::nlohmann_json tt_metal)
    set_target_properties(unit_tests_native_deployment_publish PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
    add_test(NAME unit_tests_native_deployment_publish
        COMMAND unit_tests_native_deployment_publish)
    add_executable(unit_tests_supplier_image PublishedImageTest.cpp)
    target_compile_features(unit_tests_supplier_image PRIVATE cxx_std_20)
    target_include_directories(unit_tests_supplier_image PRIVATE ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal ${PROJECT_SOURCE_DIR}/tt_metal/impl)
    target_link_libraries(unit_tests_supplier_image PRIVATE native_deployment_build GTest::gtest_main)
    set_target_properties(unit_tests_supplier_image PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
    add_test(NAME unit_tests_supplier_image COMMAND unit_tests_supplier_image)
    add_executable(unit_tests_native_firmware_publication
        BundlePublicationTest.cpp BundlePublication.cpp SourceArchiveTest.cpp SourceArchive.cpp)
    target_compile_features(unit_tests_native_firmware_publication PRIVATE cxx_std_20)
    target_include_directories(unit_tests_native_firmware_publication PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal ${PROJECT_SOURCE_DIR}/tt_metal/impl)
    target_link_libraries(unit_tests_native_firmware_publication PRIVATE
        tt_metal GTest::gtest_main nlohmann_json::nlohmann_json)
    set_target_properties(unit_tests_native_firmware_publication PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
    add_test(NAME unit_tests_native_firmware_publication COMMAND unit_tests_native_firmware_publication)
endif()
