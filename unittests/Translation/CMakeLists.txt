set(test_name "jeff-mlir-round-trip-test")

if(NOT TARGET ${test_name})
    if(NOT BUILD_JEFF_MLIR_TRANSLATION)
        message(WARNING "Cannot build ${test_name} without translation module. Skipping test.")
        return()
    endif()

    add_executable(${test_name} test_round_trip.cpp)

    target_link_libraries(${test_name} PRIVATE GTest::gtest_main MLIRJeff MLIRJeffTranslation)

    target_compile_definitions(
        ${test_name} PRIVATE TEST_INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs"
    )

    set_target_properties(${test_name} PROPERTIES FOLDER unittests)

    gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60)
endif()
