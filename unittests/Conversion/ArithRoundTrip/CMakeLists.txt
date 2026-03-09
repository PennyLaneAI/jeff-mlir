set(test_name "jeff-mlir-arith-round-trip-test")

if(NOT TARGET ${test_name})
    add_executable(${test_name} test_arith_round_trip.cpp)

    target_link_libraries(
        ${test_name} PRIVATE GTest::gtest_main ArithToJeff MLIRJeff MLIRJeffTranslation JeffToArith
    )

    target_include_directories(
        ${test_name} PRIVATE ${jeff_SOURCE_DIR}/impl/cpp/src/capnp
                             ${jeff_BINARY_DIR}/impl/cpp/src/capnp
    )

    target_sources(${test_name} PRIVATE ${jeff_SOURCE_DIR}/impl/cpp/src/capnp/jeff.capnp.c++)

    target_compile_definitions(
        ${test_name} PRIVATE TEST_INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../inputs"
    )

    set_target_properties(${test_name} PROPERTIES FOLDER unittests)

    gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60)
endif()
