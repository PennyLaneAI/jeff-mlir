add_executable(jeff-mlir-deserialize test_deserialize.cpp)

target_link_libraries(jeff-mlir-deserialize PRIVATE MLIRJeff MLIRJeffTranslation)

set(test_name "jeff-mlir-round-trip-test")

if(NOT TARGET ${test_name})
  add_executable(${test_name} test_round_trip.cpp)

  target_link_libraries(${test_name} PRIVATE GTest::gmock GTest::gtest_main MLIRJeff
                                             MLIRJeffTranslation)

  target_include_directories(${test_name} PRIVATE ${jeff_SOURCE_DIR}/impl/cpp/src/capnp
                                                  ${jeff_BINARY_DIR}/impl/cpp/src/capnp)

  target_sources(${test_name} PRIVATE ${jeff_SOURCE_DIR}/impl/cpp/src/capnp/jeff.capnp.c++)

  target_compile_definitions(${test_name}
                             PRIVATE TEST_INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/inputs")

  set_target_properties(${test_name} PROPERTIES FOLDER unittests)

  gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60)
endif()
