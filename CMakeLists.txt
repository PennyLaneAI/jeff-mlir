cmake_minimum_required(VERSION 3.20...4.2)

project(
  jeff-mlir
  DESCRIPTION "A jeff dialect in MLIR"
  LANGUAGES CXX C)

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
  set(CMAKE_CXX_STANDARD
      17
      CACHE STRING "C++ standard to conform to")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(JEFF_MLIR_MAIN_PROJECT ON)
else()
  set(JEFF_MLIR_MAIN_PROJECT OFF)
endif()

option(BUILD_JEFF_MLIR_DOCS "Build docs for the Jeff MLIR project" ${JEFF_MLIR_MAIN_PROJECT})
option(BUILD_JEFF_MLIR_TESTS "Build tests for the Jeff MLIR project" ${JEFF_MLIR_MAIN_PROJECT})
option(BUILD_JEFF_MLIR_TRANSLATION "Build the translation submodule of the Jeff MLIR project" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "Export compile commands" FORCE)

include(cmake/ExternalDependencies.cmake)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

if(BUILD_JEFF_MLIR_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(unittests)
endif()
