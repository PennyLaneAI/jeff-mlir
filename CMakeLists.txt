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

# Get Jeff
include(FetchContent)
FetchContent_Declare(
  jeff
  GIT_REPOSITORY https://github.com/unitaryfoundation/jeff/
  GIT_TAG main)
FetchContent_MakeAvailable(jeff)

# Get CapnProto
find_package(CapnProto REQUIRED)

# Get GoogleTest
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
set(INSTALL_GTEST
    OFF
    CACHE BOOL "" FORCE)
set(GTEST_VERSION
    1.17.0
    CACHE STRING "Google Test version")
set(GTEST_URL https://github.com/google/googletest/archive/refs/tags/v${GTEST_VERSION}.tar.gz)
FetchContent_Declare(googletest URL ${GTEST_URL} FIND_PACKAGE_ARGS ${GTEST_VERSION} NAMES GTest)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
enable_testing()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(unittests)
