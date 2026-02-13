add_mlir_library(
  MLIRJeffDeserialize
  Deserialize.cpp
  LINK_LIBS
  MLIRJeff
  MLIRFuncDialect
  MLIRTensorDialect
  CapnProto::capnp
  DISABLE_INSTALL)

target_include_directories(
  MLIRJeffDeserialize PRIVATE ${jeff_SOURCE_DIR}/impl/cpp/src/capnp
                              ${jeff_BINARY_DIR}/impl/cpp/src/capnp ${CAPNP_INCLUDE_DIRS})

target_compile_definitions(MLIRJeffDeserialize PRIVATE ${CAPNP_DEFINITIONS})
