add_mlir_library(
  MLIRJeffTranslation
  Deserialize.cpp
  Serialize.cpp
  LINK_LIBS
  MLIRJeff
  MLIRFuncDialect
  MLIRTensorDialect
  CapnProto::capnp
  DISABLE_INSTALL)

target_include_directories(
  MLIRJeffTranslation PRIVATE ${jeff_SOURCE_DIR}/impl/cpp/src/capnp
                              ${jeff_BINARY_DIR}/impl/cpp/src/capnp ${CAPNP_INCLUDE_DIRS})

target_compile_definitions(MLIRJeffTranslation PRIVATE ${CAPNP_DEFINITIONS})
