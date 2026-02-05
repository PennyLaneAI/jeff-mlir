add_executable(jeff-test test.cpp)

target_link_libraries(
  jeff-test
  PRIVATE MLIRJeff
          MLIRJeffDeserialize
          MLIRControlFlowInterfaces
          MLIRDialect
          MLIRInferTypeOpInterface
          MLIRIR
          MLIRSupport
          LLVMSupport
          CapnProto::capnp
          CapnProto::kj)
