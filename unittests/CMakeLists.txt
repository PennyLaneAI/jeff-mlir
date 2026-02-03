add_executable(jeff-test test.cpp)

target_link_libraries(
  jeff-test
  PRIVATE MLIRJeff
          MLIRJeffDeserialize
          MLIRIR
          MLIRSupport
          MLIRDialect
          MLIRInferTypeOpInterface
          MLIRControlFlowInterfaces
          LLVMSupport
          CapnProto::capnp
          CapnProto::kj)
