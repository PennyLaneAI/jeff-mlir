set(LLVM_TARGET_DEFINITIONS TensorToJeff.td)
mlir_tablegen(TensorToJeff.h.inc -gen-pass-decls -name TensorToJeff)
add_public_tablegen_target(TensorToJeffIncGen)

if(BUILD_JEFF_MLIR_DOCS)
    add_mlir_doc(TensorToJeff MLIRTensorToJeff Conversions/ -gen-pass-doc)
endif()
