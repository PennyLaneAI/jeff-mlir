set(LLVM_TARGET_DEFINITIONS JeffToTensor.td)
mlir_tablegen(JeffToTensor.h.inc -gen-pass-decls -name JeffToTensor)
add_public_tablegen_target(JeffToTensorIncGen)

if(BUILD_JEFF_MLIR_DOCS)
    add_mlir_doc(JeffToTensor MLIRJeffToTensor Conversions/ -gen-pass-doc)
endif()
