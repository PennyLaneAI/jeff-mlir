set(LLVM_TARGET_DEFINITIONS JeffToArith.td)
mlir_tablegen(JeffToArith.h.inc -gen-pass-decls -name JeffToArith)
add_public_tablegen_target(JeffToArithIncGen)

if(BUILD_JEFF_MLIR_DOCS)
    add_mlir_doc(JeffToArith MLIRJeffToArith Conversions/ -gen-pass-doc)
endif()
