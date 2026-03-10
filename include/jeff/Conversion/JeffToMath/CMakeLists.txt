set(LLVM_TARGET_DEFINITIONS JeffToMath.td)
mlir_tablegen(JeffToMath.h.inc -gen-pass-decls -name JeffToMath)
add_public_tablegen_target(JeffToMathIncGen)

if(BUILD_JEFF_MLIR_DOCS)
    add_mlir_doc(JeffToMath MLIRJeffToMath Conversions/ -gen-pass-doc)
endif()
