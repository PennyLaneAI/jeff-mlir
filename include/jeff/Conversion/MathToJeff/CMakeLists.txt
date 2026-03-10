set(LLVM_TARGET_DEFINITIONS MathToJeff.td)
mlir_tablegen(MathToJeff.h.inc -gen-pass-decls -name MathToJeff)
add_public_tablegen_target(MathToJeffIncGen)

if(BUILD_JEFF_MLIR_DOCS)
    add_mlir_doc(MathToJeff MLIRMathToJeff Conversions/ -gen-pass-doc)
endif()
