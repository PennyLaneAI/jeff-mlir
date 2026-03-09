set(LLVM_TARGET_DEFINITIONS ArithToJeff.td)
mlir_tablegen(ArithToJeff.h.inc -gen-pass-decls -name ArithToJeff)
add_public_tablegen_target(ArithToJeffIncGen)

if(BUILD_JEFF_MLIR_DOCS)
    add_mlir_doc(ArithToJeff MLIRArithToJeff Conversions/ -gen-pass-doc)
endif()
