set(LLVM_TARGET_DEFINITIONS JeffToNative.td)
mlir_tablegen(JeffToNative.h.inc -gen-pass-decls -name JeffToNative)
add_public_tablegen_target(JeffToNativeIncGen)

if(BUILD_JEFF_MLIR_DOCS)
    add_mlir_doc(JeffToNative MLIRJeffToNative Conversions/ -gen-pass-doc)
endif()
