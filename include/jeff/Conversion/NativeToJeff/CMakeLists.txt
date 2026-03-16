set(LLVM_TARGET_DEFINITIONS NativeToJeff.td)
mlir_tablegen(NativeToJeff.h.inc -gen-pass-decls -name NativeToJeff)
add_public_tablegen_target(NativeToJeffIncGen)

if(BUILD_JEFF_MLIR_DOCS)
    add_mlir_doc(NativeToJeff MLIRNativeToJeff Conversions/ -gen-pass-doc)
endif()
