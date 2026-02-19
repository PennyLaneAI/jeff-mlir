get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIROptLib MLIRJeff MLIRFuncDialect)

add_llvm_executable(jeff-opt jeff-opt.cpp DEPENDS ${LIBS})
target_link_libraries(jeff-opt PRIVATE ${LIBS})
llvm_update_compile_flags(jeff-opt)
