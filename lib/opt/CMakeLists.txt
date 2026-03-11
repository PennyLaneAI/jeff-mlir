get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIROptLib MLIRJeff MLIRFuncDialect)

# On LLVM 22+, the dialect and pass registration also require linking against the
# `MLIRRegisterAllDialects` and `MLIRRegisterAllPasses` targets
if(TARGET MLIRRegisterAllDialects)
    list(APPEND LIBS MLIRRegisterAllDialects)
endif()
if(TARGET MLIRRegisterAllPasses)
    list(APPEND LIBS MLIRRegisterAllPasses)
endif()

add_llvm_executable(jeff-opt jeff-opt.cpp DEPENDS ${LIBS})
target_link_libraries(jeff-opt PRIVATE ${LIBS})
llvm_update_compile_flags(jeff-opt)
