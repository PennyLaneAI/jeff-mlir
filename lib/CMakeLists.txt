add_subdirectory(IR)
add_subdirectory(opt)

if(BUILD_JEFF_MLIR_TRANSLATION)
  add_subdirectory(Translation)
endif()
