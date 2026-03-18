add_subdirectory(Conversion)
add_subdirectory(IR)

if(BUILD_JEFF_OPT)
    add_subdirectory(opt)
endif()

if(BUILD_JEFF_MLIR_TRANSLATION)
    add_subdirectory(Translation)
endif()
