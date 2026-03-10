file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
    JeffToArith
    ${CONVERSION_SOURCES}
    DEPENDS
    JeffToArithIncGen
    LINK_LIBS
    MLIRArithDialect
    MLIRDialect
    MLIRJeff
    MLIRTransforms
    DISABLE_INSTALL
)
