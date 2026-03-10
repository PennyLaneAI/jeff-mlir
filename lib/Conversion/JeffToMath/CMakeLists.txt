file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
    JeffToMath
    ${CONVERSION_SOURCES}
    DEPENDS
    JeffToMathIncGen
    LINK_LIBS
    MLIRDialect
    MLIRJeff
    MLIRMathDialect
    MLIRTransforms
    DISABLE_INSTALL
)
