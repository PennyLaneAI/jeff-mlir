file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_library(
    MathToJeff
    ${CONVERSION_SOURCES}
    DEPENDS
    MathToJeffIncGen
    LINK_LIBS
    MLIRDialect
    MLIRJeff
    MLIRMathDialect
    MLIRTransforms
    DISABLE_INSTALL
)
