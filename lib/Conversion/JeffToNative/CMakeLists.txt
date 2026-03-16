file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
    MLIRJeffToNative
    ${CONVERSION_SOURCES}
    DEPENDS
    JeffToNativeIncGen
    LINK_LIBS
    MLIRArithDialect
    MLIRDialect
    MLIRJeff
    MLIRMathDialect
    MLIRTensorDialect
    MLIRTransforms
    DISABLE_INSTALL
)
