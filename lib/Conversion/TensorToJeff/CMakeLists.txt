file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
    TensorToJeff
    ${CONVERSION_SOURCES}
    DEPENDS
    TensorToJeffIncGen
    LINK_LIBS
    MLIRArithDialect
    MLIRDialect
    MLIRJeff
    MLIRTensorDialect
    MLIRTransforms
    DISABLE_INSTALL
)
