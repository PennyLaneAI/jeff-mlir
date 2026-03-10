file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
    JeffToTensor
    ${CONVERSION_SOURCES}
    DEPENDS
    JeffToTensorIncGen
    LINK_LIBS
    MLIRDialect
    MLIRJeff
    MLIRTensorDialect
    MLIRTransforms
    DISABLE_INSTALL
)
