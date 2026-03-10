file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
    ArithToJeff
    ${CONVERSION_SOURCES}
    DEPENDS
    ArithToJeffIncGen
    LINK_LIBS
    MLIRArithDialect
    MLIRDialect
    MLIRJeff
    MLIRTransforms
    DISABLE_INSTALL
)
