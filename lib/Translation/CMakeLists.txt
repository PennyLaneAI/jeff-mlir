add_mlir_library(
    MLIRJeffTranslation
    Deserialize.cpp
    Serialize.cpp
    LINK_LIBS
    MLIRJeff
    MLIRFuncDialect
    CapnProto::capnp
    DISABLE_INSTALL
)

add_dependencies(MLIRJeffTranslation capnproto_external-install)
if(TARGET obj.MLIRJeffTranslation)
    add_dependencies(obj.MLIRJeffTranslation capnproto_external-install)
endif()

target_include_directories(
    MLIRJeffTranslation PUBLIC ${jeff_SOURCE_DIR}/impl/cpp/src/capnp
                               ${jeff_BINARY_DIR}/impl/cpp/src/capnp
)

target_sources(MLIRJeffTranslation PUBLIC ${jeff_SOURCE_DIR}/impl/cpp/src/capnp/jeff.capnp.c++)
