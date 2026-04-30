include(FetchContent)
include(ExternalProject)
set(FETCH_PACKAGES "")

if(BUILD_JEFF_MLIR_TRANSLATION)
    FetchContent_Declare(
        jeff
        GIT_REPOSITORY https://github.com/unitaryfoundation/jeff/
        GIT_TAG jeff-v0.2.0
    )
    list(APPEND FETCH_PACKAGES jeff)

    set(CAPNPROTO_PREFIX ${CMAKE_BINARY_DIR}/_deps/capnproto-install)
    file(MAKE_DIRECTORY ${CAPNPROTO_PREFIX}/include)
    file(MAKE_DIRECTORY ${CAPNPROTO_PREFIX}/lib)

    if(WIN32)
        set(CAPNP_IMPORTED_LIB ${CAPNPROTO_PREFIX}/lib/capnp.lib)
        set(KJ_IMPORTED_LIB ${CAPNPROTO_PREFIX}/lib/kj.lib)
    else()
        set(CAPNP_IMPORTED_LIB
            ${CAPNPROTO_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}capnp${CMAKE_STATIC_LIBRARY_SUFFIX}
        )
        set(KJ_IMPORTED_LIB
            ${CAPNPROTO_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kj${CMAKE_STATIC_LIBRARY_SUFFIX}
        )
    endif()

    ExternalProject_Add(
        capnproto_external
        GIT_REPOSITORY https://github.com/capnproto/capnproto.git
        GIT_TAG v1.3.0
        SOURCE_SUBDIR c++
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CAPNPROTO_PREFIX} -DBUILD_TESTING=OFF
                   -DBUILD_SHARED_LIBS=OFF
        STEP_TARGETS install
        UPDATE_DISCONNECTED ON
        BUILD_BYPRODUCTS ${CAPNP_IMPORTED_LIB} ${KJ_IMPORTED_LIB}
    )

    add_library(capnp_external_lib UNKNOWN IMPORTED)
    set_target_properties(capnp_external_lib PROPERTIES IMPORTED_LOCATION "${CAPNP_IMPORTED_LIB}")
    add_dependencies(capnp_external_lib capnproto_external)

    add_library(kj_external_lib UNKNOWN IMPORTED)
    set_target_properties(kj_external_lib PROPERTIES IMPORTED_LOCATION "${KJ_IMPORTED_LIB}")
    add_dependencies(kj_external_lib capnproto_external)

    add_library(CapnProto::capnp INTERFACE IMPORTED)
    set_target_properties(
        CapnProto::capnp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CAPNPROTO_PREFIX}/include"
                                    INTERFACE_LINK_LIBRARIES "capnp_external_lib;kj_external_lib"
    )
endif()

if(BUILD_JEFF_MLIR_TESTS)
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE
    )
    set(INSTALL_GTEST
        OFF
        CACHE BOOL "" FORCE
    )
    set(GTEST_VERSION
        1.17.0
        CACHE STRING "Google Test version"
    )
    set(GTEST_URL https://github.com/google/googletest/archive/refs/tags/v${GTEST_VERSION}.tar.gz)
    FetchContent_Declare(googletest URL ${GTEST_URL} FIND_PACKAGE_ARGS ${GTEST_VERSION} NAMES GTest)
    list(APPEND FETCH_PACKAGES googletest)
endif()

FetchContent_MakeAvailable(${FETCH_PACKAGES})
