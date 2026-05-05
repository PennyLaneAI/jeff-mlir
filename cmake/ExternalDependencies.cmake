include(FetchContent)

set(FETCH_PACKAGES "")

if(BUILD_JEFF_MLIR_TRANSLATION)
    FetchContent_Declare(
        jeff
        GIT_REPOSITORY https://github.com/unitaryfoundation/jeff/
        GIT_TAG jeff-v0.2.0
    )
    list(APPEND FETCH_PACKAGES jeff)

    if(WIN32)
        set(WITH_FIBERS
            OFF
            CACHE
                BOOL
                "Disable fiber support on Windows to avoid a build error due to disabled exceptions"
                FORCE
        )
    endif()
    FetchContent_Declare(
        capnproto
        GIT_REPOSITORY https://github.com/capnproto/capnproto.git
        GIT_TAG v1.3.0
        PATCH_COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> patch --forward -p1 -i
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/capnproto-disable-tests.patch
    )
    list(APPEND FETCH_PACKAGES capnproto)
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
