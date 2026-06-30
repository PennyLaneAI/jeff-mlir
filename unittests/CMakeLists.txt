find_program(UV_EXECUTABLE uv REQUIRED)

# Create a virtual environment
add_test(
    NAME uv_venv
    COMMAND ${UV_EXECUTABLE} venv --allow-existing
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(uv_venv PROPERTIES FIXTURES_SETUP uv_venv_success)

# Install jeff-format
add_test(
    NAME uv_pip_install_jeff_format
    COMMAND
        ${UV_EXECUTABLE} pip install
        "git+https://github.com/unitaryfoundation/jeff.git@9e15d25ddfab27a8f2bd6b21a86d069f9a39d660#subdirectory=impl/py"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(
    uv_pip_install_jeff_format PROPERTIES FIXTURES_SETUP uv_pip_install_jeff_format_success
                                          FIXTURES_REQUIRED uv_venv_success
)

# Generate test cases
add_test(
    NAME generate_test_cases
    COMMAND ${UV_EXECUTABLE} run python ${CMAKE_SOURCE_DIR}/unittests/generate_test_cases.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(
    generate_test_cases PROPERTIES FIXTURES_REQUIRED uv_pip_install_jeff_format_success
                                   FIXTURES_SETUP generate_test_cases_success
)

add_subdirectory(Conversion)
add_subdirectory(IR)
add_subdirectory(Translation)
