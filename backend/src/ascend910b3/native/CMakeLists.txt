cmake_minimum_required(VERSION 3.20)
find_package(ASC REQUIRED)
set(CMAKE_ASC_ARCHITECTURES dav-2201)
project(hyperplonk_ascend910b3_math LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_ASC_STANDARD 17)

set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
set(CANN_ARCH_ROOT "${CANN_ROOT}/aarch64-linux")
# Include only real CANN headers/libraries, never devlib runtime substitutes.
set(CANN_INCLUDE_DIRS
    "${CANN_ARCH_ROOT}/include"
    "${CANN_ARCH_ROOT}/asc/include"
    "${CANN_ARCH_ROOT}/ascendc/include/highlevel_api"
    "${CANN_ARCH_ROOT}/ascendc/include/basic_api")
set(CANN_LIBRARY_DIR "${CANN_ARCH_ROOT}/lib64")
foreach(name IN ITEMS tiling_api ascendc_runtime)
    find_library(CANN_${name}_LIBRARY NAMES "lib${name}.a"
        PATHS "${CANN_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
endforeach()
foreach(name IN ITEMS ascendcl register platform unified_dlog graph_base c_sec mmpa profapi runtime)
    find_library(CANN_${name}_LIBRARY NAMES "${name}"
        PATHS "${CANN_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
endforeach()

add_library(hyperplonk_ascend910b3_math STATIC math_kernels.asc math_tiling.cpp)
target_include_directories(hyperplonk_ascend910b3_math PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" ${CANN_INCLUDE_DIRS})
target_link_libraries(hyperplonk_ascend910b3_math PUBLIC
    "${CANN_tiling_api_LIBRARY}" "${CANN_ascendc_runtime_LIBRARY}"
    "${CANN_ascendcl_LIBRARY}" "${CANN_register_LIBRARY}"
    "${CANN_platform_LIBRARY}" "${CANN_unified_dlog_LIBRARY}"
    "${CANN_graph_base_LIBRARY}" "${CANN_c_sec_LIBRARY}" "${CANN_mmpa_LIBRARY}"
    "${CANN_profapi_LIBRARY}" "${CANN_runtime_LIBRARY}" dl m)

include(CTest)
# This target deliberately has no ACL includes, initialization, or linkage.
add_executable(field_math_tests tests/field_math_tests.cpp)
target_include_directories(field_math_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(BUILD_TESTING)
    add_test(NAME field_math_tests COMMAND field_math_tests
        --fixtures "${CMAKE_BINARY_DIR}/fixtures")
endif()

add_executable(native_math_smoke tests/math_smoke.asc tests/runtime_audit.cpp)
target_link_libraries(native_math_smoke PRIVATE hyperplonk_ascend910b3_math)
target_link_options(native_math_smoke PRIVATE
    "-Wl,--wrap=aclrtMemcpy"
    "-Wl,--wrap=aclrtMemcpyAsync"
    "-Wl,--wrap=aclrtSynchronizeStream"
    "-Wl,--wrap=aclrtSynchronizeDevice"
    "-Wl,--wrap=aclrtMalloc"
    "-Wl,--wrap=aclrtFree")
set_target_properties(native_math_smoke PROPERTIES
    BUILD_RPATH "${CANN_LIBRARY_DIR}")
