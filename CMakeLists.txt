cmake_minimum_required(VERSION 3.20)
project(OpenRisk VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.4 REQUIRED)

add_library(openrisk INTERFACE)
target_include_directories(openrisk INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${EIGEN3_INCLUDE_DIR}
)

if(MSVC)
    target_compile_options(openrisk INTERFACE /O2 /arch:AVX2 /Oi)
else()
    target_compile_options(openrisk INTERFACE 
        -O3 
        -march=native 
        -ffast-math 
        -Wall 
        -Wextra 
        -Wpedantic
    )
endif()

option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES)
    add_executable(example_garch examples/garch_forecast.cpp)
    target_link_libraries(example_garch PRIVATE openrisk Eigen3::Eigen)
endif()

option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
endif()
