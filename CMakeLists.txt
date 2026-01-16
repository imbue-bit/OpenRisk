cmake_minimum_required(VERSION 3.20)
project(OpenRisk VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.4 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system) 

add_library(openrisk STATIC
    src/core/optimization.cpp
    src/time_series/garch.cpp
    src/crash/lppl.cpp
)

target_include_directories(openrisk PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(openrisk PUBLIC Eigen3::Eigen Boost::boost)

if(MSVC)
    target_compile_options(openrisk PRIVATE /O2 /arch:AVX2 /Oi)
else()
    target_compile_options(openrisk PRIVATE -O3 -march=native -ffast-math -Wall -Wextra)
endif()

option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES)
    add_executable(market_crash_analysis examples/market_crash_analysis.cpp)
    target_link_libraries(market_crash_analysis PRIVATE openrisk Eigen3::Eigen Boost::boost)
endif()
