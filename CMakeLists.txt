cmake_minimum_required(VERSION 3.15...3.29)
project(klft LANGUAGES CXX)
find_package(Kokkos REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB LIB_SOURCES ${CMAKE_SOURCE_DIR}/lib/*.cpp)
add_library(klft ${LIB_SOURCES})
target_compile_features(klft PRIVATE cxx_std_17)
target_link_libraries(klft Kokkos::kokkos)

add_executable(metropolis metropolis.cpp)
target_compile_features(metropolis PRIVATE cxx_std_17)
target_link_libraries(metropolis klft Kokkos::kokkos)

add_executable(hmc hmc.cpp)
target_compile_features(hmc PRIVATE cxx_std_17)
target_link_libraries(hmc klft Kokkos::kokkos)
