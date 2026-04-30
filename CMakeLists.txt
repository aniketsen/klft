cmake_minimum_required(VERSION 3.21.0)
project(klft VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

# Honor already set CMAKE_CXX_STANDARD.
# Otherwise default to C++17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)


#
# Kokkos
#
# General Kokkos settings
set(Kokkos_ENABLE_COMPILER_WARNINGS "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_COMPILER_WARNINGS" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)

find_package(MPI REQUIRED COMPONENTS CXX)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/kokkos/kokkos ${CMAKE_CURRENT_BINARY_DIR}/kokkos)

#
# yaml-cpp
#
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/yaml-cpp ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp)

#
# ktune
#
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ktune ${CMAKE_CURRENT_BINARY_DIR}/ktune)

include_directories(${CMAKE_SOURCE_DIR}/include)




# build the library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR}/lib)

# build the binaries
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/binaries ${CMAKE_CURRENT_BINARY_DIR}/binaries)

# build the benchmarks
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks ${CMAKE_CURRENT_BINARY_DIR}/benchmarks)

# build the tests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR}/tests)

