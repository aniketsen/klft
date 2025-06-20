add_executable(metropolis metropolis.cpp)
target_link_libraries(metropolis PUBLIC klft Kokkos::kokkos yaml-cpp)

add_executable(hmc hmc.cpp)
target_link_libraries(hmc PUBLIC klft Kokkos::kokkos yaml-cpp)
