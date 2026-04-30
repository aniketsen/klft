add_executable(metropolis metropolis.cpp)
target_link_libraries(metropolis PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(hmc hmc.cpp)
target_link_libraries(hmc PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(ptbc ptbc.cpp)
target_link_libraries(ptbc PUBLIC klft MPI::MPI_CXX Kokkos::kokkos yaml-cpp KTune)
