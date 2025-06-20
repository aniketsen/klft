add_library(klft STATIC 
  Metropolis.cpp
  HMC_exec.cpp)
target_link_libraries(klft PUBLIC Kokkos::kokkos yaml-cpp)
