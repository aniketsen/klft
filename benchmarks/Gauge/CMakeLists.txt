add_executable(plaq_bench plaq_bench.cpp)
target_link_libraries(plaq_bench PUBLIC Kokkos::kokkos KTune)

add_executable(wloop_temporal_bench wloop_temporal_bench.cpp)
target_link_libraries(wloop_temporal_bench PUBLIC Kokkos::kokkos KTune)

add_executable(metropolis_bench metropolis_bench.cpp)
target_link_libraries(metropolis_bench PUBLIC Kokkos::kokkos KTune)