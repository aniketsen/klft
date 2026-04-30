add_executable(test_SUNAdj_conv test_SUNAdj_conv.cpp)
target_link_libraries(test_SUNAdj_conv PUBLIC Kokkos::kokkos KTune)
