# add_executable(reversability_test reversability_test.cpp)
# target_link_libraries(reversability_test PUBLIC klft Kokkos::kokkos yaml-cpp)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")

# add_executable(wilsonflow_test wilsonflow_test.cpp)
# target_link_libraries(wilsonflow_test PUBLIC klft Kokkos::kokkos yaml-cpp)

add_executable(wilsonflow_sp_test wilsonflow_sp_test.cpp)
target_link_libraries(wilsonflow_sp_test PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(topo_improved_test topo_improved_test.cpp)
target_link_libraries(topo_improved_test PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(wilsonflow_improv_testing wilsonflow_improv_testing.cpp)
target_link_libraries(wilsonflow_improv_testing PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(gauge_inv_test_Topo gauge_inv_test_Topo.cpp)
target_link_libraries(gauge_inv_test_Topo PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(gauge_inv_test_action_density gauge_inv_test_action_density.cpp)
target_link_libraries(gauge_inv_test_action_density PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(gauge_inv_test_staples gauge_inv_test_staples.cpp)
target_link_libraries(gauge_inv_test_staples PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(adaptive_wilson_flow_verify adaptive_wilson_flow_verify.cpp)
target_link_libraries(adaptive_wilson_flow_verify PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

add_executable(test_IO test_IO.cpp)
target_link_libraries(test_IO PUBLIC klft Kokkos::kokkos yaml-cpp KTune)

# build the Grouptests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Group ${CMAKE_CURRENT_BINARY_DIR}/Group)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Fermions ${CMAKE_CURRENT_BINARY_DIR}/Fermions)
