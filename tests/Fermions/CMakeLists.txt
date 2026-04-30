add_executable(test_gamma_matrices test_gamma_matricies.cpp)
target_link_libraries(test_gamma_matrices PUBLIC Kokkos::kokkos KTune)

# add_executable(test_spinor test_spinor.cpp)
# target_link_libraries(test_spinor PUBLIC Kokkos::kokkos)

# add_executable(test_deviceSpinorField test_SpinorFields.cpp)
# target_link_libraries(test_deviceSpinorField PUBLIC Kokkos::kokkos)

add_executable(test_DiracOperator test_DiracOperator.cpp)
target_link_libraries(test_DiracOperator PUBLIC Kokkos::kokkos KTune)

add_executable(test_GDircOperator test_GDircOperator.cpp)
target_link_libraries(test_GDircOperator PUBLIC Kokkos::kokkos KTune)

add_executable(test_Solver test_Solver.cpp)
target_link_libraries(test_Solver PUBLIC Kokkos::kokkos KTune)

add_executable(test_Solver_EO test_Solver_EO.cpp)
target_link_libraries(test_Solver_EO PUBLIC Kokkos::kokkos KTune)

add_executable(test_spam test_spam.cpp)
target_link_libraries(test_spam PUBLIC Kokkos::kokkos KTune)