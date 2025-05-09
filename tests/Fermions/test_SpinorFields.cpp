// test_deviceSpinorField.cpp
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>

// Include the header(s) that define the deviceSpinorField classes.
// For example, if you have one header that collects all these definitions:
// #include "deviceSpinorField.hpp"
//
// Otherwise, include each header as needed:
#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/klft.hpp"

// We'll assume that our classes are in the klft namespace.
using namespace klft;

int main(int argc, char* argv[]) {
  // Initialize Kokkos.
  Kokkos::initialize(argc, argv);
  {
    setVerbosity(5);
    std::cout << (KLFT_VERBOSITY);
    std::cout << "\n=== Testing deviceSpinorField  ===\n";
    // Dimensions for 4D field:
    index_t L0 = 8, L1 = 8, L2 = 8, L3 = 8;
    // Set an initial complex value (e.g., identity type if that makes sense,
    // here use (1,0))
    complex_t init_val(1.0, 0.0);

    // Instantiate the spinor field with Nc = 3, DimRep=4 (for example)
    deviceSpinorField<3, 4> spin(L0, L1, L2, L3, init_val);

    // Launch a parallel_for to print one field element for mu = 0
    Kokkos::fence();
  }
  Kokkos::finalize();
  return 0;
}