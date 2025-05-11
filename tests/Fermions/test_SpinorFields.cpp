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
#include "../../include/SpinorFieldHelper.hpp"
#include "../../include/klft.hpp"

// We'll assume that our classes are in the klft namespace.
using namespace klft;
template <size_t Nc, size_t Nd>
void print_spinor(const Spinor<Nc, Nd>& s, const char* name = "Spinor") {
  printf("%s:\n", name);
  for (size_t c = 0; c < Nc; ++c) {
    printf("  Color %zu:\n", c);
    for (size_t d = 0; d < Nd; ++d) {
      double re = s[c][d].real();
      double im = s[c][d].imag();
      printf("    [%zu] = (% .6f, % .6f i)\n", d, re, im);
    }
  }
}

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
    Kokkos::fence();
    print_spinor(spin(0, 0, 0, 0));
    std::cout << "\n=== Testing deviceSpinorField with random normal "
                 "distributed values  ===\n";

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    deviceSpinorField<3, 4> spinrand(L0, L1, L2, L3, random_pool, 0,
                                     1.0 / 1.41);
    print_spinor(spinrand(0, 0, 0, 0));
    Kokkos::fence();
    printf("\n=== Testing Spinor Dot Product  ===\n");
    auto val = spinor_dot_product<4, 3, 4>(spin, spinrand);
    Kokkos::fence();
    printf("% .6f+ % .6f i\n", val.real(), val.imag());
  }
  Kokkos::finalize();
  return 0;
}