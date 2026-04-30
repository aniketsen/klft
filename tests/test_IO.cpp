#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"

#include "../include/SpinorField.hpp"
#include "../include/SpinorFieldLinAlg.hpp"
#include "../include/WilsonDiracOperator.hpp"
#include "../include/klft.hpp"
#include "GaugePlaquette.hpp"

#define HLINE "=========================================================\n"

using namespace klft;
template <size_t Nc, size_t Nd>
void print_spinor(const Spinor<Nc, Nd>& s, const char* name = "Spinor") {
  printf("%s:\n", name);
  for (size_t c = 0; c < Nc; ++c) {
    printf("  Color %zu:\n", c);
    for (size_t d = 0; d < Nd; ++d) {
      Kokkos::printf("    [%zu] = (% .6f, % .6f i)\n", d, s[c][d].real(),
                     s[c][d].imag());
    }
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  KTune::initialize();

  int RETURNVALUE = 0;
  {
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing IO  ===\n");

    index_t L0 = 8, L1 = 8, L2 = 8, L3 = 8;
    diracParams params(-0.5);
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 2> gauge(L0, L1, L2, L3, random_pool, 1);
    auto plaq = GaugePlaquette<4, 2>(gauge);
    printf("Plaquette: %.21f\n", plaq);
    printf("Store Gauge Config\n");
    // gauge_U1.save("gauge_U1.dat");
    printf("Load Gauge Config\n");
    deviceGaugeField<4, 2> gauge_load(L0, L1, L2, L3,
                                      std::string("step_5_gaugeconfig.txt"));
    auto plaq1 = GaugePlaquette<4, 2>(gauge_load, true);
    printf("Plaquette: %.21f\n", plaq1);
    if (std::abs(plaq - plaq1) > 1e-14) {
      printf("Error: plaquette differs after load/save by %.21f \n",
             std::abs(plaq - plaq1));
      RETURNVALUE++;
    } else {
      printf("Passed load/save test with plaquette difference of %.21f \n",
             std::abs(plaq - plaq1));
    }
  }

  Kokkos::finalize();
  printf(HLINE);
  printf("%i Errors durring Testing\n", RETURNVALUE);
  printf(HLINE);
  RETURNVALUE = !(RETURNVALUE == 0);
  return RETURNVALUE;
}
