#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/SpinorFieldHelper.hpp"
#include "../../include/klft.hpp"
#include "../../include/DiracOperator.hpp"
#include "../../include/GammaMatrix.hpp"


using namespace klft;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
  {
    setVerbosity(5);
    printf("%i",KLFT_VERBOSITY);
  printf("\n=== Testing DiracOperator SU(3)  ===\n");
  index_t L0 = 8, L1 = 8, L2 = 8, L3 = 8;
  auto gammas = get_gammas<4>();
  printf("Lattice Dimension %i x%i x%ix%i", L0, L1, L2, L3);
  printf("Genrate SpinorFields...\n");

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  deviceSpinorField<3, 4> s1(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
  deviceSpinorField<3, 4> s2(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);

  printf("Generating Random Gauge Config\n");
    deviceGaugeField<4,3> gauge(L0,L1,L2,L3,random_pool, 1);

    printf("Apply DiracOperaotr\n");

    deviceSpinorField<3, 4> spD = apply_D<4,3,4>(s1,gauge,gammas,-0.5);
  }
  Kokkos::finalize();

  return 0;
}
