

#include <getopt.h>
#include <cassert>
#include <cmath>

#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "GaugeInvariance.hpp"
#include "IndexHelper.hpp"
#include "TopoCharge.hpp"
#define HLINE "=========================================================\n"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

using namespace klft;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  KTune::initialize();

  {
    RNGType rng(1234238120379846981);
    index_t L = 6;

    deviceGaugeField<4, 2> f(L, L, L, L, rng, 1.0);
    GaugeInv<DeviceGaugeFieldType<4, 2>> ginv(f, rng, 1.0);
    ginv.transform();
    real_t topo_Q_before =
        get_topological_charge<DeviceGaugeFieldType<4, 2>>(f);
    real_t topo_Q_after =
        get_topological_charge<DeviceGaugeFieldType<4, 2>>(ginv.field);

    Kokkos::printf(HLINE);
    Kokkos::printf("%e", topo_Q_before);
    Kokkos::printf(" -> ");
    Kokkos::printf("%e", topo_Q_after);
    Kokkos::printf("\n");
    Kokkos::printf("Difference: %e\n", topo_Q_before - topo_Q_after);
    Kokkos::printf(HLINE);

    real_t topo_Q_imp_before =
        get_topological_charge_improved<DeviceGaugeFieldType<4, 2>>(f,
                                                                    -1.0 / 12);
    real_t topo_Q_imp_after =
        get_topological_charge_improved<DeviceGaugeFieldType<4, 2>>(ginv.field,
                                                                    -1.0 / 12);

    Kokkos::printf(HLINE);
    Kokkos::printf("%e", topo_Q_imp_before);
    Kokkos::printf(" -> ");
    Kokkos::printf("%e", topo_Q_imp_after);
    Kokkos::printf("\n");
    Kokkos::printf("Difference: %e\n", topo_Q_imp_before - topo_Q_imp_after);
    Kokkos::printf(HLINE);
  }
  Kokkos::finalize();
  return 0;
}
