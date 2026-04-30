

#include <getopt.h>
#include <cassert>
#include <cmath>

#include "ActionDensity.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "GaugeInvariance.hpp"
#include "GaugePlaquette.hpp"
#include "TopoCharge.hpp"
#define HLINE "=========================================================\n"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

using namespace klft;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  KTune::initialize();

  {
    using DeviceGaugeFieldType = DeviceGaugeFieldType<4, 2>;
    RNGType rng(1234238120379846981);
    index_t L = 6;

    DeviceGaugeFieldType::type f(L, L, L, L, rng, 1.0);
    GaugeInv<DeviceGaugeFieldType> ginv(f, rng, 1.0);
    ginv.transform();

    real_t E_before_clover = getActionDensity_clover<DeviceGaugeFieldType>(f);
    real_t E_after_clover =
        getActionDensity_clover<DeviceGaugeFieldType>(ginv.field);
    Kokkos::printf(HLINE);
    Kokkos::printf("Action density (clover) before and after gauge inv:\n");
    Kokkos::printf("%e", E_before_clover);
    Kokkos::printf(" -> ");
    Kokkos::printf("%e", E_after_clover);
    Kokkos::printf("\n");
    Kokkos::printf("Difference: %e\n", E_before_clover - E_after_clover);
    Kokkos::printf(HLINE);

    real_t E_before = getActionDensity<DeviceGaugeFieldType>(f);
    real_t E_after = getActionDensity<DeviceGaugeFieldType>(ginv.field);
    Kokkos::printf(HLINE);
    Kokkos::printf("Action density (plaquette) before and after gauge inv:\n");
    Kokkos::printf("%e", E_before);
    Kokkos::printf(" -> ");
    Kokkos::printf("%e", E_after);
    Kokkos::printf("\n");
    Kokkos::printf("Difference: %e\n", E_before - E_after);
    Kokkos::printf(HLINE);

    real_t Plaq_before = GaugePlaquette<4, 2>(f);
    real_t Plaq_after = GaugePlaquette<4, 2>(ginv.field);
    Kokkos::printf(HLINE);
    Kokkos::printf("Plaquette before and after gauge inv:\n");
    Kokkos::printf("%e", Plaq_before);
    Kokkos::printf(" -> ");
    Kokkos::printf("%e", Plaq_after);
    Kokkos::printf("\n");
    Kokkos::printf("Difference: %e\n", Plaq_before - Plaq_after);
    Kokkos::printf(HLINE);

    real_t E_before_rect = getActionDensity_rect<DeviceGaugeFieldType>(f);
    real_t E_after_rect =
        getActionDensity_rect<DeviceGaugeFieldType>(ginv.field);
    Kokkos::printf(HLINE);
    Kokkos::printf("Action density (rectangle) before and after gauge inv:\n");
    Kokkos::printf("%e", E_before_rect);
    Kokkos::printf(" -> ");
    Kokkos::printf("%e", E_after_rect);
    Kokkos::printf("\n");
    Kokkos::printf("Difference: %e\n", E_before_rect - E_after_rect);
    Kokkos::printf(HLINE);
  }
  Kokkos::finalize();
  return 0;
}
