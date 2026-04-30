

#include <getopt.h>
#include <cassert>
#include <cmath>

#include "ActionDensity.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeAction.hpp"
#include "GaugeField.hpp"
#include "GaugeInvariance.hpp"
#include "GaugePlaquette.hpp"
#include "Kokkos_Macros.hpp"
#include "TopoCharge.hpp"
#define HLINE "=========================================================\n"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

using namespace klft;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  KTune::initialize();

  {
    using DeviceGaugeFieldType = DeviceGaugeFieldType<4, 2>;

    RNGType rng(12342381379846981);
    index_t L = 4;

    DeviceGaugeFieldType::type f(L, L, L, L, rng, 1.0);
    GaugeInv<DeviceGaugeFieldType> ginv(f, rng, 1.0);
    DeviceGaugeFieldType::type stap(f.field);
    ginv.transform();
    DeviceGaugeFieldType::type tstap(ginv.field.field);

    using FieldType = typename DeviceScalarFieldType<4>::type;
    FieldType E1(f.dimensions, real_t(0.0));
    FieldType E2(f.dimensions, real_t(0.0));

    KTune::parallel_for(
        "staple_ginv_test_multiply", Policy<4>(IndexArray<4>{0}, f.dimensions),
        KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3) {
          real_t local_E1 = 0;
          for (int mu = 0; mu < 4; ++mu) {
            auto tmp = traceT(
                f(i0, i1, i2, i3, mu) *
                stap.staple(
                    IndexArray<4>{static_cast<int>(i0), static_cast<int>(i1),
                                  static_cast<int>(i2), static_cast<int>(i3)},
                    mu));
            local_E1 += tr<2>(tmp, tmp);
          }
          E1(i0, i1, i2, i3) = local_E1;
        });

    KTune::parallel_for(
        "staple_ginv_test_multiply2", Policy<4>(IndexArray<4>{0}, f.dimensions),
        KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3) {
          real_t local_E2 = 0;
          for (int mu = 0; mu < 4; ++mu) {
            auto tmp = traceT(
                ginv.field(i0, i1, i2, i3, mu) *
                tstap.staple(
                    IndexArray<4>{static_cast<int>(i0), static_cast<int>(i1),
                                  static_cast<int>(i2), static_cast<int>(i3)},
                    mu));
            local_E2 += tr<2>(tmp, tmp);
          }
          E2(i0, i1, i2, i3) = local_E2;
        });

    real_t fs_before = E1.sum();
    real_t fs_after = E2.sum();
    Kokkos::printf(HLINE);
    Kokkos::printf("field * staple sum before and after gauge inv:\n");
    Kokkos::printf("%e", fs_before);
    Kokkos::printf(" -> ");
    Kokkos::printf("%e", fs_after);
    Kokkos::printf("\n");
    Kokkos::printf("Difference: %e\n", fs_before - fs_after);
    Kokkos::printf(HLINE);

    KTune::parallel_for(
        "staple_ginv_test_multiply", Policy<4>(IndexArray<4>{0}, f.dimensions),
        KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3) {
          real_t local_E1 = 0;
          for (int mu = 0; mu < 4; ++mu) {
            auto tmp = traceT(
                f(i0, i1, i2, i3, mu) *
                stap.staple_rect(
                    IndexArray<4>{static_cast<int>(i0), static_cast<int>(i1),
                                  static_cast<int>(i2), static_cast<int>(i3)},
                    mu));
            local_E1 += tr<2>(tmp, tmp);
          }
          E1(i0, i1, i2, i3) = local_E1;
        });

    KTune::parallel_for(
        "staple_ginv_test_multiply2", Policy<4>(IndexArray<4>{0}, f.dimensions),
        KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3) {
          real_t local_E2 = 0;
          for (int mu = 0; mu < 4; ++mu) {
            auto tmp = traceT(
                ginv.field(i0, i1, i2, i3, mu) *
                tstap.staple_rect(
                    IndexArray<4>{static_cast<int>(i0), static_cast<int>(i1),
                                  static_cast<int>(i2), static_cast<int>(i3)},
                    mu));
            local_E2 += tr<2>(tmp, tmp);
          }
          E2(i0, i1, i2, i3) = local_E2;
        });

    real_t fs_rect_before = E1.sum();
    real_t fs_rect_after = E2.sum();
    Kokkos::printf(HLINE);
    Kokkos::printf("field * staple_rect sum before and after gauge inv:\n");
    Kokkos::printf("%e", fs_rect_before);
    Kokkos::printf(" -> ");
    Kokkos::printf("%e", fs_rect_after);
    Kokkos::printf("\n");
    Kokkos::printf("Difference: %e\n", fs_rect_before - fs_rect_after);
    Kokkos::printf(HLINE);
  }
  Kokkos::finalize();
  return 0;
}
