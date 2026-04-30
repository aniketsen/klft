#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/DiracOperator.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/SpinorFieldLinAlg.hpp"
#include "../../include/WilsonDiracOperator.hpp"
#include "../../include/klft.hpp"
#define HLINE "=========================================================\n"

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
  Kokkos::initialize(argc, argv);
  KTune::initialize();
  int RETURNVALUE = 0;
  {
    constexpr int count = 1000;
    setVerbosity(1);
    setTuning(1);
    printf("%i", KLFT_TUNING);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing hermiticity =\n");
    index_t L0 = 64, L1 = 64, L2 = 64, L3 = 64;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    diracParams params(0.156);
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<2, 4, complex_t> u_eo(L0 / 2, L1, L2, L3, random_pool, 0,
                                            1.0 / 1.41);
    deviceSpinorField<2, 4, complex_t> u_eo_out(L0 / 2, L1, L2, L3, random_pool,
                                                0, 1.0 / 1.41);
    deviceSpinorField<2, 4, complex_t> u(L0, L1, L2, L3, random_pool, 0,
                                         1.0 / 1.41);
    deviceSpinorField<2, 4, complex_t> Mu(L0, L1, L2, L3, 0);
    deviceSpinorField<2, 4, complex_t> temp(L0, L1, L2, L3, 0);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 2> gauge(L0, L1, L2, L3, random_pool, 1);
    printf("Instantiate DiracOperator...\n");
    WilsonDiracOperator<DeviceSpinorFieldType<4, 2, 4, complex_t>,
                        DeviceGaugeFieldType<4, 2, complex_t>>
        D(gauge, params);
    D.init(u.dimensions);
    EOWilsonDiracOperator<
        DeviceSpinorFieldType<4, 2, 4, complex_t, SpinorFieldKind::Standard,
                              SpinorFieldLayout::Checkerboard>,
        DeviceGaugeFieldType<4, 2, complex_t>>
        D_eo(gauge, params);
    D_eo.init(u_eo.dimensions);
    printf("Apply DiracOperator...\n");
    DeviceSpinorFieldType<4, 2, 4, complex_t, SpinorFieldKind::Standard,
                          SpinorFieldLayout::Checkerboard>::type
        u_norm_out(L0 / 2, L1, L2, L3, 0);
    DeviceSpinorFieldType<4, 2, 4, complex_t, SpinorFieldKind::Standard,
                          SpinorFieldLayout::Checkerboard>::type
        u_axpy_out(L0 / 2, L1, L2, L3, 0);
    DeviceSpinorFieldType<4, 2, 4, complex_t, SpinorFieldKind::Standard,
                          SpinorFieldLayout::Checkerboard>::type
        u_axpy_out2(L0 / 2, L1, L2, L3, 0);
    printf("Launching Kernels for tuning...\n");
    D.template apply<Tags::TagD>(u, u_norm_out);
    printf("Tuning finished for normal dirac operator");
    D_eo.template apply<Tags::TagHeo>(u_eo, u_eo_out);
    printf("Tuning finished for eo dirac operator");
    printf("Tuning done, now timing...\n");
    Kokkos::Timer timer;
    real_t diracTime = std::numeric_limits<real_t>::max();
    for (size_t i = 0; i < count; i++) {
      D.template apply<Tags::TagD>(u, u_norm_out);
    }
    auto diracTime1 = std::min(diracTime, timer.seconds());

    printf("D Kernel Time:     %11.4e s\n", diracTime1 / count);
    printf("D_normal total time: %11.4e s\n", diracTime1);
    diracTime = std::numeric_limits<real_t>::max();
    D_eo.init(u_eo.dimensions);
    Kokkos::Timer time2;
    for (size_t i = 0; i < count; i++) {
      D_eo.template apply<Tags::TagHeo>(u, u_norm_out);
    }
    auto diracTime2 = std::min(diracTime, time2.seconds());
    printf("D_eo Heo Kernel Time:     %11.4e s\n", diracTime2 / count);
    printf("D_eo Heo_normal total time: %11.4e s\n", diracTime2);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}