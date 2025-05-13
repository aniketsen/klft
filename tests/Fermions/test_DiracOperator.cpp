#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/DiracOperator.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/SpinorFieldLinAlg.hpp"
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
  int RETURNVALUE = 0;
  {
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing hermiticity =\n");
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<3, 4> u(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<3, 4> v(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    real_t norm = spinor_norm<4, 3, 4>(u);
    norm *= spinor_norm<4, 3, 4>(v);
    norm = Kokkos::sqrt(norm);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 3> gauge(L0, L1, L2, L3, random_pool, 1);

    printf("Apply DiracOperator...\n");

    deviceSpinorField<3, 4> Mu =
        apply_HD<4, 3, 4>(u, gauge, gammas, gamma5, -0.5);
    deviceSpinorField<3, 4> Mv =
        apply_HD<4, 3, 4>(v, gauge, gammas, gamma5, -0.5);
    // deviceSpinorField<3, 4> Mu = apply_D<4, 3, 4>(u, gauge, gammas, -0.5);
    // deviceSpinorField<3, 4> Mv = apply_D<4, 3, 4>(v, gauge, gammas, -0.5);

    printf("Calculate Scalarproducts...\n");
    auto r1 = spinor_dot_product<4, 3, 4>(u, Mv);
    auto r2 = spinor_dot_product<4, 3, 4>(Mu, v);

    auto r = r1 - r2;

    real_t r3 = Kokkos::sqrt(r.real() * r.real() + r.imag() * r.imag());
    r3 /= norm;

    if (r3 < 1e-14) {
      printf("Passed hermiticity test with %.21f \n", r3);
    } else {
      printf("Error: didn't pass hermiticity test with %.21f \n", r3);
      RETURNVALUE++;
    }

    printf("\n= Testing Gaugeinvariance =\n");
    printf("Generating Random Gauge Transformation...\n");
    // Use the normal GaugeField, because it can be initialised with random
    // SU(3) matrices, however only use mu = 0
    deviceGaugeField<4, 3> gaugeTrafo(L0, L1, L2, L3, random_pool, 1);
    real_t norm_trafo = spinor_norm<4, 3, 4>(u);
    // Dont know if this function is needed again, therefore only defined here,
    // and not in an include file.
    printf("Apply Gauge Trafos...\n");
    tune_and_launch_for<4>(
        "Gauge Trafo", IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (size_t mu = 0; mu < 4; mu++) {
            gauge(i0, i1, i2, i3, mu) =
                gaugeTrafo(i0, i1, i2, i3, 1) * gauge(i0, i1, i2, i3, mu) *
                conj(gaugeTrafo(shift_index_plus<4, int>(
                                    Kokkos::Array<int, 4>{i0, i1, i2, i3}, mu,
                                    1, IndexArray<4>{L0, L1, L2, L3}),
                                1));
          }
          // Transform spinor u, and Mu
          u(i0, i1, i2, i3) = gaugeTrafo(i0, i1, i2, i3, 1) * u(i0, i1, i2, i3);
          Mu(i0, i1, i2, i3) =
              gaugeTrafo(i0, i1, i2, i3, 1) * Mu(i0, i1, i2, i3);
        });
    deviceSpinorField<3, 4> Mu_trafo =
        apply_HD<4, 3, 4>(u, gauge, gammas, gamma5, -0.5);
    tune_and_launch_for<4>(
        "Subtract Spinors", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          Mu(i0, i1, i2, i3) -= Mu_trafo(i0, i1, i2, i3);
        });
    real_t norm_trafo1 = spinor_norm<4, 3, 4>(Mu);
    norm_trafo1 = Kokkos::sqrt(norm_trafo1 / norm);
    if (norm_trafo1 < 1e-14) {
      printf("Passed invariance test with %.21f \n", norm_trafo1);
    } else {
      printf("Error: didn't pass invariance test with %.21f \n", norm_trafo1);
      RETURNVALUE++;
    }
  }
  {
    printf("\n=== Testing DiracOperator SU(2)  ===\n");

    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    printf("Lattice Dimension %ix%ix%ix%i", L0, L1, L2, L3);
    printf("\n= Testing hermiticity =\n");

    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<2, 4> u_SU2(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<2, 4> v_SU2(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    real_t norm_SU2 = spinor_norm<4, 2, 4>(u_SU2);
    norm_SU2 *= spinor_norm<4, 2, 4>(v_SU2);
    norm_SU2 = Kokkos::sqrt(norm_SU2);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 2> gauge_SU2(L0, L1, L2, L3, random_pool, 1);

    printf("Apply DiracOperator...\n");

    deviceSpinorField<2, 4> Mu_SU2 =
        apply_HD<4, 2, 4>(u_SU2, gauge_SU2, gammas, gamma5, -0.5);
    deviceSpinorField<2, 4> Mv_SU2 =
        apply_HD<4, 2, 4>(v_SU2, gauge_SU2, gammas, gamma5, -0.5);
    // deviceSpinorField<3, 4> Mu = apply_D<4, 3, 4>(u, gauge, gammas, -0.5);
    // deviceSpinorField<3, 4> Mv = apply_D<4, 3, 4>(v, gauge, gammas, -0.5);

    printf("Calculate Scalarproducts...\n");
    auto r1_SU2 = spinor_dot_product<4, 2, 4>(u_SU2, Mv_SU2);
    auto r2_SU2 = spinor_dot_product<4, 2, 4>(Mu_SU2, v_SU2);

    auto r_SU2 = r1_SU2 - r2_SU2;

    real_t r3_SU2 =
        Kokkos::sqrt(r_SU2.real() * r_SU2.real() + r_SU2.imag() * r_SU2.imag());
    r3_SU2 /= norm_SU2;

    if (r3_SU2 < 1e-14) {
      printf("Passed hermiticity test with %.21f \n", r3_SU2);
    } else {
      printf("Error: didn't pass hermiticity test with %.21f \n", r3_SU2);
      RETURNVALUE++;
    }

    printf("\n= Testing Gaugeinvariance =\n");
    printf("Generating Random Gauge Transformation...\n");
    // Use the normal GaugeField, because it can be initialised with random
    // SU(3) matrices, however only use mu = 0
    deviceGaugeField<4, 2> gaugeTrafo_SU2(L0, L1, L2, L3, random_pool, 1);
    real_t norm_trafo_SU2 = spinor_norm<4, 2, 4>(u_SU2);
    // Dont know if this function is needed again, therefore only defined here,
    // and not in an include file.
    printf("Apply Gauge Trafos...\n");
    tune_and_launch_for<4>(
        "Gauge Trafo", IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (size_t mu = 0; mu < 4; mu++) {
            gauge_SU2(i0, i1, i2, i3, mu) =
                gaugeTrafo_SU2(i0, i1, i2, i3, 1) *
                gauge_SU2(i0, i1, i2, i3, mu) *
                conj(gaugeTrafo_SU2(shift_index_plus<4, int>(
                                        Kokkos::Array<int, 4>{i0, i1, i2, i3},
                                        mu, 1, IndexArray<4>{L0, L1, L2, L3}),
                                    1));
          }
          // Transform spinor u, and Mu
          u_SU2(i0, i1, i2, i3) =
              gaugeTrafo_SU2(i0, i1, i2, i3, 1) * u_SU2(i0, i1, i2, i3);
          Mu_SU2(i0, i1, i2, i3) =
              gaugeTrafo_SU2(i0, i1, i2, i3, 1) * Mu_SU2(i0, i1, i2, i3);
        });
    deviceSpinorField<2, 4> Mu_trafo_SU2 =
        apply_HD<4, 2, 4>(u_SU2, gauge_SU2, gammas, gamma5, -0.5);
    tune_and_launch_for<4>(
        "Subtract Spinors", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          Mu_SU2(i0, i1, i2, i3) -= Mu_trafo_SU2(i0, i1, i2, i3);
        });
    real_t norm_trafo1_SU2 = spinor_norm<4, 2, 4>(Mu_SU2);
    norm_trafo1_SU2 = Kokkos::sqrt(norm_trafo1_SU2 / norm_SU2);
    if (norm_trafo1_SU2 < 1e-14) {
      printf("Passed invariance test with %.21f \n", norm_trafo1_SU2);
    } else {
      printf("Error: didn't pass invariance test with %.21f \n",
             norm_trafo1_SU2);
      RETURNVALUE++;
    }
  }

  {
    printf("\n=== Testing DiracOperator U(1)  ===\n");
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    printf("Lattice Dimension %ix%ix%ix%i", L0, L1, L2, L3);
    printf("\n= Testing hermiticity =\n");

    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<1, 4> u_U1(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<1, 4> v_U1(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    real_t norm_U1 = spinor_norm<4, 1, 4>(u_U1);
    norm_U1 *= spinor_norm<4, 1, 4>(v_U1);
    norm_U1 = Kokkos::sqrt(norm_U1);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 1> gauge_U1(L0, L1, L2, L3, random_pool, 1);

    printf("Apply DiracOperator...\n");

    deviceSpinorField<1, 4> Mu_U1 =
        apply_HD<4, 1, 4>(u_U1, gauge_U1, gammas, gamma5, -0.5);
    deviceSpinorField<1, 4> Mv_U1 =
        apply_HD<4, 1, 4>(v_U1, gauge_U1, gammas, gamma5, -0.5);
    // deviceSpinorField<3, 4> Mu = apply_D<4, 3, 4>(u, gauge, gammas, -0.5);
    // deviceSpinorField<3, 4> Mv = apply_D<4, 3, 4>(v, gauge, gammas, -0.5);

    printf("Calculate Scalarproducts...\n");
    auto r1_U1 = spinor_dot_product<4, 1, 4>(u_U1, Mv_U1);
    auto r2_U1 = spinor_dot_product<4, 1, 4>(Mu_U1, v_U1);

    auto r_U1 = r1_U1 - r2_U1;

    real_t r3_U1 =
        Kokkos::sqrt(r_U1.real() * r_U1.real() + r_U1.imag() * r_U1.imag());
    r3_U1 /= norm_U1;

    if (r3_U1 < 1e-14) {
      printf("Passed hermiticity test with %.21f \n", r3_U1);
    } else {
      printf("Error: didn't pass hermiticity test with %.21f \n", r3_U1);
      RETURNVALUE++;
    }

    printf("\n= Testing Gaugeinvariance =\n");
    printf("Generating Random Gauge Transformation...\n");
    // Use the normal GaugeField, because it can be initialised with random
    // SU(3) matrices, however only use mu = 0
    deviceGaugeField<4, 1> gaugeTrafo_U1(L0, L1, L2, L3, random_pool, 10);
    real_t norm_trafo_U1 = spinor_norm<4, 1, 4>(u_U1);
    // Dont know if this function is needed again, therefore only defined here,
    // and not in an include file.
    printf("Spinor before Gauge Trafo:\n");
    print_spinor(u_U1(0, 0, 0, 0));
    printf("Apply Gauge Trafos...\n");
    tune_and_launch_for<4>(
        "Gauge Trafo", IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (size_t mu = 0; mu < 4; mu++) {
            gauge_U1(i0, i1, i2, i3, mu) =
                gaugeTrafo_U1(i0, i1, i2, i3, 1) *
                gauge_U1(i0, i1, i2, i3, mu) *
                conj(gaugeTrafo_U1(shift_index_plus<4, int>(
                                       Kokkos::Array<int, 4>{i0, i1, i2, i3},
                                       mu, 1, IndexArray<4>{L0, L1, L2, L3}),
                                   1));
          }
          // Transform spinor u, and Mu
          u_U1(i0, i1, i2, i3) =
              gaugeTrafo_U1(i0, i1, i2, i3, 1) * u_U1(i0, i1, i2, i3);
          Mu_U1(i0, i1, i2, i3) =
              gaugeTrafo_U1(i0, i1, i2, i3, 1) * Mu_U1(i0, i1, i2, i3);
        });
    printf("Spinor after Gauge Trafo:\n");
    print_spinor(u_U1(0, 0, 0, 0));
    deviceSpinorField<1, 4> Mu_trafo_U1 =
        apply_HD<4, 1, 4>(u_U1, gauge_U1, gammas, gamma5, -0.5);
    tune_and_launch_for<4>(
        "Subtract Spinors", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          Mu_U1(i0, i1, i2, i3) -= Mu_trafo_U1(i0, i1, i2, i3);
        });
    real_t norm_trafo1_U1 = spinor_norm<4, 1, 4>(Mu_U1);
    norm_trafo1_U1 = Kokkos::sqrt(norm_trafo1_U1 / norm_U1);
    if (norm_trafo1_U1 < 1e-14) {
      printf("Passed invariance test with %.21f \n", norm_trafo1_U1);
    } else {
      printf("Error: didn't pass invariance test with %.21f \n",
             norm_trafo1_U1);
      RETURNVALUE++;
    }
  }

  Kokkos::finalize();
  printf(HLINE);
  printf("%i Errors durring Testing\n", RETURNVALUE);
  printf(HLINE);
  RETURNVALUE = !(RETURNVALUE == 0);
  return RETURNVALUE;
}
