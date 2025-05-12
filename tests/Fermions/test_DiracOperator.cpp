#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/DiracOperator.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/SpinorFieldHelper.hpp"
#include "../../include/klft.hpp"

using namespace klft;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int RETURNVALUE = 0;
  {
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    index_t L0 = 8, L1 = 8, L2 = 8, L3 = 8;
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
      RETURNVALUE = 1;
    }
  }
  {
    printf("\n=== Testing DiracOperator SU(2)  ===\n");
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    index_t L0 = 8, L1 = 8, L2 = 8, L3 = 8;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    printf("Lattice Dimension %ix%ix%ix%i", L0, L1, L2, L3);
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
      RETURNVALUE = 1;
    }
  }

  {
    printf("\n=== Testing DiracOperator U(1)  ===\n");
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    index_t L0 = 8, L1 = 8, L2 = 8, L3 = 8;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    printf("Lattice Dimension %ix%ix%ix%i", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<2, 4> u_U1(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<2, 4> v_U1(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    real_t norm_U1 = spinor_norm<4, 2, 4>(u_U1);
    norm_U1 *= spinor_norm<4, 2, 4>(v_U1);
    norm_U1 = Kokkos::sqrt(norm_U1);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 2> gauge_U1(L0, L1, L2, L3, random_pool, 1);

    printf("Apply DiracOperator...\n");

    deviceSpinorField<2, 4> Mu_U1 =
        apply_HD<4, 2, 4>(u_U1, gauge_U1, gammas, gamma5, -0.5);
    deviceSpinorField<2, 4> Mv_U1 =
        apply_HD<4, 2, 4>(v_U1, gauge_U1, gammas, gamma5, -0.5);
    // deviceSpinorField<3, 4> Mu = apply_D<4, 3, 4>(u, gauge, gammas, -0.5);
    // deviceSpinorField<3, 4> Mv = apply_D<4, 3, 4>(v, gauge, gammas, -0.5);

    printf("Calculate Scalarproducts...\n");
    auto r1_U1 = spinor_dot_product<4, 2, 4>(u_U1, Mv_U1);
    auto r2_U1 = spinor_dot_product<4, 2, 4>(Mu_U1, v_U1);

    auto r_U1 = r1_U1 - r2_U1;

    real_t r3_U1 =
        Kokkos::sqrt(r_U1.real() * r_U1.real() + r_U1.imag() * r_U1.imag());
    r3_U1 /= norm_U1;

    if (r3_U1 < 1e-14) {
      printf("Passed hermiticity test with %.21f \n", r3_U1);
    } else {
      printf("Error: didn't pass hermiticity test with %.21f \n", r3_U1);
      RETURNVALUE = 1;
    }
  }

  Kokkos::finalize();

  return RETURNVALUE;
}
