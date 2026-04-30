#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/DiracOperator.hpp"
#include "../../include/FermionMonomial.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/Solver.hpp"
#include "../../include/Spinor.hpp"
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
template <typename T, typename precision>
struct getSpinorField {
  using type = DeviceSpinorFieldType<DeviceFermionFieldTypeTraits<T>::Rank,
                                     DeviceFermionFieldTypeTraits<T>::Nc,
                                     DeviceFermionFieldTypeTraits<T>::RepDim,
                                     precision,
                                     DeviceFermionFieldTypeTraits<T>::Kind,
                                     DeviceFermionFieldTypeTraits<T>::Layout>;
};
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  KTune::initialize();
  int RETURNVALUE = 0;
  {
    constexpr int count = 1;
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing Hoe =\n");
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    diracParams params(0.156, 1);

    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<2, 4> u_for_normal(L0, L1, L2, L3, 0);

    // print_spinor_int(u_for_normal(0, 0, 0, 0), "u_for_normal");

    Kokkos::Random_XorShift64_Pool<> random_pool1(/*seed=*/1234);
    deviceSpinorField<2, 4> phi(L0 / 2, L1, L2, L3, random_pool1, 0,
                                1.0 / 1.41);
    deviceSpinorField<2, 4> test(L0 / 2, L1, L2, L3, random_pool1, 0,
                                 1.0 / 1.41);
    printf("Populate normal field\n\n\n");
    KTune::parallel_for(
        "init_deviceSpinorField",
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0 / 2, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          phi(i0, i1, i2, i3) = test(i0, i1, i2, i3);
        });
    // auto test_idx = Kokkos::Array<int, 4>({2, 0, 0, 0});
    // printf(
    //     "u_even == u_for_normal @(0,0,0,0) is equal: %i\n",
    //     u_even(index_full_to_half(test_idx).first) ==
    //     u_for_normal(test_idx));

    // deviceSpinorField<2, 4> Mu(L0 / 2, L1 / 2, L2 / 2, L3 / 2, 0);
    // deviceSpinorField<2, 4> temp(L0 / 2, L1 / 2, L2 / 2, L3 / 2, 0);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 2> gauge(L0, L1, L2, L3, random_pool, 1);
    printf("Instantiate DiracOperator...\n");

    EOWilsonDiracOperator<DeviceSpinorFieldType<4, 2, 4, complex_t>,
                          DeviceGaugeFieldType<4, 2, complex_t>, true>
        D_shift(gauge, params);
    D_shift.init(phi.dimensions);

    EOWilsonDiracOperator<DeviceSpinorFieldType<4, 2, 4, complex_t>,
                          DeviceGaugeFieldType<4, 2, complex_t>>
        D_no(gauge, params);
    D_no.init(phi.dimensions);

    printf("Apply DiracOperator...\n");
    Kokkos::Timer timer;

    real_t diracTime = std::numeric_limits<real_t>::max();
    // for (size_t i = 0; i < count; i++) {

    auto ddaggerD_via_g5 = D_shift.template apply<Tags::TagDDdagger>(phi);

    // manuell
    auto g5Se_man = D_shift.template apply<Tags::TagSe>(phi);
    auto ddaggerD_via_manually_composite =
        D_shift.template apply<Tags::TagSedagger>(g5Se_man);

    auto diracTime1 = std::min(diracTime, timer.seconds());
    printf("D^ Precondition Kernel Time:     %11.4e s\n", diracTime1);
    timer.reset();

    // KTune::parallel_for(
    //     "init_deviceSpinorField",
    //     Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0 / 2, L1, L2,
    //     L3}), KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t
    //     i2,
    //                   const index_t i3) {
    //       test(i0, i1, i2, i3) = g5Se_man(i0, i1, i2, i3) +
    //                              gamma5(params.massShift * phi(i0, i1, i2,
    //                              i3));
    //     });
    // auto out_normal = D.template apply<Tags::TagD>(out_normal1);
    // }
    diracTime1 = std::min(diracTime, timer.seconds());
    printf("D^ Kernel Time:     %11.4e s\n", diracTime1);
    // print_spinor_int(out_eo(0, 0, 0, 0), "out_eo(0,0,0,0)");
    // print_spinor_int(out_normal(0, 0, 0, 0), "out_normal(0,0,0,0)");
    // printf("out_for_eo == out_for_normal @(0,0,0,0) is equal: %i\n",
    //        out_eo(0, 2, 64, 64) == out_normal(0, 2, 64, 64));
    printf("Checking total difference:\n");
    real_t result = 1;
    KTune::parallel_reduce(
        "Reduction", Policy<4>({0, 0, 0, 0}, {L0 / 2, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3, real_t& lsum) {
          lsum += sqnorm(ddaggerD_via_g5(i0, i1, i2, i3) -
                         //  sqnorm(u_for_normal(i0, i1, i2, i3)) -
                         ddaggerD_via_manually_composite(i0, i1, i2, i3));
        },
        Kokkos::Sum<real_t>(result));
    printf("Total difference: %.16f\n", result);

    // printf("<=== Checking applying composit operator ===>\n ");
    // EOWilsonDiracOperator<
    //     DeviceSpinorFieldType<4, 2, 4, SpinorFieldKind::Standard,
    //                           SpinorFieldLayout::Checkerboard>,
    //     DeviceGaugeFieldType<4, 2>>
    //     D_pre_alt(gauge, params);
    // deviceSpinorField<2, 4> u_eo_temp(L0 / 2, L1, L2, L3, 0);
    // deviceSpinorField<2, 4> out_comp(L0 / 2, L1, L2, L3, 0);
    // // build it manually:
    // D_pre_alt.template apply<Tags::TagDDdagger>(u_even, out_comp);
    // auto temp0 = D_pre.template apply<Tags::TagSe>(u_even);
    // auto out_man = D_pre.template apply<Tags::TagSe>(temp0);
    // // auto temp2 = D_pre.template apply<Tags::TagHeo>(temp1);
    // // auto out_man =
    // //     axpyG5<DeviceSpinorFieldType<4, 2, 4, SpinorFieldKind::Standard,
    // //                                  SpinorFieldLayout::Checkerboard>>(
    // //         -params.kappa * params.kappa, temp2, u_even);
    // // built it completla maually
    // // auto temp3 = D.template apply<Tags::TagD>(u_for_normal);
    // // auto temp4 = D.template apply<Tags::TagD>(temp3);
    // // auto out_nor =
    // //     axpy<DeviceSpinorFieldType<4, 2, 4>>(-1, temp4, u_for_normal);

    // // print_spinor_int(out_man(0, 0, 0, 0), "Manually build");
    // // print_spinor_int(out_comp(0, 0, 0, 0), "Composit");
    // // print_spinor_int(out_nor(0, 0, 0, 0), "Build from normal Dirac
    // // Operator");

    // // printf("temp2 == out_comp @(0,0,0,0) is equal: %i\n",
    // //        out_man(1, 0, 0, 0) == out_comp(1, 0, 0, 0));

    // // printf("temp2 == out_nor @(0,0,0,0) is equal: %i\n",
    // //        out_man(1, 0, 0, 0) == out_nor(2, 0, 0, 0));

    // result = 0;
    // KTune::parallel_reduce(
    //     "Reduction", Policy<4>({0, 0, 0, 0}, {L0 / 2, L1, L2, L3}),
    //     KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
    //                   const index_t i3, real_t& lsum) {
    //       // auto idx = index_full_to_half(Kokkos::Array<int, 4>{i0, i1, i2,
    //       // i3}); if (idx.second == 0) {
    //       lsum += sqnorm(out_man(i0, i1, i2, i3) -
    //                      //  sqnorm(u_for_normal(i0, i1, i2, i3)) -
    //                      out_comp(i0, i1, i2, i3));
    //       // }
    //     },
    //     Kokkos::Sum<real_t>(result));
    // printf("Total difference comp to EO manually: %.16f\n", result);
    // result = -9999;
    // // KTune::parallel_reduce(
    // //     "Reduction", Policy<4>({0, 0, 0, 0}, {L0, L1, L2, L3}),
    // //     KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t
    // i2,
    // //                   const index_t i3, real_t& lsum) {
    // //       auto idx = index_full_to_half(Kokkos::Array<int, 4>{i0, i1, i2,
    // //       i3}); if (idx.second == 0) {
    // //         lsum = sqnorm(out_nor(i0, i1, i2, i3)) -
    // //                //  sqnorm(u_for_normal(i0, i1, i2, i3)) -
    // //                sqnorm(out_comp(idx.first));
    // //       }
    // //     },
    // //     Kokkos::Sum<real_t>(result));
    // printf("Total difference comp to normal manual: %.16f\n", result);

    // printf("Checking positivity: \n\n");
    // Kokkos::Random_XorShift64_Pool<> random_pool2(/*seed=*/21546542131);

    // deviceSpinorField<2, 4> x(L0 / 2, L1, L2, L3, random_pool2, 0, 1.0
    // / 1.41); auto temp_x = D_pre_alt.template apply<Tags::TagSe>(x); auto res
    // = spinor_dot_product<4, 2, 4>(x, temp_x); printf("Positivity test:
    // %f+i%f.\n ", res.real(), res.imag()); printf("Check for hermicity:\n\n");
    // deviceSpinorField<2, 4> u(L0 / 2, L1, L2, L3, random_pool, 0, 1.0
    // / 1.41); deviceSpinorField<2, 4> v(L0 / 2, L1, L2, L3, random_pool,
    // 0, 1.0 / 1.41); real_t norm = spinor_norm_sq<4, 2, 4>(u); norm *=
    // spinor_norm_sq<4, 2, 4>(v); norm = Kokkos::sqrt(norm);
    // deviceSpinorField<2, 4> Mu(L0 / 2, L1, L2, L3, complex_t(0.0, 0.0));
    // deviceSpinorField<2, 4> Mv(L0 / 2, L1, L2, L3, complex_t(0.0, 0.0));
    // D_pre_alt.template apply<Tags::TagSe>(u, Mu);
    // D_pre_alt.template apply<Tags::TagSe>(v, Mv);
    // printf("Calculate Scalarproducts...\n");
    // auto r1 = spinor_dot_product<4, 2, 4>(u, Mv);
    // auto r2 = spinor_dot_product<4, 2, 4>(Mu, v);

    // auto r = r1 - r2;

    // real_t r3 = Kokkos::sqrt(r.real() * r.real() + r.imag() * r.imag());
    // r3 /= norm;

    // if (r3 < 1e-14) {
    //   printf("Passed hermiticity test with %.21f \n", r3);
    // } else {
    //   printf("Error: didn't pass hermiticity test with %.21f \n", r3);
    //   RETURNVALUE++;
    // }

    // FermionMonomial<Kokkos::Random_XorShift64_Pool<>, DeviceAdjFieldType<4,
    // 2>,
    //                 CGSolver,
    //                 WilsonDiracOperator<DeviceSpinorFieldType<4, 2, 4>,
    //                                     DeviceGaugeFieldType<4, 2>>>
    //     test(temp_x, params, 10e-8, random_pool2, 0);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}