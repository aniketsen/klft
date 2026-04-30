#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/DiracOperator.hpp"
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

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  KTune::initialize();
  int RETURNVALUE = 0;
  {
    constexpr int count = 1;
    constexpr int N = 3;
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing hermiticity =\n");
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    diracParams params(0.156);
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");
    using DSpinorFieldType =
        DeviceSpinorFieldType<4, N, 4, complex_t, SpinorFieldKind::Standard,
                              SpinorFieldLayout::Checkerboard>;
    using SpinorFieldType = DSpinorFieldType::type;

    // Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    // deviceSpinorField<2, 4> u_for_normal(L0, L1, L2, L3, 0);

    // print_spinor_int(u_for_normal(0, 1, 0, 0), "u_for_normal");

    Kokkos::Random_XorShift64_Pool<> random_pool1(/*seed=*/1234);
    SpinorFieldType even_true(
        L0 / 2, L1, L2, L3, random_pool1, 0,
        0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367506);
    SpinorFieldType odd_true(
        L0 / 2, L1, L2, L3, random_pool1, 0,
        0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367506);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, N> gauge(L0, L1, L2, L3, random_pool1, 1);
    EOWilsonDiracOperator<DSpinorFieldType,
                          DeviceGaugeFieldType<4, N, complex_t>>
        D_pre(gauge, params);
    EOWilsonDiracOperator<DSpinorFieldType,
                          DeviceGaugeFieldType<4, N, complex_t>>
        D_pre2(gauge, params);
    D_pre.init(even_true.dimensions);
    D_pre2.init(even_true.dimensions);

    // apply DiracOperators to later verify solution:
    // D_pre.s_in_same_parity = even_true;
    // auto even_b = D_pre.template apply<Tags::TagD>(odd_true);
    // // axpy<DSpinorFieldType>(-1, even_b, even_true, even_b);

    // D_pre.s_in_same_parity = odd_true;
    // auto odd_b = D_pre.template apply<Tags::TagDdagger>(even_true);
    auto even_b = D_pre.template apply<Tags::TagSe>(even_true);
    // D_pre.s_in_same_parity = even_b_1;
    // auto even_b = D_pre.template apply<Tags::TagD>(odd_b_1);

    // D_pre.s_in_same_parity = odd_b_1;
    // auto odd_b = D_pre.template apply<Tags::TagD>(even_b_1);
    // axpy<DSpinorFieldType>(-1, odd_b, odd_true, odd_b);
    // Construct Solver:
    SpinorFieldType x(L0 / 2, L1, L2, L3, complex_t(0.0, 0.0));
    SpinorFieldType x2(L0 / 2, L1, L2, L3, complex_t(0.0, 0.0));

    BiCGStabMultiP<EOWilsonDiracOperator<DSpinorFieldType,
                                         DeviceGaugeFieldType<4, N, complex_t>>>
        solver;
    solver.init(IndexArray<4>({L0 / 2, L1, L2, L3}));
    solver.set_DiracOperator(D_pre2);
    solver.set_problem(even_b);
    BiCGStab<EOWilsonDiracOperator<DSpinorFieldType,
                                   DeviceGaugeFieldType<4, N, complex_t>>>
        solvercg(even_b, x, D_pre2);

    // Construct RHS of Prblem to solve
    // auto out_even_from_odd_b = D_pre.template apply<Tags::TagHeo>(odd_b);
    // axpy<DSpinorFieldType>(1.0, out_even_from_odd_b, even_b,
    //                        even_b);  // maybe here the other sign

    // Solver fields
    SpinorFieldType x0(L0 / 2, L1, L2, L3, complex_t(0.0, 0.0));
    SpinorFieldType x02(L0 / 2, L1, L2, L3, complex_t(0.0, 0.0));
    // BiCGStab<EOWilsonDiracOperator, DSpinorFieldType,
    //          DeviceGaugeFieldType<4, N>>
    //     solver(even_b, x, D_pre2);
    printf("Apply Solver...\n");
    auto eps = 1e-12;
    Kokkos::Timer timer;
    solvercg.solve<Tags::TagSe>(x0, eps);
    real_t diracTime = std::numeric_limits<real_t>::max();
    auto diracTime1 = std::min(diracTime, timer.seconds());
    printf("BiCGStab Solver Time:     %11.4e s\n", diracTime1);
    // CGSolver<EOWilsonDiracOperator, DSpinorFieldType,
    //          DeviceGaugeFieldType<4, N>>
    // solver2(solver.x, x2, D_pre);
    // solver2.solve<Tags::TagSe>(x02, eps);
    timer.reset();
    solver.solve<Tags::TagSe>(x0, eps);
    diracTime1 = std::min(diracTime, timer.seconds());
    printf("BiCGStabMultiP Time:     %11.4e s\n", diracTime1);
    timer.reset();
    // auto out_normal1 = D.template apply<Tags::TagD>(u_for_normal);
    // auto out_normal = D.template apply<Tags::TagD>(out_normal1);
    // }
    // print_spinor_int(out_eo(0, 0, 0, 0), "out_eo(0,0,0,0)");
    // print_spinor_int(out_normal(0, 0, 0, 0), "out_normal(0,0,0,0)");
    printf("Comparing Solver result to expected result...\n");

    auto res_norm = spinor_norm<DSpinorFieldType>(
        axpy<DSpinorFieldType>(-1, solver.x, even_true));
    auto norm = spinor_norm<DSpinorFieldType>(even_true);

    printf("Norm of Residual of the even field: %.20f\n", res_norm / norm);
    printf("Is the residual norm smaller than %.2e ? %i\n", eps,
           res_norm / norm < eps);

    // solver.reconstruct_solution(odd_b, x02);
    // auto res_norm_odd = spinor_norm<DSpinorFieldType>(
    //     axpy<DSpinorFieldType>(-1, x02, odd_true));
    // auto norm_odd = spinor_norm<DSpinorFieldType>(odd_true);
    // printf("Norm of Residual of the odd field: %.20f\n",
    //        res_norm_odd / norm_odd);
    // printf("Is the residual norm smaller than %.2e ? %i\n", eps,
    //        res_norm_odd / norm_odd < eps);
  }
  Kokkos::finalize();
  return RETURNVALUE;
}