#include <random>

#include "DiracOperator.hpp"
#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "PropagatorMatrix.hpp"
#include "Solver.hpp"
#include "Spinor.hpp"
#include "SpinorFieldLinAlg.hpp"
#include "SpinorPointSource.hpp"

namespace klft {

template <typename RNG,

          template <class DiracOpT> class _Solver,
          class DiracOpT>
std::vector<real_t> PionCorrelator(
    const typename DiracOpT::DGaugeFieldType::type& g_in,
    const diracParams& params,
    const real_t& tol,
    RNG& rng,
    const index_t& n_sources) {
  using DSpinorFieldType = typename DiracOpT::DSpinorFieldType;
  using DGaugeFieldType = typename DiracOpT::DGaugeFieldType;
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  using SpinorFieldSource =
      typename WithSpinorFieldKind<DSpinorFieldType,
                                   SpinorFieldKind::PointSource>::type;
  using SpinorField = typename DSpinorFieldType::type;
  using DiracOperator = DiracOpT;
  using Solver = _Solver<DiracOpT>;
  DiracOperator dirac_op(g_in, params);
  dirac_op.init(g_in.dimensions);
  auto Nt = g_in.field.extent(3);
  SpinorField prop(g_in.dimensions, 0);
  std::uniform_real_distribution<real_t> dist;
  std::vector<real_t> result_vec(Nt);
  IndexArray<rank> sourceIdx{};

  if constexpr (rank == 4) {
    size_t Vs = g_in.dimensions[0] * g_in.dimensions[1] * g_in.dimensions[2];
    SpinorField x(g_in.dimensions, 0);
    SpinorField x0(g_in.dimensions, 0);

    for (index_t source_i = 0; source_i < n_sources; source_i++) {
      for (index_t i = 0; i < rank; i++) {
        sourceIdx[i] = int(dist(rng) * g_in.dimensions[i]);
      }
      for (index_t alpha0 = 0; alpha0 < Nc * RepDim; alpha0++) {
        SpinorFieldSource source(g_in.dimensions, sourceIdx, alpha0);
        Solver solver(source, x, dirac_op);
        solver.template solve<Tags::TagDDdagger>(x0, tol);
        dirac_op.template apply<Tags::TagDdagger>(solver.x, prop);
        for (size_t i3 = 0; i3 < g_in.dimensions[3]; i3++) {
          real_t res = 0.0;
          KTune::parallel_reduce(
              "Reductor",
              Policy<rank - 1>(
                  IndexArray<rank - 1>{},
                  IndexArray<rank - 1>{g_in.dimensions[0], g_in.dimensions[1],
                                       g_in.dimensions[2]}),
              KOKKOS_LAMBDA(const size_t& i0, const size_t& i1,
                            const size_t& i2, real_t& upd) {
                upd += sqnorm(prop(i0, i1, i2, i3));
              },
              res);
          Kokkos::fence();
          auto t_shifted = (i3 >= sourceIdx[3]) ? (i3 - sourceIdx[3])
                                                : (Nt - (sourceIdx[3] - i3));
          result_vec[t_shifted] += (res / (Vs * n_sources));
        }
      }
    }
  }
  return result_vec;
}

template <typename RNG,

          template <class DiracOpT> class _Solver,
          class DiracOpT>
std::vector<real_t> PionCorrelatorEO(
    const typename DiracOpT::DGaugeFieldType::type& g_in,
    const diracParams& params,
    const IndexArray<
        DeviceGaugeFieldTypeTraits<typename DiracOpT::DGaugeFieldType>::Rank>&
        f_dims,
    const real_t& tol,
    RNG& rng,
    const index_t& n_sources) {
  using DSpinorFieldType = typename DiracOpT::DSpinorFieldType;
  using DGaugeFieldType = typename DiracOpT::DGaugeFieldType;
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  using SpinorFieldSource =
      typename WithSpinorFieldKind<DSpinorFieldType,
                                   SpinorFieldKind::PointSource>::type;
  using SpinorField = typename DSpinorFieldType::type;
  using DiracOperator = DiracOpT;
  using Solver = _Solver<DiracOpT>;
  DiracOperator dirac_op(g_in, params);
  dirac_op.init(f_dims);
  auto Nt = g_in.field.extent(3);
  std::uniform_real_distribution<real_t> dist;
  std::vector<real_t> result_vec(Nt);
  IndexArray<rank> sourceIdx{};

  if constexpr (rank == 4) {
    size_t Vs = g_in.dimensions[0] * g_in.dimensions[1] * g_in.dimensions[2];
    for (index_t source_i = 0; source_i < n_sources; source_i++) {
      for (index_t i = 0; i < rank; i++) {
        sourceIdx[i] = int(dist(rng) * f_dims[i]);
        // sourceIdx[i] = 0;
      }
      for (index_t alpha0 = 0; alpha0 < Nc * RepDim; alpha0++) {
        SpinorField x(f_dims, 0);
        SpinorField x0(f_dims, 0);
        SpinorField prop_even(f_dims, 0);
        SpinorField prop_odd(f_dims, 0);
        SpinorFieldSource source(f_dims, sourceIdx,
                                 alpha0);  // even source
        Solver solver(source, x, dirac_op);
        solver.init(f_dims);
        solver.set_DiracOperator(dirac_op);
        solver.set_problem(source);
        if constexpr (std::is_same_v<Solver, CGSolver<DiracOpT>>) {
          solver.template solve<Tags::TagDdaggerD>(x0, tol);
          dirac_op.template apply<Tags::TagG5Se>(solver.x, x0, prop_even);
          dirac_op.template apply<Tags::TagHoe>(prop_even, prop_odd);
          ax<DSpinorFieldType>(dirac_op.params.kappa, prop_odd, prop_odd);
        }
        if constexpr (Solver::solver_type_value ==
                      SolverType::KLFT_SOLVER_BICGSTAB) {
          // BicCGStab gives D^-1 directly
          solver.template solve<Tags::TagSe>(x0, tol);
          solver.reconstruct_solution_0(prop_odd);
          prop_even = solver.x;
        }
        for (size_t i3 = 0; i3 < g_in.dimensions[3]; i3++) {
          real_t res = 0.0;
          KTune::parallel_reduce(
              "Reductor",
              Policy<rank - 1>(
                  IndexArray<rank - 1>{},
                  IndexArray<rank - 1>{f_dims[0], f_dims[1], f_dims[2]}),
              KOKKOS_LAMBDA(const size_t& i0, const size_t& i1,
                            const size_t& i2, real_t& upd) {
                upd += sqnorm(prop_even(i0, i1, i2, i3));
                upd += sqnorm(prop_odd(i0, i1, i2, i3));
              },
              res);
          Kokkos::fence();
          auto t_shifted = (i3 >= sourceIdx[3]) ? (i3 - sourceIdx[3])
                                                : (Nt - (sourceIdx[3] - i3));
          result_vec[t_shifted] += (res / (Vs * n_sources));
        }
        // Function
      }
    }
  }
  return result_vec;
}
}  // namespace klft
