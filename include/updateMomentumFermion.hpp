//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/
#pragma once
#include "DiracOPTypeHelper.hpp"
#include "DiracOperator.hpp"
#include "FermionParams.hpp"
#include "IndexHelper.hpp"
#include "SpinorFieldLinAlg.hpp"
#include "UpdateMomentum.hpp"

namespace klft {

template <typename DAdjFieldType,
          template <class DiracOpT> class _Solver,
          class DiracOpT>
class UpdateMomentumWilson : public UpdateMomentum {
  using DSpinorFieldType = typename DiracOpT::DSpinorFieldType;
  using DGaugeFieldType = typename DiracOpT::DGaugeFieldType;
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert(rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank &&
                    rank ==
                        DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank &&
                    Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc &&
                    Nc == DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc,
                "Rank and Nc must match between gauge, adjoint, and fermion "
                "field types.");
  using DiracOp = DiracOpT;
  using Solver = _Solver<DiracOpT>;
  Solver solver;

 public:
  using FermionField = typename DSpinorFieldType::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using AdjFieldType = typename DeviceAdjFieldType<rank, Nc>::type;
  GaugeFieldType gauge_field;
  AdjFieldType momentum;
  const diracParams params;
  // \phi = D R, where R gaussian random field.
  FermionField phi;
  const Kokkos::Array<index_t, rank> dimensions;
  FermionField chi;
  FermionField chi_alt;
  const real_t tol;
  real_t eps;

  UpdateMomentumWilson() = delete;
  ~UpdateMomentumWilson() = default;

  UpdateMomentumWilson(FermionField& phi_,
                       const GaugeFieldType& gauge_field_,
                       AdjFieldType& adjoint_field_,
                       const diracParams& params_,
                       const real_t& tol_)
      : UpdateMomentum(0),
        phi(phi_),
        gauge_field(gauge_field_),
        momentum(adjoint_field_),
        params(params_),
        eps(0.0),
        tol(tol_),
        dimensions(phi_.dimensions) {
    solver.init(this->phi.dimensions);
  }

  // Implemntation of the force correspondig to the Hermitian Wilson dirac
  // Operator
  //* S_F  = \phi^{\dagger} * M^{-1} * \phi
  //* dS_F = - \phi^{\dagger} * M^{-1} * dM * M^{-1} * \phi = - \chi^{\dagger} *
  // dM * \chi
  //* where \chi = M^{-1} * \phi
  //* and M = Q*Q^{\dagger}
  // Using that dS_f = - \chi ^{\dagger} (\pdv{D}{U_\mu(x)}D^\dagger + D
  // \pdv{D^\dagger}{U_\mu(x)})\chi which can be simplified to -\chi^\dagger
  // (\pdv{D}{U_\mu(x)}D^\dagger+ (\pdv{D}{U_\mu(x)}D^\dagger)^\dagger) =
  // -2Re(\chi^\dagger \pdv{D}{U_\mu(x)}D^\dagger \chi)

  // The Derivative of the Hermitian Wilson Dirac Operator should be given as
  // \pdv{D(x,y)}{U_\mu(z)} = - \kappa \gamma_5 \delta_{x,z} (1-\gamma_mu)T_i
  // U_mu(z)\delta_{x+\mu,y}
  // +\kappa \gamma_5 \delta_{x+\mu,y} (1+\gamma_\mu) t_i
  // U_\mu(z)^\dagger\delta_{z,y} <- Have to Check this
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      // X = chi , Y = chi_alt

      auto xp = shift_index_plus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 3, -1,
          this->phi.dimensions);
      auto X_proj = project(mu, -1, this->chi(xp.first));
      // minus sign in the projector comes from the derivative of D
      auto Xplus = (-1 * this->params.kappa * xp.second) * X_proj;
      auto temp1 = reconstruct(mu, -1, gauge_field(Idcs..., mu) * Xplus);
      auto deriv = temp1 * (conj(this->chi_alt(Idcs...)));

      auto Y_proj = project_alt(mu, 1, conj(this->chi_alt(xp.first)));
      auto YPlus = (this->params.kappa * xp.second) * Y_proj;
      auto temp3 =
          reconstruct_alt(mu, 1, YPlus * conj(this->gauge_field(Idcs..., mu)));
      deriv += this->chi(Idcs...) * temp3;

      momentum(Idcs..., mu) -= 2 * eps *
                               traceT(traceLessAntiHermitian(

                                   deriv));
    }
  }

  void update(const real_t step_size) override {
    Kokkos::Profiling::pushRegion("UpdateMomentumFermion");
    eps = step_size;

    IndexArray<rank> start;
    DiracOp D(gauge_field, this->params);
    D.init(this->phi.dimensions);

    FermionField x(this->phi.dimensions, complex_t(0.0, 0.0));
    FermionField x0(this->phi.dimensions, complex_t(0.0, 0.0));

    this->solver.set_DiracOperator(D);
    this->solver.set_problem(this->phi);
    if (KLFT_VERBOSITY > 4) {
      printf("Solving insde UpdateMomentumWilson:");
    }

    solver.template solve<Tags::TagDdaggerD>(x0, this->tol);
    this->chi = solver.x;
    this->chi_alt = D.template apply<Tags::TagD>(this->chi);

    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }
    // launch the kernel
    KTune::parallel_for("UpdateMomentumWilson",
                        Policy<rank>(start, phi.dimensions), *this);
    Kokkos::fence();
    Kokkos::Profiling::popRegion();
  }
};

}  // namespace klft
