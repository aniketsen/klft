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
// Implemntation of the Hasenbusch qutient force
#pragma once
#include "DiracOPTypeHelper.hpp"
#include "DiracOperator.hpp"
#include "FermionParams.hpp"
#include "IndexHelper.hpp"
#include "SpinorFieldLinAlg.hpp"
#include "UpdateMomentum.hpp"

namespace klft {

template <typename DAdjFieldType,
          template <typename> class _Solver,
          class DiracOpT,
          class DiracOPNonShift>
class UpdateMomentumWilsonEOHasenbusch : public UpdateMomentum {
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
  static_assert(DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout ==
                    SpinorFieldLayout::Checkerboard,
                "When using Even/odd preconditioning "
                "the spinor field layout must be "
                "Checkerboard");
  static_assert(DiracOpT::HasMassShift == true,
                "Hasenbusch update requires massshifted Dirac Operator");
  using DiracOpShifted = DiracOpT;
  using DiracOpNonShift = DiracOPNonShift;

  using Solver = _Solver<DiracOpT>;
  _Solver<DiracOpNonShift> solver;

 public:
  using FermionField = typename DSpinorFieldType::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using AdjFieldType = typename DeviceAdjFieldType<rank, Nc>::type;
  GaugeFieldType gauge_field;
  AdjFieldType momentum;
  const diracParams params;
  DiracOpNonShift D_n;
  DiracOpShifted D_s;
  // \phi = D R, where R gaussian random field.
  FermionField phi;

  FermionField y;
  FermionField chi;
  FermionField rho;
  FermionField sigma;
  FermionField temp;
  FermionField x0;
  const real_t tol;
  real_t eps;
  // auxillary fields for solver
  // Solver Fields:

  UpdateMomentumWilsonEOHasenbusch() = delete;
  ~UpdateMomentumWilsonEOHasenbusch() = default;

  UpdateMomentumWilsonEOHasenbusch(FermionField& phi_,
                                   const GaugeFieldType& gauge_field_,
                                   AdjFieldType& adjoint_field_,
                                   const diracParams& params,
                                   const real_t& tol_)
      : UpdateMomentum(0),
        phi(phi_),
        gauge_field(gauge_field_),
        momentum(adjoint_field_),
        params(params),
        eps(0.0),
        tol(tol_) {
    rho = FermionField(phi.dimensions, 0);
    sigma = FermionField(phi.dimensions, 0);
    y = FermionField(phi.dimensions, 0);
    temp = FermionField(phi.dimensions, 0);
    // Solver Fields:
    x0 = FermionField(phi.dimensions, 0);
    solver.init(this->phi.dimensions);
    this->D_n = DiracOpNonShift(gauge_field_, this->params);
    this->D_s = DiracOpShifted(gauge_field_, this->params);
  }
  struct TagEvenContribution {};
  struct TagOddContribution {};
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
  KOKKOS_FORCEINLINE_FUNCTION void operator()(TagEvenContribution,
                                              const Indices... Idcs) const {
    Kokkos::Array<size_t, rank> x_half{Idcs...};

    auto idx = index_half_to_full(x_half, 0);

    // Even
    for (size_t mu = 0; mu < rank; ++mu) {
      // X = chi , Y = chi_alt

      auto xp = shift_index_plus_bc<rank, index_t>(
          Kokkos::Array<index_t, rank>{idx}, mu, 1, 3, -1,
          this->gauge_field.dimensions);  // odd
      auto xp_half = index_full_to_half(xp.first);
      KOKKOS_ASSERT(xp_half.second == 1);
      auto X_proj = project(mu, -1, this->sigma(xp_half.first));
      // minus sign in the projector comes from the derivative of D
      auto Xplus =
          (-1 * this->params.kappa * this->params.kappa * xp.second) * X_proj;
      auto temp1 = reconstruct(mu, -1, gauge_field(idx, mu) * Xplus);
      auto deriv = temp1 * gamma5(conj(this->chi(x_half)));

      auto Y_proj = project_alt(mu, 1, gamma5(conj(this->rho(xp_half.first))));
      auto YPlus =
          (this->params.kappa * this->params.kappa * xp.second) * Y_proj;
      auto temp3 =
          (reconstruct_alt(mu, 1, YPlus * conj(this->gauge_field(idx, mu))));
      deriv += this->y(x_half) * temp3;

      momentum(idx, mu) -= 2 * eps *
                           traceT(traceLessAntiHermitian(

                               deriv));
    }

    // X = chi , Y = chi_alt
    // Checkboard 0:
  }
  // iterations space is here the odd index space
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(TagOddContribution,
                                              const Indices... Idcs) const {
    Kokkos::Array<size_t, rank> x_half{Idcs...};

    auto idx = index_half_to_full(x_half, 1);

    for (size_t mu = 0; mu < rank; ++mu) {
      // X = chi , Y = chi_alt
      // Ned to go back to half index
      auto xp = shift_index_plus_bc<rank, index_t>(
          Kokkos::Array<index_t, rank>{idx}, mu, 1, 3, -1,
          this->gauge_field.dimensions);  // even
      auto xp_half = index_full_to_half(xp.first);
      KOKKOS_ASSERT(xp_half.second == 0);
      auto X_proj = project(mu, -1, this->y(xp_half.first));
      // minus sign in the projector comes from the derivative of D
      auto Xplus =
          (-1 * this->params.kappa * this->params.kappa * xp.second) * X_proj;
      auto temp1 = reconstruct(mu, -1, gauge_field(idx, mu) * Xplus);
      auto deriv = temp1 * gamma5(conj(this->rho(x_half)));

      auto Y_proj = project_alt(mu, 1, gamma5(conj(this->chi(xp_half.first))));
      auto YPlus =
          (this->params.kappa * this->params.kappa * xp.second) * Y_proj;
      auto temp3 =
          (reconstruct_alt(mu, 1, YPlus * conj(this->gauge_field(idx, mu))));
      deriv += this->sigma(x_half) * temp3;

      momentum(idx, mu) -= 2 * eps *
                           traceT(traceLessAntiHermitian(

                               deriv));
    }

    // X = chi , Y = chi_alt
    // Checkboard 0:
  }
  void update(const real_t step_size) override {
    Kokkos::Profiling::pushRegion("UpdateMomentumEO");
    eps = step_size;
    // print_spinor__int(this->phi(0, 0, 0, 0), "HB Phi at updateMoemtum");

    IndexArray<rank> start;

    D_n.init(this->phi.dimensions);

    D_s.init(this->phi.dimensions);
    // reset solver fields

    Kokkos::deep_copy(this->x0.field, zeroSpinor<Nc, RepDim>());

    if (KLFT_VERBOSITY > 4) {
      printf("Solving insde UpdateMomentumWilson:");
    }
    //
    this->solver.set_DiracOperator(D_n);
    if constexpr (Solver::solver_type_value ==
                  SolverType::KLFT_SOLVER_BICGSTAB) {
      ax<DSpinorFieldType>(this->D_s.params.massShift, this->phi, this->temp);
      solver.set_problem(this->temp);

      solver.template solve<Tags::TagSe>(
          this->x0,
          this->tol * 0.01);  // shoukld now be correct
      Kokkos::deep_copy(this->y.field, solver.x.field);
      axpyG5<DSpinorFieldType>(complex_t(1.0, 0.0), this->phi, this->y,
                               this->temp);
      solver.set_problem(this->temp);
      solver.template solve<Tags::TagSe>(this->x0, this->tol);
      this->chi = solver.x;
    } else {
      D_s.template apply<Tags::TagG5Se>(
          this->phi, this->solver.get_temp_field(),
          this->temp);  // y = (gamma_5 S_e+ pho gamma_5)Phi, // Massshift =
                        // true
      this->solver.set_problem(this->temp);
      solver.template solve<Tags::TagDdaggerD>(
          this->x0, this->tol);  // solver.x = (Q^-1 +rho q^-2 gamma5) phi //
      // massshift = false
      this->chi = solver.x;

      D_n.template apply<Tags::TagG5Se>(this->chi,
                                        this->solver.get_temp_field(), this->y);
      axpy<DSpinorFieldType>(-complex_t(1.0, 0.0), this->phi, this->y,
                             this->y);  // y = -phi + S_e solver.x = rho * Q^-1
                                        // gamma5 phi // Massshift = false
    }

    D_n.template apply<Tags::TagHoe>(this->chi, this->rho);
    D_n.template apply<Tags::TagHoe>(this->y, this->sigma);
    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }
    // print_SUNAdj(momentum(1, 0, 0, 0, 0), "Before Update Momentum");
    // launch the kernel
    KTune::parallel_for(
        "UpdateMomentumWilson",
        Policy<rank, TagEvenContribution>(start, this->phi.dimensions), *this);
    KTune::parallel_for(
        "UpdateMomentumWilson",
        Policy<rank, TagOddContribution>(start, this->phi.dimensions), *this);
    // print_SUNAdj(momentum(1, 0, 0, 0, 0), "After Update Momentum");

    Kokkos::fence();
    Kokkos::Profiling::popRegion();
  }
};

}  // namespace klft
