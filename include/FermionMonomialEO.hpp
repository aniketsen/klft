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
#include "DiracOperator.hpp"
#include "GLOBAL.hpp"
#include "Monomial.hpp"
#include "Solver.hpp"
#include "SpinorFieldLinAlg.hpp"

#define SQRT2INV \
  0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367506  // Oeis A010503
namespace klft {
template <class RNGType,
          typename DAdjFieldType,
          template <class DiracOpT> class _Solver,
          class DiracOpT>
class FermionMonomialEO
    : public Monomial<typename DiracOpT::DGaugeFieldType, DAdjFieldType> {
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
  using FermionField = typename DSpinorFieldType::type;
  using DiracOperator = DiracOpT;
  using Solver = _Solver<DiracOpT>;
  Solver solver;
  DiracOperator dirac_op;

 public:
  FermionField& phi;
  const diracParams params;
  const real_t tol;
  RNGType rng;
  FermionMonomialEO(FermionField& _phi,
                    const diracParams& params_,
                    const real_t& tol_,
                    RNGType& RNG_,
                    unsigned int _time_scale)
      : Monomial<DGaugeFieldType, DAdjFieldType>(_time_scale),
        phi(_phi),
        params(params_),
        rng(RNG_),
        tol(tol_) {
    solver.init(this->phi.dimensions);
    this->dirac_op = DiracOperator(params_);
    Monomial<DGaugeFieldType, DAdjFieldType>::monomial_type =
        KLFT_MONOMIAL_FERMION;
    printf("Created Fermion Monomial EO\n");
  }

  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("FermionHeatbathEO");
    auto dims = phi.dimensions;

    FermionField R(dims, rng, 0, SQRT2INV);

    Monomial<DGaugeFieldType, DAdjFieldType>::H_old =
        spinor_norm_sq<DSpinorFieldType>(R);
    dirac_op.set_gauge(h.gauge_field);
    dirac_op.init(dims);

    dirac_op.template apply<Tags::TagG5Se>(R, this->phi);
    Kokkos::Profiling::popRegion();
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("FermionAcceptEO");
    auto dims = phi.dimensions;

    FermionField x(dims, complex_t(0.0, 0.0));
    FermionField x0(dims, complex_t(0.0, 0.0));
    dirac_op.set_gauge(h.gauge_field);
    dirac_op.init(dims);
    this->solver.set_DiracOperator(dirac_op);
    this->solver.set_problem(this->phi);

    if (KLFT_VERBOSITY > 4) {
      printf("Solving inside Fermion Monomial accept:");
    }
    if constexpr (Solver::solver_type_value ==
                  SolverType::KLFT_SOLVER_BICGSTAB) {
      solver.template solve<Tags::TagSedagger>(x0, this->tol * 0.01);
      const FermionField chi = solver.x;

      Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
          spinor_dot_product<DSpinorFieldType>(chi, chi).real();

    } else {
      solver.template solve<Tags::TagDdaggerD>(x0, this->tol);
      const FermionField chi = solver.x;
      Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
          spinor_dot_product<DSpinorFieldType>(chi, this->phi)
              .real();  // S_F = chi^dagger chi = phi^dagger S_e^-1 S_e^-1 phi
    }

    Kokkos::Profiling::popRegion();
  }
  void print() override {
    printf("Fermion Monomial: %.20f\n", this->get_delta_H());
  }
};
}  // namespace klft
