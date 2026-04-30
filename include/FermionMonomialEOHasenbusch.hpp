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
          template <typename> class _Solver,
          class DiracOpT,
          class DiracOPNonShift>
class FermionMonomialEOHasenbusch
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
  using SolverNonShift = _Solver<DiracOPNonShift>;

 public:
  FermionField& phi;
  using DiracOpShifted = DiracOpT;
  using DiracOpNonShift = DiracOPNonShift;
  const diracParams params;
  Solver solver;
  SolverNonShift solver_nonshift;
  DiracOpNonShift D_n;
  DiracOpShifted D_s;
  FermionField x0;
  FermionField y;
  // auxillary fields

  const real_t tol;
  RNGType rng;
  FermionMonomialEOHasenbusch(FermionField& _phi,
                              const diracParams& params,
                              const real_t& tol_,
                              RNGType& RNG_,
                              unsigned int _time_scale)
      : Monomial<DGaugeFieldType, DAdjFieldType>(_time_scale),
        phi(_phi),
        params(params),

        rng(RNG_),
        tol(tol_) {
    this->x0 = FermionField(this->phi.dimensions, complex_t(0.0, 0.0));
    this->y = FermionField(this->phi.dimensions, complex_t(0.0, 0.0));
    // Auxillary
    solver.init(this->phi.dimensions);
    solver_nonshift.init(this->phi.dimensions);
    this->D_n = DiracOpNonShift(this->params);
    this->D_s = DiracOpShifted(this->params);
    Monomial<DGaugeFieldType, DAdjFieldType>::monomial_type =
        KLFT_MONOMIAL_FERMION;
    printf("Created Fermion HB Monomial EO\n");
  }

  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("FermionHeatbathEO");
    auto dims = phi.dimensions;

    D_n.set_gauge(h.gauge_field);
    D_n.init(this->phi.dimensions);

    D_s.set_gauge(h.gauge_field);
    D_s.init(this->phi.dimensions);

    FermionField R(dims, rng, 0, SQRT2INV);
    if constexpr (Solver::solver_type_value ==
                  SolverType::KLFT_SOLVER_BICGSTAB) {
      D_n.template apply<Tags::TagSe>(R, this->solver.get_temp_field(),
                                      this->phi);
    } else {
      D_n.template apply<Tags::TagG5Se>(R, this->solver.get_temp_field(),
                                        this->phi);
    }
    this->solver.set_DiracOperator(D_s);
    this->solver.set_problem(this->phi);
    if constexpr (Solver::solver_type_value ==
                  SolverType::KLFT_SOLVER_BICGSTAB) {
      solver.template solve<Tags::TagSe>(this->x0, this->tol);
      Kokkos::deep_copy(this->phi.field, this->solver.x.field);
      // axG5<DSpinorFieldType>(complex_t(1, 0), this->solver.x, this->phi);
    } else {
      solver.template solve<Tags::TagDdaggerD>(
          this->x0,
          this->tol);  // chi = S_e^-1 S_e^-1 R

      D_s.template apply<Tags::TagG5Se>(solver.x, this->x0, this->phi);
    }
    // CGMultiP<DiracOpShifted> cg;
    // FermionField cg_sol(this->phi.dimensions, complex_t(0.0, 0.0));
    // auto cg_in = D_n.template apply<Tags::TagG5Se>(R);
    // cg.init(this->phi.dimensions);
    // cg.set_DiracOperator(D_s);
    // cg.set_problem(cg_in);
    // cg.template solve<Tags::TagDdaggerD>(this->x0,
    //                                      this->tol);  // chi = S_e^-1 S_e^-1
    //                                      R

    // D_s.template apply<Tags::TagG5Se>(cg.x, this->x0, cg_sol);
    // printf("Norm between phi_cg and phi: %.20f",
    //        spinor_norm<DSpinorFieldType>(
    //            axpy<DSpinorFieldType>(-complex_t(1, 0), this->phi, cg_sol)));

    Monomial<DGaugeFieldType, DAdjFieldType>::H_old =
        spinor_norm_sq<DSpinorFieldType>(R);
    Kokkos::Profiling::popRegion();
    // print_spinor__int(this->phi(0, 0, 0, 0), "HB Phi at hatbath");s
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("FermionAcceptEO");
    auto dims = phi.dimensions;

    D_n.set_gauge(h.gauge_field);
    D_n.init(this->phi.dimensions);

    D_s.set_gauge(h.gauge_field);
    D_s.init(this->phi.dimensions);

    D_s.template apply<Tags::TagG5Se>(this->phi, this->solver.get_temp_field(),
                                      this->y);

    this->solver_nonshift.set_DiracOperator(D_n);
    this->solver_nonshift.set_problem(this->y);
    if (KLFT_VERBOSITY > 4) {
      printf("Solving inside Fermion Monomial accept:");
    }
    Kokkos::deep_copy(this->x0.field, zeroSpinor<Nc, RepDim>());
    if constexpr (_Solver<DiracOpT>::solver_type_value ==
                  SolverType::KLFT_SOLVER_BICGSTAB) {
      FermionField x(this->phi.dimensions, complex_t(0.0, 0.0));
      solver_nonshift.template solve<Tags::TagSedagger>(x0, this->tol * 0.01);
      Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
          spinor_dot_product<DSpinorFieldType>(solver_nonshift.x,
                                               solver_nonshift.x)
              .real();

    } else {
      solver_nonshift.template solve<Tags::TagDdaggerD>(this->x0, this->tol);
      Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
          spinor_dot_product<DSpinorFieldType>(y, solver_nonshift.x).real();
    }
    // CGMultiP<DiracOPNonShift> cg;
    // D_s.template apply<Tags::TagG5Se>(this->phi,
    // this->solver.get_temp_field(),
    //                                   this->y);
    // cg.init(this->phi.dimensions);
    // cg.set_DiracOperator(D_n);
    // cg.set_problem(this->y);

    // cg.template solve<Tags::TagDdaggerD>(this->x0, this->tol);
    // printf("CG Solver H_new: %.20f\n ",
    //        spinor_dot_product<DSpinorFieldType>(y, cg.x).real());
    // printf("Norm between cg  and bicgstab: %.20f",
    //        spinor_norm<DSpinorFieldType>(axpy<DSpinorFieldType>(
    //            -complex_t(1, 0), solver_nonshift.x, cg.x)));
    Kokkos::Profiling::popRegion();
  }
  void print() override {
    // printf("HB Fermion Monomial Before : %.20f\n", this->H_old);
    // printf("HB Fermion Monomial After : %.20f\n", this->H_new);

    printf("HB Fermion Monomial: %.20f\n", this->get_delta_H());
  }
};
}  // namespace klft
