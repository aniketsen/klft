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
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "MixedPrecision.hpp"
#include "SpinorFieldLinAlg.hpp"

namespace klft {
enum class SolverType { KLFT_SOLVER_CG, KLFT_SOLVER_BICGSTAB };
template <class _Solver, class DiracOpT>
class Solver {
  // using DSpinorFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DSpinorFieldType;
  // using DGaugeFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DGaugeFieldType;
  // template argument deduction and safety
 public:
  using DSpinorFieldType = typename DiracOpT::DSpinorFieldType;
  using DGaugeFieldType = typename DiracOpT::DGaugeFieldType;
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));

  // using DiracOp = DiracOpT<DSpinorFieldType, DGaugeFieldType,
  // DiracOpT::HasMassshift>;
  using DiracOp = DiracOpT;

  // using DiracOperator =
  //     DiracOperator<DerivedDiracOperator, DSpinorFieldType, DGaugeFieldType>;
  // using ConcreteSolver =
  //     _ConcreteSolver<DiracOperator, DSpinorFieldType, DGaugeFieldType>;

 public:
  using SpinorFieldType = typename DSpinorFieldType::type;
  using GaugeFieldType = typename DGaugeFieldType::type;
  SpinorFieldType b;
  SpinorFieldType x;  // Solution to DiracOP*x=b
  DiracOp dirac_op;
  // auxillary fields
  SpinorFieldType xk;
  SpinorFieldType rk;
  IndexArray<rank> dims;
  bool dirac_init = false;
  SolverType solver_type;
  Solver() = default;
  Solver(const SpinorFieldType& b, SpinorFieldType& x, const DiracOp& dirac_op)
      : b(b), x(x), dirac_op(dirac_op) {
    this->dims = this->x.dimensions;
    this->xk = SpinorFieldType(dims, complex_t(0.0, 0.0));
    this->rk = SpinorFieldType(dims, complex_t(0.0, 0.0));
    this->dirac_init = true;
  }

  Solver(const SpinorFieldType& b,
         SpinorFieldType& x,
         const DiracOp& dirac_op,
         SpinorFieldType& xk,
         SpinorFieldType& rk)
      : b(b), x(x), dirac_op(dirac_op), xk(xk), rk(rk), dims(x.dimensions) {
    this->dirac_init = true;
    this->dirac_op.init(this->b.dimensions);
  }

  template <typename Tag>
  void solve(const SpinorFieldType& x0, const real_t& tol) {
    Kokkos::Profiling::pushRegion("Solver");
    static_cast<_Solver*>(this)->template solve_int<Tag>(x0, tol);
    Kokkos::Profiling::popRegion();
  }
  void init(const IndexArray<rank>& dims) {
    this->dims = dims;
    this->xk = SpinorFieldType(dims, complex_t(0.0, 0.0));
    this->rk = SpinorFieldType(dims, complex_t(0.0, 0.0));
    static_cast<_Solver*>(this)->init_int();
  }
  void set_DiracOperator(const DiracOp& dirac_op) {
    this->dirac_op = dirac_op;
    if (!this->dirac_init) {
      /* code */
      static_cast<_Solver*>(this)->init_gauge();
    }
  }
  void set_problem(const SpinorFieldType& b) { this->b = b; }
  static constexpr SolverType get_type() { return _Solver::solver_type_value; }
  SpinorFieldType get_temp_field() {
    return static_cast<_Solver*>(this)->get_temp_field_init();
  }

  /// @brief Constructs the b vector when using an Even/Odd Precondition Field,
  /// assumes that the field saved in b is the even part of the Vector b
  /// @param odd_b
  void construct_problem(const SpinorFieldType& odd_b) {
    auto out_even_from_odd_b = dirac_op.template apply<Tags::TagHeo>(odd_b);
    axpy<DSpinorFieldType>(this->dirac_op.params.kappa, out_even_from_odd_b,
                           this->b, this->b);
  }

  /// @brief Reconstructs the Odd part of the solution
  /// with zero odd part
  void reconstruct_solution_0(SpinorFieldType& out) {
    dirac_op.template apply<Tags::TagHoe>(this->x, out);
    ax<DSpinorFieldType>(this->dirac_op.params.kappa, out, out);
  }
  SpinorFieldType reconstruct_solution_0() {
    auto out = SpinorFieldType(x.dimensions, complex_t(0.0, 0.0));
    reconstruct_solution_0(out);
    return out;
  }
  /// @brief Reconstructs the Odd part of the solution
  /// @param out
  void reconstruct_solution(const SpinorFieldType& odd_b,
                            SpinorFieldType& out) {
    dirac_op.template apply<Tags::TagHoe>(this->x, out);
    axpy<DSpinorFieldType>(this->dirac_op.params.kappa, out, odd_b, out);
  }
  SpinorFieldType reconstruct_solution(const SpinorFieldType& odd_b) {
    auto out = SpinorFieldType(x.dimensions, complex_t(0.0, 0.0));
    reconstruct_solution(odd_b, out);
    return out;
  }
};
// // Deduction guide for Solver
// template <typename Operator, typename SpinorType>
// Solver(const SpinorType&, SpinorType&, const Operator&)
//     -> Solver<typename Operator::Derived,
//               SpinorType::rank,
//               SpinorType::Nc,
//               SpinorType::RepDim>;

template <class DiracOpT>
class CGSolver : public Solver<CGSolver<DiracOpT>, DiracOpT> {
  // using DSpinorFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DSpinorFieldType;
  // using DGaugeFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DGaugeFieldType;

 public:
  static constexpr SolverType solver_type_value = SolverType::KLFT_SOLVER_CG;
  using Base = Solver<CGSolver<DiracOpT>, DiracOpT>;
  using Base::Base;
  using DSpinorFieldType = typename Base::DSpinorFieldType;
  using DGaugeFieldType = typename Base::DGaugeFieldType;
  using SpinorFieldType = typename Base::DSpinorFieldType::type;
  using GaugeFieldType = typename Base::DGaugeFieldType::type;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));

  template <typename Tag>
  void solve_int(const SpinorFieldType& x0, const real_t& tol) {
    Kokkos::deep_copy(this->xk.field, x0.field);  // x_0

    axpy<DSpinorFieldType>(-1, this->dirac_op.template apply<Tag>(this->xk),
                           this->b, this->rk);

    Kokkos::deep_copy(this->pk.field, this->rk.field);  // p_0 // d_0
    real_t rk_norm = spinor_norm<DSpinorFieldType>(
        this->rk, this->norm_per_site);  //\delta_0
    int num_iter = 0;
    while (rk_norm > tol) {
      this->dirac_op.template apply<Tag>(this->pk, this->temp_D, this->apk);
      // z = Ad_k
      const complex_t rkrk = spinor_dot_product<DSpinorFieldType>(
          this->rk, this->rk, this->dot_product_per_site);
      const complex_t alpha =
          (rkrk / spinor_dot_product<DSpinorFieldType>(
                      this->pk, this->apk,
                      this->dot_product_per_site));  // Always real
      axpy<DSpinorFieldType>(alpha, this->pk, this->xk, this->xk);
      // xk = spinor_add_mul<DSpinorFieldType>(xk, pk, alpha);
      // xk = axpy<DSpinorFieldType>(alpha, pk, xk);
      axpy<DSpinorFieldType>(-alpha, this->apk, this->rk, this->rk);
      // rk = axpy<DSpinorFieldType>(-alpha, apk, rk);
      // rk = spinor_sub_mul<DSpinorFieldType>(rk, apk, alpha);
      const complex_t beta =
          (spinor_dot_product<DSpinorFieldType>(this->rk, this->rk,
                                                this->dot_product_per_site) /
           rkrk);
      axpy<DSpinorFieldType>(beta, this->pk, this->rk, this->pk);
      // pk = axpy<DSpinorFieldType>(beta, pk, rk);
      // pk = spinor_add_mul<DSpinorFieldType>(rk, pk, beta);
      // Check if swapping is needed of pk and rk, should be correct

      rk_norm = spinor_norm<DSpinorFieldType>(this->rk, this->norm_per_site);
      num_iter++;
      if (KLFT_VERBOSITY > 2) {
        printf("CG Iteration %d: rk_norm = %.15f\n", num_iter, rk_norm);
        if (KLFT_VERBOSITY > 3) {
          printf("Norm of (b - A*x) %.15f\n",
                 spinor_norm<DSpinorFieldType>(
                     axpy<DSpinorFieldType>(
                         -1.0, this->dirac_op.template apply<Tag>(this->xk),
                         this->b),
                     this->norm_per_site));
        }
      }
    }
    this->dirac_op.template apply<Tag>(this->xk, this->apk, this->temp_D);
    axpy<DSpinorFieldType>(-1, this->temp_D, this->b, this->temp_D);
    const real_t ex_res =
        spinor_norm<DSpinorFieldType>(this->temp_D, this->norm_per_site);

    if (Kokkos::abs(ex_res / spinor_norm<DSpinorFieldType>(
                                 this->xk, this->norm_per_site)) > tol) {
      printf(
          "EX_res: %.20f, Roundoff Error, relaunching CG solver with new "
          "initial guess\n",
          ex_res);
      this->template solve<Tag>(this->xk, tol);
    } else {
      if (KLFT_VERBOSITY > 1) {
        printf("CG solver converged in %d iterations\n", num_iter);
      }
      this->x = this->xk;
    }
  }
  CGSolver() { this->solver_type = SolverType::KLFT_SOLVER_CG; };
  CGSolver(const SpinorFieldType& b,
           SpinorFieldType& x,
           const Base::DiracOp& dirac_op)
      : Base(b, x, dirac_op) {
    this->dims = this->x.dimensions;
    this->xk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->rk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->apk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->temp_D = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->pk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(this->dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(this->dims, complex_t(0.0, 0.0));
    this->solver_type = SolverType::KLFT_SOLVER_CG;
  }

  CGSolver(const SpinorFieldType& b,
           SpinorFieldType& x,
           const Base::DiracOp& dirac_op,
           SpinorFieldType& xk,
           SpinorFieldType& rk,
           SpinorFieldType& apk,
           SpinorFieldType& temp_D,
           SpinorFieldType& pk,
           typename DeviceScalarFieldType<rank>::type& norm_per_site,
           typename DeviceFieldType<rank>::type(dot_product_per_site))
      : Base(b, x, dirac_op, xk, rk),

        apk(apk),
        temp_D(temp_D),
        pk(pk),
        norm_per_site(norm_per_site),
        dot_product_per_site(dot_product_per_site) {
    this->solver_type = SolverType::KLFT_SOLVER_CG;
  }
  void init_int() {
    this->apk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->temp_D = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->pk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(this->dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(this->dims, complex_t(0.0, 0.0));
  }
  void init_gauge() {}
  SpinorFieldType get_temp_field_init() { return this->temp_D; }

 private:
  SpinorFieldType temp_D;
  SpinorFieldType apk;
  SpinorFieldType pk;
  typename DeviceScalarFieldType<rank>::type norm_per_site;
  typename DeviceFieldType<rank>::type dot_product_per_site;
};

template <class DiracOpT>
class CGMultiP : public Solver<CGMultiP<DiracOpT>, DiracOpT> {
 public:
  static constexpr SolverType solver_type_value = SolverType::KLFT_SOLVER_CG;
  using Base = Solver<CGMultiP<DiracOpT>, DiracOpT>;
  using Base::Base;
  using DSpinorFieldType = typename Base::DSpinorFieldType;
  using DGaugeFieldType = typename DiracOpT::strippedGaugeField;
  using SpinorFieldType = typename Base::DSpinorFieldType::type;
  using GaugeFieldType = typename Base::DGaugeFieldType::type;
  using DSloppyGaugeFieldType =
      WithPrecisionGaugeField<DGaugeFieldType, Kokkos::complex<float>>::type;
  using SloppyGaugFieldType = typename DSloppyGaugeFieldType::type;
  using DSploppySpinorFieldType =
      WithPrecisionSpinorField<DSpinorFieldType, Kokkos::complex<float>>::type;
  using SloppySpinorField = typename DSploppySpinorFieldType::type;
  using SloppyDiracOpT =
      DiracOpT::template rebind<DSploppySpinorFieldType, DSloppyGaugeFieldType>;
  // TODO Similar for gaugefield
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));

  template <typename Tag>
  void solve_int(const SpinorFieldType& x0, const real_t& tol) {
    // by standart all fields which are aleready part of this are sloppy
    // presicion expect x_0,this->b
    this->dirac_op.template apply<Tag>(x0, this->temp_D_full_complexity,
                                       this->rk);
    Kokkos::deep_copy(this->xk.field, x0.field);
    axpy<DSpinorFieldType>(-1, this->rk, this->b,
                           this->rk);  // r_l high precision

    changePrecisionSpinorField<DSploppySpinorFieldType, DSpinorFieldType>(
        this->r_sloppy,
        this->rk);  // wharschenlich andersrum soll von rk nach
    // r_sloppy field

    changePrecisionSpinorField<DSploppySpinorFieldType, DSpinorFieldType>(
        this->x_sloppy, x0);
    changePrecisionGaugeField<DSloppyGaugeFieldType, DGaugeFieldType>(
        this->sloppy_g_in, this->dirac_op.g_in);

    sloppy_dirac.init(r_sloppy.dimensions);
    Kokkos::deep_copy(this->pk.field, this->r_sloppy.field);  // p_0 // d_0
    real_t rk2 = spinor_norm_sq<DSploppySpinorFieldType>(
        this->r_sloppy, this->norm_per_site);  //\delta_0
    real_t r0Norm = sqrt(rk2);  // Norm at last reliable update / restart
    real_t maxrr = r0Norm;      // max recursive residual
    real_t maxrx = r0Norm;
    real_t rknorm = r0Norm;  // max residual during x updates
    int num_iter = 0;
    int num_reliable_updates = 0;
    while (rknorm > tol) {
      sloppy_dirac.template apply<Tag>(this->pk, this->temp_D, this->apk);
      // z = Ad_k

      const complex_t alpha =
          (rk2 / spinor_dot_product<DSploppySpinorFieldType>(
                     this->pk, this->apk,
                     this->dot_product_per_site));  // Always real
      axpy<DSploppySpinorFieldType>(-alpha, this->apk, this->r_sloppy,
                                    this->r_sloppy);
      auto rk2_chached = rk2;
      // reliable ipdate condition:
      rk2 = spinor_norm_sq<DSploppySpinorFieldType>(this->r_sloppy,
                                                    this->norm_per_site);
      rknorm = sqrt(rk2);
      if (rknorm > maxrr)
        maxrr = rknorm;
      if (sqrt(rk2) > maxrx)
        maxrx = sqrt(rk2);

      bool updateX = (rknorm < this->delta * r0Norm && r0Norm <= maxrx);
      bool updateR =
          ((rknorm < this->delta * maxrr && r0Norm <= maxrr) || updateX);
      if (!(updateR)) {
        axpy<DSploppySpinorFieldType>(alpha, this->pk, this->x_sloppy,
                                      this->x_sloppy);

        const complex_t beta = (rk2 / rk2_chached);
        axpy<DSploppySpinorFieldType>(beta, this->pk, this->r_sloppy, this->pk);

        /* code */
      } else {
        // reliable update
        axpy<DSploppySpinorFieldType>(alpha, this->pk, this->x_sloppy,
                                      this->x_sloppy);
        xpyMixed<DSpinorFieldType, DSploppySpinorFieldType>(this->xk,
                                                            this->x_sloppy);
        this->dirac_op.template apply<Tag>(
            this->xk, this->temp_D_full_complexity, this->rk);
        axpy<DSpinorFieldType>(-1, this->rk, this->b, this->rk);
        rk2 = spinor_norm_sq<DSpinorFieldType>(
            this->rk, this->norm_per_site);  // Full presicion
        resetSpinorField<DSploppySpinorFieldType>(this->x_sloppy);
        changePrecisionSpinorField<DSploppySpinorFieldType, DSpinorFieldType>(
            this->r_sloppy, this->rk);
        rknorm = sqrt(rk2);
        maxrr = r0Norm;
        maxrx = r0Norm;
        r0Norm = rknorm;

        const complex_t beta = rk2 / rk2_chached;
        axpy<DSploppySpinorFieldType>(beta, this->pk, this->r_sloppy, this->pk);
        num_reliable_updates++;
      }

      // pk = axpy<DSpinorFieldType>(beta, pk, rk);
      // pk = spinor_add_mul<DSpinorFieldType>(rk, pk, beta);
      // Check if swapping is needed of pk and rk, should be correct

      num_iter++;
      if (KLFT_VERBOSITY > 2) {
        printf("CGMultiP Iteration %d: rk_norm = %.15f\n", num_iter, rknorm);
        if (KLFT_VERBOSITY > 3) {
          printf("Norm of (b - A*x) %.15f\n",
                 spinor_norm<DSpinorFieldType>(
                     axpy<DSpinorFieldType>(
                         -1.0, this->dirac_op.template apply<Tag>(this->xk),
                         this->b),
                     this->norm_per_site));
        }
      }
    }
    // this->dirac_op.template apply<Tag>(this->xk, this->apk, this->temp_D);
    // axpy<DSpinorFieldType>(-1, this->temp_D, this->b, this->temp_D);
    // const real_t ex_res =
    //     spinor_norm<DSpinorFieldType>(this->temp_D, this->norm_per_site);

    // if (Kokkos::abs(ex_res / spinor_norm<DSpinorFieldType>(
    //                              this->xk, this->norm_per_site)) > tol) {
    //   printf(
    //       "EX_res: %.20f, Roundoff Error, relaunching CG solver with new "
    //       "initial guess\n",
    //       ex_res);
    //   this->template solve<Tag>(this->xk, tol);
    // } else
    {
      xpyMixed<DSpinorFieldType, DSploppySpinorFieldType>(this->xk,
                                                          this->x_sloppy);
      if (KLFT_VERBOSITY > 1) {
        printf("CG Multi Precision solver converged in %d iterations\n",
               num_iter);
      }
      this->x = this->xk;
    }
  }
  CGMultiP() { this->solver_type = SolverType::KLFT_SOLVER_CG; };
  CGMultiP(const SpinorFieldType& b,
           SpinorFieldType& x,
           const Base::DiracOp& dirac_op,
           const real_t& delta = 0.1)
      : Base(b, x, dirac_op), delta(delta) {
    this->dims = this->x.dimensions;
    this->xk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->rk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->temp_D_full_complexity =
        SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->apk = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->temp_D = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->x_sloppy = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->r_sloppy = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->sloppy_g_in = SloppyGaugFieldType(this->dirac_op.g_in.dimensions,
                                            complexsingle_t(0, 0));
    this->pk = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(this->dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(this->dims, complex_t(0.0, 0.0));
    this->sloppy_dirac = SloppyDiracOpT(sloppy_g_in, this->dirac_op.params);
    this->solver_type = SolverType::KLFT_SOLVER_CG;
  }

  CGMultiP(const SpinorFieldType& b,
           SpinorFieldType& x,
           const Base::DiracOp& dirac_op,
           SpinorFieldType& xk,
           SpinorFieldType& rk,
           SloppySpinorField& apk,
           SloppySpinorField& temp_D,
           SloppySpinorField& pk,
           typename DeviceScalarFieldType<rank>::type& norm_per_site,
           typename DeviceFieldType<rank>::type(dot_product_per_site),
           SpinorFieldType& temp_D_full_complexity,
           const SloppyGaugFieldType& sloppy_g_in,
           const real_t& delta)
      : Base(b, x, dirac_op, xk, rk),
        delta(delta),
        apk(apk),
        temp_D(temp_D),
        pk(pk),
        norm_per_site(norm_per_site),
        dot_product_per_site(dot_product_per_site),
        temp_D_full_complexity(temp_D_full_complexity),
        sloppy_g_in(sloppy_g_in) {
    this->sloppy_dirac = SloppyDiracOpT(sloppy_g_in, this->dirac_op.params);
    this->solver_type = SolverType::KLFT_SOLVER_CG;
  }
  void init_int() {
    this->delta = 0.1;
    this->temp_D_full_complexity =
        SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->apk = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->temp_D = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->x_sloppy = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->r_sloppy = SloppySpinorField(this->dims, complex_t(0.0, 0.0));

    this->pk = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(this->dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(this->dims, complex_t(0.0, 0.0));
    this->solver_type = SolverType::KLFT_SOLVER_CG;
  }
  void init_gauge() {
    if (!this->sloppy_g_in.field.is_allocated()) {
      this->sloppy_g_in = SloppyGaugFieldType(this->dirac_op.g_in.dimensions,
                                              complexsingle_t(0, 0));
    }
    if (!this->sloppy_dirac_set) {
      this->sloppy_dirac = SloppyDiracOpT(sloppy_g_in, this->dirac_op.params);
      this->sloppy_dirac_set = true;
    }
  }
  SpinorFieldType get_temp_field_init() { return this->temp_D_full_complexity; }

 private:
  SloppySpinorField r_sloppy;
  SloppySpinorField x_sloppy;
  SloppySpinorField apk;
  SloppySpinorField pk;
  SloppySpinorField temp_D;
  SpinorFieldType temp_D_full_complexity;
  real_t delta;
  SloppyGaugFieldType sloppy_g_in;
  SloppyDiracOpT sloppy_dirac;
  bool sloppy_dirac_set = false;
  typename DeviceScalarFieldType<rank>::type norm_per_site;
  typename DeviceFieldType<rank>::type dot_product_per_site;
};

template <class DiracOpT>
class BiCGStab : public Solver<BiCGStab<DiracOpT>, DiracOpT> {
  // using DSpinorFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DSpinorFieldType;
  // using DGaugeFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DGaugeFieldType;

 public:
  static constexpr SolverType solver_type_value =
      SolverType::KLFT_SOLVER_BICGSTAB;
  using Base = Solver<BiCGStab<DiracOpT>, DiracOpT>;
  using Base::Base;
  using DSpinorFieldType = typename Base::DSpinorFieldType;
  using DGaugeFieldType = typename Base::DGaugeFieldType;
  using SpinorFieldType = typename Base::DSpinorFieldType::type;
  using GaugeFieldType = typename Base::DGaugeFieldType::type;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));

  template <typename Tag>
  void solve_int(const SpinorFieldType& x0, const real_t& tol) {
    auto dims = x0.dimensions;

    Kokkos::deep_copy(this->xk.field, x0.field);  // x_0
    axpy<DSpinorFieldType>(-1, this->dirac_op.template apply<Tag>(this->xk),
                           this->b, this->r0);
    Kokkos::deep_copy(this->rk.field, r0.field);

    Kokkos::deep_copy(this->pk.field, r0.field);  // p_0 // d_0
                                                  //\delta_0

    complex_t rho = spinor_dot_product<DSpinorFieldType>(this->r0, this->rk);
    real_t rk_norm = spinor_norm<DSpinorFieldType>(this->rk);
    size_t num_iter = 0;

    while (rk_norm > tol) {
      // apk = A * pk
      this->dirac_op.template apply<Tag>(this->pk, this->temp_D, this->apk);

      const complex_t rho_old = rho;

      const complex_t alpha =
          rho_old / spinor_dot_product<DSpinorFieldType>(this->r0, this->apk);

      // rk = rk - alpha * apk
      axpy<DSpinorFieldType>(-alpha, this->apk, this->rk, this->rk);

      // t = A * rk
      this->dirac_op.template apply<Tag>(this->rk, this->temp_D, t);

      const complex_t omega =
          spinor_dot_product<DSpinorFieldType>(this->t, this->rk) /
          spinor_dot_product<DSpinorFieldType>(this->t, this->t);

      // // xk += omega * rk +alpha * pk

      axpby<DSpinorFieldType>(alpha, this->pk, omega, this->rk, this->xk);

      // rk = rk - omega * t
      axpy<DSpinorFieldType>(-omega, this->t, this->rk, this->rk);

      // rho = (r0, rk)
      rho = spinor_dot_product<DSpinorFieldType>(this->r0, this->rk);

      const complex_t beta = (rho / rho_old) * (alpha / omega);

      // pk = rk + beta * pk - beta * omega * apk
      axpbypcz<DSpinorFieldType>(beta,  // beta * pk
                                 this->pk,
                                 -beta * omega,  // -beta*omega * apk
                                 this->apk, complex_t(1, 0),  // + rk
                                 this->rk, this->pk);

      rk_norm = spinor_norm<DSpinorFieldType>(this->rk);
      num_iter++;

      if (KLFT_VERBOSITY > 2) {
        printf("BiCGStab Iteration %zu: rk_norm = %.15f\n", num_iter, rk_norm);
      }
    }

    this->dirac_op.template apply<Tag>(this->xk, this->apk, this->temp_D);
    axpy<DSpinorFieldType>(-1, this->temp_D, this->b, this->temp_D);
    const real_t ex_res = spinor_norm<DSpinorFieldType>(this->temp_D);

    if (Kokkos::abs(ex_res / spinor_norm<DSpinorFieldType>(this->xk)) > tol) {
      printf(
          "EX_res: %.20f, Roundoff Error, relaunching CG solver with new "
          "initial guess\n",
          ex_res);
      this->template solve<Tag>(this->xk, tol);
    } else {
      if (KLFT_VERBOSITY > 1) {
        printf("BiCGstab solver converged in %zu iterations\n", num_iter);
      }
      this->x = this->xk;
    }
  }
  BiCGStab() { this->solver_type = SolverType::KLFT_SOLVER_BICGSTAB; };
  BiCGStab(const SpinorFieldType& b,
           SpinorFieldType& x,
           const Base::DiracOp& dirac_op)
      : Base(b, x, dirac_op) {
    this->xk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->rk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->apk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->temp_D = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->pk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(this->dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(this->dims, complex_t(0.0, 0.0));
    this->t = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->r0 = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
  }
  void init_int() {
    this->xk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->rk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->apk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->temp_D = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->pk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(this->dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(this->dims, complex_t(0.0, 0.0));
    this->t = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->r0 = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->solver_type = SolverType::KLFT_SOLVER_BICGSTAB;
  }
  void init_gauge() {}
  SpinorFieldType get_temp_field_init() { return this->temp_D; }
  BiCGStab(const SpinorFieldType& b,
           SpinorFieldType& x,
           const Base::DiracOp& dirac_op,
           SpinorFieldType& xk,
           SpinorFieldType& rk,
           SpinorFieldType& apk,
           SpinorFieldType& temp_D,
           SpinorFieldType& pk,
           SpinorFieldType& t,
           SpinorFieldType& r0,

           typename DeviceScalarFieldType<rank>::type& norm_per_site,
           typename DeviceFieldType<rank>::type(dot_product_per_site))
      : Base(b, x, dirac_op, xk, rk),

        apk(apk),
        temp_D(temp_D),
        pk(pk),
        norm_per_site(norm_per_site),
        t(t),
        r0(r0),
        dot_product_per_site(dot_product_per_site) {
    this->solver_type = SolverType::KLFT_SOLVER_BICGSTAB;
  }

 private:
  SpinorFieldType temp_D;
  SpinorFieldType apk;
  SpinorFieldType pk;
  SpinorFieldType t;
  SpinorFieldType r0;

  typename DeviceScalarFieldType<rank>::type norm_per_site;
  typename DeviceFieldType<rank>::type dot_product_per_site;
};
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

template <class DiracOpT>
class BiCGStabMultiP : public Solver<BiCGStabMultiP<DiracOpT>, DiracOpT> {
  // using DSpinorFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DSpinorFieldType;
  // using DGaugeFieldType =
  //     typename DiracOpFieldTypeTraits<DiracOperator>::DGaugeFieldType;

 public:
  static constexpr SolverType solver_type_value =
      SolverType::KLFT_SOLVER_BICGSTAB;
  using Base = Solver<BiCGStabMultiP<DiracOpT>, DiracOpT>;
  using Base::Base;
  using DSpinorFieldType = typename Base::DSpinorFieldType;
  using DGaugeFieldType = typename DiracOpT::strippedGaugeField;
  using SpinorFieldType = typename Base::DSpinorFieldType::type;
  using GaugeFieldType = typename Base::DGaugeFieldType::type;
  using DSloppyGaugeFieldType =
      WithPrecisionGaugeField<DGaugeFieldType, Kokkos::complex<float>>::type;
  using SloppyGaugFieldType = typename DSloppyGaugeFieldType::type;
  using DSploppySpinorFieldType =
      WithPrecisionSpinorField<DSpinorFieldType, Kokkos::complex<float>>::type;
  using SloppySpinorField = typename DSploppySpinorFieldType::type;
  using SloppyDiracOpT =
      DiracOpT::template rebind<DSploppySpinorFieldType, DSloppyGaugeFieldType>;
  // TODO Similar for gaugefield
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<typename Base::DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));

  template <typename Tag>
  void solve_int(const SpinorFieldType& x0, const real_t& tol) {
    auto dims = x0.dimensions;

    Kokkos::deep_copy(this->xk.field, x0.field);  // x_0
    changePrecisionSpinorField<DSploppySpinorFieldType, DSpinorFieldType>(
        this->x_sloppy, x0);
    this->dirac_op.template apply<Tag>(x0, this->temp_D_full_complexity,
                                       this->rk);
    axpy<DSpinorFieldType>(-1, this->rk, this->b, this->rk);
    changePrecisionSpinorField<DSploppySpinorFieldType, DSpinorFieldType>(
        this->r0, this->rk);

    Kokkos::deep_copy(this->r_sloppy.field, this->r0.field);
    Kokkos::deep_copy(this->pk.field, this->r_sloppy.field);
    changePrecisionGaugeField<DSloppyGaugeFieldType, DGaugeFieldType>(
        this->sloppy_g_in, this->dirac_op.g_in);
    // rk, xk full percison pk,apk,t,x_sloppy, r_sloppy, r0_sloppy
    sloppy_dirac.init(r_sloppy.dimensions);
    complex_t rho =
        spinor_dot_product<DSploppySpinorFieldType>(this->r0, this->r_sloppy);
    real_t rk2 = spinor_norm_sq<DSploppySpinorFieldType>(
        this->r_sloppy, this->norm_per_site);  //\delta_0
    real_t r0Norm = sqrt(rk2);  // Norm at last reliable update / restart
    real_t maxrr = r0Norm;      // max recursive residual
    real_t maxrx = r0Norm;
    real_t rknorm = r0Norm;  // max residual during x updates
    size_t num_iter = 0;

    while (rknorm > tol) {
      // apk = A * pk
      sloppy_dirac.template apply<Tag>(this->pk, this->temp_D, this->apk);

      const complex_t rho_old = rho;

      const complex_t alpha =
          rho_old / spinor_dot_product<DSploppySpinorFieldType>(
                        this->r0, this->apk, this->dot_product_per_site);

      // rk = rk - alpha * apk
      axpy<DSploppySpinorFieldType>(-alpha, this->apk, this->r_sloppy,
                                    this->r_sloppy);

      // t = A * rk
      sloppy_dirac.template apply<Tag>(this->r_sloppy, this->temp_D, t);

      const complex_t omega =
          spinor_dot_product<DSploppySpinorFieldType>(
              this->t, this->r_sloppy, this->dot_product_per_site) /
          spinor_dot_product<DSploppySpinorFieldType>(
              this->t, this->t, this->dot_product_per_site);

      // // xk += omega * rk +alpha * pk

      axpby<DSploppySpinorFieldType>(alpha, this->pk, omega, this->r_sloppy,
                                     this->x_sloppy);

      // rk = rk - omega * t
      axpy<DSploppySpinorFieldType>(-omega, this->t, this->r_sloppy,
                                    this->r_sloppy);

      // rho = (r0, rk)
      rho = spinor_dot_product<DSploppySpinorFieldType>(
          this->r0, this->r_sloppy, this->dot_product_per_site);

      const complex_t beta = (rho / rho_old) * (alpha / omega);

      // pk = rk + beta * pk - beta * omega * apk
      axpbypcz<DSploppySpinorFieldType>(beta,  // beta * pk
                                        this->pk,
                                        -beta * omega,  // -beta*omega * apk
                                        this->apk, complex_t(1, 0),  // + rk
                                        this->r_sloppy, this->pk);

      rknorm = spinor_norm<DSploppySpinorFieldType>(this->r_sloppy,
                                                    this->norm_per_site);
      if (rknorm > maxrx)
        maxrx = rknorm;
      if (rknorm > maxrr)
        maxrr = rknorm;
      bool updateX = (rknorm < this->delta * r0Norm && r0Norm <= maxrx);
      bool updateR =
          ((rknorm < this->delta * maxrr && r0Norm <= maxrr) || updateX);
      if (updateR)  // do reliable update
      {
        xpyMixed<DSpinorFieldType, DSploppySpinorFieldType>(this->xk,
                                                            this->x_sloppy);
        this->dirac_op.template apply<Tag>(
            this->xk, this->temp_D_full_complexity, this->rk);
        axpy<DSpinorFieldType>(-1, this->rk, this->b, this->rk);
        rk2 = spinor_norm_sq<DSpinorFieldType>(this->rk, this->norm_per_site);
        resetSpinorField<DSploppySpinorFieldType>(this->x_sloppy);
        changePrecisionSpinorField<DSploppySpinorFieldType, DSpinorFieldType>(
            this->r_sloppy, this->rk);
        rknorm = sqrt(rk2);
        r0Norm = rknorm;
        maxrr = r0Norm;
        maxrx = r0Norm;
      }

      num_iter++;

      if (KLFT_VERBOSITY > 2) {
        printf("BiCGStabMultiP Iteration %zu: rk_norm = %.15f\n", num_iter,
               rknorm);
        if (KLFT_VERBOSITY > 3) {
          printf("Norm of (b - A*x) %.15f\n",
                 spinor_norm<DSpinorFieldType>(
                     axpy<DSpinorFieldType>(
                         -1.0, this->dirac_op.template apply<Tag>(this->xk),
                         this->b),
                     this->norm_per_site));
        }
      }
    }

    xpyMixed<DSpinorFieldType, DSploppySpinorFieldType>(this->xk,
                                                        this->x_sloppy);
    if (KLFT_VERBOSITY > 1) {
      printf("BiCGStabMultiP solver converged in %zu iterations\n", num_iter);
    }
    this->x = this->xk;
  }
  BiCGStabMultiP() { this->solver_type = SolverType::KLFT_SOLVER_BICGSTAB; };
  BiCGStabMultiP(const SpinorFieldType& b,
                 SpinorFieldType& x,
                 const Base::DiracOp& dirac_op,
                 const real_t& delta = 0.1)
      : Base(b, x, dirac_op), delta(delta) {
    this->xk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->rk = SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->temp_D_full_complexity =
        SpinorFieldType(this->dims, complex_t(0.0, 0.0));
    this->apk = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->temp_D = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->pk = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(this->dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(this->dims, complex_t(0.0, 0.0));
    this->t = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->r0 = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->sloppy_g_in = SloppyGaugFieldType(this->dirac_op.g_in.dimensions,
                                            complexsingle_t(0, 0));
    this->sloppy_dirac = SloppyDiracOpT(sloppy_g_in, this->dirac_op.params);
    this->solver_type = SolverType::KLFT_SOLVER_BICGSTAB;
  }
  void init_int() {
    this->delta = 0.1;
    this->temp_D_full_complexity =
        SpinorFieldType(this->dims, complex_t(0.0, 0.0));

    this->apk = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->temp_D = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->pk = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->x_sloppy = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->r_sloppy = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->norm_per_site =
        typename DeviceScalarFieldType<rank>::type(this->dims, 0.0);
    this->dot_product_per_site =
        typename DeviceFieldType<rank>::type(this->dims, complex_t(0.0, 0.0));
    this->t = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
    this->r0 = SloppySpinorField(this->dims, complex_t(0.0, 0.0));
  }
  void init_gauge() {
    if (!this->sloppy_g_in.field.is_allocated()) {
      this->sloppy_g_in = SloppyGaugFieldType(this->dirac_op.g_in.dimensions,
                                              complexsingle_t(0, 0));
    }
    if (!this->sloppy_dirac_set) {
      this->sloppy_dirac = SloppyDiracOpT(sloppy_g_in, this->dirac_op.params);
      this->sloppy_dirac_set = true;
    }
  }
  SpinorFieldType get_temp_field_init() { return this->temp_D_full_complexity; }
  BiCGStabMultiP(const SpinorFieldType& b,
                 SpinorFieldType& x,
                 const Base::DiracOp& dirac_op,
                 SpinorFieldType& xk,
                 SpinorFieldType& rk,
                 SloppySpinorField& apk,
                 SloppySpinorField& temp_D,
                 SloppySpinorField& pk,
                 SloppySpinorField& t,
                 SloppySpinorField& r0,
                 SpinorFieldType& temp_D_full_complexity,
                 const SloppyGaugFieldType& sloppy_g_in,
                 const real_t& delta,
                 typename DeviceScalarFieldType<rank>::type& norm_per_site,
                 typename DeviceFieldType<rank>::type(dot_product_per_site))
      : Base(b, x, dirac_op, xk, rk),
        delta(delta),
        apk(apk),
        temp_D(temp_D),
        pk(pk),
        norm_per_site(norm_per_site),
        t(t),
        r0(r0),
        dot_product_per_site(dot_product_per_site),
        temp_D_full_complexity(temp_D_full_complexity),
        sloppy_g_in(sloppy_g_in) {
    this->sloppy_dirac = SloppyDiracOpT(sloppy_g_in, this->dirac_op.params);
  }

 private:
  SloppySpinorField r_sloppy;
  SloppySpinorField x_sloppy;
  SloppySpinorField apk;
  SloppySpinorField pk;
  SloppySpinorField temp_D;
  SpinorFieldType temp_D_full_complexity;
  real_t delta;
  SloppyGaugFieldType sloppy_g_in;
  SloppyDiracOpT sloppy_dirac;
  bool sloppy_dirac_set = false;
  SloppySpinorField t;
  SloppySpinorField r0;  // not sure about this one

  typename DeviceScalarFieldType<rank>::type norm_per_site;
  typename DeviceFieldType<rank>::type dot_product_per_site;
};
}  // namespace klft
