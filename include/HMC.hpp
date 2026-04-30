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
#include <random>

#include "AdjointFieldHelper.hpp"
#include "FermionMonomial.hpp"
#include "FermionMonomialEO.hpp"
#include "FermionMonomialEOHasenbusch.hpp"
#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeMonomial.hpp"
#include "Gauge_Util.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "IOParams.hpp"
#include "Integrator.hpp"
#include "Monomial.hpp"
namespace klft {

template <typename DGaugeFieldType, typename DAdjFieldType, class RNG>
class HMC {
 public:
  // template argument deduction and safety
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  typedef DGaugeFieldType DeviceGaugeFieldType;
  typedef DAdjFieldType DeviceAdjFieldType;
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeFieldType = typename DGaugeFieldType::type;
  using AdjFieldType = typename DAdjFieldType::type;

  struct randomize_momentum_s {};
  Integrator_Params params;
  HamiltonianField<DGaugeFieldType, DAdjFieldType>& hamiltonian_field;
  std::vector<std::unique_ptr<Monomial<DGaugeFieldType, DAdjFieldType>>>
      monomials;
  std::shared_ptr<Integrator> integrator;
  RNG rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;
  real_t delta_H;
  IOParams ioParams;

  HMC() = default;

  HMC(const Integrator_Params params_,
      const IOParams ioParams_,
      HamiltonianField<DGaugeFieldType, DAdjFieldType>& hamiltonian_field_,
      std::shared_ptr<Integrator> integrator_,
      RNG rng_,
      std::uniform_real_distribution<real_t> dist_,
      std::mt19937 mt_)
      : params(params_),
        rng(rng_),
        dist(dist_),
        mt(mt_),
        hamiltonian_field(hamiltonian_field_),
        integrator(std::move(integrator_)),
        ioParams(ioParams_) {}

  void add_gauge_monomial(const real_t _beta, const unsigned int _time_scale) {
    monomials.emplace_back(
        std::make_unique<GaugeMonomial<DGaugeFieldType, DAdjFieldType>>(
            _beta, _time_scale));
  }

  void add_kinetic_monomial(const unsigned int _time_scale) {
    monomials.emplace_back(
        std::make_unique<KineticMonomial<DGaugeFieldType, DAdjFieldType>>(
            _time_scale));
  }
  template <template <class DiracOpT> class _Solver, class DiracOpT>
  void add_fermion_monomial(
      typename DiracOpT::DSpinorFieldType::type& spinorField,
      const diracParams& params_,
      const real_t& tol_,
      RNG& rng,
      const unsigned int _time_scale) {
    monomials.emplace_back(
        std::make_unique<
            FermionMonomial<RNG, DAdjFieldType, _Solver, DiracOpT>>(
            spinorField, params_, tol_, rng, _time_scale));
  }
  template <template <class DiracOpT> class _Solver, class DiracOpT>
  void add_fermion_monomialEO(
      typename DiracOpT::DSpinorFieldType::type& spinorField,
      const diracParams& params_,
      const real_t& tol_,
      RNG& rng,
      const unsigned int _time_scale) {
    monomials.emplace_back(
        std::make_unique<
            FermionMonomialEO<RNG, DAdjFieldType, _Solver, DiracOpT>>(
            spinorField, params_, tol_, rng, _time_scale));
  }
  template <template <class DiracOpT> class _Solver,
            class DiracOpT,
            class DiracOpNonShift>
  void add_fermion_monomialEOHasenbusch(
      typename DiracOpT::DSpinorFieldType::type& spinorField,
      const diracParams& params_light,

      const real_t& tol_,
      RNG& rng,
      const unsigned int _time_scale) {
    monomials.emplace_back(
        std::make_unique<FermionMonomialEOHasenbusch<
            RNG, DAdjFieldType, _Solver, DiracOpT, DiracOpNonShift>>(
            spinorField, params_light, tol_, rng, _time_scale));
  }

  bool hmc_step(const bool& check_Reversibility = false) {
    Kokkos::fence();
    hamiltonian_field.randomize_momentum(rng);
    // print_SUNAdj(hamiltonian_field.adjoint_field(0, 0, 0, 0, 0),
    //              "Randomized Momentum");

    Kokkos::fence();
    GaugeFieldType gauge_old(hamiltonian_field.gauge_field.dimensions,
                             complex_t(0));
    Kokkos::deep_copy(gauge_old.field, hamiltonian_field.gauge_field.field);
    Kokkos::fence();
    for (int i = 0; i < monomials.size(); ++i) {
      monomials[i]->heatbath(hamiltonian_field);
    }
    integrator->integrate(params.tau, false);
    Kokkos::fence();
    restoreSUN<DGaugeFieldType>(hamiltonian_field.gauge_field);
    delta_H = 0.0;
    for (int i = 0; i < monomials.size(); ++i) {
      monomials[i]->accept(hamiltonian_field);
      delta_H += monomials[i]->get_delta_H();
      if (KLFT_VERBOSITY > 1) {
        monomials[i]->print();
      }

      // Kokkos::printf("delta_H_monomial: %.20f\n",
      // monomials[i]->get_delta_H());
    }
    // Kokkos::printf("delta_H_ges %.20f \n", delta_H);
    bool accept = true;
    if (delta_H > 0.0) {
      if (dist(mt) > Kokkos::exp(-delta_H)) {
        accept = false;
      }
    }
    if (check_Reversibility) {
      GaugeFieldType gauge_save(hamiltonian_field.gauge_field.dimensions,
                                complex_t(0));
      Kokkos::deep_copy(gauge_save.field, hamiltonian_field.gauge_field.field);
      Kokkos::fence();
      // for (int i = 0; i < monomials.size(); ++i) {
      //   monomials[i]->reset();
      //   monomials[i]->heatbath(hamiltonian_field);
      // }
      real_t delta_H_revers = 0;
      flip_sign<DAdjFieldType>(hamiltonian_field.adjoint_field);
      integrator->integrate(params.tau, false);

      for (int i = 0; i < monomials.size(); ++i) {
        monomials[i]->accept(hamiltonian_field);
        delta_H_revers += monomials[i]->get_delta_H();
        if (KLFT_VERBOSITY > 0) {
          printf("ReverseCheck Monomial\n");
          monomials[i]->print();
        }
      }
      Kokkos::printf("Deltadelta_H_ges %.20f \n", delta_H_revers);
      Kokkos::deep_copy(hamiltonian_field.gauge_field.field, gauge_save.field);
    }
    if (!accept) {
      Kokkos::deep_copy(hamiltonian_field.gauge_field.field, gauge_old.field);
    }

    return accept;
  }
};

// compile time check for the appropriate types
template <typename T>
struct isHMCClass : std::false_type {};

template <size_t rank,
          size_t Nc,
          GaugeFieldKind k,
          class RNG,
          typename precision>
struct isHMCClass<HMC<DeviceGaugeFieldType<rank, Nc, precision, k>,
                      DeviceAdjFieldType<rank, Nc>,
                      RNG>> : std::true_type {};

}  // namespace klft
